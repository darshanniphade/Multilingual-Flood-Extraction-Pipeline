"""
translator.py
=============

The GPU half of the pipeline: loads NLLB-200-distilled-1.3B from a local
directory and translates length-sorted, language-homogeneous batches of chunks
into English.

Tuning notes (measured on this exact machine -- RTX 4060 Ti 16 GB, Windows/WDDM,
torch 2.11+cu128, transformers 5.13)
------------------------------------------------------------------------------
Two measurements shaped this module and are worth stating plainly, because both
contradict the obvious approach:

1. **Filling VRAM makes it slower, not faster.** Under WDDM the NVIDIA driver
   silently spills oversized allocations into host RAM instead of raising OOM.
   A batch whose peak "VRAM" reached 21.8 GiB on a 16 GiB card did not crash --
   it ran at 617 out-tok/s versus 5509 for a batch that fit. Throughput peaks at
   a padded-token budget of ~24-32k (peak ~7.5-9.3 GiB) and then *degrades*.
   So the goal is the throughput knee, not maximum occupancy.

2. **``set_per_process_memory_fraction`` restores real OOM.** Capping the
   allocator turns that silent spill into a catchable ``OutOfMemoryError``,
   which is what makes "reduce the batch and continue" work at all. Verified:
   after a caught OOM the model still produces correct output.

Consequently the batcher targets the measured knee and adapts within a band,
rather than probing for the largest batch that does not crash.

Measured throughput reference (uniform chunks, greedy decode):

===========  ============  ===============  ==========
chunk tokens  token budget  out tokens/sec   peak VRAM
===========  ============  ===============  ==========
48            24576         6935             7.6 GiB
64            32768         6073             8.4 GiB
128           32768         4714             8.8 GiB
192           49152         3880             10.7 GiB
===========  ============  ===============  ==========
"""

from __future__ import annotations

import gc
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Sequence

# Allocator tuning must be set before the CUDA allocator initialises (i.e.
# before the first CUDA call), hence above the torch import rather than in main().
#
# Every decode step allocates and frees a ~batch x 256206 logits buffer (~700 MiB
# at batch 463), which fragments the caching allocator: a profile showed 2.13 GiB
# "reserved but unallocated" and an OOM while only 11.6 GiB was live.
#
# NOTE: `expandable_segments:True` -- PyTorch's own suggested remedy -- is
# **silently unsupported on Windows** (it warns "not supported on this platform"
# and is ignored), so it is deliberately NOT set here. On Linux, adding it is
# worthwhile.
#
# Absent that, fragmentation is bounded by `gpu.max_batch_size`, because the
# logits buffer scales with batch *width* alone. At 512 it is ~750 MiB churned
# per decode step; at 256, ~376 MiB. The logits term inside utils.batch_kv_slots
# does NOT bound this -- at batch 512 it is 9% of the slot cost, far too little
# to bind -- so the explicit width cap is what does the work.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.9")

import torch  # noqa: E402  (import order is deliberate -- see above)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: E402

from utils import (  # noqa: E402
    Chunk,
    FLORES_200,
    LOGITS_SLOTS_PER_SEQ,
    TARGET_LANG,
    batch_kv_slots,
    build_batches,
    kv_bytes_per_slot,
    projected_max_new,
    slots_for_vram,
    vram_for_slots,
)

log = logging.getLogger("translator.model")


@dataclass(slots=True)
class BatchResult:
    """Outcome of translating one batch."""

    texts: list[str]
    src_tokens: int
    out_tokens: int
    seconds: float
    batch_size: int
    retries: int
    oom: bool = False


class NLLBTranslator:
    """Offline NLLB-200 translator with adaptive, OOM-safe dynamic batching.

    Parameters
    ----------
    model_dir:
        Local directory holding ``config.json``/``model.safetensors``/tokenizer.
        Nothing is ever downloaded; ``local_files_only=True`` is forced.
    cfg:
        The ``gpu`` + ``generation`` sections of ``config.json``.
    """

    def __init__(self, model_dir: str | os.PathLike[str], cfg: dict[str, Any]) -> None:
        self.model_dir = os.fspath(model_dir)
        self.cfg = cfg

        # Budget is in KV-cache slots (batch x (src + generated)), not source
        # tokens -- see utils.batch_kv_slots for why that distinction matters.
        self.slot_budget = int(cfg.get("kv_slot_budget", 90000))
        self.min_slot_budget = int(cfg.get("min_kv_slot_budget", 4096))
        self.max_slot_budget = int(cfg.get("max_kv_slot_budget", 120000))
        self.max_batch_size = int(cfg.get("max_batch_size", 512))
        self.target_vram_gib = float(cfg.get("target_peak_vram_gib", 9.5))
        self.max_retries = int(cfg.get("max_retries", 4))
        self.num_beams = int(cfg.get("num_beams", 1))
        self.len_ratio = float(cfg.get("max_new_token_ratio", 1.8))
        self.len_pad = int(cfg.get("max_new_token_pad", 24))
        self.hard_max_new = int(cfg.get("hard_max_new_tokens", 384))

        # The length we *reserve KV for* is deliberately not the length we *allow*
        # (len_ratio above). See _budget_max_new.
        self.budget_ratio = float(cfg.get("budget_len_ratio", 1.25))
        self.budget_pad = int(cfg.get("budget_len_pad", 16))
        self.oom_count = 0
        self._src_lang: str | None = None

        self._assert_offline_model()
        self._load()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def _assert_offline_model(self) -> None:
        """Verify the checkpoint is present locally before touching the network."""
        needed = ("config.json", "tokenizer_config.json")
        missing = [f for f in needed if not os.path.exists(os.path.join(self.model_dir, f))]
        weights = [
            f for f in os.listdir(self.model_dir)
            if f.endswith((".safetensors", ".bin")) or f.startswith("model-0")
        ] if os.path.isdir(self.model_dir) else []
        if missing or not weights:
            raise FileNotFoundError(
                f"No usable local NLLB checkpoint in {self.model_dir!r}. "
                f"Missing: {missing or 'model weights'}. "
                "The pipeline runs fully offline and will not download the model; "
                "point config.json:model.dir at a complete local checkpoint."
            )
        size_gib = sum(
            os.path.getsize(os.path.join(self.model_dir, f)) for f in weights
        ) / 1024 ** 3
        log.info("model: found local checkpoint (%.2f GiB) at %s", size_gib, self.model_dir)

    def _load(self) -> None:
        """Load tokenizer + model onto CUDA in float16."""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required but torch.cuda.is_available() is False. "
                "This pipeline is CUDA-only by design (see README)."
            )

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)

        attn = self.cfg.get("attn_implementation", "sdpa")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_dir,
            dtype=torch.float16,          # transformers>=5 spelling (was torch_dtype)
            device_map="cuda",
            local_files_only=True,
            attn_implementation=attn,
        )
        self.model.eval()

        # generation_config ships max_length=200, which otherwise fights our
        # explicit max_new_tokens and emits a warning on every single batch.
        self.model.generation_config.max_length = None
        self.model.generation_config.early_stopping = False

        self.device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        self.total_vram_gib = props.total_memory / 1024 ** 3

        # Cap the allocator so oversized batches raise a catchable OOM instead of
        # silently spilling to host RAM (see module docstring).
        frac = float(self.cfg.get("memory_fraction", 0.88))
        if 0 < frac < 1:
            torch.cuda.set_per_process_memory_fraction(frac)
            log.info("gpu: allocator capped at %.0f%% of %.1f GiB (=%.1f GiB) "
                     "so OOM is raised instead of spilling to host RAM",
                     frac * 100, self.total_vram_gib, self.total_vram_gib * frac)

        if self.cfg.get("tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self._eng_id = self._lang_token_id(TARGET_LANG)

        # Cross-check the static table in utils against this checkpoint.
        vocab_langs = {
            t for t in getattr(self.tokenizer, "additional_special_tokens", []) or []
            if len(t) == 8 and t[3] == "_"
        }
        if vocab_langs:
            drift = FLORES_200 ^ vocab_langs
            if drift:
                log.warning("model: language table drift vs tokenizer (%d code(s)): %s",
                            len(drift), sorted(drift)[:8])

        log.info("model: loaded in %.1fs | %s | %s | attn=%s | weights %.2f GiB",
                 time.time() - t0, self.model.dtype, props.name,
                 self.model.config._attn_implementation,
                 torch.cuda.memory_allocated() / 1024 ** 3)
        log.info("gpu: alloc_conf=%s | vocab=%s (%.2f MiB of logits per sequence, per step)",
                 os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<default>"),
                 f"{self.model.config.vocab_size:,}",
                 self.model.config.vocab_size * 6 / 1024 ** 2)

        if self.cfg.get("autotune", True):
            self._autotune_budget()

    def _lang_token_id(self, code: str) -> int:
        """Resolve a FLORES language code to its token id.

        ``lang_code_to_id`` was removed in modern transformers; the supported
        path is a plain vocabulary lookup.
        """
        tid = self.tokenizer.convert_tokens_to_ids(code)
        unk = getattr(self.tokenizer, "unk_token_id", None)
        if tid is None or tid == unk:
            raise KeyError(f"language code {code!r} is not in the tokenizer vocabulary")
        return int(tid)

    # ------------------------------------------------------------------ #
    # Batching + generation
    # ------------------------------------------------------------------ #

    def _budget_max_new(self, max_src: int) -> int:
        """Output length to **reserve KV for** -- not the cap handed to generate().

        These are two different questions and conflating them cost throughput:

        * ``projected_max_new`` (ratio 1.8) is a *correctness* bound. Hit it and a
          translation is silently truncated, so it must stay generous.
        * This is a *memory* estimate. KV is allocated lazily (``DynamicCache``
          cats one step at a time), so a batch's real peak tracks the length it
          actually generates, not the length it was permitted to.

        Measured over 6,336 articles: 2.4M source tokens produced 1.9M output
        tokens, an aggregate ratio of **0.79** -- NLLB output is *shorter* than
        its input here. Reserving at 1.8 therefore over-books every batch by
        ~2.3x against reality, which shrinks batch width for exactly the long
        chunks where width is worth most.

        Reserving at 1.25 keeps ~1.6x headroom over the measured mean. A batch
        that overshoots simply OOMs and is split -- cheap now that the budget
        actually binds (see _split_to_budget), and the whole point of capping the
        allocator so OOM is catchable in the first place.
        """
        return projected_max_new(max_src, ratio=self.budget_ratio,
                                 pad=self.budget_pad, hard_cap=self.hard_max_new)

    def plan(self, chunks: Sequence[Chunk]) -> list[list[Chunk]]:
        """Group *chunks* into language-homogeneous, length-sorted batches.

        NLLB encodes the source language as a prefix token, so a batch must not
        mix languages. Within a language, chunks are sorted by token length so
        padding -- and therefore wasted decode steps -- stays minimal.
        """
        by_lang: dict[str, list[Chunk]] = {}
        for ch in chunks:
            by_lang.setdefault(ch.lang, []).append(ch)

        batches: list[list[Chunk]] = []
        for lang, items in by_lang.items():
            # build_batches only ever *costs* a batch; it never feeds generate().
            # So its length ratio is a budgeting parameter, and takes the
            # reservation ratio rather than the correctness cap.
            batches.extend(build_batches(
                items,
                slot_budget=self.slot_budget,
                max_batch_size=self.max_batch_size,
                out_ratio=self.budget_ratio,
                out_pad=self.budget_pad,
                hard_max_new=self.hard_max_new,
            ))
        # Heaviest batches first: if a budget is too aggressive, find out early
        # (and adapt) rather than after hours of successful small batches.
        batches.sort(
            key=lambda b: batch_kv_slots(
                len(b), max(c.n_tokens for c in b),
                self._budget_max_new(max(c.n_tokens for c in b)),
            ),
            reverse=True,
        )
        return batches

    def _set_src_lang(self, lang: str) -> None:
        if self._src_lang != lang:
            self.tokenizer.src_lang = lang
            self._src_lang = lang

    def translate_batch(self, batch: Sequence[Chunk]) -> BatchResult:
        """Translate one homogeneous batch, halving it on OOM until it fits.

        Never raises for OOM: on repeated failure the batch is split and retried
        down to single items, and a chunk that still cannot be translated yields
        an empty string so the caller can record the failure and continue.
        """
        if not batch:
            return BatchResult([], 0, 0, 0.0, 0, 0)

        lang = batch[0].lang
        self._set_src_lang(lang)
        max_src = max(c.n_tokens for c in batch)
        max_new = projected_max_new(max_src, ratio=self.len_ratio, pad=self.len_pad,
                                    hard_cap=self.hard_max_new)

        retries = 0
        pieces: list[Sequence[Chunk]] = self._split_to_budget(batch)
        out: list[str] = []
        src_tokens = out_tokens = 0
        t0 = time.time()
        oom_seen = False

        while pieces:
            piece = pieces.pop(0)
            try:
                got, s_tok, o_tok = self._generate(piece, max_new)
                out.extend(got)
                src_tokens += s_tok
                out_tokens += o_tok
            except torch.cuda.OutOfMemoryError:
                oom_seen = True
                self.oom_count += 1
                self._recover()
                if len(piece) == 1:
                    # A single chunk that will not fit: give up on it, keep going.
                    log.error("oom: single chunk of %d tokens will not fit; skipping",
                              piece[0].n_tokens)
                    out.append("")
                    retries += 1
                    continue
                retries += 1
                mid = len(piece) // 2
                pieces.insert(0, piece[mid:])
                pieces.insert(0, piece[:mid])
                self._shrink_budget()
                log.warning("oom: split batch %d -> %d+%d | kv_slot_budget now %d",
                            len(piece), mid, len(piece) - mid, self.slot_budget)

        secs = time.time() - t0
        if not oom_seen:
            self._grow_budget()
        return BatchResult(out, src_tokens, out_tokens, secs, len(batch), retries, oom_seen)

    def _generate(self, batch: Sequence[Chunk], max_new: int) -> tuple[list[str], int, int]:
        """Tokenise, generate and decode a batch. May raise ``OutOfMemoryError``."""
        enc = self.tokenizer(
            [c.text for c in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.cfg.get("chunk_max_tokens", 192)) + 8,
        )
        # pin_memory + non_blocking overlaps the H2D copy with GPU work.
        enc = {
            k: v.pin_memory().to(self.device, non_blocking=True) if v.is_cpu else v
            for k, v in enc.items()
        }
        src_tokens = int(enc["attention_mask"].sum().item())

        with torch.inference_mode():
            gen = self.model.generate(
                **enc,
                forced_bos_token_id=self._eng_id,
                max_new_tokens=max_new,
                num_beams=self.num_beams,
                do_sample=False,
                use_cache=True,
            )
        pad = self.tokenizer.pad_token_id
        out_tokens = int((gen != pad).sum().item())
        texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        return [t.strip() for t in texts], src_tokens, out_tokens

    # ------------------------------------------------------------------ #
    # Adaptive budget
    # ------------------------------------------------------------------ #

    def _split_to_budget(self, batch: Sequence[Chunk]) -> list[Sequence[Chunk]]:
        """Split *batch* into pieces that each fit the budget in force *now*.

        ``plan()`` sizes a whole block's batches up front, using the budget as it
        stood when the block began. An OOM part-way through the block lowers
        ``slot_budget``, but nothing re-plans the batches already queued behind
        it -- so without this, a shrunk budget never reaches them.

        That is not hypothetical: a production block logged the budget decaying
        90000 -> 60000 -> ... -> 4096 (its floor) while batch size stayed pinned
        at 512, because every batch had been planned at 90000. Each one paid a
        failed 512-wide forward pass, an ``empty_cache()``, and only then split
        into 256+256. Roughly half the block's wall-clock went to forward passes
        whose results were discarded.

        Splitting on the way in makes the adaptive budget actually adaptive: the
        first OOM in a block is the only one the block pays for.
        """
        if len(batch) <= 1:
            return [batch]
        max_src = max(c.n_tokens for c in batch)
        per_seq = max_src + self._budget_max_new(max_src) + LOGITS_SLOTS_PER_SEQ
        width = max(1, min(self.max_batch_size, self.slot_budget // max(1, per_seq)))
        if width >= len(batch):
            return [batch]
        pieces = [batch[i:i + width] for i in range(0, len(batch), width)]
        log.debug("presplit: %d -> %d piece(s) of <=%d at budget %d",
                  len(batch), len(pieces), width, self.slot_budget)
        return pieces

    def _autotune_budget(self) -> None:
        """Derive the KV-slot budget from the checkpoint's real memory shape.

        Replaces a hand-set ``kv_slot_budget`` with arithmetic over this GPU and
        this model, which is what makes the budget correct rather than merely
        calibrated for one machine::

            usable = total * memory_fraction - weights - logits - activations
            budget = usable / kv_bytes_per_slot

        ``config.json`` may still pin ``kv_slot_budget`` explicitly; set
        ``gpu.autotune`` to false to use it verbatim.
        """
        mc = self.model.config
        self.bytes_per_slot = kv_bytes_per_slot(mc.decoder_layers, mc.d_model)

        # Widest batch we allow, times the fp32 logits copy generate() makes each
        # decode step. This is transient, but it is live at peak, so it is booked.
        logits_gib = self.max_batch_size * mc.vocab_size * 6 / 1024 ** 3
        reserve_gib = float(self.cfg.get("activation_reserve_gib", 0.75))
        weights_gib = torch.cuda.memory_allocated() / 1024 ** 3

        usable = self.target_vram_gib - weights_gib - logits_gib - reserve_gib
        if usable <= 0:
            log.warning("gpu: no KV headroom at target %.1f GiB (weights %.2f + "
                        "logits %.2f + reserve %.2f); keeping budget %d",
                        self.target_vram_gib, weights_gib, logits_gib,
                        reserve_gib, self.slot_budget)
            return

        tuned = slots_for_vram(usable, bytes_per_slot=self.bytes_per_slot)
        self.slot_budget = min(tuned, self.max_slot_budget)
        log.info("gpu: autotuned budget %d slots (%.0f KiB/slot -> KV %.1f GiB) | "
                 "target %.1f = weights %.2f + logits %.2f + reserve %.2f + KV %.1f",
                 self.slot_budget, self.bytes_per_slot / 1024, usable,
                 self.target_vram_gib, weights_gib, logits_gib, reserve_gib, usable)
        log.info("gpu: widest batch predicted at %.1f GiB peak",
                 vram_for_slots(self.slot_budget, weights_gib=weights_gib,
                                bytes_per_slot=self.bytes_per_slot) + logits_gib)

    def _recover(self) -> None:
        """Clear CUDA caches after an OOM so the next attempt has room."""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def _shrink_budget(self) -> None:
        """Halve the KV budget after an OOM."""
        self.slot_budget = max(self.min_slot_budget, self.slot_budget // 2)

    def _grow_budget(self) -> None:
        """Creep the budget back up while measured peak VRAM stays under target.

        Growth is deliberately slow (+12.5%) and bounded by *max_slot_budget*.
        The gate is the *measured* peak from the last batch, not a prediction, so
        this self-corrects if the slot model drifts from reality on some batch
        shape -- which is exactly the failure that motivated
        :func:`utils.vram_for_slots`.

        Occupancy is not the goal. Under WDDM an oversized batch does not fail
        fast, it spills into host RAM and crawls (measured: 617 out-tok/s at a
        21.8 GiB peak versus 5509 for a batch that fit), so a high VRAM gauge is
        a symptom to avoid rather than a target to hit.
        """
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        if peak < self.target_vram_gib * 0.75 and self.slot_budget < self.max_slot_budget:
            self.slot_budget = min(self.max_slot_budget, int(self.slot_budget * 1.125) + 256)
        torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def vram(self) -> tuple[float, float]:
        """Return ``(allocated_gib, peak_gib)``."""
        return (torch.cuda.memory_allocated() / 1024 ** 3,
                torch.cuda.max_memory_allocated() / 1024 ** 3)

    def warmup(self, lang: str = "spa_Latn") -> None:
        """Run one tiny batch so the first real batch is not paying init costs."""
        try:
            self._set_src_lang(lang if lang in FLORES_200 else "spa_Latn")
            enc = self.tokenizer(["Hola mundo."], return_tensors="pt").to(self.device)
            with torch.inference_mode():
                self.model.generate(**enc, forced_bos_token_id=self._eng_id, max_new_tokens=8)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            log.info("gpu: warmup complete")
        except Exception as exc:  # pragma: no cover
            log.warning("gpu: warmup failed (%s) -- continuing", exc)

    def close(self) -> None:
        """Release the model and empty the CUDA cache."""
        try:
            del self.model
        except Exception:  # pragma: no cover
            pass
        gc.collect()
        torch.cuda.empty_cache()
