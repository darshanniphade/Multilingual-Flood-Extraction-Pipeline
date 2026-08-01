"""Translate the ``title`` field of an article dataset into English.

This program walks a dataset directory recursively, reads every ``.json`` and
``.jsonl`` file, and translates *only* the ``title`` field of each article into
English using a local NLLB translation model (GPU if available, otherwise CPU).

For every article it writes exactly three fields to ``translated_titles.jsonl``:

    {
        "article_id":       "2022_01/article_000094047",
        "original_title":   "...",
        "translated_title": "..."
    }

Nothing else is read, matched, filtered or classified here. The resulting file
is meant to be consumed later by a keyword-search stage that operates only on
``translated_title``.

All input/output paths and tuning knobs come from ``config.json`` next to this
file; command-line flags exist only to override a value for a one-off run.

Performance design - the goal is a permanently busy GPU, so every stage that is
not the GPU runs somewhere else:

  * **Reading** happens on a pool of threads; the files are tiny and numerous,
    so this stage is disk-bound and threads are enough.
  * **Language detection** happens in a pool of separate *processes*. The
    dataset carries no ``language`` field, so every title has to be detected,
    and ``langdetect`` is pure Python: on threads the GIL would serialise it and
    starve the GPU (this is what pinned the card at ~5% utilisation). Processes
    sidestep the GIL entirely and put every core to work.
  * **Translation** runs on its own thread fed by a small queue of ready
    buffers, so the GPU never waits for the reader/detector stages to catch up.
  * **Writing** runs on yet another thread, so translation never stalls to save.
  * Batches are sized by **KV-cache budget** rather than a fixed count, so short
    titles pack into much larger batches than long ones and VRAM stays full
    either way; anything that still overshoots is split on out-of-memory.
  * The model is loaded strictly from the local cache (no network round-trips),
    and the output file itself doubles as the resume checkpoint.

Run with the project virtual environment, e.g.::

    C:\\darsh\\AI_MODELS\\translator_env\\Scripts\\python.exe translate_titles.py
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import signal
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

# Configure HuggingFace before importing transformers. Offline mode stops the
# library from contacting the Hub (which otherwise tries to fetch an alternative
# safetensors copy and stalls on the network); Xet/symlink tweaks are Windows
# robustness. The model is already fully present in the local cache.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

from tqdm import tqdm

# torch and transformers are imported lazily, by _import_backend(). The
# detection workers are separate processes, and on Windows each one re-imports
# this module to find its worker function - keeping the heavy imports out of
# module scope means those workers start in milliseconds and never touch CUDA.
torch: Any = None
AutoModelForSeq2SeqLM: Any = None
AutoTokenizer: Any = None


def _import_backend(cfg: "Config") -> None:
    """Import torch/transformers once, in the parent process only."""
    global torch, AutoModelForSeq2SeqLM, AutoTokenizer
    if torch is not None:
        return
    if cfg.expandable_segments:
        # Batch sizes vary a lot here (KV budgeting), and expandable segments
        # keep the allocator from fragmenting across those size changes.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch as _torch
    from transformers import AutoModelForSeq2SeqLM as _model_cls
    from transformers import AutoTokenizer as _tokenizer_cls
    from transformers.utils import logging as hf_logging

    # Silence the per-generation "max_new_tokens vs max_length" notice so it does
    # not spam the log once per batch across a long run.
    hf_logging.set_verbosity_error()
    torch = _torch
    AutoModelForSeq2SeqLM = _model_cls
    AutoTokenizer = _tokenizer_cls


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Every tunable lives in this file, alongside the program.
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")

# File extensions that may hold articles.
DATA_EXTENSIONS = (".json", ".jsonl")

# How often to refresh the progress-bar postfix (file / gpu / counters).
POSTFIX_EVERY = 2_000

# An input folder named like this (e.g. ``.../ssd/2022``) supplies the
# ``article_id`` prefix when the config asks for "auto".
YEAR_DIR_RE = re.compile(r"^\d{4}$")

# Fallbacks used when a key is absent from config.json, so an older or trimmed
# config still runs. See config.json for what each one means.
CONFIG_DEFAULTS: dict[str, Any] = {
    "paths": {
        "input_dir": r"C:\darsh\pipeline\data\ssd\2022",
        "output_dir": r"C:\darsh\pipeline\data\translated titles\2022",
        "output_file": "translated_titles.jsonl",
        "articles_subdir": "articles",
    },
    "article_id": {"prefix": "auto", "separator": "_"},
    "fields": {"id": "article_id", "title": "title", "language": "language"},
    "language": {"detect_when_missing": True, "detect_min_chars": 10},
    "model": {
        "name": "facebook/nllb-200-distilled-1.3B",
        # NLLB FLORES-200 target code for English.
        "target_language": "eng_Latn",
        # Titles are short; cap input/generation length for speed and safety.
        "max_input_tokens": 128,
        "max_new_tokens": 48,
    },
    "gpu": {
        # One slot = one cached token = 96 KiB for this model, so ~110k slots
        # targets ~10.5 GB of KV cache alongside the ~2.6 GB of weights.
        "kv_slot_budget": 110_000,
        # Fragmentation guard on the per-decode-step logits buffer, not a
        # throughput knob.
        "max_batch_size": 1536,
        "attn_implementation": "sdpa",
        "expandable_segments": True,
    },
    "cpu": {
        # Threads parsing the (many, tiny) json files - a disk-bound stage.
        "readers": 24,
        # Processes running language detection - the CPU-bound stage.
        "detect_workers": 20,
        # Titles handed to a detection worker in one round trip.
        "detect_chunk": 4096,
    },
    "runtime": {
        # Titles pooled and length-sorted before a translation pass. Bigger
        # pools pad less per batch and keep the GPU busier.
        "translate_chunk": 16_384,
        # Ready pools kept queued for the GPU thread.
        "buffer_queue": 3,
        # Article hand-off queue bound. Readers preload the dataset into this
        # RAM queue, then go idle so the GPU thread gets the machine to itself.
        "queue_maxsize": 2_000_000,
        # Finished-record queue handed to the background writer thread.
        "write_queue_maxsize": 200_000,
        # Durability vs speed: a light OS-level flush is cheap and frequent; the
        # costly fsync (real disk sync) happens rarely so it never bottlenecks.
        "flush_every": 10_000,
        "checkpoint_every": 50_000,
        "limit": None,
    },
}

# ISO 639-1 (and a few common variants) -> NLLB FLORES-200 code.
# Titles whose language is unknown/unmapped are copied through unchanged so the
# original text is still searchable, rather than risking a wrong translation.
LANG_TO_FLORES: dict[str, str] = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "nl": "nld_Latn",
    "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl",
    "pl": "pol_Latn",
    "tr": "tur_Latn",
    "ar": "arb_Arab",
    "fa": "pes_Arab",
    "he": "heb_Hebr",
    "iw": "heb_Hebr",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ur": "urd_Arab",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "zh": "zho_Hans",
    "zh-cn": "zho_Hans",
    "zh-hans": "zho_Hans",
    "zh-tw": "zho_Hant",
    "zh-hant": "zho_Hant",
    "sv": "swe_Latn",
    "no": "nob_Latn",
    "nb": "nob_Latn",
    "nn": "nno_Latn",
    "da": "dan_Latn",
    "fi": "fin_Latn",
    "cs": "ces_Latn",
    "sk": "slk_Latn",
    "hu": "hun_Latn",
    "ro": "ron_Latn",
    "bg": "bul_Cyrl",
    "el": "ell_Grek",
    "hr": "hrv_Latn",
    "sr": "srp_Cyrl",
    "sl": "slv_Latn",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    "et": "est_Latn",
    "ca": "cat_Latn",
    "gl": "glg_Latn",
    "eu": "eus_Latn",
    "is": "isl_Latn",
    "ga": "gle_Latn",
    "cy": "cym_Latn",
    "sq": "als_Latn",
    "mk": "mkd_Cyrl",
    "af": "afr_Latn",
    "sw": "swh_Latn",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ml": "mal_Mlym",
    "kn": "kan_Knda",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "pa": "pan_Guru",
    "ne": "npi_Deva",
    "si": "sin_Sinh",
    "my": "mya_Mymr",
    "km": "khm_Khmr",
    "lo": "lao_Laoo",
    "ka": "kat_Geor",
    "am": "amh_Ethi",
    "hy": "hye_Armn",
    "az": "azj_Latn",
    "kk": "kaz_Cyrl",
    "uz": "uzn_Latn",
    "mn": "khk_Cyrl",
    "be": "bel_Cyrl",
    "tl": "tgl_Latn",
    "fil": "tgl_Latn",
    "ku": "kmr_Latn",
    "ps": "pbt_Arab",
    "sd": "snd_Arab",
    "yi": "ydd_Hebr",
    "jv": "jav_Latn",
    "jw": "jav_Latn",
    "su": "sun_Latn",
    "ceb": "ceb_Latn",
}

# Sentinels for the inter-thread queues.
_QUEUE_DONE = object()
_WRITER_STOP = object()
_GPU_STOP = object()


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class Config:
    """Everything the run needs, read from ``config.json``."""

    input_dir: Path
    output_path: Path
    articles_subdir: str

    id_prefix: str
    id_prefix_setting: str      # raw config value, kept so "auto" can re-resolve
    id_separator: str

    id_field: str
    title_field: str
    language_field: str

    detect_when_missing: bool
    detect_min_chars: int
    detect_workers: int
    detect_chunk: int

    model_name: str
    target_language: str
    max_input_tokens: int
    max_new_tokens: int

    kv_slot_budget: int
    max_batch_size: int
    attn_implementation: str
    expandable_segments: bool

    readers: int
    translate_chunk: int
    buffer_queue: int
    queue_maxsize: int
    write_queue_maxsize: int
    flush_every: int
    checkpoint_every: int
    limit: Optional[int]


def _section(raw: dict, name: str) -> dict:
    """Merge one config section over its defaults (missing keys fall back)."""
    merged = dict(CONFIG_DEFAULTS[name])
    value = raw.get(name)
    if isinstance(value, dict):
        merged.update({k: v for k, v in value.items() if not k.startswith("_")})
    return merged


def resolve_id_prefix(setting: str, input_dir: Path) -> str:
    """Work out the ``article_id`` prefix, expanding the "auto" setting.

    "auto" means: use the input folder's own name when it looks like a year
    (``.../ssd/2022`` -> ``2022``), otherwise no prefix.
    """
    setting = (setting or "").strip()
    if setting.lower() != "auto":
        return setting
    name = input_dir.name
    return name if YEAR_DIR_RE.match(name) else ""


def load_config(path: Path) -> Config:
    """Read and validate ``config.json`` into a :class:`Config`."""
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        raise SystemExit(f"[error] config file not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[error] config file '{path}' is not valid JSON: {exc}")
    if not isinstance(raw, dict):
        raise SystemExit(f"[error] config file '{path}' must contain a JSON object")

    paths = _section(raw, "paths")
    article_id = _section(raw, "article_id")
    fields = _section(raw, "fields")
    language = _section(raw, "language")
    model = _section(raw, "model")
    gpu = _section(raw, "gpu")
    cpu = _section(raw, "cpu")
    runtime = _section(raw, "runtime")

    input_dir = Path(str(paths["input_dir"])).expanduser()
    output_dir = Path(str(paths["output_dir"])).expanduser()

    limit = runtime.get("limit")
    prefix_setting = str(article_id["prefix"])
    return Config(
        input_dir=input_dir,
        output_path=output_dir / str(paths["output_file"]),
        articles_subdir=str(paths["articles_subdir"]),
        id_prefix=resolve_id_prefix(prefix_setting, input_dir),
        id_prefix_setting=prefix_setting,
        id_separator=str(article_id["separator"]),
        id_field=str(fields["id"]),
        title_field=str(fields["title"]),
        language_field=str(fields["language"]),
        detect_when_missing=bool(language["detect_when_missing"]),
        detect_min_chars=int(language["detect_min_chars"]),
        detect_workers=int(cpu["detect_workers"]),
        detect_chunk=int(cpu["detect_chunk"]),
        model_name=str(model["name"]),
        target_language=str(model["target_language"]),
        max_input_tokens=int(model["max_input_tokens"]),
        max_new_tokens=int(model["max_new_tokens"]),
        kv_slot_budget=int(gpu["kv_slot_budget"]),
        max_batch_size=int(gpu["max_batch_size"]),
        attn_implementation=str(gpu["attn_implementation"]),
        expandable_segments=bool(gpu["expandable_segments"]),
        readers=int(cpu["readers"]),
        translate_chunk=int(runtime["translate_chunk"]),
        buffer_queue=int(runtime["buffer_queue"]),
        queue_maxsize=int(runtime["queue_maxsize"]),
        write_queue_maxsize=int(runtime["write_queue_maxsize"]),
        flush_every=int(runtime["flush_every"]),
        checkpoint_every=int(runtime["checkpoint_every"]),
        limit=int(limit) if limit else None,
    )


@dataclass
class Article:
    """A single article's identity, its title, and its source language.

    ``flores`` is the NLLB FLORES-200 code the title should be translated from,
    or ``None`` when the language is unknown and the title must be copied
    through unchanged. ``needs_detect`` marks the ones still waiting on a
    detection worker to fill ``flores`` in.
    """

    article_id: str
    title: str
    flores: Optional[str]
    needs_detect: bool
    source_file: str


@dataclass
class Stats:
    """Running counters displayed in the progress bar and final summary."""

    processed: int = 0          # articles read & handled (drives the progress bar)
    translated: int = 0         # titles actually sent through the model
    copied: int = 0             # english / unknown-language / empty titles copied
    detected: int = 0           # languages recovered by detecting from the title
    undetected: int = 0         # titles whose language stayed unknown
    file_errors: int = 0        # files that failed to open/parse
    record_errors: int = 0      # individual records that failed to parse
    batch_errors: int = 0       # translation batches that fell back to copy
    oom_splits: int = 0         # batches halved after a CUDA out-of-memory
    start_time: float = field(default_factory=time.time)


# --------------------------------------------------------------------------- #
# Language helpers
# --------------------------------------------------------------------------- #


def to_flores(language: Optional[str]) -> Optional[str]:
    """Map a dataset language code to an NLLB FLORES-200 code.

    Returns ``None`` when the language is missing or not recognised, signalling
    that the title should be copied through untouched.
    """
    if not language:
        return None
    code = language.strip().lower()
    if code in LANG_TO_FLORES:
        return LANG_TO_FLORES[code]
    # Fall back to the primary subtag, e.g. "pt-br" -> "pt".
    if "-" in code:
        base = code.split("-", 1)[0]
        return LANG_TO_FLORES.get(base)
    return None


# Bound once per worker process by _detect_init().
_worker_detect: Any = None


def _detect_init() -> None:
    """Per-process setup: load the langdetect profiles exactly once."""
    global _worker_detect
    from langdetect import DetectorFactory, detect

    # Make detection reproducible across runs, and force the lazy profile load
    # to happen now rather than on the first (timed) title.
    DetectorFactory.seed = 0
    try:
        detect("warm up the profile cache")
    except Exception:  # noqa: BLE001 - a failed warm-up is not fatal
        pass
    _worker_detect = detect


def _detect_chunk(titles: list[str]) -> list[Optional[str]]:
    """Detect a whole chunk of titles inside a worker process.

    Chunked rather than one-per-call so the inter-process round trip is
    amortised over thousands of titles.
    """
    out: list[Optional[str]] = []
    for text in titles:
        try:
            out.append(_worker_detect(text))
        except Exception:  # noqa: BLE001 - undetectable titles are not errors
            out.append(None)
    return out


class LanguageResolver:
    """Decides which FLORES-200 code a title should be translated from.

    The declared ``language`` field is trusted first, in-process and for free.
    Datasets that do not carry one (the 2022 ``ssd`` export is one - every
    record holds only ``article_id``, ``title`` and ``text``) would otherwise
    have every title copied through untranslated, so the language is detected
    from the title text itself.

    That detection is the single most CPU-hungry stage of the run and
    ``langdetect`` is pure Python, so it is farmed out to a pool of worker
    *processes*: on threads the GIL serialises it and the GPU sits idle.

    Detection is deliberately conservative - very short titles are not worth
    guessing at, and anything that stays unknown is copied through unchanged
    rather than translated from a wrong source language.
    """

    def __init__(self, cfg: Config, stats: Stats) -> None:
        self.stats = stats
        self.enabled = cfg.detect_when_missing
        self.min_chars = cfg.detect_min_chars
        self.pool: Optional[ProcessPoolExecutor] = None
        if not self.enabled:
            return
        try:
            import langdetect  # noqa: F401  - checked here for a clear error
        except ImportError:
            raise SystemExit(
                "[error] language detection is enabled but 'langdetect' is not "
                "installed. Install it (pip install langdetect) or set "
                "language.detect_when_missing to false in config.json."
            )
        workers = max(1, cfg.detect_workers)
        self.pool = ProcessPoolExecutor(
            max_workers=workers, initializer=_detect_init
        )

    def classify(self, title: str, declared: str) -> tuple[Optional[str], bool]:
        """Resolve what can be resolved cheaply, on the calling thread.

        Returns ``(flores, needs_detect)``: a code straight from the declared
        language when there is one, otherwise a request for detection.
        """
        flores = to_flores(declared)
        if flores is not None:
            return flores, False
        if not self.enabled or len(title.strip()) < self.min_chars:
            return None, False
        return None, True

    def resolve_pending(self, pending: list[Article]) -> None:
        """Fill in ``flores`` for every article still awaiting detection.

        Blocks on the worker pool, which releases the GIL - so the calling
        reader thread stops competing for it while detection runs.
        """
        todo = [a for a in pending if a.needs_detect]
        if not todo or self.pool is None:
            return
        codes = self.pool.submit(
            _detect_chunk, [a.title.strip() for a in todo]
        ).result()
        detected = 0
        for article, code in zip(todo, codes):
            article.flores = to_flores(code)
            article.needs_detect = False
            if article.flores is not None:
                detected += 1
        self.stats.detected += detected
        self.stats.undetected += len(todo) - detected

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.pool = None


# --------------------------------------------------------------------------- #
# Model / translation
# --------------------------------------------------------------------------- #


class Translator:
    """Wraps the NLLB model and performs cross-language batched translation."""

    def __init__(self, cfg: Config, stats: Stats) -> None:
        _import_backend(cfg)
        self.stats = stats
        self.max_input_tokens = cfg.max_input_tokens
        self.max_new_tokens = cfg.max_new_tokens
        self.target_language = cfg.target_language
        self.kv_slot_budget = cfg.kv_slot_budget
        self.max_batch_size = cfg.max_batch_size
        self.attn_implementation = cfg.attn_implementation
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda":
            self.dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            # Faster matmuls on Ada GPUs; harmless if unsupported.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        else:
            self.dtype = torch.float32

        print(f"[model] loading '{cfg.model_name}' on {self.device} ({self.dtype}) ...")
        t0 = time.time()
        self.tokenizer, self.model = self._load(cfg.model_name)
        self.target_id = self.tokenizer.convert_tokens_to_ids(self.target_language)
        self.pad_id = self.tokenizer.pad_token_id
        print(f"[model] ready in {time.time() - t0:.1f}s")
        if self.device == "cuda":
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(
                f"[model] {torch.cuda.get_device_name(0)} - {total:.1f} GB total, "
                f"batching to a {self.kv_slot_budget} KV-slot budget "
                f"(~{self.kv_slot_budget * 96 / 1024**2:.1f} GB of cache)"
            )

    def _load(self, model_name: str):
        """Load tokenizer+model from the local cache, downloading only if needed."""
        for local_only in (True, False):
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, local_files_only=local_only
                )
                kwargs = {
                    "low_cpu_mem_usage": True,
                    "local_files_only": local_only,
                    "attn_implementation": self.attn_implementation,
                }
                try:
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name, dtype=self.dtype, **kwargs
                    )
                except TypeError:
                    # Older transformers use the legacy dtype keyword.
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name, torch_dtype=self.dtype, **kwargs
                    )
                model.to(self.device).eval()
                return tokenizer, model
            except Exception as exc:  # noqa: BLE001
                if local_only:
                    tqdm.write(
                        f"[model] not fully cached ({exc}); trying to download ..."
                    )
                    continue
                raise
        raise RuntimeError("unreachable")

    def _tokenize_by_language(
        self, buffer: list[tuple[str, str, str]]
    ) -> list[list[int]]:
        """Tokenize each title with its own source language token.

        NLLB prepends a source-language token, so tokenisation is grouped by
        language (a fast batched call per language, run in parallel by the Rust
        tokenizer) while the resulting token ids can then be freely mixed into
        one generation batch.
        """
        input_ids: list[Optional[list[int]]] = [None] * len(buffer)
        by_lang: dict[str, list[int]] = {}
        for idx, (_, _, flores) in enumerate(buffer):
            by_lang.setdefault(flores, []).append(idx)

        for flores, indices in by_lang.items():
            self.tokenizer.src_lang = flores
            texts = [buffer[i][1] for i in indices]
            encoded = self.tokenizer(
                texts, truncation=True, max_length=self.max_input_tokens
            )["input_ids"]
            for i, ids in zip(indices, encoded):
                input_ids[i] = ids
        return input_ids  # type: ignore[return-value]

    def _plan_batches(
        self, order: list[int], tokenized: list[list[int]]
    ) -> list[list[int]]:
        """Group length-sorted titles into batches that fit the KV budget.

        A fixed batch size wastes VRAM on short titles and overflows it on long
        ones. Budgeting by cache slots instead - ``rows x (source + generated
        tokens)`` - means a batch of 20-token headlines can be several times
        wider than a batch of 128-token ones while both peak at the same VRAM.
        """
        batches: list[list[int]] = []
        current: list[int] = []
        longest = 0
        for index in order:
            length = len(tokenized[index])
            widest = length if length > longest else longest
            slots = (len(current) + 1) * (widest + self.max_new_tokens)
            if current and (
                slots > self.kv_slot_budget or len(current) >= self.max_batch_size
            ):
                batches.append(current)
                current = [index]
                longest = length
            else:
                current.append(index)
                longest = widest
        if current:
            batches.append(current)
        return batches

    def _generate_ids(self, ids_list: list[list[int]]) -> list[str]:
        """Pad a list of token-id sequences into one batch and translate."""
        max_len = max(len(ids) for ids in ids_list)
        batch = len(ids_list)
        input_ids = torch.full((batch, max_len), self.pad_id, dtype=torch.long)
        attention = torch.zeros((batch, max_len), dtype=torch.long)
        for row, ids in enumerate(ids_list):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention[row, : len(ids)] = 1
        if self.device == "cuda":
            # Pinned staging + async copy lets the host queue the next batch
            # while this one is still crossing the bus.
            input_ids = input_ids.pin_memory()
            attention = attention.pin_memory()
        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=input_ids.to(self.device, non_blocking=True),
                attention_mask=attention.to(self.device, non_blocking=True),
                forced_bos_token_id=self.target_id,
                max_new_tokens=self.max_new_tokens,
                num_beams=1,
                use_cache=True,
            )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def _generate_adaptive(self, ids_list: list[list[int]]) -> list[str]:
        """Generate, halving the batch and retrying on CUDA out-of-memory."""
        try:
            return self._generate_ids(ids_list)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and len(ids_list) > 1:
                self.stats.oom_splits += 1
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                mid = len(ids_list) // 2
                return self._generate_adaptive(ids_list[:mid]) + self._generate_adaptive(
                    ids_list[mid:]
                )
            raise

    def translate_buffer(
        self, buffer: list[tuple[str, str, str]]
    ) -> Iterator[tuple[str, str, str]]:
        """Translate a mixed-language buffer.

        ``buffer`` holds ``(article_id, original_title, flores_code)`` tuples,
        all of which require translation. Yields ``(article_id, original_title,
        translated_title)`` incrementally, one batch at a time, so the caller
        can report progress as each batch completes instead of only at the end.
        Titles are packed length-sorted across languages; on any failure the
        original title is used as a safe fallback so the run continues.
        """
        stats = self.stats
        tokenized = self._tokenize_by_language(buffer)
        # Sort by token length so each batch pads to a similar width.
        order = sorted(range(len(buffer)), key=lambda i: len(tokenized[i]))

        for window in self._plan_batches(order, tokenized):
            ids_list = [tokenized[i] for i in window]
            try:
                decoded = self._generate_adaptive(ids_list)
                stats.translated += len(window)
            except Exception as exc:  # noqa: BLE001 - keep going on any model error
                stats.batch_errors += 1
                tqdm.write(
                    f"[warn] translation failed for a batch of {len(window)} "
                    f"titles: {exc}. Falling back to originals."
                )
                decoded = [buffer[i][1] for i in window]
            for i, text in zip(window, decoded):
                article_id, original, _ = buffer[i]
                yield article_id, original, text

    def gpu_memory_str(self) -> str:
        """Human-readable CUDA memory usage, or empty string on CPU."""
        if self.device != "cuda":
            return ""
        reserved = torch.cuda.memory_reserved() / 1024**3
        allocated = torch.cuda.memory_allocated() / 1024**3
        return f"{allocated:.1f}/{reserved:.1f}GB"


# --------------------------------------------------------------------------- #
# Dataset reading
# --------------------------------------------------------------------------- #


def discover_files(root: Path, subdir: str = "") -> list[Path]:
    """Recursively find all ``.json`` and ``.jsonl`` files under ``root``.

    When ``subdir`` is set only files living inside a folder of that name are
    returned. The ssd export keeps crawler bookkeeping next to the articles in
    every month folder (``progress.json``, ``dns_cache.json``, ``metrics/``,
    ``logs/``); those are json but not article data, so restricting the scan to
    ``articles`` keeps them out of the dataset.
    """
    files: list[Path] = []
    wanted = subdir.strip().lower()
    for dirpath, _dirnames, filenames in os.walk(root):
        if wanted:
            parts = Path(dirpath).relative_to(root).parts
            if not any(part.lower() == wanted for part in parts):
                continue
        for name in filenames:
            if name.lower().endswith(DATA_EXTENSIONS):
                files.append(Path(dirpath) / name)
    files.sort()
    return files


def id_prefix(path: Path, cfg: Config) -> str:
    """The ``article_id`` prefix for a file, e.g. ``2022_01``.

    The raw ``article_id`` restarts numbering in every month folder, so it is
    prefixed with the configured dataset prefix plus the month folder the file
    lives under, forming a globally-unique, traceable id.
    """
    try:
        rel = path.relative_to(cfg.input_dir)
    except ValueError:
        return cfg.id_prefix
    group = rel.parts[0] if len(rel.parts) > 1 else ""
    if cfg.id_prefix and group:
        # Some exports name the month folder with the year already in it
        # (``2023/2023_01``) — don't repeat it, ids stay ``2023_01/...``.
        if group == cfg.id_prefix or group.startswith(
            f"{cfg.id_prefix}{cfg.id_separator}"
        ):
            return group
        return f"{cfg.id_prefix}{cfg.id_separator}{group}"
    return cfg.id_prefix or group


def _articles_from_obj(
    obj: object,
    source_file: str,
    prefix: str,
    cfg: Config,
    languages: LanguageResolver,
) -> Iterator[Article]:
    """Yield ``Article`` records from a parsed JSON object (dict or list).

    ``prefix`` (dataset + month folder) is prepended to each ``article_id`` so
    the stored id is unique across the whole dataset.
    """
    if isinstance(obj, dict):
        records: list[object] = [obj]
    elif isinstance(obj, list):
        records = obj
    else:
        return

    stem = Path(source_file).stem
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        raw_id = record.get(cfg.id_field)
        if not raw_id:
            # Fall back to a stable id derived from the file / position.
            raw_id = stem if len(records) == 1 else f"{stem}#{index}"
        article_id = f"{prefix}/{raw_id}" if prefix else str(raw_id)
        title = record.get(cfg.title_field)
        title = title if isinstance(title, str) else ""
        language = record.get(cfg.language_field)
        language = language if isinstance(language, str) else ""
        flores, needs_detect = languages.classify(title, language)
        yield Article(article_id, title, flores, needs_detect, source_file)


def iter_articles(
    path: Path, stats: Stats, cfg: Config, languages: LanguageResolver
) -> Iterator[Article]:
    """Yield all articles contained in a single file, tolerating bad data."""
    prefix = id_prefix(path, cfg)
    name = path.name
    try:
        # utf-8-sig transparently strips a leading BOM if present, and behaves
        # exactly like utf-8 otherwise, so no file is dropped over a stray BOM.
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        stats.record_errors += 1
                        continue
                    yield from _articles_from_obj(obj, name, prefix, cfg, languages)
        else:
            with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
                obj = json.load(handle)
            yield from _articles_from_obj(obj, name, prefix, cfg, languages)
    except (OSError, json.JSONDecodeError) as exc:
        stats.file_errors += 1
        tqdm.write(f"[warn] skipping unreadable file '{path}': {exc}")


# --------------------------------------------------------------------------- #
# Resume / output
# --------------------------------------------------------------------------- #


def load_done_ids(output_path: Path) -> set[str]:
    """Read already-translated article ids from a previous run (for resume)."""
    done: set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Tolerate a torn final line from an interrupted run.
                continue
            article_id = record.get("article_id")
            if article_id:
                done.add(str(article_id))
    return done


class BackgroundWriter:
    """Append-only JSONL writer running on its own thread.

    Finished records are handed over through an in-memory queue, so the
    translation loop never blocks on disk I/O. A cheap OS flush happens often
    and a durable fsync happens rarely.
    """

    def __init__(self, cfg: Config) -> None:
        self.path = cfg.output_path
        self.flush_every = cfg.flush_every
        self.checkpoint_every = cfg.checkpoint_every
        self.handle = self.path.open("a", encoding="utf-8", buffering=1 << 20)
        self.queue: "queue.Queue" = queue.Queue(maxsize=cfg.write_queue_maxsize)
        self.written = 0
        self._thread = threading.Thread(target=self._run, name="writer", daemon=True)
        self._thread.start()

    def submit(self, article_id: str, original: str, translated: str) -> None:
        self.queue.put((article_id, original, translated))

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            if item is _WRITER_STOP:
                return
            article_id, original, translated = item
            record = {
                "article_id": article_id,
                "original_title": original,
                "translated_title": translated,
            }
            self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.written += 1
            if self.written % self.flush_every == 0:
                self.handle.flush()
            if self.written % self.checkpoint_every == 0:
                os.fsync(self.handle.fileno())

    def close(self) -> None:
        """Drain the queue, sync to disk and close the file."""
        self.queue.put(_WRITER_STOP)
        self._thread.join()
        try:
            self.handle.flush()
            os.fsync(self.handle.fileno())
        finally:
            self.handle.close()


# --------------------------------------------------------------------------- #
# Parallel readers
# --------------------------------------------------------------------------- #


def start_readers(
    files: list[Path],
    cfg: Config,
    languages: LanguageResolver,
    article_q: "queue.Queue",
    done_ids: set[str],
    stats: Stats,
) -> None:
    """Spawn reader threads that parse files and feed ``article_q``.

    Each reader batches its articles up to ``detect_chunk`` and hands the ones
    with no usable language to the detection processes in a single round trip,
    then forwards the whole batch. A watcher thread joins the readers and pushes
    a single sentinel so the consumer knows when every file has been read.
    """
    next_index = {"i": 0}
    index_lock = threading.Lock()

    def claim_file() -> Optional[Path]:
        with index_lock:
            i = next_index["i"]
            if i >= len(files):
                return None
            next_index["i"] = i + 1
            return files[i]

    def reader_worker() -> None:
        pending: list[Article] = []

        def dispatch() -> None:
            if not pending:
                return
            languages.resolve_pending(pending)
            for article in pending:
                article_q.put(article)
            pending.clear()

        while True:
            path = claim_file()
            if path is None:
                break
            for article in iter_articles(path, stats, cfg, languages):
                # Cheap pre-skip of work already done in a previous run.
                if article.article_id in done_ids:
                    continue
                pending.append(article)
                if len(pending) >= cfg.detect_chunk:
                    dispatch()
        dispatch()

    readers = [
        threading.Thread(target=reader_worker, name=f"reader-{n}", daemon=True)
        for n in range(cfg.readers)
    ]
    for thread in readers:
        thread.start()

    def watcher() -> None:
        for thread in readers:
            thread.join()
        article_q.put(_QUEUE_DONE)

    threading.Thread(target=watcher, name="reader-watcher", daemon=True).start()


# --------------------------------------------------------------------------- #
# Main processing loop
# --------------------------------------------------------------------------- #


def process(
    files: list[Path],
    cfg: Config,
    languages: LanguageResolver,
    translator: Translator,
    writer: BackgroundWriter,
    done_ids: set[str],
    stats: Stats,
    total_estimate: int,
) -> None:
    """Stream every article through translation and hand results to the writer.

    Three stages run at once: reader threads (plus the detection processes)
    fill ``article_q``, this thread sorts articles into copy-through or
    translate and packs the latter into buffers, and a dedicated GPU thread
    drains those buffers. The GPU therefore keeps working while the next buffer
    is being assembled.
    """
    article_q: "queue.Queue" = queue.Queue(maxsize=cfg.queue_maxsize)
    start_readers(files, cfg, languages, article_q, done_ids, stats)

    buffer_q: "queue.Queue" = queue.Queue(maxsize=cfg.buffer_queue)
    abort = threading.Event()
    progress = tqdm(
        total=total_estimate,
        initial=len(done_ids),
        unit="title",
        dynamic_ncols=True,
        smoothing=0.1,
        mininterval=0.5,
    )
    state = {"file": "", "next_postfix": POSTFIX_EVERY}
    # The fill thread and the GPU thread both report completed titles.
    progress_lock = threading.Lock()

    def advance() -> None:
        # Advance the bar per *completed* title so it never freezes during a
        # translation batch (it moves one batch-worth at a time), and the rate /
        # ETA reflect real finished work.
        with progress_lock:
            stats.processed += 1
            progress.update(1)
            if stats.processed >= state["next_postfix"]:
                state["next_postfix"] += POSTFIX_EVERY
                progress.set_postfix(
                    file=state["file"],
                    tr=stats.translated,
                    cp=stats.copied,
                    gpu=translator.gpu_memory_str() or "cpu",
                    q=buffer_q.qsize(),
                    refresh=False,
                )

    def gpu_worker() -> None:
        while True:
            item = buffer_q.get()
            if item is _GPU_STOP:
                return
            if abort.is_set():
                # Interrupted: drop whatever is still queued instead of
                # spending minutes finishing work that was cancelled.
                continue
            for article_id, original, translated in translator.translate_buffer(item):
                writer.submit(article_id, original, translated)
                advance()

    gpu_thread = threading.Thread(target=gpu_worker, name="gpu", daemon=True)
    gpu_thread.start()

    buffer: list[tuple[str, str, str]] = []
    queued = len(done_ids)
    try:
        while True:
            article = article_q.get()
            if article is _QUEUE_DONE:
                break
            state["file"] = article.source_file

            if article.article_id in done_ids:
                continue
            done_ids.add(article.article_id)
            queued += 1

            flores = article.flores
            if (
                not article.title.strip()
                or flores is None
                or flores == cfg.target_language
            ):
                # Empty, already English, or unknown language: copy through.
                writer.submit(article.article_id, article.title, article.title)
                stats.copied += 1
                advance()
            else:
                buffer.append((article.article_id, article.title, flores))
                if len(buffer) >= cfg.translate_chunk:
                    buffer_q.put(buffer)
                    buffer = []

            if cfg.limit is not None and queued >= cfg.limit:
                break

        # Hand over whatever is left in the buffer.
        if buffer:
            buffer_q.put(buffer)
    except KeyboardInterrupt:
        abort.set()
        raise
    finally:
        buffer_q.put(_GPU_STOP)
        gpu_thread.join()
        progress.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Translate only the 'title' field of an article dataset to English. "
            "Settings come from config.json; the flags below override it for one run."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the JSON config file.",
    )
    # Every override defaults to None so "not given" is distinguishable from a
    # value that happens to match the config.
    parser.add_argument("--input-dir", default=None, help="Override paths.input_dir.")
    parser.add_argument("--output-dir", default=None, help="Override paths.output_dir.")
    parser.add_argument("--output-file", default=None, help="Override paths.output_file.")
    parser.add_argument(
        "--articles-subdir",
        default=None,
        help=(
            "Override paths.articles_subdir — only read files inside folders of "
            "this name. Pass \"\" to scan every folder."
        ),
    )
    parser.add_argument("--model", default=None, help="Override model.name.")
    parser.add_argument(
        "--kv-budget", type=int, default=None, help="Override gpu.kv_slot_budget."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override gpu.max_batch_size."
    )
    parser.add_argument(
        "--readers", type=int, default=None, help="Override cpu.readers."
    )
    parser.add_argument(
        "--detect-workers", type=int, default=None, help="Override cpu.detect_workers."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after roughly this many new articles (for a quick test run).",
    )
    parser.add_argument(
        "--no-detect",
        action="store_true",
        help="Disable detecting the language from the title when the field is missing.",
    )
    return parser.parse_args(argv)


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Fold any command-line overrides into the loaded config."""
    if args.input_dir is not None:
        cfg.input_dir = Path(args.input_dir).expanduser()
        # An "auto" prefix was derived from the old input folder's name.
        cfg.id_prefix = resolve_id_prefix(cfg.id_prefix_setting, cfg.input_dir)
    if args.output_dir is not None or args.output_file is not None:
        directory = (
            Path(args.output_dir).expanduser()
            if args.output_dir is not None
            else cfg.output_path.parent
        )
        name = args.output_file if args.output_file is not None else cfg.output_path.name
        cfg.output_path = directory / name
    if args.articles_subdir is not None:
        cfg.articles_subdir = args.articles_subdir
    if args.model is not None:
        cfg.model_name = args.model
    if args.kv_budget is not None:
        cfg.kv_slot_budget = args.kv_budget
    if args.batch_size is not None:
        cfg.max_batch_size = args.batch_size
    if args.readers is not None:
        cfg.readers = args.readers
    if args.detect_workers is not None:
        cfg.detect_workers = args.detect_workers
    if args.limit is not None:
        cfg.limit = args.limit
    if args.no_detect:
        cfg.detect_when_missing = False
    return cfg


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    config_path = Path(args.config).expanduser()
    print(f"[config] {config_path}")
    cfg = apply_overrides(load_config(config_path), args)

    if not cfg.input_dir.is_dir():
        print(f"[error] input directory not found: {cfg.input_dir}", file=sys.stderr)
        return 1
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[config] input  : {cfg.input_dir}")
    print(f"[config] output : {cfg.output_path}")
    print(f"[config] id form: {cfg.id_prefix or '<none>'}<sep>MM/<article_id>")

    # Make Ctrl+C raise KeyboardInterrupt promptly even during long batches.
    signal.signal(signal.SIGINT, signal.default_int_handler)

    scope = f"*/{cfg.articles_subdir}" if cfg.articles_subdir.strip() else "every folder"
    print(f"[scan] discovering files under {cfg.input_dir} ({scope}) ...")
    files = discover_files(cfg.input_dir, cfg.articles_subdir)
    print(f"[scan] found {len(files)} json/jsonl files")
    if not files:
        print("[done] nothing to do.")
        return 0

    done_ids = load_done_ids(cfg.output_path)
    if done_ids:
        print(f"[resume] {len(done_ids)} articles already translated; skipping them.")

    # Each .json holds one article; .jsonl may hold many. Files vastly dominate
    # here, so the file count is a good ETA basis without a second scan.
    total_estimate = len(files)

    stats = Stats()
    # Built before the model so the worker processes fork/spawn off a small
    # parent that has not yet loaded torch or touched CUDA.
    languages = LanguageResolver(cfg, stats)
    if cfg.detect_when_missing:
        print(
            f"[lang] detecting source language from the title on "
            f"{cfg.detect_workers} worker processes "
            f"({os.cpu_count()} cores available)"
        )
    print(f"[cpu] {cfg.readers} reader threads")
    translator = Translator(cfg, stats)
    writer = BackgroundWriter(cfg)

    try:
        process(
            files=files,
            cfg=cfg,
            languages=languages,
            translator=translator,
            writer=writer,
            done_ids=done_ids,
            stats=stats,
            total_estimate=total_estimate,
        )
    except KeyboardInterrupt:
        print("\n[interrupt] stopping; progress is saved. Re-run to resume.")
        return 130
    finally:
        writer.close()
        languages.close()

    elapsed = time.time() - stats.start_time
    speed = stats.processed / elapsed if elapsed > 0 else 0.0
    print(
        "\n[summary]\n"
        f"  articles processed : {stats.processed}\n"
        f"  titles translated  : {stats.translated}\n"
        f"  titles copied      : {stats.copied}\n"
        f"  languages detected : {stats.detected}\n"
        f"  language unknown   : {stats.undetected}\n"
        f"  file errors        : {stats.file_errors}\n"
        f"  record errors      : {stats.record_errors}\n"
        f"  batch fallbacks    : {stats.batch_errors}\n"
        f"  oom batch splits   : {stats.oom_splits}\n"
        f"  elapsed            : {elapsed:.1f}s\n"
        f"  speed              : {speed:.1f} titles/s\n"
        f"  output             : {cfg.output_path.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
