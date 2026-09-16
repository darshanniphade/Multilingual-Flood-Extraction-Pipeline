"""
dataset.py
==========

Everything between "a flood event file on disk" and "a batch of chunks ready
for the GPU". Deliberately **torch-free** so it can be imported by ``spawn``ed
multiprocessing workers without paying CUDA init costs in every child.

Responsibilities
----------------
* Read the flood-event JSONL files and collect the article ids to translate.
* Build (once) and cache an ``article_id -> file path`` index.
* Load/parse/normalise/chunk articles in a process pool, overlapping that CPU
  work with GPU inference via a prefetching block iterator.

Index strategy
--------------
The corpus stores articles at ``<data>/<month>/articles/<article_id>.json`` and
event ids are already ``"<month>/<article_id>"``, so a path can be *derived*
arithmetically. That means the index costs one ``os.path.exists`` per needed
article (parallelised) instead of a directory walk over ~1.25M files. A month
directory is scanned exactly once, and only if its convention-derived paths
miss. The result is cached to disk and reused on every subsequent run.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Callable, Iterator, Sequence

from utils import Chunk, detect_language, normalize_text, pack_chunks

log = logging.getLogger("translator.dataset")

# --------------------------------------------------------------------------- #
# Flood event loading
# --------------------------------------------------------------------------- #


def load_event_ids(events_dir: str | os.PathLike[str]) -> dict[str, str | None]:
    """Collect ``article_id -> translated_title`` from every flood event file.

    Handles both JSONL (one record per line -- what this corpus actually uses)
    and plain JSON files containing a list or a dict of records, since the spec
    describes the directory as "JSON files".

    Corrupt lines are skipped and counted rather than aborting the run.
    """
    events_dir = Path(events_dir)
    if not events_dir.is_dir():
        raise FileNotFoundError(f"flood events directory not found: {events_dir}")

    out: dict[str, str | None] = {}
    bad = 0
    files = sorted(p for p in events_dir.rglob("*") if p.suffix.lower() in (".jsonl", ".json"))
    if not files:
        raise FileNotFoundError(f"no .json/.jsonl files under {events_dir}")

    def take(rec: Any) -> None:
        nonlocal bad
        if not isinstance(rec, dict):
            bad += 1
            return
        aid = rec.get("article_id") or rec.get("id") or rec.get("articleId")
        if not aid:
            bad += 1
            return
        title = rec.get("translated_title")
        # Keep the first non-empty translated_title seen for an id.
        if aid not in out or (out[aid] in (None, "") and title):
            out[aid] = title

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                if path.suffix.lower() == ".jsonl":
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            take(json.loads(line))
                        except json.JSONDecodeError:
                            bad += 1
                else:
                    data = json.load(fh)
                    if isinstance(data, list):
                        for rec in data:
                            take(rec)
                    elif isinstance(data, dict):
                        # {"articles": [...]} or {"id": {...}} shapes
                        if any(isinstance(v, list) for v in data.values()):
                            for v in data.values():
                                if isinstance(v, list):
                                    for rec in v:
                                        take(rec)
                        else:
                            take(data)
        except (OSError, json.JSONDecodeError) as exc:
            log.error("event file unreadable, skipping: %s (%s)", path, exc)
            continue

    log.info(
        "flood events: %d file(s) -> %d unique article id(s)%s",
        len(files), len(out), f" ({bad} bad record(s) skipped)" if bad else "",
    )
    return out


# --------------------------------------------------------------------------- #
# Article index
# --------------------------------------------------------------------------- #


def _derive_path(data_dir: Path, article_id: str, subdir: str) -> Path | None:
    """Derive the on-disk path for ``"<month>/<article>"`` ids."""
    if "/" not in article_id:
        return None
    month, name = article_id.split("/", 1)
    if not month or not name or ".." in month or ".." in name:
        return None
    return data_dir / month / subdir / f"{name}.json"


def build_index(
    article_ids: Sequence[str],
    data_dir: str | os.PathLike[str],
    *,
    cache_path: str | os.PathLike[str] | None = None,
    articles_subdir: str = "articles",
    workers: int = 16,
    rebuild: bool = False,
    with_languages: bool = True,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build (or load) the ``article_id -> absolute path`` index.

    The index is built **once** and cached to *cache_path*; later runs reuse it
    and never touch the dataset directory again. Only ids not resolvable by the
    naming convention trigger a one-time scan of their month directory.

    Returns ``(index, lang_hints)`` where *index* contains only ids actually
    found on disk, and *lang_hints* maps id -> declared language code (empty
    dict when *with_languages* is False). See :func:`scan_languages` for why the
    hints matter so much for throughput.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"article dataset directory not found: {data_dir}")

    wanted = set(article_ids)

    # ---- reuse cache when it covers everything we need -------------------- #
    if cache_path and not rebuild and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as fh:
                cached = json.load(fh)
            idx = cached.get("index", {})
            missing = cached.get("missing", [])
            langs = cached.get("langs", {})
            covered = set(idx) | set(missing)
            if wanted <= covered and (not with_languages or langs):
                hit = {k: v for k, v in idx.items() if k in wanted}
                hints = {k: langs.get(k, "") for k in hit} if with_languages else {}
                log.info("index: loaded from cache (%s) -> %d path(s)", cache_path, len(hit))
                return hit, hints
            log.info("index: cache does not cover %d new id(s); rebuilding",
                     len(wanted - covered))
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            log.warning("index: cache unreadable (%s); rebuilding", exc)

    # ---- pass 1: derive paths by convention, verify existence in parallel -- #
    t0 = time.time()
    ids = sorted(wanted)
    derived: dict[str, Path] = {}
    unresolvable: list[str] = []
    for aid in ids:
        p = _derive_path(data_dir, aid, articles_subdir)
        if p is None:
            unresolvable.append(aid)
        else:
            derived[aid] = p

    index: dict[str, str] = {}
    misses: list[str] = []

    def check(item: tuple[str, Path]) -> tuple[str, Path, bool]:
        aid, path = item
        return aid, path, path.is_file()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for aid, path, ok in pool.map(check, derived.items(), chunksize=256):
            if ok:
                index[aid] = str(path)
            else:
                misses.append(aid)

    log.info(
        "index: convention resolved %d/%d id(s) in %.1fs",
        len(index), len(ids), time.time() - t0,
    )

    # ---- pass 2: one scan per month that still has misses ------------------ #
    stragglers = misses + unresolvable
    if stragglers:
        by_month: dict[str, list[str]] = {}
        for aid in stragglers:
            month = aid.split("/", 1)[0] if "/" in aid else ""
            by_month.setdefault(month, []).append(aid)

        log.info("index: scanning %d month dir(s) for %d unresolved id(s)",
                 len(by_month), len(stragglers))
        for month, aids in by_month.items():
            found = _scan_month(data_dir, month, articles_subdir)
            for aid in aids:
                name = aid.split("/", 1)[-1]
                hit = found.get(name) or found.get(aid)
                if hit:
                    index[aid] = hit

    missing = sorted(wanted - set(index))
    if missing:
        log.warning("index: %d article id(s) have no file on disk (first 5: %s)",
                    len(missing), missing[:5])

    # ---- language hints (drives batch-friendly ordering) ------------------- #
    hints: dict[str, str] = scan_languages(index, workers=workers) if with_languages else {}

    # ---- persist ---------------------------------------------------------- #
    if cache_path:
        try:
            os.makedirs(os.path.dirname(os.fspath(cache_path)) or ".", exist_ok=True)
            tmp = f"{cache_path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(
                    {"data_dir": str(data_dir), "built": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "index": index, "missing": missing, "langs": hints},
                    fh,
                )
            os.replace(tmp, cache_path)
            log.info("index: cached %d path(s) -> %s", len(index), cache_path)
        except OSError as exc:
            log.warning("index: could not write cache (%s)", exc)

    log.info("index: built %d path(s) in %.1fs", len(index), time.time() - t0)
    return index, hints


#: Matches the ``"language": "xx"`` field without parsing the whole document.
_LANG_RE = re.compile(rb'"language"\s*:\s*"([^"]*)"')

#: The language field sits near the top of these records (before the large
#: ``text`` field), so a small head read almost always finds it.
_LANG_HEAD_BYTES = 4096


def scan_languages(
    index: dict[str, str],
    *,
    workers: int = 32,
    cache_path: str | os.PathLike[str] | None = None,
    rebuild: bool = False,
) -> dict[str, str]:
    """Read each article's declared ``language`` field cheaply.

    Why this exists
    ---------------
    NLLB encodes the source language as a prefix token, so **a batch cannot mix
    languages**. This corpus spans ~55 languages, so processing articles in id
    order shatters every block into dozens of tiny per-language batches -- and
    small batches are catastrophically slow here (measured: ~550 out-tok/s at
    ~22 chunks/batch versus ~6900 at ~500). Sorting the work list by language
    makes each block nearly homogeneous and restores full-size batches.

    The value is only a *hint* used for ordering: real detection still happens
    per article in the workers, so a wrong or missing hint costs a little batch
    efficiency, never correctness.

    Implementation note: the field is extracted with a byte regex over the first
    few KB rather than a full ``json.loads``, which keeps a 233k-file pass to
    well under a minute and, being pure I/O, parallelises across threads.
    """
    if cache_path and not rebuild and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as fh:
                cached = json.load(fh)
            langs = cached.get("langs") or {}
            if langs and set(index) <= set(langs):
                log.info("languages: loaded %d hint(s) from cache", len(langs))
                return {k: langs[k] for k in index}
        except (OSError, json.JSONDecodeError, AttributeError):
            pass

    t0 = time.time()

    def one(item: tuple[str, str]) -> tuple[str, str]:
        aid, path = item
        try:
            with open(path, "rb") as fh:
                head = fh.read(_LANG_HEAD_BYTES)
                m = _LANG_RE.search(head)
                if m is None and len(head) == _LANG_HEAD_BYTES:
                    m = _LANG_RE.search(head + fh.read())
        except OSError:
            return aid, ""
        return aid, (m.group(1).decode("utf-8", "ignore") if m else "")

    out: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for aid, lang in pool.map(one, index.items(), chunksize=256):
            out[aid] = lang

    found = sum(1 for v in out.values() if v)
    log.info("languages: scanned %d article(s) in %.1fs (%d with a declared language)",
             len(out), time.time() - t0, found)
    return out


def _scan_month(data_dir: Path, month: str, articles_subdir: str) -> dict[str, str]:
    """Scan one month directory once, returning ``stem -> path``."""
    base = data_dir / month / articles_subdir if month else data_dir
    if not base.is_dir():
        base = data_dir / month
        if not base.is_dir():
            return {}
    out: dict[str, str] = {}
    try:
        with os.scandir(base) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(".json"):
                    out[entry.name[:-5]] = entry.path
    except OSError as exc:
        log.error("index: cannot scan %s (%s)", base, exc)
    return out


# --------------------------------------------------------------------------- #
# Article preparation (runs in worker processes)
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class PreparedArticle:
    """An article loaded, language-detected and split into chunks."""

    article_id: str
    path: str
    out_path: str
    data: dict[str, Any] = field(default_factory=dict)
    lang: str | None = None
    lang_source: str = "unknown"
    text_chunks: list[tuple[str, int]] = field(default_factory=list)
    title_chunks: list[tuple[str, int]] = field(default_factory=list)
    translated_title: str | None = None      # carried over from the event file
    status: str = "ok"                       # ok | skipped_english | corrupt | failed
    error: str | None = None

    @property
    def n_src_tokens(self) -> int:
        """Total source tokens across all chunks of this article."""
        return sum(n for _, n in self.text_chunks) + sum(n for _, n in self.title_chunks)


# Per-worker singletons, initialised lazily in each spawned process.
_WORKER: dict[str, Any] = {}


def _worker_init(model_dir: str, cfg: dict[str, Any]) -> None:
    """Initialise per-process state (tokenizer + config).

    The tokenizer is loaded once per worker. It is a *fast* (Rust-backed)
    tokenizer, measured at ~81k sentences/s, so counting exact token lengths on
    the CPU is far cheaper than the GPU time it saves through tight batching.
    """
    import warnings

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Workers only *count* tokens; they never run the model. Silence the noise
    # that would otherwise be emitted once per worker (torch's pynvml notice)
    # or once per long paragraph (the >1024 length notice -- harmless here,
    # because over-long units are hard-split immediately afterwards).
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    from transformers import AutoTokenizer  # local import: keeps parent import cheap
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()

    _WORKER["tok"] = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    _WORKER["cfg"] = cfg


def _count_tokens(text: str) -> int:
    """Exact NLLB source-token count for *text* (without special tokens)."""
    tok = _WORKER["tok"]
    return len(tok(text, add_special_tokens=False, verbose=False)["input_ids"])


def prepare_article(task: tuple[str, str, str, str | None]) -> PreparedArticle:
    """Load, validate, language-detect and chunk one article.

    Runs inside a worker process. Never raises: any failure is reported through
    the returned object's ``status``/``error`` so the pipeline can log and move
    on (the spec requires corrupt files be skipped, not fatal).

    Parameters
    ----------
    task:
        ``(article_id, source_path, output_path, translated_title_from_event)``
    """
    article_id, path, out_path, event_title = task
    cfg = _WORKER["cfg"]
    art = PreparedArticle(article_id=article_id, path=path, out_path=out_path,
                          translated_title=event_title)

    # ---- read -------------------------------------------------------------- #
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        art.status = "corrupt"
        art.error = f"{type(exc).__name__}: {exc}"
        return art
    if not isinstance(data, dict):
        art.status = "corrupt"
        art.error = "top-level JSON is not an object"
        return art
    art.data = data

    text_field = cfg.get("text_field", "text")
    title_field = cfg.get("title_field", "title")
    raw_text = data.get(text_field) or ""
    raw_title = data.get(title_field) or ""
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)
    if not isinstance(raw_title, str):
        raw_title = str(raw_title)

    # ---- language ---------------------------------------------------------- #
    probe = raw_text if len(raw_text) >= 40 else f"{raw_title} {raw_text}"
    lang, source = detect_language(
        probe,
        data.get(cfg.get("language_field", "language")),
        trust_metadata=cfg.get("trust_metadata_language", True),
    )
    art.lang, art.lang_source = lang, source

    # ---- English fast path ------------------------------------------------- #
    # ~39% of the flood set is already English. Round-tripping English through
    # NLLB costs GPU hours and can only degrade the text, so it is copied.
    if lang == "eng_Latn" and cfg.get("skip_english", True):
        art.status = "skipped_english"
        return art
    if lang is None:
        art.status = "failed"
        art.error = "language could not be determined"
        return art

    # ---- normalise + chunk ------------------------------------------------- #
    repair = cfg.get("repair_mojibake", True)
    target = int(cfg.get("chunk_target_tokens", 64))
    maxtok = int(cfg.get("chunk_max_tokens", 192))
    max_chars = int(cfg.get("max_article_chars", 0))

    try:
        text = normalize_text(raw_text, repair_mojibake=repair)
        if max_chars and len(text) > max_chars:
            text = text[:max_chars]
        if text:
            art.text_chunks = pack_chunks(text, _count_tokens,
                                          target_tokens=target, max_tokens=maxtok)
        # Only translate the title if the event file did not already supply one.
        if not art.translated_title and raw_title.strip():
            title = normalize_text(raw_title, repair_mojibake=repair)
            art.title_chunks = pack_chunks(title, _count_tokens,
                                           target_tokens=maxtok, max_tokens=maxtok)
    except Exception as exc:  # pragma: no cover - defensive
        art.status = "failed"
        art.error = f"chunking failed: {type(exc).__name__}: {exc}"
    return art


# --------------------------------------------------------------------------- #
# Prefetching block loader
# --------------------------------------------------------------------------- #


class ArticleLoader:
    """Streams blocks of :class:`PreparedArticle` from a process pool.

    A background thread keeps *prefetch* blocks in flight so that CPU work
    (JSON parse, language detection, sentence splitting, tokenisation) overlaps
    with GPU inference on the previous block. The queue bounds memory: at most
    ``prefetch * block_size`` articles are resident at once.

    Blocks -- rather than a flat stream -- are the unit of work because an
    article can only be written once *all* of its chunks are translated, and
    pooling a whole block's chunks is what makes length-sorted batching
    effective.
    """

    def __init__(
        self,
        tasks: Sequence[tuple[str, str, str, str | None]],
        model_dir: str,
        cfg: dict[str, Any],
        *,
        block_size: int = 2000,
        workers: int = 12,
        prefetch: int = 2,
    ) -> None:
        self.tasks = tasks
        self.model_dir = model_dir
        self.cfg = cfg
        self.block_size = max(1, block_size)
        self.workers = max(1, workers)
        self.prefetch = max(1, prefetch)
        self._queue: Queue = Queue(maxsize=self.prefetch)
        self._thread: Thread | None = None
        self._stop = False

    @property
    def n_blocks(self) -> int:
        """Number of blocks this loader will yield."""
        return (len(self.tasks) + self.block_size - 1) // self.block_size

    def _produce(self) -> None:
        """Background producer: submits blocks to the pool, pushes results."""
        try:
            with ProcessPoolExecutor(
                max_workers=self.workers,
                initializer=_worker_init,
                initargs=(self.model_dir, self.cfg),
            ) as pool:
                for start in range(0, len(self.tasks), self.block_size):
                    if self._stop:
                        break
                    block = self.tasks[start : start + self.block_size]
                    chunksize = max(1, len(block) // (self.workers * 4))
                    try:
                        results = list(pool.map(prepare_article, block, chunksize=chunksize))
                    except Exception as exc:  # pool died -- degrade, don't crash
                        log.error("loader: worker pool failure (%s); falling back inline", exc)
                        _worker_init(self.model_dir, self.cfg)
                        results = [prepare_article(t) for t in block]
                    self._queue.put(results)
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("loader: producer thread died: %s", exc)
            self._queue.put(exc)
        finally:
            self._queue.put(None)  # sentinel

    def __iter__(self) -> Iterator[list[PreparedArticle]]:
        """Yield successive blocks of prepared articles."""
        self._thread = Thread(target=self._produce, name="article-loader", daemon=True)
        self._thread.start()
        while True:
            item = self._queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def close(self) -> None:
        """Ask the producer to stop and drain the queue."""
        self._stop = True
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                break


def chunks_from_block(block: Sequence[PreparedArticle]) -> list[Chunk]:
    """Flatten a block of prepared articles into a list of GPU work units."""
    out: list[Chunk] = []
    for i, art in enumerate(block):
        if art.status != "ok" or not art.lang:
            continue
        for j, (text, n) in enumerate(art.text_chunks):
            out.append(Chunk(article_idx=i, field="text", order=j, text=text,
                             n_tokens=n, lang=art.lang))
        for j, (text, n) in enumerate(art.title_chunks):
            out.append(Chunk(article_idx=i, field="title", order=j, text=text,
                             n_tokens=n, lang=art.lang))
    return out


def build_tasks(
    ids_to_titles: dict[str, str | None],
    index: dict[str, str],
    out_root: str | os.PathLike[str],
    data_root: str | os.PathLike[str],
    done: set[str],
    lang_hints: dict[str, str] | None = None,
) -> tuple[list[tuple[str, str, str, str | None]], list[str]]:
    """Assemble the work list, preserving the source folder structure.

    Returns ``(tasks, unresolved_ids)`` where each task is
    ``(article_id, src_path, out_path, translated_title)`` and *unresolved_ids*
    are event ids with no corresponding file on disk.

    When *lang_hints* is supplied the list is ordered by language, which is what
    lets each block form large single-language GPU batches (see
    :func:`scan_languages`); otherwise it is ordered by id. Both orders are
    deterministic, so a resumed run picks up exactly where the last one stopped.
    """
    out_root = Path(out_root)
    data_root_res = Path(data_root).resolve()
    tasks: list[tuple[str, str, str, str | None]] = []
    unresolved: list[str] = []

    for aid, title in ids_to_titles.items():
        if aid in done:
            continue
        src = index.get(aid)
        if not src:
            unresolved.append(aid)
            continue
        try:
            rel = Path(src).resolve().relative_to(data_root_res)
        except ValueError:
            rel = Path(aid + ".json")
        tasks.append((aid, src, str(out_root / rel), title))

    if lang_hints:
        # Cluster by language, then id. Unknown-language articles sort last so
        # the (rare) detect-at-runtime cases do not fragment the good blocks.
        tasks.sort(key=lambda t: (lang_hints.get(t[0]) or "zzz~unknown", t[0]))
    else:
        tasks.sort(key=lambda t: t[0])
    return tasks, unresolved
