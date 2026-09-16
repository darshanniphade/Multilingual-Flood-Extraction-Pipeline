"""
logger.py
=========

Logging for the flood-article translation pipeline.

Two distinct sinks are provided:

``setup_logging``
    Configures the root logger: a rotating ``translation.log`` file plus a
    console handler that plays nicely with ``tqdm`` progress bars (log lines are
    routed through ``tqdm.write`` so they never smear a live bar).

``RecordLogger``
    Appends one structured JSON line per article to ``translation_records.jsonl``
    containing the per-article facts the spec calls for -- id, language,
    translation time, retry count, GPU memory, batch size and any error.

The structured sink is separate from the human log on purpose: 233k articles of
per-article detail is data to be queried, not prose to be read, and keeping it
out of ``translation.log`` keeps that file useful for humans.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, TextIO

try:  # tqdm is a hard dependency, but logging must never be the thing that fails
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


class _TqdmHandler(logging.StreamHandler):
    """A console handler that writes via ``tqdm.write`` to protect progress bars.

    Also hardened against the Windows console's cp1252 default: this corpus is
    ~60% non-English, so a log line quoting article text (Cyrillic, Arabic,
    Thai...) would otherwise raise ``UnicodeEncodeError`` *inside logging* and
    take down a long run. Unencodable characters are replaced instead.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            try:
                if tqdm is not None:
                    tqdm.write(msg, file=sys.stderr)
                else:  # pragma: no cover
                    sys.stderr.write(msg + "\n")
            except UnicodeEncodeError:
                enc = getattr(sys.stderr, "encoding", None) or "ascii"
                safe = msg.encode(enc, errors="replace").decode(enc, errors="replace")
                if tqdm is not None:
                    tqdm.write(safe, file=sys.stderr)
                else:  # pragma: no cover
                    sys.stderr.write(safe + "\n")
            self.flush()
        except Exception:  # pragma: no cover
            self.handleError(record)


def setup_logging(
    log_path: str | os.PathLike[str] = "translation.log",
    *,
    level: str = "INFO",
    console: bool = True,
    max_bytes: int = 64 * 1024 * 1024,
    backups: int = 3,
) -> logging.Logger:
    """Configure root logging and return the pipeline logger.

    Parameters
    ----------
    log_path:
        Destination for the human-readable log. Rotated at *max_bytes*.
    level:
        Logging level name for both sinks (``DEBUG``/``INFO``/...).
    console:
        Whether to also emit to stderr through ``tqdm.write``.
    """
    log_path = os.fspath(log_path)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    # Prefer real UTF-8 on the console when the terminal supports it; the
    # replace-fallback in _TqdmHandler covers terminals that do not.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError, ValueError):  # pragma: no cover
            pass

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for h in list(root.handlers):  # idempotent across re-invocations
        root.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-12s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)

    if console:
        ch = _TqdmHandler()
        ch.setFormatter(logging.Formatter("%(levelname)-7s | %(message)s"))
        root.addHandler(ch)

    # Third-party noise: keep the log about *our* pipeline.
    for noisy in ("transformers", "torch", "accelerate", "urllib3", "filelock", "langdetect"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    return logging.getLogger("translator")


@dataclass(slots=True)
class ArticleRecord:
    """One structured per-article log record.

    Mirrors the fields the specification requires be captured per article.
    """

    article_id: str
    language: str | None = None
    lang_source: str = "unknown"       # metadata | langdetect | unknown
    status: str = "ok"                 # ok | skipped_english | failed | corrupt
    chunks: int = 0
    src_tokens: int = 0
    out_tokens: int = 0
    translate_seconds: float = 0.0
    retries: int = 0
    batch_size: int = 0
    gpu_mem_gib: float = 0.0
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class RecordLogger:
    """Thread-safe append-only JSONL sink for :class:`ArticleRecord`.

    Buffered and flushed in batches to keep per-article overhead negligible at
    233k articles, while still flushing on every checkpoint so a crash loses at
    most one buffer.
    """

    def __init__(self, path: str | os.PathLike[str], *, buffer_size: int = 500) -> None:
        self.path = os.fspath(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._buf: list[str] = []
        self._buffer_size = buffer_size
        self._lock = threading.Lock()
        self._fh: TextIO = open(self.path, "a", encoding="utf-8", buffering=1 << 20)

    def log(self, rec: ArticleRecord) -> None:
        """Buffer one record, flushing when the buffer fills."""
        line = json.dumps(asdict(rec), ensure_ascii=False)
        with self._lock:
            self._buf.append(line)
            if len(self._buf) >= self._buffer_size:
                self._flush_locked()

    def log_many(self, recs: list[ArticleRecord]) -> None:
        """Buffer many records at once."""
        lines = [json.dumps(asdict(r), ensure_ascii=False) for r in recs]
        with self._lock:
            self._buf.extend(lines)
            if len(self._buf) >= self._buffer_size:
                self._flush_locked()

    def _flush_locked(self) -> None:
        if self._buf:
            self._fh.write("\n".join(self._buf) + "\n")
            self._fh.flush()
            self._buf.clear()

    def flush(self) -> None:
        """Force-flush the buffer to disk."""
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        """Flush and close the underlying file."""
        with self._lock:
            self._flush_locked()
            try:
                self._fh.close()
            except Exception:  # pragma: no cover
                pass

    def __enter__(self) -> "RecordLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
