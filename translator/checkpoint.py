"""
checkpoint.py
=============

Crash-safe resume support.

Design
------
The checkpoint is an **append-only JSONL journal** of completed article ids
plus a small periodically-rewritten summary file. Append-only is the important
part: at 233k articles, rewriting a full state file every N articles would cost
more I/O than the translation itself, and a rewrite is exactly the moment a
crash corrupts state.

* ``checkpoint.jsonl``  -- one line per completed article: ``{"id": ..., "s": ...}``
  Appended and ``flush``ed on every checkpoint interval. A torn final line is
  detected and discarded on load.
* ``checkpoint.json``   -- human-readable summary (counts, timing, config hash).
  Written atomically; purely informational, never required for correctness.

Recovery is therefore "replay the journal, ignore the last line if it is torn",
which cannot lose more than one checkpoint interval of work.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable

log = logging.getLogger("translator.checkpoint")


@dataclass(slots=True)
class CheckpointStats:
    """Aggregate counters persisted alongside the completed-id journal."""

    completed: int = 0
    skipped_english: int = 0
    failed: int = 0
    corrupt: int = 0
    src_tokens: int = 0
    out_tokens: int = 0
    gpu_seconds: float = 0.0
    wall_seconds: float = 0.0

    def merge(self, other: "CheckpointStats") -> None:
        """Accumulate *other* into this instance."""
        self.completed += other.completed
        self.skipped_english += other.skipped_english
        self.failed += other.failed
        self.corrupt += other.corrupt
        self.src_tokens += other.src_tokens
        self.out_tokens += other.out_tokens
        self.gpu_seconds += other.gpu_seconds
        self.wall_seconds += other.wall_seconds


class CheckpointManager:
    """Tracks which article ids are already done and survives interruption.

    Usage::

        cp = CheckpointManager("state/checkpoint.jsonl", interval=500)
        done = cp.load()                 # set[str] of completed ids
        ...
        cp.mark_many([("2021_01/article_1", "ok"), ...])
        cp.maybe_flush()                 # flushes every `interval` marks
        cp.close()

    All public methods are thread-safe.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        interval: int = 500,
        summary_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.path = os.fspath(path)
        self.summary_path = os.fspath(summary_path) if summary_path else (
            os.path.splitext(self.path)[0] + ".json"
        )
        self.interval = max(1, interval)
        self.stats = CheckpointStats()

        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._pending: list[str] = []
        self._since_flush = 0
        self._fh = None  # type: ignore[assignment]
        self._completed: set[str] = set()
        self._start = time.time()

    # ------------------------------------------------------------------ #
    # Load / resume
    # ------------------------------------------------------------------ #

    def load(self) -> set[str]:
        """Replay the journal and return the set of already-completed ids.

        A truncated final line (the classic "killed mid-write" artefact) is
        discarded with a warning rather than raising.
        """
        done: set[str] = set()
        if not os.path.exists(self.path):
            self._completed = done
            return done

        torn = 0
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    aid = rec["id"]
                except Exception:
                    torn += 1
                    continue
                done.add(aid)
                status = rec.get("s", "ok")
                if status == "skipped_english":
                    self.stats.skipped_english += 1
                elif status == "failed":
                    self.stats.failed += 1
                elif status == "corrupt":
                    self.stats.corrupt += 1
        if torn:
            log.warning("checkpoint: discarded %d unreadable journal line(s)", torn)

        self.stats.completed = len(done)
        self._completed = done
        if done:
            log.info("checkpoint: resuming with %d article(s) already done", len(done))
        return done

    # ------------------------------------------------------------------ #
    # Marking progress
    # ------------------------------------------------------------------ #

    def _ensure_open(self) -> None:
        if self._fh is None:
            self._fh = open(self.path, "a", encoding="utf-8", buffering=1 << 20)

    def mark(self, article_id: str, status: str = "ok") -> None:
        """Record a single article as finished."""
        self.mark_many([(article_id, status)])

    def mark_many(self, items: Iterable[tuple[str, str]]) -> None:
        """Record many ``(article_id, status)`` pairs as finished."""
        with self._lock:
            for aid, status in items:
                if aid in self._completed:
                    continue
                self._completed.add(aid)
                self._pending.append(json.dumps({"id": aid, "s": status}, ensure_ascii=False))
                self._since_flush += 1
                self.stats.completed += 1
                if status == "skipped_english":
                    self.stats.skipped_english += 1
                elif status == "failed":
                    self.stats.failed += 1
                elif status == "corrupt":
                    self.stats.corrupt += 1

    def maybe_flush(self, force: bool = False) -> bool:
        """Flush the journal if *interval* marks have accumulated (or *force*).

        Returns ``True`` when a flush actually happened.
        """
        with self._lock:
            if not force and self._since_flush < self.interval:
                return False
            if not self._pending:
                if force:
                    self._write_summary_locked()
                return False
            self._ensure_open()
            self._fh.write("\n".join(self._pending) + "\n")  # type: ignore[union-attr]
            self._fh.flush()                                  # type: ignore[union-attr]
            os.fsync(self._fh.fileno())                       # type: ignore[union-attr]
            self._pending.clear()
            self._since_flush = 0
            self._write_summary_locked()
            return True

    def is_done(self, article_id: str) -> bool:
        """Whether *article_id* has already been completed."""
        with self._lock:
            return article_id in self._completed

    @property
    def completed_count(self) -> int:
        """Number of articles recorded as complete."""
        with self._lock:
            return len(self._completed)

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #

    def _write_summary_locked(self) -> None:
        self.stats.wall_seconds = time.time() - self._start
        summary: dict[str, Any] = {
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "completed": self.stats.completed,
            "skipped_english": self.stats.skipped_english,
            "failed": self.stats.failed,
            "corrupt": self.stats.corrupt,
            "src_tokens": self.stats.src_tokens,
            "out_tokens": self.stats.out_tokens,
            "gpu_seconds": round(self.stats.gpu_seconds, 1),
            "session_wall_seconds": round(self.stats.wall_seconds, 1),
        }
        tmp = f"{self.summary_path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)
            os.replace(tmp, self.summary_path)
        except OSError as exc:  # summary is informational -- never fatal
            log.debug("checkpoint: could not write summary: %s", exc)

    def close(self) -> None:
        """Final flush and close."""
        self.maybe_flush(force=True)
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception:  # pragma: no cover
                    pass
                self._fh = None

    def __enter__(self) -> "CheckpointManager":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
