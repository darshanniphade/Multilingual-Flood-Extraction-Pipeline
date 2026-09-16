"""Shared infrastructure: filesystem streaming, logging, I/O and formatting.

Nothing in this module holds the corpus in memory.  Directory traversal is a
generator over :func:`os.scandir`, and batching yields fixed-size lists of
*relative paths* only -- never file contents.
"""

from __future__ import annotations

import codecs
import logging
import os
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Final, TypeVar

import orjson

T = TypeVar("T")

JSON_SUFFIX: Final[str] = ".json"

_LOGGER_NAME: Final[str] = "flood_cleaner"


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------


def setup_logging(log_path: Path, verbose: bool = False) -> logging.Logger:
    """Configure and return the pipeline logger.

    Writes DEBUG-level detail to ``log_path`` and INFO-level progress to
    stderr.  stderr is used rather than stdout so that tqdm's bar (stderr) and
    the log stream interleave predictably and stdout stays clean for piping.

    Args:
        log_path: File that receives the full run log. Parent dirs are created.
        verbose: When True, console output drops to DEBUG level.

    Returns:
        The configured logger instance.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    # Close before dropping: a multi-year batch calls this once per year, and
    # bare .clear() would leak an open file handle per previous run -- which on
    # Windows leaves the earlier log locked.
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(file_handler)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
    logger.addHandler(console)

    return logger


def get_logger() -> logging.Logger:
    """Return the pipeline logger configured by :func:`setup_logging`.

    Returns:
        The shared logger; usable (silent) even before setup in worker
        processes, which deliberately do not log.
    """
    return logging.getLogger(_LOGGER_NAME)


# --------------------------------------------------------------------------
# Filesystem streaming
# --------------------------------------------------------------------------


def iter_relative_json_paths(root: Path) -> Iterator[str]:
    """Yield every ``*.json`` file under ``root`` as a root-relative path.

    Traversal is depth-first with directory and file names sorted at each
    level.  The deterministic order matters: deduplication keeps the *first*
    occurrence, so a stable walk makes runs reproducible.

    Memory stays flat -- only the current directory's entry list is resident,
    never the full corpus.

    Args:
        root: Directory to walk.

    Yields:
        Paths relative to ``root``, using the host path separator.
    """
    stack: list[str] = [""]
    while stack:
        rel_dir = stack.pop()
        abs_dir = root / rel_dir if rel_dir else root
        try:
            with os.scandir(abs_dir) as entries:
                dirs: list[str] = []
                files: list[str] = []
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            dirs.append(entry.name)
                        elif entry.name.lower().endswith(JSON_SUFFIX):
                            files.append(entry.name)
                    except OSError:
                        continue
        except OSError as exc:
            get_logger().warning("Cannot scan directory %s: %s", abs_dir, exc)
            continue

        for name in sorted(files):
            yield os.path.join(rel_dir, name) if rel_dir else name
        # Reversed push keeps pop() order alphabetical.
        for name in sorted(dirs, reverse=True):
            stack.append(os.path.join(rel_dir, name) if rel_dir else name)


def count_json_files(root: Path) -> int:
    """Count ``*.json`` files under ``root`` without retaining their paths.

    Args:
        root: Directory to walk.

    Returns:
        Total number of JSON files found.
    """
    return sum(1 for _ in iter_relative_json_paths(root))


def batched(iterable: Iterable[T], size: int) -> Iterator[list[T]]:
    """Group an iterable into lists of at most ``size`` items.

    Args:
        iterable: Source of items.
        size: Maximum items per batch; must be positive.

    Yields:
        Lists of up to ``size`` items, the last possibly shorter.

    Raises:
        ValueError: If ``size`` is not positive.
    """
    if size < 1:
        raise ValueError("batch size must be >= 1")
    batch: list[T] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def limited(iterable: Iterable[T], limit: int | None) -> Iterator[T]:
    """Yield at most ``limit`` items from ``iterable``.

    Args:
        iterable: Source of items.
        limit: Maximum items to yield; ``None`` means unlimited.

    Yields:
        Items from the source, truncated to ``limit``.
    """
    if limit is None:
        yield from iterable
        return
    for index, item in enumerate(iterable):
        if index >= limit:
            return
        yield item


# --------------------------------------------------------------------------
# JSON I/O
# --------------------------------------------------------------------------


def read_json(path: Path) -> dict[str, Any]:
    """Read and parse a UTF-8 JSON document.

    Args:
        path: File to read.

    Returns:
        The decoded object.

    Raises:
        OSError: If the file cannot be read.
        orjson.JSONDecodeError: If the payload is not valid JSON.
        TypeError: If the payload is valid JSON but not an object.
    """
    raw = path.read_bytes()
    # A UTF-8 BOM is legal at the head of a UTF-8 file and scrapers do emit it,
    # but orjson treats it as an unexpected character -- so an otherwise valid
    # article would be discarded as malformed_json. Skip it explicitly.
    if raw.startswith(codecs.BOM_UTF8):
        raw = raw[len(codecs.BOM_UTF8):]
    payload = orjson.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object, got {type(payload).__name__}")
    return payload


def write_json(path: Path, document: dict[str, Any], pretty: bool = True) -> int:
    """Serialise ``document`` to ``path`` as UTF-8 JSON.

    Args:
        path: Destination file. The parent directory must already exist.
        document: Object to serialise.
        pretty: Whether to indent the output with two spaces.

    Returns:
        Number of bytes written.

    Raises:
        OSError: If the file cannot be written.
    """
    option = orjson.OPT_INDENT_2 if pretty else 0
    blob = orjson.dumps(document, option=option)
    path.write_bytes(blob)
    return len(blob)


class DirectoryCache:
    """Memoises ``mkdir`` calls so each output directory is created once.

    At corpus scale the same directory is written thousands of times in a row;
    without this, every article costs a redundant ``mkdir`` syscall.
    """

    __slots__ = ("_seen",)

    def __init__(self) -> None:
        """Initialise an empty set of already-created directories."""
        self._seen: set[str] = set()

    def ensure(self, directory: Path) -> None:
        """Create ``directory`` (and parents) unless already ensured.

        Args:
            directory: Directory that must exist.

        Raises:
            OSError: If creation fails.
        """
        key = str(directory)
        if key in self._seen:
            return
        directory.mkdir(parents=True, exist_ok=True)
        self._seen.add(key)


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------


def format_duration(seconds: float) -> str:
    """Render a duration as ``HH:MM:SS`` with fractional seconds under a minute.

    Args:
        seconds: Elapsed wall-clock seconds.

    Returns:
        Human-readable duration string.
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    return f"{minutes:d}m {secs:02d}s"


def format_count(value: int) -> str:
    """Render an integer with thousands separators.

    Args:
        value: Number to format.

    Returns:
        Grouped decimal string.
    """
    return f"{value:,}"
