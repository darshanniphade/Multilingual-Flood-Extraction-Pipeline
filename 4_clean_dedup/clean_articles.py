r"""Parallel cleaning pipeline for the translated flood-news corpus.

Reads every ``*.json`` article beneath the input root, cleans and validates it,
removes exact duplicates, and writes a three-field document to a mirrored
directory tree under the output root.

Architecture
------------
``main`` (parent process)
    Walks the input tree lazily in sorted order, feeds fixed-size batches of
    *relative paths* to a process pool, consumes results **in submission
    order**, owns the deduplication index, and writes the run report.

``_process_batch`` (worker processes)
    Reads, cleans, validates, hashes and writes each article in its batch,
    returning one small record per file. Cleaned text is never sent back
    across the process boundary.

Duplicates are written by the worker and unlinked by the parent once the
digest collision is known. This keeps the run single-pass over the corpus;
the cost is a redundant write for the duplicate fraction (~10% of this
corpus), which is far cheaper than a second full read pass.

Memory is flat regardless of corpus size: only in-flight batches of paths and
the digest index are resident.

Usage:
    python clean_articles.py
    python clean_articles.py --limit 2000 --workers 8
    python clean_articles.py --input D:\\corpus --output D:\\clean --overwrite
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final, NamedTuple

import orjson
from tqdm import tqdm

import cleaner  # noqa: F401  (imported so worker processes pre-compile patterns)
import config
import deduplicator
import utils
import validators
from config import PipelineSettings
from deduplicator import Deduplicator
from validators import SkipReason

#: Result status codes returned by workers.
STATUS_OK: Final[str] = "ok"
STATUS_SKIPPED: Final[str] = "skipped"
STATUS_ERROR: Final[str] = "error"


class FileOutcome(NamedTuple):
    """Per-file result returned from a worker to the parent process.

    Attributes:
        rel_path: Path of the source file relative to the input root.
        status: One of :data:`STATUS_OK`, :data:`STATUS_SKIPPED`,
            :data:`STATUS_ERROR`.
        reason: Skip/error reason code, empty when ``status`` is ok.
        detail: Exception text for error outcomes, otherwise empty.
        digest: SHA-256 digest of the cleaned text; empty unless ok.
        word_count: Word count of the cleaned text.
        bytes_written: Size of the written output document in bytes.
    """

    rel_path: str
    status: str
    reason: str
    detail: str
    digest: bytes
    word_count: int
    bytes_written: int


# --------------------------------------------------------------------------
# Worker side
# --------------------------------------------------------------------------

_WORKER_INPUT_ROOT: Path
_WORKER_OUTPUT_ROOT: Path
_WORKER_MIN_WORDS: int
_WORKER_PRETTY: bool
_WORKER_DIRS: utils.DirectoryCache
_WORKER_TITLE_FIELD: str
_WORKER_TEXT_FIELD: str


def _init_worker(
    input_root: str,
    output_root: str,
    min_words: int,
    pretty: bool,
    title_field: str,
    text_field: str,
) -> None:
    """Initialise per-process worker state.

    Runs once per worker process. The regex modules are compiled at import
    time, so this only needs to install the run's paths and thresholds.

    Args:
        input_root: Absolute path of the input tree.
        output_root: Absolute path of the output tree.
        min_words: Post-cleaning word-count floor.
        pretty: Whether to indent output JSON.
        title_field: Source field to read as the article title.
        text_field: Source field to read as the article body.
    """
    global _WORKER_INPUT_ROOT, _WORKER_OUTPUT_ROOT
    global _WORKER_MIN_WORDS, _WORKER_PRETTY, _WORKER_DIRS
    global _WORKER_TITLE_FIELD, _WORKER_TEXT_FIELD
    _WORKER_INPUT_ROOT = Path(input_root)
    _WORKER_OUTPUT_ROOT = Path(output_root)
    _WORKER_MIN_WORDS = min_words
    _WORKER_PRETTY = pretty
    _WORKER_TITLE_FIELD = title_field
    _WORKER_TEXT_FIELD = text_field
    _WORKER_DIRS = utils.DirectoryCache()

    # Workers must never emit progress noise or compete for the log file.
    import logging

    logging.getLogger("flood_cleaner").handlers.clear()
    logging.getLogger("flood_cleaner").addHandler(logging.NullHandler())


def _process_one(rel_path: str) -> FileOutcome:
    """Clean, validate and write a single article.

    Every failure mode is captured and returned as a record; this function
    does not raise.

    Args:
        rel_path: Source path relative to the input root.

    Returns:
        The outcome for this file.
    """
    source = _WORKER_INPUT_ROOT / rel_path
    try:
        document: dict[str, Any] = utils.read_json(source)
    # Reason codes are coerced to plain str: they become dict keys in the
    # report, and orjson rejects str *subclasses* (such as StrEnum) as keys
    # even though it accepts them as values.
    except orjson.JSONDecodeError as exc:
        return FileOutcome(
            rel_path, STATUS_ERROR, str(SkipReason.MALFORMED_JSON), str(exc), b"", 0, 0
        )
    except TypeError as exc:
        return FileOutcome(
            rel_path, STATUS_ERROR, str(SkipReason.NOT_AN_OBJECT), str(exc), b"", 0, 0
        )
    except OSError as exc:
        return FileOutcome(
            rel_path, STATUS_ERROR, str(SkipReason.UNREADABLE_FILE), str(exc), b"", 0, 0
        )
    except Exception as exc:  # pragma: no cover - defensive catch-all
        return FileOutcome(
            rel_path, STATUS_ERROR, str(SkipReason.MALFORMED_JSON), repr(exc), b"", 0, 0
        )

    try:
        result = validators.validate_and_clean(
            document,
            Path(rel_path).stem,
            _WORKER_MIN_WORDS,
            _WORKER_TITLE_FIELD,
            _WORKER_TEXT_FIELD,
        )
    except Exception as exc:  # pragma: no cover - defensive catch-all
        return FileOutcome(rel_path, STATUS_ERROR, "cleaning_failed", repr(exc), b"", 0, 0)

    if not result.ok:
        reason = result.reason or SkipReason.MISSING_TEXT
        return FileOutcome(
            rel_path, STATUS_SKIPPED, str(reason), "", b"", result.word_count, 0
        )

    destination = _WORKER_OUTPUT_ROOT / rel_path
    # Output keys mirror the source keys: this stage cleans text, it does not
    # rename fields. A raw tree read via title/text is written back as
    # title/text; a translated tree keeps translated_title/translated_text.
    payload = {
        "article_id": result.article_id,
        _WORKER_TITLE_FIELD: result.title,
        _WORKER_TEXT_FIELD: result.text,
    }
    try:
        _WORKER_DIRS.ensure(destination.parent)
        written = utils.write_json(destination, payload, _WORKER_PRETTY)
    except OSError as exc:
        return FileOutcome(
            rel_path, STATUS_ERROR, str(SkipReason.WRITE_FAILED), str(exc), b"", 0, 0
        )

    return FileOutcome(
        rel_path,
        STATUS_OK,
        "",
        "",
        deduplicator.hash_text(result.text),
        result.word_count,
        written,
    )


def _process_batch(rel_paths: list[str]) -> list[FileOutcome]:
    """Process a batch of files in one IPC round-trip.

    Args:
        rel_paths: Source paths relative to the input root.

    Returns:
        One outcome per input path, in the same order.
    """
    return [_process_one(rel_path) for rel_path in rel_paths]


# --------------------------------------------------------------------------
# Parent side
# --------------------------------------------------------------------------


class RunStatistics:
    """Accumulates counters and examples across a run.

    Attributes:
        total: Files examined.
        valid: Files that passed validation, before deduplication.
        duplicates: Files rejected as exact duplicates.
        written: Files present in the output tree.
        errors: Files that failed with an error.
        bytes_written: Total output bytes retained.
    """

    __slots__ = (
        "total", "valid", "duplicates", "written", "errors", "bytes_written",
        "skip_reasons", "error_reasons", "examples", "duplicate_examples",
        "word_total",
    )

    def __init__(self) -> None:
        """Initialise all counters to zero and the example buffers to empty."""
        self.total = 0
        self.valid = 0
        self.duplicates = 0
        self.written = 0
        self.errors = 0
        self.bytes_written = 0
        self.word_total = 0
        self.skip_reasons: Counter[str] = Counter()
        self.error_reasons: Counter[str] = Counter()
        self.examples: defaultdict[str, list[str]] = defaultdict(list)
        self.duplicate_examples: list[dict[str, str]] = []

    def record_example(self, reason: str, rel_path: str, detail: str = "") -> None:
        """Retain a bounded sample of files per rejection reason.

        Args:
            reason: Reason code.
            rel_path: Source path relative to the input root.
            detail: Optional supplementary text, e.g. an exception message.
        """
        bucket = self.examples[reason]
        if len(bucket) < config.MAX_LOGGED_EXAMPLES:
            bucket.append(f"{rel_path}: {detail}" if detail else rel_path)

    @property
    def skipped(self) -> int:
        """Total files excluded for any reason other than duplication.

        Returns:
            Count of skipped files including hard errors.
        """
        return sum(self.skip_reasons.values()) + self.errors


def _iter_batches(settings: PipelineSettings) -> Iterator[list[str]]:
    """Yield batches of relative source paths from the input tree.

    Args:
        settings: Resolved run settings.

    Yields:
        Lists of at most ``settings.batch_size`` relative paths.
    """
    paths = utils.iter_relative_json_paths(settings.input_root)
    yield from utils.batched(utils.limited(paths, settings.limit), settings.batch_size)


def _throttled(
    batches: Iterator[list[str]], permits: threading.Semaphore
) -> Iterator[list[str]]:
    """Gate batch submission so only a bounded number are ever in flight.

    ``Pool.imap`` otherwise drains the input iterator as fast as it can, which
    would materialise the whole corpus's path list in the task queue. The
    semaphore is released as results are consumed.

    Args:
        batches: Source of batches.
        permits: Semaphore sized to the desired queue depth.

    Yields:
        Batches, blocking once the in-flight limit is reached.
    """
    for batch in batches:
        permits.acquire()
        yield batch


def _consume_outcome(
    outcome: FileOutcome,
    stats: RunStatistics,
    dedup: Deduplicator | None,
    output_root: Path,
    logger: Any,
) -> None:
    """Fold one worker outcome into the run statistics.

    Duplicate output files written speculatively by the worker are removed
    here, once the parent's index proves the collision.

    Args:
        outcome: Result record from a worker.
        stats: Accumulator to update.
        dedup: Deduplication index, or ``None`` when disabled.
        output_root: Root of the output tree, used to unlink duplicates.
        logger: Logger for per-file diagnostics.
    """
    stats.total += 1

    if outcome.status == STATUS_ERROR:
        stats.errors += 1
        stats.error_reasons[outcome.reason] += 1
        stats.record_example(outcome.reason, outcome.rel_path, outcome.detail)
        logger.debug("ERROR   %s (%s): %s", outcome.rel_path, outcome.reason, outcome.detail)
        return

    if outcome.status == STATUS_SKIPPED:
        stats.skip_reasons[outcome.reason] += 1
        stats.record_example(outcome.reason, outcome.rel_path)
        logger.debug("SKIP    %s (%s)", outcome.rel_path, outcome.reason)
        return

    stats.valid += 1

    if dedup is not None:
        record = dedup.check(outcome.digest, outcome.rel_path)
        if record is not None:
            stats.duplicates += 1
            if len(stats.duplicate_examples) < config.MAX_LOGGED_EXAMPLES:
                stats.duplicate_examples.append(
                    {
                        "duplicate": record.duplicate_path,
                        "original": record.original_path,
                        "sha256": record.digest,
                    }
                )
            try:
                (output_root / outcome.rel_path).unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Could not remove duplicate output %s: %s", outcome.rel_path, exc)
            logger.debug(
                "DUP     %s duplicates %s", record.duplicate_path, record.original_path
            )
            return

    stats.written += 1
    stats.bytes_written += outcome.bytes_written
    stats.word_total += outcome.word_count


def _build_report(
    settings: PipelineSettings, stats: RunStatistics, elapsed: float
) -> dict[str, Any]:
    """Assemble the machine-readable processing report.

    Args:
        settings: Resolved run settings.
        stats: Accumulated counters.
        elapsed: Wall-clock runtime in seconds.

    Returns:
        A JSON-serialisable report document.
    """
    speed = stats.total / elapsed if elapsed > 0 else 0.0
    reasons = {
        # str() is load-bearing: orjson refuses str subclasses as dict keys,
        # and a report that cannot serialise would discard a completed run.
        str(reason): {
            "count": count,
            "description": validators.describe_reason(
                reason,
                settings.text_field,
                settings.title_field,
                settings.min_word_count,
            ),
            "examples": stats.examples.get(reason, [])[:10],
        }
        for reason, count in (stats.skip_reasons + stats.error_reasons).most_common()
    }
    return {
        "summary": {
            "total_files": stats.total,
            "valid_files": stats.valid,
            "skipped_files": stats.skipped,
            "duplicate_files": stats.duplicates,
            "output_files": stats.written,
            "error_files": stats.errors,
            "runtime_seconds": round(elapsed, 3),
            "runtime_human": utils.format_duration(elapsed),
            "average_files_per_second": round(speed, 2),
        },
        "skip_reasons": reasons,
        "duplicates": {
            "hash_algorithm": deduplicator.HASH_ALGORITHM,
            "matching": "exact",
            "count": stats.duplicates,
            "unique_texts": stats.valid - stats.duplicates,
            "examples": stats.duplicate_examples[:10],
        },
        "output": {
            "bytes_written": stats.bytes_written,
            "megabytes_written": round(stats.bytes_written / (1024 * 1024), 2),
            "total_words": stats.word_total,
            "average_words_per_article": (
                round(stats.word_total / stats.written, 1) if stats.written else 0.0
            ),
        },
        "configuration": {
            "input_root": str(settings.input_root),
            "output_root": str(settings.output_root),
            "workers": settings.workers,
            "batch_size": settings.batch_size,
            "min_word_count": settings.min_word_count,
            "source_title_field": settings.title_field,
            "source_text_field": settings.text_field,
            "deduplication_enabled": settings.deduplicate,
            "unicode_normal_form": config.UNICODE_NORMAL_FORM,
            "ellipsis_policy": config.ELLIPSIS_POLICY,
            "garbled_text_check": config.ENABLE_GARBLED_TEXT_CHECK,
            "python": sys.version.split()[0],
        },
    }


def _check_output_root(settings: PipelineSettings, overwrite: bool) -> None:
    """Verify the output tree is safe to write into.

    Args:
        settings: Resolved run settings.
        overwrite: Whether the caller has authorised writing into a
            non-empty output tree.

    Raises:
        SystemExit: If the output tree already holds files and ``overwrite``
            was not requested.
    """
    root = settings.output_root
    if not root.exists():
        return
    try:
        occupied = next(root.rglob("*.json"), None) is not None
    except OSError:
        occupied = False
    if occupied and not overwrite:
        raise SystemExit(
            f"Output directory {root} already contains JSON files.\n"
            "Re-run with --overwrite to write into it, or choose another path."
        )


def run(settings: PipelineSettings, overwrite: bool, verbose: bool) -> dict[str, Any]:
    """Execute the pipeline end to end.

    Args:
        settings: Resolved run settings.
        overwrite: Permit writing into a non-empty output tree.
        verbose: Emit DEBUG-level console output.

    Returns:
        The processing report document.

    Raises:
        SystemExit: If the input tree is missing or the output tree is
            occupied and ``overwrite`` was not given.
    """
    logger = utils.setup_logging(settings.log_path, verbose)

    if not settings.input_root.is_dir():
        raise SystemExit(f"Input directory does not exist: {settings.input_root}")
    _check_output_root(settings, overwrite)
    settings.output_root.mkdir(parents=True, exist_ok=True)

    logger.info("Input : %s", settings.input_root)
    logger.info("Output: %s", settings.output_root)
    logger.info("Workers: %d | batch size: %d", settings.workers, settings.batch_size)

    total_files: int | None = None
    if settings.count_first:
        logger.info("Scanning input tree...")
        scan_start = time.perf_counter()
        total_files = utils.count_json_files(settings.input_root)
        if settings.limit is not None:
            total_files = min(total_files, settings.limit)
        logger.info(
            "Found %s files in %s",
            utils.format_count(total_files),
            utils.format_duration(time.perf_counter() - scan_start),
        )

    stats = RunStatistics()
    dedup = Deduplicator() if settings.deduplicate else None
    permits = threading.Semaphore(settings.workers * config.QUEUE_DEPTH_BATCHES)

    start = time.perf_counter()
    context = mp.get_context("spawn")
    progress = tqdm(
        total=total_files,
        unit="file",
        unit_scale=True,
        smoothing=0.05,
        desc="Cleaning",
        dynamic_ncols=True,
    )

    pool = context.Pool(
        processes=settings.workers,
        initializer=_init_worker,
        initargs=(
            str(settings.input_root),
            str(settings.output_root),
            settings.min_word_count,
            settings.pretty,
            settings.title_field,
            settings.text_field,
        ),
    )
    try:
        batches = _throttled(_iter_batches(settings), permits)
        # imap (ordered) rather than imap_unordered: deduplication keeps the
        # first occurrence, and ordered consumption over a sorted walk makes
        # "first" deterministic across runs.
        for outcomes in pool.imap(_process_batch, batches):
            permits.release()
            for outcome in outcomes:
                _consume_outcome(outcome, stats, dedup, settings.output_root, logger)
            progress.update(len(outcomes))
            progress.set_postfix(
                kept=stats.written, dup=stats.duplicates, skip=stats.skipped, refresh=False
            )
    except KeyboardInterrupt:
        logger.warning("Interrupted - terminating workers and reporting partial results.")
        pool.terminate()
    else:
        pool.close()
    finally:
        pool.join()
        progress.close()

    elapsed = time.perf_counter() - start
    report = _build_report(settings, stats, elapsed)

    settings.report_path.parent.mkdir(parents=True, exist_ok=True)
    utils.write_json(settings.report_path, report, pretty=True)

    logger.info("-" * 62)
    logger.info("Total files     : %s", utils.format_count(stats.total))
    logger.info("Valid           : %s", utils.format_count(stats.valid))
    logger.info("Skipped         : %s", utils.format_count(stats.skipped))
    logger.info("Duplicates      : %s", utils.format_count(stats.duplicates))
    logger.info("Written         : %s", utils.format_count(stats.written))
    logger.info("Runtime         : %s", utils.format_duration(elapsed))
    logger.info(
        "Speed           : %.1f files/sec",
        report["summary"]["average_files_per_second"],
    )
    if stats.skip_reasons or stats.error_reasons:
        logger.info("Skip breakdown:")
        for reason, count in (stats.skip_reasons + stats.error_reasons).most_common():
            logger.info("  %-38s %s", reason, utils.format_count(count))
    logger.info("Report          : %s", settings.report_path)
    logger.info("Log             : %s", settings.log_path)

    if dedup is not None:
        dedup.clear()
    return report


def parse_args(argv: list[str] | None = None) -> tuple[PipelineSettings, bool, bool]:
    """Parse command-line arguments into pipeline settings.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        A tuple of (settings, overwrite flag, verbose flag).
    """
    parser = argparse.ArgumentParser(
        prog="clean_articles.py",
        description="Clean, validate and deduplicate translated flood-news articles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=config.INPUT_ROOT,
                        help="Root of the input article tree.")
    parser.add_argument("--output", type=Path, default=config.OUTPUT_ROOT,
                        help="Root of the mirrored output tree.")
    parser.add_argument("--report", type=Path, default=None,
                        help="Path of processing_report.json (default: beside output).")
    parser.add_argument("--log", type=Path, default=None,
                        help="Path of the run log (default: beside output).")
    parser.add_argument("--workers", type=int, default=config.resolve_worker_count(),
                        help="Worker process count.")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                        help="Files per IPC batch.")
    parser.add_argument("--min-words", type=int, default=config.MIN_WORD_COUNT,
                        help="Minimum words required after cleaning.")
    parser.add_argument("--title-field", type=str, default=config.SOURCE_TITLE_FIELD,
                        help="Source field read as the title (raw trees: 'title').")
    parser.add_argument("--text-field", type=str, default=config.SOURCE_TEXT_FIELD,
                        help="Source field read as the body (raw trees: 'text').")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N files (smoke testing).")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Disable exact-duplicate removal.")
    parser.add_argument("--no-count", action="store_true",
                        help="Skip the pre-scan; progress bar runs without a total.")
    parser.add_argument("--compact", action="store_true",
                        help="Write output JSON without indentation.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow writing into a non-empty output directory.")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-file DEBUG output on the console.")
    args = parser.parse_args(argv)

    output_root: Path = args.output
    settings = PipelineSettings(
        input_root=args.input,
        output_root=output_root,
        report_path=args.report or (output_root.parent / "processing_report.json"),
        log_path=args.log or (output_root.parent / "processing.log"),
        workers=max(1, args.workers),
        batch_size=max(1, args.batch_size),
        min_word_count=max(0, args.min_words),
        pretty=not args.compact,
        deduplicate=not args.no_dedup,
        count_first=not args.no_count,
        limit=args.limit,
        title_field=args.title_field,
        text_field=args.text_field,
    )
    return settings, args.overwrite, args.verbose


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 on fatal configuration errors.
    """
    settings, overwrite, verbose = parse_args(argv)
    try:
        run(settings, overwrite, verbose)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
