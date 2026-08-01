r"""Stage-4 field extractor: keep only the English fields.

Reads the stage-3 translator output and rewrites each article carrying nothing
but the three fields downstream stages need::

    {
      "article_id":       "article_000094073",
      "translated_title": "...",
      "translated_text":  "..."
    }

Everything else -- the original ``title`` / ``text`` and the
``translation_meta`` block -- is dropped. Field names are preserved; this stage
copies values, it never renames, cleans or translates.

    python extract_english.py 2022                 # one year
    python extract_english.py 2022 2023            # several
    python extract_english.py 2019-2024            # an inclusive range
    python extract_english.py --list               # what is available / done

Layout
------
The output tree mirrors the input tree, month directories and all::

    in   C:\darsh\pipeline\data\translated_articles\<year>\<year>_MM\*.json
    out  C:\darsh\pipeline\data\only eng\<year>\<year>_MM\*.json
         C:\darsh\pipeline\data\only eng\_reports\<year>_report.json

Adding a year is nothing but a new directory under the source root -- no edit
to this file is required.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterator

# --------------------------------------------------------------------------
# Site configuration -- edit these values, not the logic below
# --------------------------------------------------------------------------

#: Stage-3 translator output: one subdirectory per year.
SOURCE_ROOT: Final[Path] = Path(r"C:\darsh\pipeline\data\translated_articles")

#: Where the trimmed year trees are written.
DEST_ROOT: Final[Path] = Path(r"C:\darsh\pipeline\data\only eng")

#: Subdirectory of DEST_ROOT holding per-year reports. The leading underscore
#: keeps it sorted away from the year directories.
REPORTS_DIRNAME: Final[str] = "_reports"

#: The only fields carried over, in the order they are written.
ID_FIELD: Final[str] = "article_id"
TITLE_FIELD: Final[str] = "translated_title"
TEXT_FIELD: Final[str] = "translated_text"

#: Fallbacks used when a translated field is absent -- articles already in
#: English are sometimes written with ``status: skipped_english`` and only the
#: source fields populated.
TITLE_FALLBACK: Final[str] = "title"
TEXT_FALLBACK: Final[str] = "text"

#: Lowest and highest plausible year, used to validate CLI input.
YEAR_RANGE: Final[tuple[int, int]] = (1990, 2100)


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything one year's run needs to know.

    Attributes:
        source: Year tree to read.
        output: Year tree to write.
        workers: Worker process count; 1 runs in-process.
        pretty: Indent the output JSON.
        allow_fallback: Fall back to ``title`` / ``text`` when the translated
            field is missing.
        limit: Process at most this many files, for smoke testing.
    """

    source: Path
    output: Path
    workers: int
    pretty: bool
    allow_fallback: bool
    limit: int | None


@dataclass(slots=True)
class Counts:
    """Tally of one run's outcomes.

    Attributes:
        total: Files seen.
        written: Files written to the output tree.
        skipped: Files with no usable text, or no ``article_id``.
        errors: Files that could not be read or parsed.
    """

    total: int = 0
    written: int = 0
    skipped: int = 0
    errors: int = 0

    def add(self, other: "Counts") -> None:
        """Fold another tally into this one.

        Args:
            other: Tally to absorb.
        """
        self.total += other.total
        self.written += other.written
        self.skipped += other.skipped
        self.errors += other.errors


# --------------------------------------------------------------------------
# Per-file work
# --------------------------------------------------------------------------


def extract(record: dict[str, Any], allow_fallback: bool) -> dict[str, str] | None:
    """Reduce one article record to the three fields worth keeping.

    Args:
        record: Parsed article JSON.
        allow_fallback: Accept ``title`` / ``text`` when the translated field
            is missing or blank.

    Returns:
        A dict of ``article_id`` / ``translated_title`` / ``translated_text``,
        or None when the record carries no id or no body text.
    """
    article_id = record.get(ID_FIELD)
    if not isinstance(article_id, str) or not article_id.strip():
        return None

    def pick(primary: str, fallback: str) -> str:
        value = record.get(primary)
        if isinstance(value, str) and value.strip():
            return value
        if allow_fallback:
            value = record.get(fallback)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    text = pick(TEXT_FIELD, TEXT_FALLBACK)
    if not text:
        return None

    return {
        ID_FIELD: article_id,
        TITLE_FIELD: pick(TITLE_FIELD, TITLE_FALLBACK),
        TEXT_FIELD: text,
    }


def process_batch(job: tuple[list[str], str, str, bool, bool]) -> tuple[Counts, list[str]]:
    """Extract every file in one batch. Runs in a worker process.

    Args:
        job: Tuple of (source paths, source root, output root, pretty,
            allow_fallback). Plain strings so the job pickles cheaply.

    Returns:
        Tuple of (tally, problem messages) for this batch.
    """
    paths, source_root, output_root, pretty, allow_fallback = job
    source, output = Path(source_root), Path(output_root)
    counts = Counts()
    problems: list[str] = []
    indent = 2 if pretty else None
    made: set[Path] = set()

    for raw in paths:
        path = Path(raw)
        counts.total += 1
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            counts.errors += 1
            problems.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue

        if not isinstance(record, dict):
            counts.skipped += 1
            problems.append(f"{path.name}: top level is {type(record).__name__}, not object")
            continue

        trimmed = extract(record, allow_fallback)
        if trimmed is None:
            counts.skipped += 1
            continue

        destination = output / path.relative_to(source)
        if destination.parent not in made:
            destination.parent.mkdir(parents=True, exist_ok=True)
            made.add(destination.parent)
        try:
            destination.write_text(
                json.dumps(trimmed, ensure_ascii=False, indent=indent),
                encoding="utf-8",
            )
        except OSError as exc:
            counts.errors += 1
            problems.append(f"{path.name}: write failed: {exc}")
            continue
        counts.written += 1

    return counts, problems


# --------------------------------------------------------------------------
# Driving a year
# --------------------------------------------------------------------------


def iter_batches(paths: list[Path], size: int) -> Iterator[list[str]]:
    """Slice a path list into batches of stringified paths.

    Args:
        paths: Source files.
        size: Maximum files per batch.

    Yields:
        Lists of at most ``size`` path strings.
    """
    for start in range(0, len(paths), size):
        yield [str(p) for p in paths[start:start + size]]


def run(settings: Settings, verbose: bool) -> dict[str, Any]:
    """Extract one year tree.

    Args:
        settings: Resolved paths and options for this year.
        verbose: Print every problem instead of the first few.

    Returns:
        A report dict, also written to the reports directory by the caller.
    """
    started = time.perf_counter()
    print("  scanning...", end="", flush=True)
    paths = sorted(settings.source.rglob("*.json"))
    if settings.limit is not None:
        paths = paths[:settings.limit]
    print(f"\r  {len(paths):,} source files")

    counts = Counts()
    problems: list[str] = []
    if not paths:
        return report_for(settings, counts, problems, time.perf_counter() - started)

    # Batches large enough that IPC overhead stays negligible next to the
    # per-file read/parse/write, small enough to keep every worker fed.
    batch_size = max(50, min(500, len(paths) // (settings.workers * 8) or 1))
    jobs = [
        (batch, str(settings.source), str(settings.output),
         settings.pretty, settings.allow_fallback)
        for batch in iter_batches(paths, batch_size)
    ]

    done = 0
    if settings.workers > 1:
        with ProcessPoolExecutor(max_workers=settings.workers) as pool:
            for batch_counts, batch_problems in pool.map(process_batch, jobs):
                counts.add(batch_counts)
                problems.extend(batch_problems)
                done += 1
                print(f"\r  {counts.total:,}/{len(paths):,} files"
                      f"  ({done}/{len(jobs)} batches)", end="", flush=True)
    else:
        for job in jobs:
            batch_counts, batch_problems = process_batch(job)
            counts.add(batch_counts)
            problems.extend(batch_problems)
            print(f"\r  {counts.total:,}/{len(paths):,} files", end="", flush=True)
    print()

    if problems:
        shown = problems if verbose else problems[:10]
        for line in shown:
            print(f"    ! {line}", file=sys.stderr)
        if len(problems) > len(shown):
            print(f"    ! ...and {len(problems) - len(shown):,} more "
                  f"(see the report, or use --verbose)", file=sys.stderr)

    return report_for(settings, counts, problems, time.perf_counter() - started)


def report_for(settings: Settings, counts: Counts, problems: list[str],
               elapsed: float) -> dict[str, Any]:
    """Assemble the per-year report.

    Args:
        settings: Settings the run used.
        counts: Final tally.
        problems: Problem messages collected by the workers.
        elapsed: Wall-clock seconds.

    Returns:
        The report dict.
    """
    return {
        "source": str(settings.source),
        "output": str(settings.output),
        "fields": [ID_FIELD, TITLE_FIELD, TEXT_FIELD],
        "fallback_to_source_fields": settings.allow_fallback,
        "summary": {
            "total_files": counts.total,
            "written_files": counts.written,
            "skipped_files": counts.skipped,
            "error_files": counts.errors,
            "runtime_seconds": round(elapsed, 1),
            "runtime_human": format_duration(elapsed),
        },
        "problems": problems,
    }


def format_duration(seconds: float) -> str:
    """Render a duration as ``1h02m03s`` / ``2m03s`` / ``3.4s``.

    Args:
        seconds: Elapsed seconds.

    Returns:
        Compact human-readable duration.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


# --------------------------------------------------------------------------
# Year arguments and status
# --------------------------------------------------------------------------


def parse_years(tokens: list[str]) -> list[int]:
    """Expand year tokens into a sorted, de-duplicated list of years.

    Accepts single years (``2022``) and inclusive ranges (``2019-2024``).

    Args:
        tokens: Raw command-line year tokens.

    Returns:
        Sorted unique years.

    Raises:
        SystemExit: If a token is not a year or a well-formed range.
    """
    low, high = YEAR_RANGE
    years: set[int] = set()

    def _one(token: str) -> int:
        if not token.isdigit():
            raise SystemExit(f"Not a year: {token!r}")
        value = int(token)
        if not low <= value <= high:
            raise SystemExit(f"Year out of range ({low}-{high}): {value}")
        return value

    for token in tokens:
        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2:
                raise SystemExit(f"Malformed year range: {token!r} (expected 2019-2024)")
            start, end = _one(parts[0]), _one(parts[1])
            if start > end:
                raise SystemExit(f"Reversed year range: {token!r}")
            years.update(range(start, end + 1))
        else:
            years.add(_one(token))
    return sorted(years)


def discover_years(source_root: Path) -> list[int]:
    """List years present under the source root.

    Args:
        source_root: Directory holding one subdirectory per year.

    Returns:
        Sorted years whose directory name is a plain four-digit year.
    """
    if not source_root.is_dir():
        return []
    low, high = YEAR_RANGE
    found: list[int] = []
    for entry in source_root.iterdir():
        if entry.is_dir() and entry.name.isdigit() and len(entry.name) == 4:
            if low <= int(entry.name) <= high:
                found.append(int(entry.name))
    return sorted(found)


def count_json_files(root: Path) -> int:
    """Count ``*.json`` files anywhere under a directory.

    Args:
        root: Directory to walk.

    Returns:
        Number of JSON files, or 0 when the directory is missing.
    """
    if not root.is_dir():
        return 0
    return sum(1 for _ in root.rglob("*.json"))


def output_is_populated(output_root: Path) -> bool:
    """Report whether an output tree already holds extracted articles.

    Args:
        output_root: Year output tree.

    Returns:
        True when at least one ``*.json`` file is present.
    """
    if not output_root.is_dir():
        return False
    try:
        return next(output_root.rglob("*.json"), None) is not None
    except OSError:
        return False


def print_status(source_root: Path, dest_root: Path) -> None:
    """Print which years exist at the source and which have been extracted.

    Args:
        source_root: Directory holding one subdirectory per year.
        dest_root: Root under which year trees are written.
    """
    print(f"source : {source_root}")
    print(f"dest   : {dest_root}")
    print(f"fields : {ID_FIELD}, {TITLE_FIELD}, {TEXT_FIELD}")
    print()
    years = discover_years(source_root)
    if not years:
        print(f"No year directories found under {source_root}")
        return
    print(f"{'year':<8}{'source files':>14}   {'status':<12}{'output files':>14}")
    print("-" * 52)
    for year in years:
        output = dest_root / str(year)
        done = output_is_populated(output)
        print(f"{year:<8}{count_json_files(source_root / str(year)):>14,}   "
              f"{'done' if done else 'pending':<12}"
              f"{count_json_files(output) if done else 0:>14,}")


def resolve_worker_count() -> int:
    """Choose a default worker count that leaves the machine usable.

    Returns:
        Number of worker processes.
    """
    return max(1, min(8, (os.cpu_count() or 2) - 1))


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="extract_english.py",
        description=f"Keep only {ID_FIELD}, {TITLE_FIELD} and {TEXT_FIELD} "
                    f"from each translated article.",
        epilog="examples: extract_english.py 2022 | extract_english.py 2019-2024 "
               "| extract_english.py --list",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("years", nargs="*",
                        help="Years to process: 2022, or a range 2019-2024.")
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT,
                        help="Root holding one <year> directory per year.")
    parser.add_argument("--dest-root", type=Path, default=DEST_ROOT,
                        help="Root under which <year> output trees are written.")
    parser.add_argument("--list", action="store_true",
                        help="Show available and already-extracted years, then exit.")
    parser.add_argument("--workers", type=int, default=resolve_worker_count(),
                        help="Worker process count; 1 runs in-process.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N files per year (smoke testing).")
    parser.add_argument("--compact", action="store_true",
                        help="Write output JSON without indentation.")
    parser.add_argument("--no-fallback", action="store_true",
                        help=f"Do not fall back to {TITLE_FALLBACK} / {TEXT_FALLBACK} "
                             f"when a translated field is missing.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Redo years whose output tree is already populated.")
    parser.add_argument("--verbose", action="store_true",
                        help="List every problem file, not just the first ten.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 when every requested year succeeded or was
        deliberately skipped, 1 otherwise.
    """
    args = parse_args(argv)

    if args.list:
        print_status(args.source_root, args.dest_root)
        return 0

    if not args.years:
        available = discover_years(args.source_root)
        hint = ", ".join(str(y) for y in available) if available else "none found"
        print(f"No years given. Available under {args.source_root}: {hint}",
              file=sys.stderr)
        print("Try: python extract_english.py 2022     (or --list)", file=sys.stderr)
        return 1

    years = parse_years(args.years)
    reports_dir = args.dest_root / REPORTS_DIRNAME

    print(f"source  : {args.source_root}\\<year>")
    print(f"dest    : {args.dest_root}\\<year>")
    print(f"fields  : {ID_FIELD}, {TITLE_FIELD}, {TEXT_FIELD}  (names preserved)")
    print(f"years   : {', '.join(str(y) for y in years)}")
    print(f"workers : {args.workers}")
    print()

    results: list[tuple[int, dict[str, Any] | None, str]] = []
    failures = 0
    batch_start = time.perf_counter()

    for year in years:
        source = args.source_root / str(year)
        output = args.dest_root / str(year)

        if not source.is_dir():
            print(f"[{year}] SKIP - source not found: {source}", file=sys.stderr)
            results.append((year, None, "no source"))
            failures += 1
            continue

        if output_is_populated(output) and not args.overwrite:
            print(f"[{year}] SKIP - already extracted: {output} (use --overwrite to redo)")
            results.append((year, None, "already done"))
            continue

        print(f"[{year}] {source}  ->  {output}")
        settings = Settings(
            source=source,
            output=output,
            workers=max(1, args.workers),
            pretty=not args.compact,
            allow_fallback=not args.no_fallback,
            limit=args.limit,
        )
        try:
            report = run(settings, args.verbose)
        except KeyboardInterrupt:
            print(f"\n[{year}] interrupted - stopping.", file=sys.stderr)
            results.append((year, None, "interrupted"))
            failures += 1
            break

        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / f"{year}_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        results.append((year, report, "ok"))
        if report["summary"]["error_files"]:
            failures += 1
        print()

    print_summary(results, time.perf_counter() - batch_start)
    return 1 if failures else 0


def print_summary(results: list[tuple[int, dict[str, Any] | None, str]],
                  elapsed: float) -> None:
    """Print one line per year plus a corpus-wide total.

    Args:
        results: Tuples of (year, report or None, status text).
        elapsed: Wall-clock seconds for the whole batch.
    """
    print("=" * 64)
    print(f"{'year':<8}{'input':>12}{'written':>12}{'skipped':>10}{'errors':>10}{'time':>12}")
    print("-" * 64)
    totals = dict.fromkeys(("total_files", "written_files", "skipped_files", "error_files"), 0)
    for year, report, status in results:
        if report is None:
            print(f"{year:<8}{status:>44}")
            continue
        summary = report["summary"]
        for key in totals:
            totals[key] += summary[key]
        print(f"{year:<8}{summary['total_files']:>12,}{summary['written_files']:>12,}"
              f"{summary['skipped_files']:>10,}{summary['error_files']:>10,}"
              f"{summary['runtime_human']:>12}")
    print("-" * 64)
    print(f"{'ALL':<8}{totals['total_files']:>12,}{totals['written_files']:>12,}"
          f"{totals['skipped_files']:>10,}{totals['error_files']:>10,}"
          f"{format_duration(elapsed):>12}")
    print("=" * 64)


if __name__ == "__main__":
    mp.freeze_support()
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
