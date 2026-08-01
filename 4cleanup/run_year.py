r"""Reusable per-year driver for the cleaning stage.

Wraps :mod:`clean_articles` so a whole calendar year -- or several -- can be
cleaned with one command, writing each year into its own mirrored output tree
with its own report and log.

    python run_year.py 2022                  # one year
    python run_year.py 2022 2023 2024        # several
    python run_year.py 2019-2024             # an inclusive range
    python run_year.py --list                # what is available / already done

Layout
------
Source and field names come from a *profile*, because the corpus exists in two
shapes and the fields differ between them::

    raw         H:\<year>\<MM>\articles\*.json          title / text
                (stage-0 GDELT scrape, multilingual)

    translated  ...\translated_articles\<year>\...      translated_title /
                (stage-3 translator output, English)    translated_text

Field names are preserved -- this stage cleans text, it never renames and it
never translates -- so a cleaned raw tree is ``article_id`` / ``title`` /
``text`` and a cleaned translated tree is ``article_id`` /
``translated_title`` / ``translated_text``.  Output mirrors the input tree::

    C:\darsh\pipeline\data\ssd\<year>\<MM>\articles\*.json
    C:\darsh\pipeline\data\ssd\_reports\<year>_report.json
    C:\darsh\pipeline\data\ssd\_reports\<year>.log

Deduplication is **per year**: each year builds a fresh SHA-256 index, so a
story republished across a year boundary survives in both years. Runs are
therefore independent and order does not matter.

Adding a year is nothing but a new directory under the source root -- no edit
to this file is required.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import clean_articles
import config
import utils
from config import PipelineSettings

# --------------------------------------------------------------------------
# Site configuration -- edit these four values, not the logic below
# --------------------------------------------------------------------------

#: Where cleaned year trees are written.
DEST_ROOT: Final[Path] = Path(r"C:\darsh\pipeline\data\ssd")

#: Subdirectory of DEST_ROOT holding per-year reports and logs. Leading
#: underscore keeps it sorted away from the year directories, and it sits
#: outside any year tree so the non-empty-output guard never sees it.
REPORTS_DIRNAME: Final[str] = "_reports"

#: Lowest and highest plausible year, used to validate CLI input.
YEAR_RANGE: Final[tuple[int, int]] = (1990, 2100)


@dataclass(frozen=True, slots=True)
class Profile:
    """A source corpus shape: where it lives and which fields carry the text.

    Attributes:
        source_root: Directory containing one subdirectory per year.
        title_field: Source field read as the article title.
        text_field: Source field read as the article body.
        note: One-line description shown in the run banner.
    """

    source_root: Path
    title_field: str
    text_field: str
    note: str


#: Known corpus shapes. Add an entry here to support a new source layout.
PROFILES: Final[dict[str, Profile]] = {
    "raw": Profile(
        # Forward slash on purpose: Path("H:") means "current directory on
        # drive H", not the drive root. Path("H:/") is the root.
        source_root=Path("H:/"),
        title_field="title",
        text_field="text",
        note="stage-0 GDELT scrape; MULTILINGUAL -- text is cleaned, not translated",
    ),
    "translated": Profile(
        source_root=config.INPUT_ROOT,
        title_field="translated_title",
        text_field="translated_text",
        note="stage-3 translator output; English",
    ),
}

#: Profile used when ``--profile`` is not given.
DEFAULT_PROFILE: Final[str] = "raw"


# --------------------------------------------------------------------------
# Year arguments
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


def discover_years(profile: Profile) -> list[int]:
    """List years present under a profile's source root.

    Args:
        profile: Corpus shape to inspect.

    Returns:
        Sorted years whose directory name is a plain four-digit year.
    """
    if not profile.source_root.is_dir():
        return []
    low, high = YEAR_RANGE
    found: list[int] = []
    for entry in profile.source_root.iterdir():
        if entry.is_dir() and entry.name.isdigit() and len(entry.name) == 4:
            if low <= int(entry.name) <= high:
                found.append(int(entry.name))
    return sorted(found)


# --------------------------------------------------------------------------
# Paths and status
# --------------------------------------------------------------------------


def paths_for(year: int, profile: Profile, dest_root: Path) -> tuple[Path, Path, Path, Path]:
    """Resolve the four paths a year's run reads from and writes to.

    Args:
        year: Calendar year.
        profile: Corpus shape supplying the source root.
        dest_root: Root under which year trees are written.

    Returns:
        Tuple of (source tree, output tree, report path, log path).
    """
    reports = dest_root / REPORTS_DIRNAME
    return (
        profile.source_root / str(year),
        dest_root / str(year),
        reports / f"{year}_report.json",
        reports / f"{year}.log",
    )


def output_is_populated(output_root: Path) -> bool:
    """Report whether an output tree already holds cleaned articles.

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


def print_status(profile_name: str, profile: Profile, dest_root: Path) -> None:
    """Print which years exist at the source and which have been cleaned.

    Args:
        profile_name: Name of the selected profile.
        profile: The selected profile.
        dest_root: Root under which year trees are written.
    """
    years = discover_years(profile)
    print(f"profile   : {profile_name}  ({profile.note})")
    print(f"source    : {profile.source_root}")
    print(f"dest      : {dest_root}")
    print(f"fields    : {profile.title_field} / {profile.text_field}")
    print()
    if not years:
        print(f"No year directories found under {profile.source_root}")
        return
    print(f"{'year':<8}{'source files':>14}   {'status':<12}{'cleaned files':>14}")
    print("-" * 52)
    for year in years:
        source, output, report, _ = paths_for(year, profile, dest_root)
        n_in = utils.count_json_files(source)
        done = output_is_populated(output)
        n_out = utils.count_json_files(output) if done else 0
        status = "cleaned" if done else "pending"
        if done and not report.exists():
            status = "cleaned*"
        print(f"{year:<8}{n_in:>14,}   {status:<12}{n_out:>14,}")


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


def build_settings(year: int, profile: Profile, args: argparse.Namespace) -> PipelineSettings:
    """Assemble the pipeline settings for one year.

    Args:
        year: Calendar year to process.
        profile: Corpus shape supplying source root and field names.
        args: Parsed command-line arguments.

    Returns:
        Settings ready to hand to :func:`clean_articles.run`.
    """
    source, output, report, log = paths_for(year, profile, args.dest_root)
    return PipelineSettings(
        input_root=source,
        output_root=output,
        report_path=report,
        log_path=log,
        workers=max(1, args.workers),
        batch_size=max(1, args.batch_size),
        min_word_count=max(0, args.min_words),
        pretty=not args.compact,
        deduplicate=not args.no_dedup,
        count_first=not args.no_count,
        limit=args.limit,
        title_field=args.title_field or profile.title_field,
        text_field=args.text_field or profile.text_field,
    )


def print_summary(results: list[tuple[int, dict[str, Any] | None, str]], elapsed: float) -> None:
    """Print one line per year plus a corpus-wide total.

    Args:
        results: Tuples of (year, report or None, status text).
        elapsed: Wall-clock seconds for the whole batch.
    """
    print()
    print("=" * 78)
    print(
        f"{'year':<7}{'input':>11}{'written':>11}{'skipped':>10}"
        f"{'dupes':>10}{'errors':>8}{'time':>11}"
    )
    print("-" * 78)
    totals = dict.fromkeys(("total_files", "output_files", "skipped_files",
                            "duplicate_files", "error_files"), 0)
    for year, report, status in results:
        if report is None:
            print(f"{year:<7}{status:>51}")
            continue
        summary = report["summary"]
        for key in totals:
            totals[key] += summary[key]
        print(
            f"{year:<7}{summary['total_files']:>11,}{summary['output_files']:>11,}"
            f"{summary['skipped_files']:>10,}{summary['duplicate_files']:>10,}"
            f"{summary['error_files']:>8,}{summary['runtime_human']:>11}"
        )
    print("-" * 78)
    print(
        f"{'ALL':<7}{totals['total_files']:>11,}{totals['output_files']:>11,}"
        f"{totals['skipped_files']:>10,}{totals['duplicate_files']:>10,}"
        f"{totals['error_files']:>8,}{utils.format_duration(elapsed):>11}"
    )
    print("=" * 78)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="run_year.py",
        description="Clean one or more calendar years into per-year output trees.",
        epilog="examples: run_year.py 2022 | run_year.py 2019-2024 | run_year.py --list",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("years", nargs="*",
                        help="Years to process: 2022, or a range 2019-2024.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default=DEFAULT_PROFILE,
                        help="Source corpus shape (sets source root and field names).")
    parser.add_argument("--dest-root", type=Path, default=DEST_ROOT,
                        help="Root under which <year> output trees are written.")
    parser.add_argument("--source-root", type=Path, default=None,
                        help="Override the profile's source root.")
    parser.add_argument("--title-field", type=str, default=None,
                        help="Override the profile's title field.")
    parser.add_argument("--text-field", type=str, default=None,
                        help="Override the profile's text field.")
    parser.add_argument("--list", action="store_true",
                        help="Show available and already-cleaned years, then exit.")
    parser.add_argument("--workers", type=int, default=config.resolve_worker_count(),
                        help="Worker process count.")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                        help="Files per IPC batch.")
    parser.add_argument("--min-words", type=int, default=config.MIN_WORD_COUNT,
                        help="Minimum words required after cleaning.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N files per year (smoke testing).")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Disable exact-duplicate removal.")
    parser.add_argument("--no-count", action="store_true",
                        help="Skip the pre-scan; progress bar runs without a total.")
    parser.add_argument("--compact", action="store_true",
                        help="Write output JSON without indentation.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-clean years whose output tree is already populated.")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-file DEBUG output on the console.")
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
    profile = PROFILES[args.profile]
    if args.source_root is not None:
        profile = Profile(args.source_root, profile.title_field,
                          profile.text_field, profile.note)

    if args.list:
        print_status(args.profile, profile, args.dest_root)
        return 0

    if not args.years:
        available = discover_years(profile)
        hint = ", ".join(str(y) for y in available) if available else "none found"
        print(f"No years given. Available under {profile.source_root}: {hint}",
              file=sys.stderr)
        print("Try: python run_year.py 2022     (or --list)", file=sys.stderr)
        return 1

    years = parse_years(args.years)
    title_field = args.title_field or profile.title_field
    text_field = args.text_field or profile.text_field

    print(f"profile : {args.profile}  --  {profile.note}")
    print(f"source  : {profile.source_root}\\<year>")
    print(f"dest    : {args.dest_root}\\<year>")
    print(f"fields  : article_id, {title_field}, {text_field}  (names preserved, no translation)")
    print(f"years   : {', '.join(str(y) for y in years)}")
    print(f"dedup   : {'per year (independent)' if not args.no_dedup else 'disabled'}")
    print()

    results: list[tuple[int, dict[str, Any] | None, str]] = []
    failures = 0
    batch_start = time.perf_counter()

    for year in years:
        source, output, _, _ = paths_for(year, profile, args.dest_root)

        if not source.is_dir():
            print(f"[{year}] SKIP - source not found: {source}", file=sys.stderr)
            results.append((year, None, "no source"))
            failures += 1
            continue

        if output_is_populated(output) and not args.overwrite:
            print(f"[{year}] SKIP - already cleaned: {output} (use --overwrite to redo)")
            results.append((year, None, "already done"))
            continue

        print(f"[{year}] {source}  ->  {output}")
        try:
            report = clean_articles.run(
                build_settings(year, profile, args), args.overwrite, args.verbose
            )
        except SystemExit as exc:
            # One bad year must not abort the rest of the batch.
            print(f"[{year}] FAILED - {exc}", file=sys.stderr)
            results.append((year, None, "failed"))
            failures += 1
            continue
        except KeyboardInterrupt:
            print(f"\n[{year}] interrupted - stopping batch.", file=sys.stderr)
            results.append((year, None, "interrupted"))
            failures += 1
            break
        results.append((year, report, "ok"))
        print()

    print_summary(results, time.perf_counter() - batch_start)
    return 1 if failures else 0


if __name__ == "__main__":
    mp.freeze_support()
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
