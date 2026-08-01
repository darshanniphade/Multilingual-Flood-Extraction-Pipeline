"""Build a flood-article index from translated titles.

Reads the translated-title JSONL produced by the title_tranlator stage, keeps
only the records whose `translated_title` describes a real flood / hydro-met
event, and writes them out partitioned by month:

    C:\\darsh\\pipeline\\flood_events\\2021_01.jsonl
    ...
    C:\\darsh\\pipeline\\flood_events\\2021_12.jsonl

Only `article_id` and `translated_title` are emitted - titles are copied
verbatim, never translated, rewritten or summarised. Detection logic lives in
flood_lexicon.py.

Run:
    C:\\darsh\\AI_MODELS\\translator_env\\Scripts\\python.exe filter_flood_titles.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from flood_lexicon import is_flood_title

DEFAULT_INPUT = Path(r"C:\darsh\pipeline\ttilte_filter\translated_titles.jsonl")
DEFAULT_OUTPUT_DIR = Path(r"C:\darsh\pipeline\flood_events")

# article_id looks like "2021_07/article_000000002"; the prefix decides which
# monthly output file the record lands in.
_MONTH_RE = re.compile(r"^(\d{4}_\d{2})[/\\]")


def iter_input_files(root: Path) -> list[Path]:
    """A single .jsonl file, or every .jsonl under a directory, recursively."""
    if root.is_file():
        return [root]
    if root.is_dir():
        return sorted(p for p in root.rglob("*.jsonl") if p.is_file())
    raise SystemExit("input path does not exist: %s" % root)


def count_lines(path: Path) -> int:
    """Fast binary line count, used to give tqdm a real total."""
    total = 0
    with path.open("rb") as fh:
        while True:
            block = fh.read(8 << 20)
            if not block:
                break
            total += block.count(b"\n")
    return total


class MonthWriter:
    """Lazily opens one output file per month and routes records to it."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.handles: dict[str, object] = {}
        self.counts: Counter = Counter()

    def write(self, month: str, record: dict) -> None:
        fh = self.handles.get(month)
        if fh is None:
            fh = (self.out_dir / ("%s.jsonl" % month)).open(
                "w", encoding="utf-8", newline="\n"
            )
            self.handles[month] = fh
        fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        fh.write("\n")
        self.counts[month] += 1

    def close(self) -> None:
        for fh in self.handles.values():
            fh.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                    help="translated-title .jsonl file, or a dir to scan recursively")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help="directory for the per-month flood index")
    ap.add_argument("--sample-misses", type=int, default=0, metavar="N",
                    help="also dump N near-miss titles for tuning review")
    args = ap.parse_args()

    files = iter_input_files(args.input)
    if not files:
        raise SystemExit("no .jsonl files found under %s" % args.input)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = MonthWriter(args.output_dir)

    scanned = 0
    matched = 0
    malformed = 0
    missing_field = 0
    started = time.perf_counter()

    total_lines = sum(count_lines(p) for p in files)

    with tqdm(total=total_lines, unit=" titles", unit_scale=True,
              desc="scanning", smoothing=0.05) as bar:
        for path in files:
            fallback_month = path.stem
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    bar.update(1)
                    line = line.strip()
                    if not line:
                        continue
                    scanned += 1
                    try:
                        rec = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        malformed += 1
                        continue
                    if not isinstance(rec, dict):
                        malformed += 1
                        continue

                    article_id = rec.get("article_id")
                    title = rec.get("translated_title")
                    if not article_id or not isinstance(title, str):
                        missing_field += 1
                        continue

                    if not is_flood_title(title):
                        continue

                    m = _MONTH_RE.match(article_id)
                    month = m.group(1) if m else fallback_month
                    writer.write(month, {"article_id": article_id,
                                         "translated_title": title})
                    matched += 1
                    bar.set_postfix(found=matched, refresh=False)

    writer.close()
    elapsed = time.perf_counter() - started

    pct = (matched / scanned * 100) if scanned else 0.0
    print("\n" + "=" * 58)
    print("Files processed          : %d" % len(files))
    print("Titles scanned           : %d" % scanned)
    print("Flood-related titles     : %d  (%.2f%%)" % (matched, pct))
    if malformed:
        print("Malformed lines skipped  : %d" % malformed)
    if missing_field:
        print("Records missing fields   : %d" % missing_field)
    print("Processing time          : %.2fs" % elapsed)
    print("Output directory         : %s" % args.output_dir)
    print("-" * 58)
    for month in sorted(writer.counts):
        print("  %s.jsonl%s%6d" % (month, " " * 8, writer.counts[month]))
    print("=" * 58)
    return 0


if __name__ == "__main__":
    sys.exit(main())
