"""Audit tool: which rules are driving matches, and do the hits look right?

    python audit_matches.py --top 30 --samples 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from flood_lexicon import is_flood_title

# The corpus is multilingual and the Windows console defaults to cp1252, which
# raises UnicodeEncodeError on the first Polish/Turkish/Vietnamese title.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_INPUT = Path(r"C:\darsh\pipeline\2_filter_titles\translated_titles.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--limit", type=int, default=400000,
                    help="scan only the first N lines (0 = all)")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    reasons: Counter = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    scanned = 0
    matched = 0

    with args.input.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if args.limit and scanned >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            scanned += 1
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            title = rec.get("translated_title")
            if not isinstance(title, str):
                continue
            hit, reason = is_flood_title(title, explain=True)
            if not hit:
                continue
            matched += 1
            # Bucket by the rule that fired, not the specific text.
            key = reason
            reasons[key] += 1
            bucket = examples[key]
            if len(bucket) < 40:
                bucket.append(title)

    print("scanned %d, matched %d (%.2f%%)\n" % (
        scanned, matched, matched / scanned * 100 if scanned else 0))
    print("top %d firing rules" % args.top)
    print("=" * 78)
    for key, n in reasons.most_common(args.top):
        print("%7d  %5.2f%%  %s" % (n, n / matched * 100, key))
        pool = examples[key]
        for t in rng.sample(pool, min(args.samples, len(pool))):
            print("           - %s" % t[:110])
        print()


if __name__ == "__main__":
    main()
