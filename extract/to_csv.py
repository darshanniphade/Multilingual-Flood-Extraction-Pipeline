"""Flatten the extraction JSONL into a single CSV.

The prompt emits only fields that exist in each article, so the column set is not
known ahead of time - this makes two passes: one to discover every event field
that ever appears, one to write. Event columns are ordered by how often they
occur, so the densest data sits on the left.

One row per event. Articles with no event still get a row (contains_flood_event
false), so the CSV accounts for every article processed.

    python to_csv.py                        # all months -> data/extracted/extractions.csv
    python to_csv.py --months 2021_07       # one month
    python to_csv.py --only-floods          # skip rejected articles
    python to_csv.py --only-verifiable      # only rows with a date AND a location
    python to_csv.py --output C:/tmp/x.csv

By default, model-invented synonym field names (death_toll, description, ...) are
folded onto their canonical prompt field, and top-level keys the model leaked into
events (flooded_locations, flood_dates) are dropped as redundant. Pass
--no-canonicalize to keep the raw field names untouched.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.json"

# Identity columns copied straight off the record, in this order. The title key and
# any corpus passthrough fields are config-driven, so this is built per run.
def record_columns(cfg: dict) -> list[tuple[str, str]]:
    """[(csv column, record key)] for the leading identity block."""
    out, inp = cfg["output"], cfg["input"]
    columns = [
        ("article_id", "article_id"),
        # extract.py always writes the partition under "month"; it holds a year for
        # corpora partitioned that way, so the column can be renamed without
        # changing the on-disk records.
        (out.get("partition_column", "month"), "month"),
        ("publish_date", "publish_date"),
    ]
    title_key = out.get("title_key", "translated_title")
    columns.append((title_key, title_key))
    columns += [(key, key) for key in inp.get("passthrough_fields", [])]
    return columns


# summary columns derived from the extraction, appended after the identity block
EXTRACTION_COLUMNS = [
    "contains_flood_event",
    "is_verifiable_flood",
    "flood_dates",
    "flooded_locations",
    "event_index",
]

# Model-invented field names folded into their canonical prompt field. Deliberately
# conservative - only unambiguous synonyms. Unit-suffixed names (water_level_cm,
# flood_depth_cm) are intentionally NOT mapped because collapsing them onto
# water_level/water_depth would silently lose the unit. Ambiguous names (casualties,
# response, event, affected_areas) are left alone rather than guessed.
FIELD_ALIASES = {
    "description": "summary",
    "date": "event_date",
    "flood_date": "event_date",
    "flood_cause": "cause",
    "flood_depth": "water_depth",
    "death_toll": "deaths",
    "total_deaths": "deaths",
    "number_of_deaths": "deaths",
    "num_deaths": "deaths",
    "fatalities": "deaths",
    "people_dead": "deaths",
    "missing_people": "missing",
    "people_missing": "missing",
    "injured_people": "injured",
    "people_injured": "injured",
    "displaced_people": "displaced",
    "people_displaced": "displaced",
    "evacuated_people": "evacuated",
    "people_evacuated": "evacuated",
    "people_affected": "affected_people",
    "affected_people_count": "affected_people",
    "houses_damaged_count": "houses_damaged",
    "homes_damaged": "houses_damaged",
    "homes_destroyed": "houses_destroyed",
}

# Top-level record keys the model leaks verbatim into individual event objects.
# They already exist as base columns, so drop them from events to avoid redundant,
# confusingly-duplicated columns.
DROP_IN_EVENT = {
    "contains_flood_event",
    "is_verifiable_flood",
    "flood_dates",
    "flooded_locations",
    "events",
    "event_details",
}

# how many times each alias/drop fired, for the run summary
CONSOLIDATED = Counter()


def normalize_event(event: dict) -> dict:
    """Fold synonym field names onto canonical ones and drop leaked top-level keys.

    Canonical always wins: if an event carries both `deaths` and `death_toll`, the
    real `deaths` value is kept and the alias is discarded rather than overwriting."""
    out = {}
    for key, value in event.items():
        if key in DROP_IN_EVENT or key in FIELD_ALIASES:
            continue
        out[key] = value
    for key, value in event.items():
        if key in DROP_IN_EVENT:
            CONSOLIDATED[f"drop {key}"] += 1
            continue
        if key in FIELD_ALIASES:
            target = FIELD_ALIASES[key]
            if target not in out:
                out[target] = value
            CONSOLIDATED[f"{key} -> {target}"] += 1
    return out


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def jsonl_files(cfg: dict, months: list[str] | None) -> list[Path]:
    root = Path(cfg["paths"]["output_root"])
    files = sorted(p for p in root.glob("*.jsonl") if not p.name.endswith(".failed.jsonl"))
    if months:
        wanted = set(months)
        files = [p for p in files if p.stem in wanted]
    return files


def iter_records(files: list[Path]):
    for path in files:
        with path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(f"  skipping malformed line {path.name}:{line_no}", file=sys.stderr)


def cell(value, sep: str) -> str:
    """Scalars pass through; lists join on sep; anything nested becomes JSON."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    if isinstance(value, list):
        if all(isinstance(v, (str, int, float, bool)) for v in value):
            return sep.join(str(v) for v in value)
        return json.dumps(value, ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False)


def rows_for(record: dict, field: str, sep: str, id_columns: list[tuple[str, str]],
             normalize: bool = True):
    """Yields one row dict per event, or a single row if there are no events."""
    extraction = record.get(field) or {}
    base = {column: cell(record.get(key, ""), sep) for column, key in id_columns}
    base |= {
        "contains_flood_event": cell(extraction.get("contains_flood_event"), sep),
        "is_verifiable_flood": cell(extraction.get("is_verifiable_flood"), sep),
        "flood_dates": cell(extraction.get("flood_dates"), sep),
        "flooded_locations": cell(extraction.get("flooded_locations"), sep),
    }

    events = extraction.get("events")
    # tolerate the earlier flat-schema records, which used event_details
    if not events and isinstance(extraction.get("event_details"), dict):
        events = [extraction["event_details"]] if extraction["event_details"] else []

    if not events:
        yield {**base, "event_index": ""}, {}
        return

    for i, event in enumerate(events):
        if not isinstance(event, dict):
            event = {"value": event}
        if normalize:
            event = normalize_event(event)
        yield {**base, "event_index": i}, event


def main() -> int:
    parser = argparse.ArgumentParser(description="Flatten extraction JSONL into CSV")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--months", nargs="*", help="e.g. 2021_01 (default: all)")
    parser.add_argument("--output", type=Path, help="default: <output_root>/extractions.csv")
    parser.add_argument("--sep", default="; ", help="separator for list values (default '; ')")
    parser.add_argument("--only-floods", action="store_true", help="drop rejected articles")
    parser.add_argument("--only-verifiable", action="store_true", help="keep is_verifiable_flood only")
    parser.add_argument(
        "--min-fill",
        type=int,
        default=0,
        help="event fields occurring fewer than N times are folded into one "
        "extra_fields JSON column instead of getting their own (0 = keep all)",
    )
    parser.add_argument(
        "--no-canonicalize",
        dest="canonicalize",
        action="store_false",
        help="keep raw model field names; skip synonym folding and leak dropping",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    field = cfg["output"]["extraction_field"]
    root = cfg["paths"]["output_root"]
    id_columns = record_columns(cfg)
    files = jsonl_files(cfg, args.months)
    if not files:
        available = [p.stem for p in jsonl_files(cfg, None)]
        if args.months and available:
            print(
                f"no .jsonl files for month(s) {' '.join(args.months)}.\n"
                f"available: {' '.join(available)}",
                file=sys.stderr,
            )
        else:
            print(f"no .jsonl files under {root} - run extract.py first", file=sys.stderr)
        return 1

    out_path = args.output or Path(root) / "extractions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def wanted(base_row: dict) -> bool:
        if args.only_verifiable and base_row["is_verifiable_flood"] != "true":
            return False
        if args.only_floods and base_row["contains_flood_event"] != "true":
            return False
        return True

    # pass 1 - discover event columns and how common each one is
    print(f"scanning {len(files)} file(s)...", flush=True)
    counts: Counter[str] = Counter()
    total = kept = event_rows = 0
    for record in iter_records(files):
        for base_row, event in rows_for(record, field, args.sep, id_columns, args.canonicalize):
            total += 1
            if not wanted(base_row):
                continue
            kept += 1
            if event:
                event_rows += 1
            counts.update(event.keys())

    if not kept:
        print("nothing to write after filtering", file=sys.stderr)
        return 1

    # normalize_event runs again in pass 2, so freeze the tally from pass 1 now
    consolidated = CONSOLIDATED.copy()

    # The prompt lets the model name its own fields, so a long tail of one-off
    # names would otherwise become thousands of near-empty columns.
    rare = {k for k, n in counts.items() if n < args.min_fill} if args.min_fill else set()

    # densest columns first, ties broken alphabetically so the order is stable
    event_columns = [
        k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if k not in rare
    ]
    columns = (
        [column for column, _ in id_columns]
        + EXTRACTION_COLUMNS
        + event_columns
        + (["extra_fields"] if rare else [])
    )

    # pass 2 - write. The target is often open in Excel, which locks it on Windows;
    # rather than fail, fall back to a sibling filename so a run is never wasted.
    try:
        fh = out_path.open("w", encoding="utf-8-sig", newline="")
    except PermissionError:
        alt = out_path.with_name(f"{out_path.stem}.new{out_path.suffix}")
        print(
            f"  {out_path.name} is locked (open in Excel?) - writing {alt.name} instead",
            file=sys.stderr,
        )
        out_path = alt
        fh = out_path.open("w", encoding="utf-8-sig", newline="")
    with fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in iter_records(files):
            for base_row, event in rows_for(record, field, args.sep, id_columns, args.canonicalize):
                if not wanted(base_row):
                    continue
                row = dict(base_row)
                extras = {}
                for key, value in event.items():
                    if key in rare:
                        extras[key] = value
                    else:
                        row[key] = cell(value, args.sep)
                if extras:
                    row["extra_fields"] = json.dumps(extras, ensure_ascii=False)
                writer.writerow(row)

    print(f"wrote {kept} rows x {len(columns)} columns -> {out_path}")
    if kept != total:
        print(f"  ({total - kept} rows filtered out)")
    if rare:
        folded = sum(counts[k] for k in rare)
        print(
            f"  folded {len(rare)} field(s) seen <{args.min_fill}x into extra_fields "
            f"({folded} values preserved as JSON)"
        )
    if args.canonicalize and consolidated:
        merges = {k: v for k, v in consolidated.items() if not k.startswith("drop ")}
        drops = {k: v for k, v in consolidated.items() if k.startswith("drop ")}
        merged_vals = sum(merges.values())
        dropped_vals = sum(drops.values())
        print(
            f"  canonicalized {merged_vals:,} synonym value(s) across {len(merges)} name(s); "
            f"dropped {dropped_vals:,} leaked top-level value(s)"
        )
        for name, n in sorted(consolidated.items(), key=lambda kv: -kv[1])[:8]:
            print(f"     {name:28s} {n:,}")
    # Fill rate is measured against rows that actually carry an event, NOT all rows -
    # most rows are rejected non-flood articles with no event fields at all, and
    # including them would understate every column several-fold.
    print(
        f"\ntop event columns by fill rate "
        f"(share of the {event_rows:,} rows that have an event; "
        f"{kept - event_rows:,} rows have none):"
    )
    for name, n in counts.most_common(15):
        print(f"  {name:22s} {n:6d}  {100 * n / event_rows:5.1f}%" if event_rows else name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
