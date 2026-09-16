"""Stage 7g - grounding: is the extraction actually supported by the article?

Stage 5 asks a 14B model for structured JSON and gets structured JSON back. That
the JSON is well-formed says nothing about whether "deaths: 31" appears anywhere
in the article it was extracted from. This script checks every field back
against the source text and records, per field, whether it is supported and by
which span.

What grounding is and is not:

    grounded = true    the value is stated in the article the model read
    grounded = false   it is not - the model inferred, aggregated, or invented it

Neither says the value is TRUE in the world; the article can be wrong. Grounding
separates "the model read this" from "the model produced this", which is the
distinction that matters when an LLM is in the loop, and it is the only claim
this file makes.

How each field is checked:

    dates        the ISO date, or any surface form of it ("5 July 2021",
                 "July 5, 2021", "05/07/2021"), appears in the text
    counts       the number appears as digits, with thousands separators
                 (Western or Indian grouping), scaled ("1.2 million", "5 lakh"),
                 or spelled out ("twelve")
    places,      the string appears, or enough of its content tokens do
    rivers,      (grounding.token_match_min_ratio); a LOCATION is additionally
    country      cross-checked against what the NER tagger sees as a place
    flood_type   the cue phrase that decided the label is in the text
    cause        the phrase, or its content tokens, appear

    python ground.py --sample 2000
    python ground.py --years 2021
    python ground.py --force

Outputs under paths.output_root:

    grounding.jsonl      per article: every field with grounded/evidence
    grounding_report.md  measured support rate per field, worked failures
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from common import (
    DEFAULT_CONFIG,
    NUMERIC_SLOTS,
    Event,
    JsonlAppender,
    fmt_duration,
    iter_bodies,
    iter_events,
    load_config,
    load_sidecar,
    location_tokens,
    md_table,
    normalize_text,
    number_surface_forms,
    output_root,
    parse_iso,
    seed_everything,
    setup_logging,
    to_number,
    write_report,
)

log = logging.getLogger("events.ground")

_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def date_surface_forms(d: date) -> list[str]:
    """Every way a newsroom writes one date. Deliberately does not include a
    bare weekday - "Thursday" is not evidence for a specific date."""
    month = _MONTH_NAMES[d.month - 1]
    short = month[:3]
    return [
        d.isoformat(),
        f"{d.day} {month} {d.year}",
        f"{d.day} {month}",
        f"{month} {d.day}, {d.year}",
        f"{month} {d.day} {d.year}",
        f"{month} {d.day}",
        f"{short} {d.day}, {d.year}",
        f"{short} {d.day}",
        f"{d.day}/{d.month}/{d.year}",
        f"{d.month}/{d.day}/{d.year}",
        f"{d.day:02d}/{d.month:02d}/{d.year}",
        f"{d.day:02d}-{d.month:02d}-{d.year}",
    ]


class Grounder:
    def __init__(self, cfg: dict, ner_index: dict[str, dict] | None = None):
        gc = cfg["grounding"]
        self.cfg = cfg
        self.window = int(gc.get("evidence_window", 160))
        self.max_chars = int(gc.get("max_body_chars", 20000))
        self.min_ratio = float(gc.get("token_match_min_ratio", 0.6))
        self.ner_enabled = bool(gc.get("ner_enabled", True))
        self.ner_index = ner_index or {}

    # -- primitives ---------------------------------------------------------

    def _evidence(self, text: str, start: int, length: int) -> str:
        lo = max(0, start - self.window // 2)
        hi = min(len(text), start + length + self.window // 2)
        return " ".join(text[lo:hi].split())

    def find_any(self, text_lower: str, text: str, forms: list[str]) -> tuple[bool, str, str]:
        for form in forms:
            if not form:
                continue
            idx = text_lower.find(form.lower())
            if idx >= 0:
                return True, self._evidence(text, idx, len(form)), form
        return False, "", ""

    def token_support(self, text_lower: str, text: str, phrase: str) -> tuple[bool, str, float]:
        """Fraction of the phrase's content tokens present in the text. Catches
        "Kishtwar district" when the article writes "district of Kishtwar"."""
        tokens = location_tokens(phrase) or set(normalize_text(phrase).split())
        if not tokens:
            return False, "", 0.0
        present = [t for t in tokens if re.search(rf"\b{re.escape(t)}\b", text_lower)]
        ratio = len(present) / len(tokens)
        evidence = ""
        if present:
            m = re.search(rf"\b{re.escape(present[0])}\b", text_lower)
            if m:
                evidence = self._evidence(text, m.start(), len(present[0]))
        return ratio >= self.min_ratio, evidence, round(ratio, 3)

    # -- field checks -------------------------------------------------------

    def check_string(self, text: str, text_lower: str, field: str, value: str, ner_labels: set[str] | None = None) -> dict:
        ok, evidence, form = self.find_any(text_lower, text, [value])
        method = "exact"
        ratio = 1.0 if ok else 0.0
        if not ok:
            ok, evidence, ratio = self.token_support(text_lower, text, value)
            method = "tokens"
        row = {
            "field": field,
            "value": value,
            "grounded": bool(ok),
            "method": method if ok else "none",
            "evidence": evidence,
            "token_ratio": ratio,
        }
        if ner_labels is not None:
            row["ner_confirms_place"] = normalize_text(value) in ner_labels
        return row

    def check_number(self, text: str, text_lower: str, field: str, value) -> dict:
        number = to_number(value)
        if number is None:
            return {"field": field, "value": value, "grounded": False, "method": "unparseable", "evidence": ""}
        forms = number_surface_forms(number)
        ok, evidence, form = self.find_any(text_lower, text, forms)
        return {
            "field": field,
            "value": number,
            "grounded": bool(ok),
            "method": f"surface:{form}" if ok else "none",
            "evidence": evidence,
        }

    def check_date(self, text: str, text_lower: str, field: str, value: str, publish_date: str | None = None) -> dict:
        parsed = parse_iso(value)
        forms = date_surface_forms(parsed) if parsed else [value]
        ok, evidence, form = self.find_any(text_lower, text, forms)
        row = {
            "field": field,
            "value": value,
            "grounded": bool(ok),
            "method": f"surface:{form}" if ok else "none",
            "evidence": evidence,
        }
        if not ok and parsed:
            # The dominant failure mode is a weekday-relative date: the article
            # writes "flooding on Saturday" and the model resolved it against
            # the publication date. Still ungrounded - the date is inferred, not
            # stated - but worth separating from an invented date, so the report
            # can count how much of the gap is this.
            weekday = _WEEKDAYS[parsed.weekday()]
            published = parse_iso(publish_date or "")
            near_publication = published is not None and abs((published - parsed).days) <= 7
            if re.search(rf"\b{weekday.lower()}\b", text_lower) and near_publication:
                m = re.search(rf"\b{weekday.lower()}\b", text_lower)
                row["method"] = "weekday_relative"
                row["evidence"] = self._evidence(text, m.start(), len(weekday))
        return row

    # -- one article --------------------------------------------------------

    def ground_event(self, ev: Event, text: str, flood_type_row: dict | None) -> dict:
        text = text[: self.max_chars]
        text_lower = text.lower()
        fields: list[dict] = []

        ner_places: set[str] | None = None
        if self.ner_enabled:
            row = self.ner_index.get(ev.uid)
            if row:
                ner_places = {
                    normalize_text(e["text"])
                    for e in row.get("entities", [])
                    if e.get("label") in {"LOCATION", "RIVER"} and e.get("source") == "model"
                }

        for value in ev.all_dates():
            fields.append(self.check_date(text, text_lower, "event_date", value, ev.publish_date))
        if ev.country:
            fields.append(self.check_string(text, text_lower, "country", ev.country, ner_places))
        for place in ev.places():
            fields.append(self.check_string(text, text_lower, "locations", place, ner_places))
        for river in ev.rivers():
            fields.append(self.check_string(text, text_lower, "rivers", river, ner_places))
        for cause in ev.causes():
            fields.append(self.check_string(text, text_lower, "cause", cause))
        for slot, value in ev.counts().items():
            fields.append(self.check_number(text, text_lower, slot, value))
        if flood_type_row and flood_type_row.get("flood_type") not in (None, "", "Unknown"):
            cue = flood_type_row.get("cue") or ""
            ok, evidence, _ = self.find_any(text_lower, text, [cue]) if cue else (False, "", "")
            fields.append(
                {
                    "field": "flood_type",
                    "value": flood_type_row["flood_type"],
                    "grounded": bool(ok),
                    "method": f"cue:{cue}" if ok else "none",
                    "evidence": evidence,
                }
            )

        grounded = sum(1 for f in fields if f["grounded"])
        return {
            "uid": ev.uid,
            "year": int(ev.year),
            "n_fields": len(fields),
            "n_grounded": grounded,
            "grounded_ratio": round(grounded / len(fields), 4) if fields else None,
            "text_len": len(text),
            "fields": fields,
        }


def reservoir(stream, k: int, rng: random.Random) -> list:
    out: list = []
    for i, item in enumerate(stream):
        if i < k:
            out.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                out[j] = item
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Check every extracted field against its source text")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--years", nargs="*")
    ap.add_argument("--select", choices=["verifiable", "flood", "all"], default="verifiable")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sample", type=int)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    seed = seed_everything(cfg)
    root = output_root(cfg)
    out_path = root / "grounding.jsonl"

    appender = JsonlAppender(out_path)
    done = set() if args.force else appender.done_ids()
    if args.force and out_path.exists():
        out_path.unlink()

    ner_index = load_sidecar(root / "ner.jsonl")
    flood_types = load_sidecar(root / "flood_type.jsonl")
    log.info("sidecars: ner=%d flood_type=%d; already grounded=%d", len(ner_index), len(flood_types), len(done))
    if cfg["grounding"].get("ner_enabled", True) and not ner_index:
        log.warning("grounding.ner_enabled is on but ner.jsonl is empty - the NER cross-check will be skipped. Run ner.py first.")

    stream = iter_events(cfg, args.years, select=args.select, limit=args.limit, skip_uids=done)
    events = reservoir(stream, args.sample, random.Random(seed)) if args.sample else stream
    if args.sample:
        log.info("sampled %d extractions (seed=%d)", len(events), seed)

    grounder = Grounder(cfg, ner_index)
    per_field = defaultdict(lambda: [0, 0])   # field -> [grounded, total]
    ungrounded_examples: dict[str, list[dict]] = defaultdict(list)
    ner_confirm = [0, 0]
    stats = Counter()
    ratios: list[float] = []
    started = time.time()

    with appender:
        for ev, body in iter_bodies(cfg, events, workers=int(cfg["grounding"].get("io_workers", 16))):
            if not body.strip():
                stats["body_missing"] += 1
                continue
            row = grounder.ground_event(ev, body, flood_types.get(ev.uid))
            appender.write(row)
            stats["articles"] += 1
            if row["grounded_ratio"] is not None:
                ratios.append(row["grounded_ratio"])
            for field in row["fields"]:
                if field.get("method") == "weekday_relative":
                    stats["weekday_relative_dates"] += 1
                bucket = per_field[field["field"]]
                bucket[1] += 1
                if field["grounded"]:
                    bucket[0] += 1
                elif len(ungrounded_examples[field["field"]]) < 8:
                    ungrounded_examples[field["field"]].append({"uid": row["uid"], "value": field["value"]})
                if "ner_confirms_place" in field:
                    ner_confirm[1] += 1
                    ner_confirm[0] += int(bool(field["ner_confirms_place"]))
            if stats["articles"] % 2000 == 0:
                log.info("  %d grounded (%.0f/s)", stats["articles"], stats["articles"] / max(time.time() - started, 1e-6))

    seconds = time.time() - started
    field_rows = []
    for field, (ok, total) in sorted(per_field.items(), key=lambda kv: -kv[1][1]):
        field_rows.append([field, total, ok, total - ok, f"{100 * ok / max(total, 1):.1f}%"])
    numeric_rows = [r for r in field_rows if r[0] in NUMERIC_SLOTS]
    text_rows = [r for r in field_rows if r[0] not in NUMERIC_SLOTS]

    mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    fully = sum(1 for r in ratios if r >= 0.999)
    sections = [
        (
            "Run",
            md_table(
                ["", "value"],
                [
                    ["select", args.select],
                    ["years", " ".join(args.years) if args.years else "all"],
                    ["sample", args.sample or "-"],
                    ["articles checked", stats["articles"]],
                    ["bodies missing on disk", stats["body_missing"]],
                    ["field assertions checked", sum(t for _ok, t in per_field.values())],
                    ["mean share of an article's fields that are grounded", f"{100 * mean_ratio:.1f}%"],
                    ["articles with every field grounded", f"{fully} ({100 * fully / max(len(ratios), 1):.1f}%)"],
                    ["NER cross-check available", "yes" if ner_index else "no (run ner.py)"],
                    ["wall clock", fmt_duration(seconds)],
                ],
                align=["---", "---:"],
            ),
        ),
        (
            "Support rate by field - text fields",
            md_table(
                ["field", "assertions", "grounded", "ungrounded", "support rate"],
                text_rows,
                align=["---", "---:", "---:", "---:", "---:"],
            ),
        ),
        (
            "Support rate by field - counts",
            md_table(
                ["field", "assertions", "grounded", "ungrounded", "support rate"],
                numeric_rows,
                align=["---", "---:", "---:", "---:", "---:"],
            )
            + [
                "",
                "An ungrounded count is the interesting case: the model produced a number the article "
                "never printed. Common legitimate causes are a total the article gives in words this "
                "checker does not generate, and a figure summed across paragraphs - both are still "
                "unsupported *as extracted*.",
            ],
        ),
        (
            "Why dates fail",
            md_table(
                ["", "count"],
                [
                    ["event_date assertions", per_field.get("event_date", [0, 0])[1]],
                    ["grounded (a surface form of the date is in the text)", per_field.get("event_date", [0, 0])[0]],
                    ["weekday-relative: the text says a weekday within 7 days of publication", stats["weekday_relative_dates"]],
                ],
                align=["---", "---:"],
            )
            + [
                "",
                "A weekday-relative date is one the article expresses as 'flooding on Saturday' and the model "
                "resolved against the publication date. It stays ungrounded - the ISO date is inferred, not "
                "stated - but it is a different failure from an invented date, and this row measures how much "
                "of the gap it accounts for.",
            ],
        ),
        (
            "Ungrounded countries",
            [
                "The country is the field the model most often supplies from world knowledge rather than "
                "from the text: a Malaysian outlet writing for Malaysian readers rarely prints the word "
                "'Malaysia'. That inference is usually right and always unsupported, which is exactly the "
                "distinction this stage exists to record.",
            ],
        ),
        (
            "NER cross-check on locations",
            md_table(
                ["", "count"],
                [
                    ["location/river assertions with an NER opinion", ner_confirm[1]],
                    ["also tagged as a place by the NER model", ner_confirm[0]],
                    ["agreement", f"{100 * ner_confirm[0] / max(ner_confirm[1], 1):.1f}%"],
                ],
                align=["---", "---:"],
            )
            + [
                "",
                "Disagreement is not automatically an extraction error: the tagger is a generic "
                "CoNLL-2003 model and misses South and Southeast Asian place names constantly.",
            ],
        ),
    ]
    for field, examples in list(ungrounded_examples.items())[:6]:
        if not examples:
            continue
        sections.append(
            (
                f"Ungrounded examples: {field}",
                md_table(["uid", "extracted value"], [[e["uid"], str(e["value"])[:70]] for e in examples], align=["---", "---"]),
            )
        )
    path = write_report(
        root / "grounding_report.md",
        "Grounding / evidence verification report",
        sections,
        preamble=(
            "Produced by `7_semantic_layer/ground.py`. 'Grounded' means the value appears in the article it was "
            "extracted from - not that it is true. Nothing here is a gold-standard accuracy figure; "
            "for that see `evaluation/score.py` against human annotation."
        ),
    )
    log.info("grounded %d articles -> %s, %s (%s)", stats["articles"], out_path.name, path.name, fmt_duration(seconds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
