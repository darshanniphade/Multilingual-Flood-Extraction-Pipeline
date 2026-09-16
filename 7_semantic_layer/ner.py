"""Stage 7b - named-entity recognition with a flood-domain normalisation layer.

WHAT THIS IS, PRECISELY
-----------------------
Two components, kept separate on purpose and labelled as such in the output:

1. A generic pretrained tagger. `dslim/bert-base-NER` is BERT-base fine-tuned on
   CoNLL-2003; it knows PER, ORG, LOC and MISC and nothing about floods. Its
   spans are emitted with source="model" and the model's own confidence.

2. A flood-domain normalisation layer written for this corpus: lexicon and
   pattern rules over the same text, plus the structured slots stage 5 already
   extracted. This is what produces RIVER, FLOOD_TYPE, WEATHER_EVENT,
   INFRASTRUCTURE, CASUALTY, EVACUATION and DAMAGE. Those spans carry
   source="rule" or source="slot" and a fixed confidence from the config, NOT a
   learned probability.

This is NOT a fine-tuned flood NER model and must not be described as one. It is
a generic tagger plus domain rules. Training a real flood NER model needs an
annotated span corpus; `evaluation/make_annotation_sample.py` produces the sample to
start one, and until it exists this file is the honest alternative.

    python ner.py --sample 2000            # random sample, reproducible via runtime.seed
    python ner.py --years 2021 --limit 500
    python ner.py                          # every verifiable extraction (slow)
    python ner.py --force                  # ignore what is already on disk

Output under paths.output_root:

    ner.jsonl        one line per article: uid + entities [{text,label,start,end,confidence,source}]
    ner_report.md    measured entity counts by label and by source

Offsets are into the string the tagger actually read: title + "\\n" + body,
truncated to ner.max_chars. That exact string is recorded as `text_len` so a
span can always be resolved back.
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from common import (
    DEFAULT_CONFIG,
    Event,
    JsonlAppender,
    apply_offline_env,
    fmt_duration,
    gpu_note,
    iter_bodies,
    iter_events,
    load_config,
    load_sidecar,
    md_table,
    normalize_text,
    output_root,
    read_jsonl,
    seed_everything,
    setup_logging,
    write_report,
)

log = logging.getLogger("events.ner")

# CoNLL-2003 tags -> the names used in this project's output.
MODEL_LABEL_MAP = {"PER": "PERSON", "ORG": "ORGANIZATION", "LOC": "LOCATION", "MISC": "MISC"}

DOMAIN_LABELS = ["RIVER", "FLOOD_TYPE", "WEATHER_EVENT", "INFRASTRUCTURE", "CASUALTY", "EVACUATION", "DAMAGE"]

_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december|"
    "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(rf"\b\d{{1,2}}\s+(?:{_MONTHS})\b\.?(?:,?\s+\d{{4}})?", re.I),
    re.compile(rf"\b(?:{_MONTHS})\.?\s+\d{{1,2}}(?:,?\s+\d{{4}})?\b", re.I),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
]
NUMBER_PATTERN = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?\s*(?:hundred|thousand|lakh|lakhs|crore|crores|million|billion)?\b", re.I
)

# "the Brahmaputra river", "River Severn". Case-sensitive on purpose: the name
# has to be capitalised, or "river overflowed" and "river basin" become rivers.
RIVER_AFTER = re.compile(
    r"\b([A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+){0,2})\s+(?:[Rr]iver|[Cc]reek|[Cc]anal|[Ss]tream|[Nn]adi|[Nn]ala|[Kk]hal|[Bb]urn|[Bb]eck|[Bb]rook)\b"
)
RIVER_BEFORE = re.compile(r"\b[Rr]ivers?\s+(?:the\s+)?([A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+){0,2})\b")
# Words that are capitalised but are never a river name, so "River Wednesday"
# (from "the river Wednesday morning") does not enter the index.
_RIVER_STOPWORDS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "the", "a", "an", "this",
    "that", "it", "he", "she", "they", "we", "i", "there", "water", "flood",
    "floods", "flooding", "police", "officials", "government",
}


def compile_lexicon(lexicon: dict[str, list[str]]) -> list[tuple[str, re.Pattern]]:
    """One alternation per label, longest phrase first so 'heavy rainfall' wins
    over 'rain'. Word boundaries on both ends: 'dam' must not fire on 'damage'."""
    out = []
    for label, phrases in lexicon.items():
        ordered = sorted({p.lower() for p in phrases}, key=len, reverse=True)
        if not ordered:
            continue
        pattern = r"\b(?:" + "|".join(re.escape(p) for p in ordered) + r")\b"
        out.append((label, re.compile(pattern, re.I)))
    return out


def spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def dedup_entities(entities: list[dict]) -> list[dict]:
    """Keep the longest span per (label, overlapping region); a shorter rule hit
    inside a model span is redundant noise."""
    out: list[dict] = []
    by_label: dict[str, list[dict]] = defaultdict(list)
    for ent in sorted(entities, key=lambda e: (-(e["end"] - e["start"]), e["start"])):
        bucket = by_label[ent["label"]]
        if any(spans_overlap((ent["start"], ent["end"]), (o["start"], o["end"])) for o in bucket):
            continue
        bucket.append(ent)
        out.append(ent)
    return sorted(out, key=lambda e: (e["start"], e["label"]))


class FloodNER:
    """Generic tagger + the flood-domain normalisation layer around it."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        nc = cfg["ner"]
        self.max_chars = int(nc.get("max_chars", 3000))
        self.min_score = float(nc.get("min_score", 0.5))
        self.rule_conf = float(nc.get("rule_confidence", 0.6))
        self.slot_conf = float(nc.get("slot_confidence", 0.9))
        self.lexicon = compile_lexicon(nc.get("lexicon", {}))
        self.river_suffixes = {s.lower() for s in nc.get("river_suffixes", [])}
        self.flood_type_cues = compile_lexicon(
            {"FLOOD_TYPE": [p for cues in cfg["flood_type"]["lexicon"].values() for p in cues]}
        )
        apply_offline_env(cfg)
        from transformers import pipeline  # heavy import, keep local

        device = 0 if str(nc.get("device", "cuda")).startswith("cuda") else -1
        self.pipe = pipeline(
            "token-classification",
            model=nc["model"],
            aggregation_strategy=nc.get("aggregation", "first"),
            device=device,
        )
        log.info("tagger %s on %s (generic CoNLL-2003 model)", nc["model"], nc.get("device"))

    # -- text --------------------------------------------------------------

    def text_for(self, ev: Event, body: str) -> str:
        head = (ev.title.strip() + "\n") if self.cfg["ner"].get("use_title", True) else ""
        return (head + body)[: self.max_chars]

    # -- the three sources -------------------------------------------------

    def model_entities(self, batch_out: list[dict]) -> list[dict]:
        out = []
        for ent in batch_out:
            score = float(ent.get("score", 0.0))
            if score < self.min_score:
                continue
            label = MODEL_LABEL_MAP.get(ent.get("entity_group", ""), ent.get("entity_group", ""))
            out.append(
                {
                    "text": ent["word"],
                    "label": label,
                    "start": int(ent["start"]),
                    "end": int(ent["end"]),
                    "confidence": round(score, 4),
                    "source": "model",
                }
            )
        return out

    def rule_entities(self, text: str) -> list[dict]:
        out: list[dict] = []
        for pattern in DATE_PATTERNS:
            for m in pattern.finditer(text):
                out.append(self._ent(m.group(0), "DATE", m.start(), m.end(), "rule"))
        for m in NUMBER_PATTERN.finditer(text):
            token = m.group(0).strip()
            if token and any(c.isdigit() for c in token):
                out.append(self._ent(token, "NUMBER", m.start(), m.start() + len(token), "rule"))
        for label, pattern in self.lexicon:
            for m in pattern.finditer(text):
                out.append(self._ent(m.group(0), label, m.start(), m.end(), "rule"))
        for label, pattern in self.flood_type_cues:
            for m in pattern.finditer(text):
                out.append(self._ent(m.group(0), label, m.start(), m.end(), "rule"))
        for pattern in (RIVER_AFTER, RIVER_BEFORE):
            for m in pattern.finditer(text):
                name = m.group(1).lower().split()[0]
                if name in _RIVER_STOPWORDS:
                    continue
                out.append(self._ent(m.group(0), "RIVER", m.start(), m.end(), "rule"))
        return out

    def promote_rivers(self, text: str, entities: list[dict]) -> list[dict]:
        """A LOCATION the tagger found is a RIVER when the word right after it is
        a watercourse noun. This is the domain-mapping step: the generic model
        cannot make that distinction, the corpus needs it constantly."""
        promoted = []
        for ent in entities:
            if ent["label"] != "LOCATION" or ent["source"] != "model":
                continue
            m = re.match(r"\s+([A-Za-z]+)", text[ent["end"] : ent["end"] + 24])
            if not m or m.group(1).lower() not in self.river_suffixes:
                continue
            end = ent["end"] + m.end()
            promoted.append(
                {
                    **ent,
                    "label": "RIVER",
                    "text": text[ent["start"] : end],
                    "end": end,
                    "source": "rule",
                    "confidence": round(min(ent["confidence"], self.rule_conf), 4),
                }
            )
        return promoted

    def slot_entities(self, ev: Event, text: str, flood_type: str | None) -> list[dict]:
        """Entities the pipeline already knows from stage 5. Offsets are given
        when the value is literally present in the text, otherwise -1: the fact
        is real, the span is not, and pretending otherwise would corrupt any
        span-level evaluation."""
        out: list[dict] = []
        lowered = text.lower()

        def add(value: str, label: str) -> None:
            if not value or not str(value).strip():
                return
            value = str(value).strip()
            idx = lowered.find(value.lower())
            out.append(
                {
                    "text": value,
                    "label": label,
                    "start": idx,
                    "end": idx + len(value) if idx >= 0 else -1,
                    "confidence": self.slot_conf,
                    "source": "slot",
                    "in_text": idx >= 0,
                }
            )

        for river in ev.rivers():
            add(river, "RIVER")
        for place in ev.places():
            add(place, "LOCATION")
        if ev.country:
            add(ev.country, "LOCATION")
        for d in ev.all_dates():
            add(d, "DATE")
        if flood_type and flood_type != "Unknown":
            add(flood_type, "FLOOD_TYPE")
        return out

    def _ent(self, text: str, label: str, start: int, end: int, source: str) -> dict:
        return {
            "text": text,
            "label": label,
            "start": start,
            "end": end,
            "confidence": self.rule_conf,
            "source": source,
        }

    # -- batch driver ------------------------------------------------------

    def run_batch(self, items: list[tuple[Event, str, str | None]]) -> list[dict]:
        texts = [t for _ev, t, _ft in items]
        results = self.pipe(texts, batch_size=int(self.cfg["ner"].get("batch_size", 32)))
        if items and isinstance(results, list) and results and isinstance(results[0], dict):
            results = [results]  # a single-item batch comes back unwrapped
        rows = []
        for (ev, text, ftype), model_out in zip(items, results):
            model_ents = self.model_entities(model_out or [])
            ents = model_ents + self.promote_rivers(text, model_ents) + self.rule_entities(text)
            ents = dedup_entities(ents)
            slot_ents = self.slot_entities(ev, text, ftype)
            # slot entities are additive knowledge; keep them even where a model
            # span already covers the same characters, but flag the duplication
            model_spans = {(e["start"], e["end"], e["label"]) for e in ents}
            for se in slot_ents:
                se["duplicate_of_model_span"] = (se["start"], se["end"], se["label"]) in model_spans
            rows.append(
                {
                    "uid": ev.uid,
                    "year": int(ev.year),
                    "text_len": len(text),
                    "entities": ents + slot_ents,
                }
            )
        return rows


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def reservoir(stream, k: int, rng: random.Random) -> list:
    """Uniform sample of k items from a stream of unknown length, O(k) memory."""
    out: list = []
    for i, item in enumerate(stream):
        if i < k:
            out.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                out[j] = item
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Flood-domain NER over article text")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--years", nargs="*")
    ap.add_argument("--select", choices=["verifiable", "flood", "all"], default="verifiable")
    ap.add_argument("--limit", type=int, help="first N matching extractions")
    ap.add_argument("--sample", type=int, help="uniform random sample of N, reproducible")
    ap.add_argument("--force", action="store_true", help="re-tag uids already in ner.jsonl")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    seed = seed_everything(cfg)
    root = output_root(cfg)
    out_path = root / "ner.jsonl"

    appender = JsonlAppender(out_path)
    done = set() if args.force else appender.done_ids()
    if args.force and out_path.exists():
        out_path.unlink()
    log.info("already tagged: %d", len(done))

    flood_types = load_sidecar(root / "flood_type.jsonl")

    stream = iter_events(cfg, args.years, select=args.select, limit=args.limit, skip_uids=done)
    if args.sample:
        rng = random.Random(seed)
        events = reservoir(stream, args.sample, rng)
        log.info("sampled %d extractions (seed=%d)", len(events), seed)
    else:
        events = stream

    tagger = FloodNER(cfg)
    batch_size = int(cfg["ner"].get("batch_size", 32))
    label_counts: Counter = Counter()
    source_counts: Counter = Counter()
    river_counts: Counter = Counter()
    stats = Counter()
    started = time.time()
    buf: list[tuple[Event, str, str | None]] = []

    with appender:
        def flush() -> None:
            if not buf:
                return
            for row in tagger.run_batch(buf):
                appender.write(row)
                stats["docs"] += 1
                for ent in row["entities"]:
                    label_counts[ent["label"]] += 1
                    source_counts[ent["source"]] += 1
                    if ent["label"] == "RIVER":
                        river_counts[normalize_text(ent["text"])] += 1
            buf.clear()
            if stats["docs"] % (batch_size * 20) < batch_size:
                rate = stats["docs"] / max(time.time() - started, 1e-6)
                log.info("  %d docs tagged (%.1f doc/s)", stats["docs"], rate)

        for ev, body in iter_bodies(cfg, events, workers=int(cfg["grounding"].get("io_workers", 16))):
            if not body.strip():
                stats["body_missing"] += 1
                if not ev.title.strip():
                    continue
            ft = (flood_types.get(ev.uid) or {}).get("flood_type")
            text = tagger.text_for(ev, body)
            if not text.strip():
                stats["empty"] += 1
                continue
            buf.append((ev, text, ft))
            if len(buf) >= batch_size:
                flush()
        flush()

    seconds = time.time() - started
    # Aggregate over the whole file so a resumed run reports the corpus, not the
    # slice this process happened to tag.
    label_counts = Counter()
    source_counts = Counter()
    river_counts = Counter()
    n_docs_total = 0
    for row in read_jsonl(out_path):
        n_docs_total += 1
        for ent in row.get("entities", []):
            label_counts[ent.get("label", "?")] += 1
            source_counts[ent.get("source", "?")] += 1
            if ent.get("label") == "RIVER":
                river_counts[normalize_text(ent.get("text", ""))] += 1
    total_ents = sum(label_counts.values())
    sections = [
        (
            "What produced these entities",
            [
                "| source | entities | what it is |",
                "|---|---:|---|",
                f"| model | {source_counts['model']} | `{cfg['ner']['model']}`, a generic CoNLL-2003 tagger (PER/ORG/LOC/MISC). Carries the model's own confidence. |",
                f"| rule | {source_counts['rule']} | flood-domain lexicon and pattern layer written for this corpus. Fixed confidence {cfg['ner']['rule_confidence']}. |",
                f"| slot | {source_counts['slot']} | values stage 5 already extracted, re-anchored in the text where they literally occur. Fixed confidence {cfg['ner']['slot_confidence']}. |",
                "",
                "This is a **generic pretrained NER model plus a domain-specific normalisation layer**, "
                "not a flood-trained NER model. No span-annotated flood corpus exists for this project yet.",
            ],
        ),
        (
            "Run",
            md_table(
                ["", "value"],
                [
                    ["select", args.select],
                    ["years", " ".join(args.years) if args.years else "all"],
                    ["sample", args.sample or "-"],
                    ["already tagged before this run", len(done)],
                    ["documents tagged this run", stats["docs"]],
                    ["rows in ner.jsonl (what the tables below describe)", n_docs_total],
                    ["bodies missing on disk (title only)", stats["body_missing"]],
                    ["skipped: no text at all", stats["empty"]],
                    ["entities", total_ents],
                    ["entities per document", f"{total_ents / max(n_docs_total, 1):.1f}"],
                    ["max chars read per document", cfg["ner"]["max_chars"]],
                    ["device", gpu_note(cfg["ner"].get("device", "cuda"))],
                    ["wall clock", fmt_duration(seconds)],
                    ["throughput", f"{stats['docs'] / max(seconds, 1e-6):.1f} doc/s"],
                ],
                align=["---", "---:"],
            ),
        ),
        (
            "Entities by label",
            md_table(
                ["label", "count", "per doc", "origin"],
                [
                    [
                        label,
                        n,
                        f"{n / max(n_docs_total, 1):.2f}",
                        "domain layer" if label in DOMAIN_LABELS else "generic model / regex",
                    ]
                    for label, n in label_counts.most_common()
                ],
                align=["---", "---:", "---:", "---"],
            ),
        ),
        (
            "Most frequent watercourses (RIVER, normalised)",
            md_table(
                ["river", "mentions"],
                river_counts.most_common(25),
                align=["---", "---:"],
            ),
        ),
    ]
    path = write_report(
        root / "ner_report.md",
        "Flood-domain NER report",
        sections,
        preamble="Produced by `7_semantic_layer/ner.py`. All counts measured on the documents this run tagged.",
    )
    log.info("wrote %d rows to %s, report -> %s (%s)", stats["docs"], out_path.name, path.name, fmt_duration(seconds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
