"""The canonical flood-event record: what one REAL-WORLD flood looks like.

Stage 5 works per article. This module defines the object the semantic layer
produces instead - one record per real-world flood, assembled from every
article that reported it - plus the code that builds one from a coreference
cluster, validates it, and flattens it for CSV.

Article vs event, restated because the whole stage turns on it:

    uid        "2021/article_000000204"   one article's extraction
    event_id   "E-2021-000123"            one flood, cited by 1..n uids

Counts are the awkward part of merging. Fifty outlets report a death toll that
climbs from 7 to 31 over three days; averaging that is meaningless and taking
the first is arbitrary. So every count keeps its full disagreement in
`counts[slot]` (min, max, reports, conflict) and the scalar field alongside it
is the maximum reported value, which is what a disaster database wants. The
scalar is a summary of the evidence, never a substitute for it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from common import NUMERIC_SLOTS, normalize_text

EVENT_TYPE = "Flood"

# Order matters: the first band whose floor is cleared wins.
DEFAULT_SEVERITY_BANDS = [
    {"name": "catastrophic", "deaths": 100, "affected_people": 1000000},
    {"name": "major", "deaths": 20, "affected_people": 100000},
    {"name": "moderate", "deaths": 3, "affected_people": 10000},
    {"name": "minor", "deaths": 0, "affected_people": 0},
]


@dataclass
class CountEstimate:
    """One count slot as the corpus actually reports it."""

    value: float | int | None = None      # max across reports
    min: float | int | None = None
    max: float | int | None = None
    reports: int = 0                       # how many articles stated it
    conflict: bool = False                 # did they disagree

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FloodEvent:
    """One real-world flood event."""

    event_id: str
    source_uids: list[str] = field(default_factory=list)
    representative_uid: str | None = None
    event_type: str = EVENT_TYPE
    flood_type: str = "Unknown"
    flood_type_confidence: float = 0.0
    flood_type_evidence: str = ""

    country: str | None = None
    locations: list[str] = field(default_factory=list)
    cities: list[str] = field(default_factory=list)
    rivers: list[str] = field(default_factory=list)
    geo: list[dict] = field(default_factory=list)      # geocode.py output, one per linked place

    event_dates: list[str] = field(default_factory=list)
    date_start: str | None = None
    date_end: str | None = None
    publish_dates: list[str] = field(default_factory=list)

    causes: list[str] = field(default_factory=list)

    deaths: float | int | None = None
    injuries: float | int | None = None
    missing: float | int | None = None
    displaced: float | int | None = None
    evacuated_population: float | int | None = None
    affected_population: float | int | None = None
    infrastructure_damage: list[str] = field(default_factory=list)
    economic_damage: str | None = None
    counts: dict[str, dict] = field(default_factory=dict)

    severity: str | None = None
    n_articles: int = 0
    title: str = ""
    summary: str = ""
    years: list[int] = field(default_factory=list)
    months: list[str] = field(default_factory=list)

    confidence: float = 0.0
    grounded_fields: dict[str, bool] = field(default_factory=dict)
    evidence: list[dict] = field(default_factory=list)

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict) -> "FloodEvent":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in row.items() if k in known})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    # -- validation -------------------------------------------------------

    def validate(self) -> list[str]:
        """Structural problems only. Returns a list of human-readable issues;
        empty means the record is well-formed (not that it is true)."""
        problems: list[str] = []
        if not self.event_id:
            problems.append("missing event_id")
        if not self.source_uids:
            problems.append("no source_uids: an event with no article is not evidence of anything")
        if self.n_articles != len(self.source_uids):
            problems.append(f"n_articles={self.n_articles} but {len(self.source_uids)} source_uids")
        if self.date_start and self.date_end and self.date_start > self.date_end:
            problems.append("date_start after date_end")
        for slot, est in self.counts.items():
            lo, hi = est.get("min"), est.get("max")
            if lo is not None and hi is not None and lo > hi:
                problems.append(f"{slot}: min {lo} > max {hi}")
        for value, name in ((self.deaths, "deaths"), (self.affected_population, "affected_population")):
            if value is not None and value < 0:
                problems.append(f"{name} is negative")
        if self.flood_type_confidence < 0 or self.flood_type_confidence > 1:
            problems.append("flood_type_confidence outside [0, 1]")
        return problems


# --------------------------------------------------------------------------
# Building an event from a cluster of article-level extractions
# --------------------------------------------------------------------------

# Which FloodEvent scalar each stage-5 count slot lands on.
SCALAR_FOR_SLOT = {
    "deaths": "deaths",
    "injured": "injuries",
    "missing": "missing",
    "displaced": "displaced",
    "evacuated": "evacuated_population",
    "affected_people": "affected_population",
}


def merge_counts(rows: Sequence[dict]) -> dict[str, dict]:
    """rows are per-article metadata dicts using -1 for 'not reported'."""
    out: dict[str, dict] = {}
    for slot in NUMERIC_SLOTS:
        values = [r[slot] for r in rows if isinstance(r.get(slot), (int, float)) and not isinstance(r.get(slot), bool) and r[slot] >= 0]
        if not values:
            continue
        out[slot] = CountEstimate(
            value=max(values),
            min=min(values),
            max=max(values),
            reports=len(values),
            conflict=len(set(values)) > 1,
        ).to_dict()
    return out


def dedup_keep_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        key = normalize_text(v)
        if key and key not in seen:
            seen.add(key)
            out.append(v)
    return out


def severity_for(deaths: Any, affected: Any, bands: Sequence[dict] | None = None) -> str | None:
    """Ordinal severity from the two counts most consistently reported. Returns
    None when neither is known - an unlabelled event, not a 'minor' one."""
    bands = bands or DEFAULT_SEVERITY_BANDS
    if deaths is None and affected is None:
        return None
    d = deaths if isinstance(deaths, (int, float)) else -1
    a = affected if isinstance(affected, (int, float)) else -1
    for band in bands:
        if d >= band.get("deaths", 0) or a >= band.get("affected_people", 0):
            return band["name"]
    return None


def build_event(
    event_id: str,
    rows: Sequence[dict],
    representative: dict,
    severity_bands: Sequence[dict] | None = None,
) -> FloodEvent:
    """Assemble one FloodEvent from the per-article metadata rows of a cluster.

    `rows` are events_meta.jsonl records (the flat metadata written by
    build_index.py); `representative` is the cluster medoid's row.
    """
    counts = merge_counts(rows)
    dates = sorted({d for r in rows for d in (r.get("date_min"), r.get("date_max")) if d})
    flood_types = [r.get("flood_type") for r in rows if r.get("flood_type") and r.get("flood_type") != "Unknown"]
    top_type = max(set(flood_types), key=flood_types.count) if flood_types else "Unknown"
    type_rows = [r for r in rows if r.get("flood_type") == top_type]
    type_conf = (
        sum(float(r.get("flood_type_confidence") or 0.0) for r in type_rows) / len(type_rows)
        if type_rows and top_type != "Unknown"
        else 0.0
    )

    countries = [r["country"] for r in rows if r.get("country")]
    country = max(set(countries), key=countries.count) if countries else None

    event = FloodEvent(
        event_id=event_id,
        source_uids=[r["uid"] for r in rows],
        representative_uid=representative.get("uid"),
        flood_type=top_type,
        flood_type_confidence=round(type_conf, 4),
        flood_type_evidence=str(representative.get("flood_type_evidence") or ""),
        country=country,
        locations=dedup_keep_order([loc for r in rows for loc in _split_semi(r.get("locations"))]),
        cities=dedup_keep_order([r["city"] for r in rows if r.get("city")]),
        rivers=dedup_keep_order([riv for r in rows for riv in _split_semi(r.get("rivers"))]),
        event_dates=dates,
        date_start=dates[0] if dates else None,
        date_end=dates[-1] if dates else None,
        publish_dates=sorted({r["publish_date"] for r in rows if r.get("publish_date")}),
        counts=counts,
        n_articles=len(rows),
        title=str(representative.get("title") or ""),
        years=sorted({int(r["year"]) for r in rows if r.get("year") is not None}),
        months=sorted({r["month"] for r in rows if r.get("month")}),
    )
    for slot, attr in SCALAR_FOR_SLOT.items():
        if slot in counts:
            setattr(event, attr, counts[slot]["value"])
    event.severity = severity_for(event.deaths, event.affected_population, severity_bands)
    return event


def _split_semi(value) -> list[str]:
    if not isinstance(value, str):
        return []
    return [x.strip() for x in value.split(";") if x.strip()]


# --------------------------------------------------------------------------
# Flat views
# --------------------------------------------------------------------------

CSV_COLUMNS = [
    "event_id", "n_articles", "date_start", "date_end", "country", "flood_type",
    "flood_type_confidence", "locations", "cities", "rivers", "causes", "severity",
    "deaths", "injuries", "missing", "displaced", "evacuated_population",
    "affected_population", "economic_damage", "title", "representative_uid",
    "source_uids",
]


def to_csv_row(event: FloodEvent) -> dict:
    row = event.to_dict()
    for key in ("locations", "cities", "rivers", "causes", "infrastructure_damage"):
        row[key] = "; ".join(row.get(key) or [])
    row["source_uids"] = " ".join(row.get("source_uids") or [])
    return {k: row.get(k) for k in CSV_COLUMNS}


# --------------------------------------------------------------------------
# The schema handed to the LLM
# --------------------------------------------------------------------------

LLM_EVENT_SCHEMA: dict = {
    "flood_type": "one of: River Flood, Flash Flood, Urban Flood, Coastal Flood, Pluvial Flood, Dam/Reservoir Flood, Other, Unknown",
    "country": "string or null",
    "locations": ["place names stated in the evidence"],
    "rivers": ["watercourse names stated in the evidence"],
    "event_dates": ["YYYY-MM-DD, only dates the evidence states the flooding happened on"],
    "causes": ["short cause phrases quoted from the evidence"],
    "deaths": "integer or null",
    "injuries": "integer or null",
    "missing": "integer or null",
    "displaced": "integer or null",
    "evacuated_population": "integer or null",
    "affected_population": "integer or null",
    "infrastructure_damage": ["short phrases"],
    "economic_damage": "string with the currency as written, or null",
    "summary": "2-3 sentences, only facts present in the evidence",
    "evidence": [{"field": "name of a field above", "quote": "verbatim span from the evidence supporting it"}],
}

LLM_SYSTEM_PROMPT = (
    "You are a flood-event extraction engine. You are given numbered evidence passages taken "
    "verbatim from news articles, and a question.\n"
    "Rules:\n"
    "1. Use ONLY the evidence passages. Never use outside knowledge.\n"
    "2. Never infer, never calculate, never guess a value that is not stated.\n"
    "3. Use null for an unknown scalar and [] for an unknown list. An empty answer is correct "
    "when the evidence is silent.\n"
    "4. Every non-null field you fill must have a matching entry in `evidence` whose `quote` is a "
    "verbatim substring of one passage.\n"
    "5. Return ONLY a JSON object matching the given schema. No markdown, no explanation.\n"
)
