"""Shared helpers for the semantic / NLP layer (stage 7).

Everything here is about reading what earlier stages wrote and turning one
extraction record into the small, typed object the rest of this stage works
on. No model is loaded in this module, so every script can import it cheaply.

Layout this stage consumes (all produced upstream, none of it in the repo):

    <extracted_root>/<year>/<YYYY_MM>.jsonl        stage-5 output, one line per article
    <articles_root>/<article_id>.json              stage-4b three-field article

The 2021 article tree has an extra "articles/" level that later years dropped,
so article roots are format strings resolved per year, with an override table.

Two levels of identity are used throughout and must not be confused:

    uid   = "<year>/<month>/<article_id>"   one ARTICLE's extraction (a *mention*)
    event_id                                one REAL-WORLD flood, assigned by
                                            cluster_events.py to a set of uids

`article_id` alone is never a key: every year's crawl restarts its numbering.
Neither is "<year>/<article_id>", which was the original plan for this stage and
does not survive contact with the data - measured over the 74,447 verifiable
extractions, 2,075 article_ids occur in two different months of the SAME year,
and they are different articles:

    2021/2021_01/article_000001527  "At least 96 killed ... as quake, floods hit Indonesia"
    2021/2021_05/article_000001527  "Parapat City Paralyzed, Hit by Floods and Landslides"

So the month - which is also what addresses the article on disk - is part of the
key. A uid therefore maps 1:1 onto exactly one article file. Separately, some
month files contain the same article twice (identical re-extractions, mostly in
2023); `iter_events` drops the repeat and counts it.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.json"

log = logging.getLogger("events")

# Event slots that hold a count. Used by grounding, consolidation and the CSV.
NUMERIC_SLOTS = [
    "deaths",
    "injured",
    "missing",
    "displaced",
    "evacuated",
    "affected_people",
    "houses_damaged",
    "houses_destroyed",
    "rainfall_mm",
]

# Model-invented synonyms folded onto canonical slot names, mirroring
# 6_extract_events/to_csv.py so both views of the data agree.
SLOT_ALIASES = {
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
    "people_displaced": "displaced",
    "displaced_people": "displaced",
    "people_evacuated": "evacuated",
    "evacuated_people": "evacuated",
    "people_affected": "affected_people",
    "affected": "affected_people",
    "houses_damaged_count": "houses_damaged",
    "homes_damaged": "houses_damaged",
    "homes_destroyed": "houses_destroyed",
}

# Slots that name a watercourse, in the order they are trusted.
RIVER_SLOTS = ("river", "water_body", "dam")
# Slots that carry a free-text place name, coarse to fine.
PLACE_SLOTS = ("location", "city", "district", "county", "state", "province", "region")
# Slots that say why the water arrived.
CAUSE_SLOTS = ("cause", "trigger")

FLOOD_TYPE_UNKNOWN = "Unknown"


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def setup_logging(cfg: dict | None = None, level: str | None = None) -> logging.Logger:
    """One log format for every script in this stage. Idempotent."""
    lvl = level or (cfg or {}).get("runtime", {}).get("log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, str(lvl).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    return log


def seed_everything(cfg: dict) -> int:
    seed = int((cfg.get("runtime") or {}).get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    return seed


def apply_offline_env(cfg: dict) -> None:
    """Hugging Face never reaches the network in this project. Called before any
    transformers import so a missing local model fails loudly instead of
    silently downloading a few hundred megabytes."""
    if (cfg.get("embedding") or {}).get("offline", True):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


def output_root(cfg: dict) -> Path:
    root = Path(cfg["paths"]["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    return root


def extraction_files(cfg: dict, years: list[str] | None = None) -> list[tuple[str, str, Path]]:
    """[(year, month, path)] for every stage-5 output file, sorted."""
    root = Path(cfg["paths"]["extracted_root"])
    out = []
    for ydir in sorted(p for p in root.iterdir() if p.is_dir()):
        if years and ydir.name not in years:
            continue
        for f in sorted(ydir.glob("*.jsonl")):
            if ".failed." in f.name:
                continue
            out.append((ydir.name, f.stem, f))
    return out


def articles_root(cfg: dict, year: str, month: str) -> Path:
    paths = cfg["paths"]
    pattern = paths.get("articles_root_overrides", {}).get(year) or paths["articles_root"]
    return Path(pattern.format(year=year, month=month))


def article_path(cfg: dict, year: str, month: str, article_id: str) -> Path:
    return articles_root(cfg, year, month) / f"{article_id}.json"


def load_article(cfg: dict, year: str, month: str, article_id: str) -> dict | None:
    path = article_path(cfg, year, month, article_id)
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def make_uid(year: str, month: str, article_id: str) -> str:
    """The global key for one article's extraction. See the module docstring for
    why the month is in it."""
    return f"{year}/{month}/{article_id}"


def parse_uid(uid: str) -> tuple[str, str, str]:
    """uid -> (year, month, article_id). Accepts the older two-part form so an
    artefact written before the key changed still resolves; the month is then
    unknown and returned empty."""
    parts = uid.split("/")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], "", parts[1]
    raise ValueError(f"not a uid: {uid!r}")


def article_text(article: dict | None) -> str:
    if not article:
        return ""
    return str(article.get("translated_text") or article.get("text") or "")


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass
class Event:
    """One ARTICLE's extraction.

    Stage 5 emits `events: [...]`; the prompt asks for exactly one flood, but
    ~10% of accepted articles list several sub-events (one per affected
    district, usually). `slots` is the first sub-event, kept for backward
    compatibility with the CSV view; `all_slots` is every sub-event, and the
    list accessors below union across all of them.

    `enrich` holds anything a later script attached (flood_type, ner counts).
    It is never read from stage-5 output.
    """

    uid: str
    year: str
    month: str
    article_id: str
    title: str
    publish_date: str | None
    contains_flood: bool
    verifiable: bool
    dates: list[str]
    locations: list[str]
    slots: dict = field(default_factory=dict)
    all_slots: list[dict] = field(default_factory=list)
    n_events: int = 0
    enrich: dict = field(default_factory=dict)

    # -- scalars ----------------------------------------------------------

    @property
    def country(self) -> str | None:
        for s in self.all_slots or [self.slots]:
            value = s.get("country")
            if value not in (None, "", [], {}):
                return str(value).strip()
        return None

    @property
    def summary(self) -> str:
        parts = []
        for s in self.all_slots or [self.slots]:
            value = s.get("summary")
            if isinstance(value, (str, int, float)) and str(value).strip():
                parts.append(str(value).strip())
        return " ".join(parts)

    @property
    def flood_type(self) -> str:
        return str(self.enrich.get("flood_type") or FLOOD_TYPE_UNKNOWN)

    # -- unions across sub-events ----------------------------------------

    def _slot_union(self, slots: Sequence[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for sub in self.all_slots or [self.slots]:
            for name in slots:
                for value in _as_str_list(sub.get(name)):
                    key = normalize_text(value)
                    if key and key not in seen:
                        seen.add(key)
                        out.append(value)
        return out

    def rivers(self) -> list[str]:
        return self._slot_union(RIVER_SLOTS) + list(self.enrich.get("rivers") or [])

    def places(self) -> list[str]:
        """Every place string: the article-level flooded_locations plus the
        per-sub-event location/city/district/state slots."""
        seen = {normalize_text(x) for x in self.locations}
        out = list(self.locations)
        for value in self._slot_union(PLACE_SLOTS):
            key = normalize_text(value)
            if key and key not in seen:
                seen.add(key)
                out.append(value)
        return out

    def causes(self) -> list[str]:
        return self._slot_union(CAUSE_SLOTS)

    def counts(self) -> dict[str, float | int]:
        """Max of each count slot across sub-events. Sub-events are usually
        districts inside one flood, so summing would double-count any
        article that also states a total; max is the conservative read."""
        out: dict[str, float | int] = {}
        for sub in self.all_slots or [self.slots]:
            for slot in NUMERIC_SLOTS:
                value = to_number(sub.get(slot))
                if value is not None and (slot not in out or value > out[slot]):
                    out[slot] = value
        return out

    # -- dates ------------------------------------------------------------

    def all_dates(self) -> list[str]:
        seen = set(self.dates)
        out = list(self.dates)
        for sub in self.all_slots or [self.slots]:
            for name in ("event_date", "start_date", "end_date"):
                for value in _as_str_list(sub.get(name)):
                    if value not in seen:
                        seen.add(value)
                        out.append(value)
        return out

    def first_date(self) -> date | None:
        parsed = self.parsed_dates()
        return parsed[0] if parsed else None

    def parsed_dates(self) -> list[date]:
        return sorted({d for d in (parse_iso(x) for x in self.all_dates()) if d})

    # -- derived text -----------------------------------------------------

    def text_for_embedding(self) -> str:
        """The string the embedding model sees. Title carries the most signal;
        summary, locations, rivers, flood type, country, cause and date make two
        reports of the same flood land close together even when the headlines
        share no words."""
        parts = [self.title.strip()]
        if self.summary:
            parts.append(self.summary)
        places = self.places()
        if places:
            parts.append("Flooded: " + "; ".join(places[:12]))
        rivers = self.rivers()
        if rivers:
            parts.append("River: " + "; ".join(rivers[:6]))
        if self.country:
            parts.append("Country: " + self.country)
        ftype = self.flood_type
        if ftype and ftype != FLOOD_TYPE_UNKNOWN:
            parts.append("Flood type: " + ftype)
        dates = self.all_dates()
        if dates:
            parts.append("Date: " + ", ".join(dates[:6]))
        causes = self.causes()
        if causes:
            parts.append("Cause: " + "; ".join(causes[:4]))
        return ". ".join(p.rstrip(".") for p in parts if p)

    def metadata(self) -> dict:
        """Flat, scalar-only view for the vector store (Chroma rejects None and
        nested values). Missing counts are stored as -1, not omitted, so a
        filter on the field never silently drops rows."""
        dates = self.parsed_dates()
        counts = self.counts()
        meta = {
            "uid": self.uid,
            "year": int(self.year),
            "month": self.month,
            "article_id": self.article_id,
            "publish_date": self.publish_date or "",
            "event_date": dates[0].isoformat() if dates else "",
            "date_min": dates[0].isoformat() if dates else "",
            "date_max": dates[-1].isoformat() if dates else "",
            "country": self.country or "",
            "city": str((self.all_slots or [self.slots])[0].get("city") or "") if (self.all_slots or self.slots) else "",
            "locations": "; ".join(self.places()[:20]),
            "n_locations": len(self.places()),
            "rivers": "; ".join(self.rivers()[:10]),
            "flood_type": self.flood_type,
            "flood_type_confidence": float(self.enrich.get("flood_type_confidence") or 0.0),
            "flood_type_evidence": str(self.enrich.get("flood_type_evidence") or "")[:300],
            "n_events": self.n_events,
            "verifiable": bool(self.verifiable),
            "contains_flood": bool(self.contains_flood),
            "title": self.title[:300],
        }
        for slot in NUMERIC_SLOTS:
            value = counts.get(slot)
            meta[slot] = value if value is not None else -1
        return meta


ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")


def parse_iso(value) -> date | None:
    if not isinstance(value, str):
        return None
    m = ISO.match(value.strip())
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def to_number(value, default=None):
    """Counts arrive as 14, "14", "at least 14", "1,200" or "two". Return an
    int/float for the first three, `default` otherwise. Never guesses words."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        m = _NUM.search(value)
        if m:
            raw = m.group(0).replace(",", "")
            try:
                num = float(raw)
                return int(num) if num.is_integer() else num
            except ValueError:
                return default
    return default


def canonical_slots(event: dict) -> dict:
    """Fold synonyms onto canonical names; first value wins on collision."""
    out = {}
    for key, value in event.items():
        canon = SLOT_ALIASES.get(key, key)
        if canon not in out:
            out[canon] = value
    return out


def _as_str_list(value) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if isinstance(v, (str, int, float)) and str(v).strip()]
    return []


def record_to_event(year: str, record: dict, extraction_field: str = "extraction") -> Event | None:
    ext = record.get(extraction_field)
    if not isinstance(ext, dict):
        return None
    raw = ext.get("events") if isinstance(ext.get("events"), list) else []
    subs = [canonical_slots(e) for e in raw if isinstance(e, dict)]
    article_id = str(record.get("article_id", ""))
    month = str(record.get("month", ""))
    return Event(
        uid=make_uid(year, month, article_id),
        year=year,
        month=month,
        article_id=article_id,
        title=str(record.get("translated_title") or record.get("title") or ""),
        publish_date=record.get("publish_date") or None,
        contains_flood=bool(ext.get("contains_flood_event")),
        verifiable=bool(ext.get("is_verifiable_flood")),
        dates=_as_str_list(ext.get("flood_dates")),
        locations=_as_str_list(ext.get("flooded_locations")),
        slots=subs[0] if subs else {},
        all_slots=subs,
        n_events=len(raw),
    )


def iter_events(
    cfg: dict,
    years: list[str] | None = None,
    select: str = "verifiable",
    limit: int | None = None,
    skip_uids: set[str] | None = None,
    enrich: dict[str, dict] | None = None,
    dedupe: bool = True,
) -> Iterator[Event]:
    """Stream Event objects from the stage-5 output. Never holds more than one
    month's file in memory.

    select: "verifiable" (date AND location), "flood" (model said one real flood
    event), "all" (every article the model looked at, prefiltered included).
    skip_uids: already-processed ids, for resumable stages.
    enrich: uid -> dict merged onto Event.enrich (flood_type, rivers from NER).
    dedupe: drop a uid seen twice in the stream. Some month files contain the
    same article twice from a re-extraction; the second copy is identical and
    would otherwise be rejected by the vector store as a duplicate id.
    """
    n = 0
    seen: set[str] = set()
    repeats = 0
    for year, _month, path in extraction_files(cfg, years):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ev = record_to_event(year, record)
                if ev is None:
                    continue
                if select == "verifiable" and not ev.verifiable:
                    continue
                if select == "flood" and not ev.contains_flood:
                    continue
                if skip_uids and ev.uid in skip_uids:
                    continue
                if dedupe:
                    if ev.uid in seen:
                        repeats += 1
                        continue
                    seen.add(ev.uid)
                if enrich:
                    extra = enrich.get(ev.uid)
                    if extra:
                        ev.enrich.update(extra)
                yield ev
                n += 1
                if limit and n >= limit:
                    if repeats:
                        log.info("dropped %d repeated uid(s) already seen in this stream", repeats)
                    return
    if repeats:
        log.info("dropped %d repeated uid(s) already seen in this stream", repeats)


def iter_bodies(cfg: dict, events: Iterable[Event], workers: int = 16, lookahead: int = 256) -> Iterator[tuple[Event, str]]:
    """Yield (event, body_text) with the article files read ahead on a thread
    pool. One article is one small JSON file, so a GPU stage that reads them
    inline spends most of its time waiting on the filesystem; a bounded
    lookahead window keeps the reads in flight without loading the corpus.
    Order is preserved, and memory is bounded by `lookahead` bodies."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    def read(ev: Event) -> tuple[Event, str]:
        return ev, article_text(load_article(cfg, ev.year, ev.month, ev.article_id))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: deque = deque()
        stream = iter(events)
        for ev in stream:
            pending.append(pool.submit(read, ev))
            if len(pending) >= lookahead:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def count_events(cfg: dict, years: list[str] | None = None, select: str = "verifiable") -> int:
    """Cheap pass for progress bars; reads the JSONL but builds no objects."""
    seen: set[str] = set()
    for year, _month, path in extraction_files(cfg, years):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ext = record.get("extraction") or {}
                if select == "verifiable" and not ext.get("is_verifiable_flood"):
                    continue
                if select == "flood" and not ext.get("contains_flood_event"):
                    continue
                seen.add(make_uid(year, str(record.get("month", "")), str(record.get("article_id", ""))))
    return len(seen)


# --------------------------------------------------------------------------
# Text normalisation used by clustering, grounding and geocoding
# --------------------------------------------------------------------------

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_LOCATION_STOP = {
    "the", "of", "in", "at", "and", "district", "districts", "city", "town", "village",
    "villages", "province", "state", "region", "county", "area", "areas", "municipality",
    "prefecture", "department", "governorate", "island", "islands", "along", "near",
    "parts", "part", "highway", "road", "river", "valley", "north", "south", "east",
    "west", "northern", "southern", "eastern", "western", "central",
}


def normalize_text(text: str) -> str:
    folded = unicodedata.normalize("NFKC", str(text)).lower()
    return " ".join(_NON_ALNUM.sub(" ", folded).split())


def location_tokens(location: str) -> set[str]:
    return {t for t in normalize_text(location).split() if t not in _LOCATION_STOP and len(t) > 1}


def token_jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def locations_overlap(a: list[str], b: list[str], min_jaccard: float = 0.5) -> bool:
    """True when any pair of location strings names the same place, judged by
    exact match after normalisation or by token Jaccard over content words."""
    return location_similarity(a, b) >= min_jaccard


def location_similarity(a: list[str], b: list[str]) -> float:
    """Best-pair similarity in [0, 1] between two location lists. 1.0 on an
    exact normalised match, otherwise the highest content-token Jaccard."""
    if not a or not b:
        return 0.0
    norm_a = {normalize_text(x) for x in a}
    norm_b = {normalize_text(x) for x in b}
    if norm_a & norm_b:
        return 1.0
    best = 0.0
    toks_b = [location_tokens(x) for x in b]
    for x in a:
        ta = location_tokens(x)
        if not ta:
            continue
        for tb in toks_b:
            best = max(best, token_jaccard(ta, tb))
    return best


_RIVER_TAIL = re.compile(r"\b(river|rivers|nadi|creek|canal|stream|nala|khal)\b")


def river_key(name: str) -> str:
    """'the Brahmaputra River' and 'Brahmaputra' are the same watercourse."""
    return " ".join(t for t in _RIVER_TAIL.sub(" ", normalize_text(name)).split() if t not in {"the", "of"})


def river_similarity(a: list[str], b: list[str]) -> float:
    ka = {river_key(x) for x in a if river_key(x)}
    kb = {river_key(x) for x in b if river_key(x)}
    if not ka or not kb:
        return 0.0
    return 1.0 if ka & kb else 0.0


# --------------------------------------------------------------------------
# Numbers in prose, for grounding
# --------------------------------------------------------------------------

_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000,
    "lakh": 100_000, "lakhs": 100_000, "crore": 10_000_000, "crores": 10_000_000,
}

_MULTIPLIERS = {
    "hundred": 100, "thousand": 1_000, "lakh": 100_000, "lakhs": 100_000,
    "crore": 10_000_000, "crores": 10_000_000, "million": 1_000_000,
    "billion": 1_000_000_000,
}


def number_surface_forms(value: float | int) -> list[str]:
    """Every plausible way `value` could be written in a news article, used to
    decide whether an extracted count is actually supported by the text."""
    forms: list[str] = []
    if value is None:
        return forms
    if isinstance(value, float) and not float(value).is_integer():
        forms.append(f"{value:g}")
        return forms
    n = int(value)
    forms.append(str(n))
    if n >= 1000:
        forms.append(f"{n:,}")
        forms.append(f"{n:,}".replace(",", " "))
        # Indian grouping: 1,20,000
        s = str(n)
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        forms.append(",".join(groups + [tail]) if groups else s)
    for word, mult in _MULTIPLIERS.items():
        if n >= mult and n % mult == 0:
            forms.append(f"{n // mult} {word}")
        elif n >= mult and (n / mult) * 10 % 10 == 0:
            forms.append(f"{n / mult:g} {word}")
    for word, val in _WORD_NUMBERS.items():
        if val == n:
            forms.append(word)
    return list(dict.fromkeys(forms))


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(‘“])|\n{2,}")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s and s.strip()]


def chunk_text(text: str, target_chars: int, overlap_chars: int, min_chars: int, max_chunks: int) -> list[tuple[int, int, str]]:
    """Sentence-aware chunking with character overlap.

    Returns [(start, end, chunk_text)] with offsets into `text`, so a retrieved
    passage can always be pointed back at its exact place in the article. A
    trailing chunk shorter than `min_chars` is folded into the previous one
    rather than emitted on its own.
    """
    text = text or ""
    if not text.strip():
        return []
    if len(text) <= target_chars:
        return [(0, len(text), text)]

    spans: list[tuple[int, int]] = []
    cursor = 0
    for sent in split_sentences(text):
        idx = text.find(sent, cursor)
        if idx < 0:
            idx = cursor
        spans.append((idx, idx + len(sent)))
        cursor = idx + len(sent)
    if not spans:
        spans = [(0, len(text))]

    chunks: list[tuple[int, int, str]] = []
    start = spans[0][0]
    end = start
    for s_start, s_end in spans:
        if s_end - start > target_chars and end > start:
            chunks.append((start, end, text[start:end]))
            if len(chunks) >= max_chunks:
                return chunks
            start = max(end - overlap_chars, s_start - overlap_chars, 0)
            start = max(start, chunks[-1][0] + 1)
        end = s_end
    if end > start:
        tail = text[start:end]
        if chunks and len(tail) < min_chars:
            p_start, _p_end, _ = chunks[-1]
            chunks[-1] = (p_start, end, text[p_start:end])
        else:
            chunks.append((start, end, tail))
    return chunks[:max_chunks]


# --------------------------------------------------------------------------
# JSONL I/O, resumability, reports
# --------------------------------------------------------------------------


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    """Atomic full rewrite: write to <name>.tmp, then replace. A killed run
    never leaves a half-written artefact for the next stage to read."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def read_jsonl(path: Path) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("skipping malformed line in %s", path.name)


class JsonlAppender:
    """Append-only writer for resumable stages. Flushes every `flush_every`
    rows so a killed run loses at most that many, and exposes the ids already
    on disk so the caller can skip them."""

    def __init__(self, path: Path, key: str = "uid", flush_every: int = 200):
        self.path = Path(path)
        self.key = key
        self.flush_every = flush_every
        self._fh = None
        self._since_flush = 0
        self.written = 0

    def done_ids(self) -> set[str]:
        return {str(r[self.key]) for r in read_jsonl(self.path) if self.key in r}

    def __enter__(self) -> "JsonlAppender":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        return self

    def write(self, row: dict) -> None:
        assert self._fh is not None, "use JsonlAppender as a context manager"
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.written += 1
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self._fh.flush()
            self._since_flush = 0

    def __exit__(self, *exc) -> None:
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None


def load_sidecar(path: Path, key: str = "uid") -> dict[str, dict]:
    """Read an enrichment file (flood_type.jsonl, ner.jsonl, ...) into a dict.
    Returns {} when the file has not been produced yet, so every stage runs
    standalone and only gets richer as the ones before it are run."""
    path = Path(path)
    if not path.exists():
        return {}
    return {str(r[key]): r for r in read_jsonl(path) if key in r}


# --------------------------------------------------------------------------
# Local LLM (Ollama)
# --------------------------------------------------------------------------


class LLMUnavailable(RuntimeError):
    """Raised when the local Ollama server cannot be reached. Callers decide
    whether that is fatal; nothing in this stage silently falls back to a
    remote API, and no model is ever downloaded."""


def ollama_client(cfg: dict):
    """Thin wrapper so every LLM call in this stage shares one configuration.
    `ollama` is imported here rather than at module scope so scripts that never
    talk to the LLM stay cheap to import."""
    import ollama

    llm = cfg.get("llm") or {}
    kwargs = {"timeout": llm.get("timeout_seconds", 300)}
    if llm.get("host"):
        kwargs["host"] = llm["host"]
    return ollama.Client(**kwargs)


def ollama_available(cfg: dict) -> tuple[bool, str]:
    """(reachable, message). Also checks the configured model is actually pulled."""
    llm = cfg.get("llm") or {}
    want = llm.get("model", "")
    try:
        client = ollama_client(cfg)
        models = [m.model for m in client.list().models]
    except Exception as exc:  # server down, package missing, wrong host
        return False, f"Ollama not reachable: {exc}"
    if want and want not in models and want.split(":")[0] not in [m.split(":")[0] for m in models]:
        return False, f"Ollama is up but model {want!r} is not pulled (have: {', '.join(models) or 'none'})"
    return True, f"Ollama up, model {want}"


def ollama_json(cfg: dict, system: str, user: str, client=None) -> dict:
    """One deterministic, JSON-mode chat completion. Returns the parsed object.

    Raises LLMUnavailable when the server cannot be reached and ValueError when
    the model returns something that is not JSON after the configured retries.
    """
    llm = cfg.get("llm") or {}
    client = client or ollama_client(cfg)
    options = {
        "temperature": llm.get("temperature", 0),
        "top_p": llm.get("top_p", 1),
        "num_ctx": llm.get("num_ctx", 8192),
        "num_predict": llm.get("num_predict", 1024),
    }
    last: Exception | None = None
    for attempt in range(int(llm.get("retries", 2)) + 1):
        try:
            response = client.chat(
                model=llm["model"],
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                format=llm.get("format", "json"),
                options=options,
                keep_alive=llm.get("keep_alive", "30m"),
                think=llm.get("think", False),
            )
        except Exception as exc:
            raise LLMUnavailable(str(exc)) from exc
        content = (response.get("message") or {}).get("content", "") if isinstance(response, dict) else response.message.content
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            last = exc
            log.warning("LLM returned non-JSON (attempt %d/%d)", attempt + 1, int(llm.get("retries", 2)) + 1)
    raise ValueError(f"LLM did not return JSON after retries: {last}")


def md_table(header: Sequence[str], rows: Iterable[Sequence], align: Sequence[str] | None = None) -> list[str]:
    align = align or ["---"] * len(header)
    out = ["| " + " | ".join(str(h) for h in header) + " |", "|" + "|".join(align) + "|"]
    for row in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in row) + " |")
    return out


def write_report(path: Path, title: str, sections: Sequence[tuple[str, Sequence[str]]], preamble: str = "") -> Path:
    """Every stage writes one of these. Numbers in them are measured by the run
    that wrote them - nothing here is illustrative."""
    path = Path(path)
    lines = [f"# {title}", ""]
    if preamble:
        lines += [preamble, ""]
    for heading, body in sections:
        if heading:
            lines += [f"## {heading}", ""]
        lines += list(body) + [""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def gpu_note(device: str) -> str:
    """One line about what actually ran the model, for the reports."""
    try:
        import torch

        if str(device).startswith("cuda") and torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            peak = torch.cuda.max_memory_allocated() / 1024**3
            return f"{name} (CUDA), peak allocated {peak:.2f} GiB"
    except Exception:  # torch missing or no CUDA - not an error here
        pass
    return f"CPU (requested device: {device})"
