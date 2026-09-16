"""Stage 7f - geospatial entity linking.

Stage 5 extracts location strings as the article wrote them: "Honzar Dachhan
area of the Kishtwar district", "parts of Bombay", "Bengaluru". Those are
mentions, not places. This script links each distinct mention to a GeoNames
record so that two spellings of one city collapse, and so an event can be put
on a map.

Everything runs offline from the `geonamescache` tables (~34k cities with
population, every country, US states). GeoNames' own alternate-name lists are
what make "Bombay" resolve to Mumbai and "Bangalore" to Bengaluru - no alias
table is hand-maintained here, though `geocode.aliases` in config.json can
override or add one.

Resolution, in order, stopping at the first step that finds candidates:

    1. the whole mention, normalised
    2. the mention with administrative nouns stripped ("Kishtwar district" -> "kishtwar")
    3. each comma- or parenthesis-separated part, most specific first
    4. the longest trailing run of capitalised words

Candidates are then filtered by the country stage 5 extracted, if it extracted
one. What is left is ranked by population - but only accepted when the top
candidate is `ambiguity_ratio` times larger than the runner-up. Otherwise the
mention is written out with status="ambiguous" and every candidate listed. An
unresolved mention is a correct answer; a confidently wrong point on a map is
not.

    python geocode.py
    python geocode.py --years 2021 --limit 5000
    python geocode.py --force

Outputs under paths.output_root:

    geonames.jsonl        one row per distinct (country context, mention)
    event_geo.jsonl       per consolidated event: resolved places + a centroid
    geocode_report.md     measured resolution/ambiguity rates
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from common import (
    DEFAULT_CONFIG,
    fmt_duration,
    load_config,
    md_table,
    normalize_text,
    output_root,
    read_jsonl,
    setup_logging,
    write_jsonl,
    write_report,
)

log = logging.getLogger("events.geocode")

_PARENS = re.compile(r"\(([^)]*)\)")
_SPLIT = re.compile(r"[,/;]| - ")


class GeoLinker:
    def __init__(self, cfg: dict):
        import geonamescache

        gc = geonamescache.GeonamesCache()
        self.cfg = cfg["geocode"]
        self.min_pop = int(self.cfg.get("min_population", 0))
        self.ratio = float(self.cfg.get("ambiguity_ratio", 5.0))
        self.max_candidates = int(self.cfg.get("max_candidates", 8))
        self.strip_tokens = {t.lower() for t in self.cfg.get("strip_tokens", [])}

        self.cities = {
            cid: rec for cid, rec in gc.get_cities().items() if int(rec.get("population") or 0) >= self.min_pop
        }
        # Two tables, because a primary name and a foreign transliteration are
        # not equally good evidence. GeoNames lists "Cherbourg" as an alternate
        # name of Port Angeles and "Patani" as one of Putney; without the split,
        # every such pair looks like a genuine ambiguity.
        self.by_name: dict[str, list[dict]] = defaultdict(list)
        self.by_alt: dict[str, list[dict]] = defaultdict(list)
        for rec in self.cities.values():
            primary = normalize_text(rec["name"])
            self.by_name[primary].append(rec)
            for alt in rec.get("alternatenames") or []:
                key = normalize_text(alt)
                if key and key != primary:
                    self.by_alt[key].append(rec)

        countries = gc.get_countries()
        self.country_by_name: dict[str, str] = {}
        self.country_name_by_code: dict[str, str] = {}
        for code, rec in countries.items():
            self.country_by_name[normalize_text(rec["name"])] = code
            self.country_name_by_code[code] = rec["name"]
        for name, rec in gc.get_countries_by_names().items():
            self.country_by_name[normalize_text(name)] = rec["iso"]
        # A few forms news copy uses that GeoNames does not list as the country name.
        for extra, code in {
            "usa": "US", "u s a": "US", "us": "US", "united states of america": "US",
            "uk": "GB", "u k": "GB", "britain": "GB", "great britain": "GB",
            "south korea": "KR", "north korea": "KP", "russia": "RU", "iran": "IR",
            "syria": "SY", "vietnam": "VN", "laos": "LA", "bolivia": "BO", "tanzania": "TZ",
            "venezuela": "VE", "moldova": "MD", "czech republic": "CZ", "ivory coast": "CI",
            "drc": "CD", "dr congo": "CD", "democratic republic of congo": "CD",
        }.items():
            self.country_by_name.setdefault(extra, code)

        self.us_states = {normalize_text(s["name"]): s for s in gc.get_us_states().values()}
        self.countries_by_code = countries
        self.aliases = {normalize_text(k): v for k, v in (self.cfg.get("aliases") or {}).items()}
        log.info(
            "geonamescache: %d cities (population >= %d), %d primary names, %d alternate names, %d countries",
            len(self.cities), self.min_pop, len(self.by_name), len(self.by_alt), len(self.country_by_name),
        )

    # -- mention -> candidate strings ---------------------------------------

    def surface_forms(self, mention: str) -> list[tuple[str, str]]:
        """Progressively more aggressive readings of one location string, each
        tagged with how it was derived.

        The tag matters: a form that is the WHOLE mention (possibly minus
        administrative nouns) may resolve to a country, but a fragment of it may
        not. Without that rule the capitalised-run reading of "US 281" - a Texas
        highway - matched the country code "us" and put the event in the middle
        of the United States.
        """
        forms: list[tuple[str, str]] = []
        seen: set[str] = set()

        def add(value: str, kind: str) -> None:
            key = normalize_text(value)
            if key and key not in seen:
                seen.add(key)
                forms.append((key, kind))

        add(mention, "full")
        add(self.aliases.get(normalize_text(mention), ""), "full")
        outer = _PARENS.sub(" ", mention)
        add(outer, "full")
        stripped = " ".join(t for t in normalize_text(outer).split() if t not in self.strip_tokens)
        add(stripped, "stripped")
        for part in _PARENS.findall(mention):
            add(part, "part")
        for part in _SPLIT.split(outer):
            add(part, "part")
            add(" ".join(t for t in normalize_text(part).split() if t not in self.strip_tokens), "part")
        # longest run of capitalised words: "flooding in Lower Assam" -> "lower assam"
        caps = re.findall(r"(?:[A-Z][\w'\-]+)(?:\s+[A-Z][\w'\-]+)*", outer)
        for run in sorted(caps, key=len, reverse=True):
            add(run, "caps")
        return forms

    def country_code(self, country: str | None) -> str | None:
        if not country:
            return None
        return self.country_by_name.get(normalize_text(country))

    # -- resolution ---------------------------------------------------------

    def resolve_admin(self, form: str, mention: str, country: str | None) -> dict | None:
        """Countries and US states, which the city table cannot answer. Anything
        coarser than a city but finer than a country (Indian states, Malaysian
        states, Philippine provinces) is simply not in the offline dataset and
        stays unresolved rather than being snapped to a nearby city."""
        code = self.country_by_name.get(form)
        if code and code in self.countries_by_code:
            rec = self.countries_by_code[code]
            return {
                "mention": mention,
                "matched_form": form,
                "match_type": "primary",
                "feature": "country",
                "status": "resolved",
                "geonames_id": rec.get("geonameid"),
                "canonical_name": rec["name"],
                "country_code": code,
                "country": rec["name"],
                "admin1_code": None,
                "latitude": None,
                "longitude": None,
                "population": int(rec.get("population") or 0),
                "n_candidates": 1,
                "candidates": [],
                "context_country": country or None,
            }
        state = self.us_states.get(form)
        if state:
            return {
                "mention": mention,
                "matched_form": form,
                "match_type": "primary",
                "feature": "admin1",
                "status": "resolved",
                "geonames_id": state.get("geonameid"),
                "canonical_name": state["name"],
                "country_code": "US",
                "country": self.country_name_by_code.get("US"),
                "admin1_code": state.get("code"),
                "latitude": None,
                "longitude": None,
                "population": None,
                "n_candidates": 1,
                "candidates": [],
                "context_country": country or None,
            }
        return None

    def resolve(self, mention: str, country: str | None = None) -> dict:
        code = self.country_code(country)
        for form, kind in self.surface_forms(mention):
            # A country or a US state named outright is a real answer the city
            # table cannot give - but only when the whole mention names it.
            if kind in {"full", "stripped"}:
                admin = self.resolve_admin(form, mention, country)
                if admin:
                    return admin
            # Primary names first: an exact city name beats a foreign alias of
            # some other city that happens to be spelled the same.
            candidates = self.by_name.get(form) or self.by_alt.get(form)
            match_type = "primary" if self.by_name.get(form) else "alternate"
            if not candidates:
                continue
            pool = [c for c in candidates if c["countrycode"] == code] if code else list(candidates)
            if not pool:
                if code:
                    # The name exists but not in the stated country: trust the
                    # country, report the mismatch rather than moving the event.
                    continue
                pool = list(candidates)
            pool = sorted(pool, key=lambda c: -int(c.get("population") or 0))
            unique: list[dict] = []
            seen_ids = set()
            for c in pool:
                if c["geonameid"] not in seen_ids:
                    seen_ids.add(c["geonameid"])
                    unique.append(c)
            top = unique[0]
            second_pop = int(unique[1].get("population") or 0) if len(unique) > 1 else 0
            top_pop = int(top.get("population") or 0)
            ambiguous = len(unique) > 1 and not (second_pop == 0 or top_pop >= self.ratio * max(second_pop, 1))
            return {
                "mention": mention,
                "matched_form": form,
                "match_type": match_type,
                "feature": "city",
                "status": "ambiguous" if ambiguous else "resolved",
                "geonames_id": None if ambiguous else top["geonameid"],
                "canonical_name": None if ambiguous else top["name"],
                "country_code": None if ambiguous else top["countrycode"],
                "country": None if ambiguous else self.country_name_by_code.get(top["countrycode"]),
                "admin1_code": None if ambiguous else top.get("admin1code"),
                "latitude": None if ambiguous else top["latitude"],
                "longitude": None if ambiguous else top["longitude"],
                "population": None if ambiguous else top_pop,
                "n_candidates": len(unique),
                "candidates": [
                    {
                        "geonames_id": c["geonameid"],
                        "name": c["name"],
                        "country_code": c["countrycode"],
                        "population": int(c.get("population") or 0),
                        "latitude": c["latitude"],
                        "longitude": c["longitude"],
                    }
                    for c in unique[: self.max_candidates]
                ],
                "context_country": country or None,
            }
        return {
            "mention": mention,
            "matched_form": None,
            "match_type": None,
            "feature": None,
            "status": "unresolved",
            "geonames_id": None,
            "canonical_name": None,
            "country_code": code,
            "country": self.country_name_by_code.get(code) if code else None,
            "n_candidates": 0,
            "candidates": [],
            "context_country": country or None,
        }


def mention_key(mention: str, country: str | None) -> str:
    return f"{normalize_text(country or '')}|{normalize_text(mention)}"


def _split_semi(value) -> list[str]:
    return [x.strip() for x in str(value or "").split(";") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Link extracted location strings to GeoNames")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--years", nargs="*")
    ap.add_argument("--limit", type=int, help="only the first N distinct mentions")
    ap.add_argument("--force", action="store_true", help="re-resolve mentions already on disk")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    root = output_root(cfg)
    meta_path = root / "events_meta.jsonl"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} not found - run build_index.py first")

    started = time.time()
    mentions: dict[str, tuple[str, str | None]] = {}
    per_uid: dict[str, list[str]] = {}
    years = set(args.years or [])
    n_rows = 0
    for row in read_jsonl(meta_path):
        if years and str(row.get("year")) not in years:
            continue
        n_rows += 1
        country = row.get("country") or None
        places = _split_semi(row.get("locations"))
        if row.get("city"):
            places.append(row["city"])
        keys = []
        for place in places:
            key = mention_key(place, country)
            mentions.setdefault(key, (place, country))
            keys.append(key)
        per_uid[row["uid"]] = keys
    log.info("%d indexed extractions -> %d distinct (country, mention) pairs", n_rows, len(mentions))

    out_path = root / "geonames.jsonl"
    existing = {} if args.force else {r["key"]: r for r in read_jsonl(out_path) if "key" in r}
    todo = [k for k in mentions if k not in existing]
    if args.limit:
        todo = todo[: args.limit]
    log.info("%d already resolved, %d to resolve", len(existing), len(todo))

    linker = GeoLinker(cfg)
    resolved = dict(existing)
    for i, key in enumerate(todo, 1):
        mention, country = mentions[key]
        row = linker.resolve(mention, country)
        row["key"] = key
        resolved[key] = row
        if i % 5000 == 0:
            log.info("  %d/%d resolved", i, len(todo))
    write_jsonl(out_path, resolved.values())

    status = Counter(r["status"] for r in resolved.values())
    with_country = sum(1 for r in resolved.values() if r.get("context_country"))
    feature = Counter(r.get("feature") or "-" for r in resolved.values() if r["status"] == "resolved")
    # Per-record coverage is the number that matters downstream: distinct
    # mentions are dominated by hyper-specific village strings, but an event
    # only needs one of its places to resolve to be mappable.
    uid_any = sum(
        1 for keys in per_uid.values() if any(resolved.get(k, {}).get("status") == "resolved" for k in keys)
    )
    uid_all = sum(
        1
        for keys in per_uid.values()
        if keys and all(resolved.get(k, {}).get("status") == "resolved" for k in keys)
    )

    # Per-event geography, when coreference has already run.
    events_path = root / "consolidated_events.jsonl"
    event_rows: list[dict] = []
    if events_path.exists():
        for event in read_jsonl(events_path):
            places = list(event.get("locations") or []) + list(event.get("cities") or [])
            hits = []
            for place in places:
                row = resolved.get(mention_key(place, event.get("country")))
                if row and row["status"] == "resolved":
                    hits.append(row)
            # Only point features carry coordinates; a resolved country has no
            # meaningful centroid contribution and is excluded from the average.
            points = [h for h in hits if h.get("latitude") is not None]
            centroid = None
            if points:
                centroid = {
                    "latitude": round(sum(float(h["latitude"]) for h in points) / len(points), 5),
                    "longitude": round(sum(float(h["longitude"]) for h in points) / len(points), 5),
                    "from_n_places": len(points),
                }
            event_rows.append(
                {
                    "event_id": event["event_id"],
                    "country": event.get("country"),
                    "n_place_mentions": len(places),
                    "n_resolved": len(hits),
                    # Prefer an actual point: a resolved country is true but
                    # useless as an event's primary location, and it always wins
                    # on population.
                    "primary": (
                        max(points or hits, key=lambda h: int(h.get("population") or 0)) if hits else None
                    ),
                    "places": [
                        {
                            "mention": h["mention"],
                            "canonical_name": h["canonical_name"],
                            "geonames_id": h["geonames_id"],
                            "latitude": h["latitude"],
                            "longitude": h["longitude"],
                        }
                        for h in hits
                    ],
                    "centroid": centroid,
                }
            )
        write_jsonl(root / "event_geo.jsonl", event_rows)
        log.info("wrote event_geo.jsonl for %d consolidated events", len(event_rows))

    total = max(len(resolved), 1)
    sections = [
        (
            "Resolution",
            md_table(
                ["status", "mentions", "share"],
                [[s, n, f"{100 * n / total:.1f}%"] for s, n in status.most_common()],
                align=["---", "---:", "---:"],
            )
            + [
                "",
                f"Distinct (country, mention) pairs: {total}. "
                f"{with_country} ({100 * with_country / total:.0f}%) had a country from stage 5 to "
                "narrow candidates with.",
                "",
                "`ambiguous` means several GeoNames records match the name and no one of them is "
                f"{cfg['geocode']['ambiguity_ratio']}x more populous than the next - the mention is kept "
                "with all candidates listed rather than being pinned to a guess. `unresolved` means no "
                "gazetteer entry matched any reading of the string, which is expected for villages and "
                "wards below the 15k-population cut of the offline dataset.",
            ],
        ),
        (
            "Coverage per indexed extraction",
            md_table(
                ["", "records", "share"],
                [
                    ["extractions with at least one place resolved", uid_any, f"{100 * uid_any / max(len(per_uid), 1):.1f}%"],
                    ["extractions with every place resolved", uid_all, f"{100 * uid_all / max(len(per_uid), 1):.1f}%"],
                    ["extractions read", len(per_uid), "100%"],
                ],
                align=["---", "---:", "---:"],
            )
            + [
                "",
                "This is the number that matters downstream: an event is mappable if one of its places "
                "resolves, and the distinct-mention rate above is dragged down by hyper-specific strings "
                "('Kampung Laut Batu 10', 'Taman Kemang') that no 34k-city gazetteer contains.",
                "",
                "Resolved by feature type: "
                + ", ".join(f"{name} {n}" for name, n in feature.most_common()),
            ],
        ),
        (
            "Run",
            md_table(
                ["", "value"],
                [
                    ["indexed extractions read", n_rows],
                    ["distinct mentions", len(mentions)],
                    ["resolved this run", len(todo)],
                    ["gazetteer", f"geonamescache, population >= {cfg['geocode']['min_population']}"],
                    ["consolidated events geocoded", len(event_rows) or "-"],
                    ["wall clock", fmt_duration(time.time() - started)],
                ],
                align=["---", "---:"],
            ),
        ),
        (
            "Most frequent resolved places",
            md_table(
                ["canonical name", "country", "geonames id", "mentions mapped to it"],
                [
                    [name, country, gid, n]
                    for (name, country, gid), n in Counter(
                        (r["canonical_name"], r["country"], r["geonames_id"])
                        for r in resolved.values()
                        if r["status"] == "resolved"
                    ).most_common(25)
                ],
                align=["---", "---", "---:", "---:"],
            ),
        ),
        (
            "Ambiguous examples (kept unresolved on purpose)",
            md_table(
                ["mention", "context country", "candidates"],
                [
                    [
                        r["mention"][:50],
                        r.get("context_country") or "-",
                        ", ".join(f"{c['name']} ({c['country_code']}, {c['population']:,})" for c in r["candidates"][:3]),
                    ]
                    for r in list(x for x in resolved.values() if x["status"] == "ambiguous")[:15]
                ],
                align=["---", "---", "---"],
            ),
        ),
        (
            "Unresolved examples",
            md_table(
                ["mention", "context country"],
                [
                    [r["mention"][:60], r.get("context_country") or "-"]
                    for r in list(x for x in resolved.values() if x["status"] == "unresolved")[:15]
                ],
                align=["---", "---"],
            ),
        ),
    ]
    path = write_report(
        root / "geocode_report.md",
        "Geospatial entity linking report",
        sections,
        preamble="Produced by `7_semantic_layer/geocode.py` against the offline geonamescache gazetteer.",
    )
    log.info("wrote %s and %s (%s)", out_path.name, path.name, fmt_duration(time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
