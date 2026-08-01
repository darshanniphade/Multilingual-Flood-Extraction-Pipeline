"""Scan the extraction JSONL and write ONE analytics report.

    python analytics.py                 # -> data/extracted/analytics_report.md
    python analytics.py --output x.md
    python analytics.py --top 30        # longer top-N lists

The report is descriptive, not authoritative: every number is only as good as the
model's extraction, and the corpus is NOT deduplicated - one flood reported by ten
outlets counts ten times. Casualty sums are therefore an upper bound on article
mentions, not a death toll. Caveats are printed inline in the report.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.json"
ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
LEADING_NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*")

# fields whose values are counts we may want to total
COUNT_FIELDS = [
    "deaths", "injured", "missing", "displaced", "evacuated", "affected_people",
    "houses_damaged", "houses_destroyed", "roads_damaged", "bridges_damaged",
]
PLACE_FIELDS = ["country", "state", "province", "district", "city", "village", "location"]
WATER_FIELDS = ["river", "stream", "lake", "dam", "water_body"]


def load_config(path):
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def iter_records(files):
    for path in files:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        pass


def events_of(extraction):
    evs = extraction.get("events")
    if not evs and isinstance(extraction.get("event_details"), dict):
        evs = [extraction["event_details"]] if extraction["event_details"] else []
    return [e for e in (evs or []) if isinstance(e, dict)]


def as_number(value):
    """Best-effort numeric read. Returns (number|None, is_clean).

    is_clean is True only when the whole value was already a plain number, so we
    can separate 'the model gave us 14' from 'we scraped 200 out of more than 200
    families'."""
    if isinstance(value, bool):
        return None, False
    if isinstance(value, (int, float)):
        return float(value), True
    if isinstance(value, str):
        m = LEADING_NUM.search(value)
        if m:
            try:
                n = float(m.group(0).replace(",", ""))
            except ValueError:
                return None, False
            return n, value.strip() == m.group(0)
    return None, False


def date_quality(d):
    if not isinstance(d, str):
        return "non-string"
    m = ISO.match(d)
    if not m:
        return "not-iso"
    y, mo, dy = (int(x) for x in m.groups())
    if mo == 0 or dy == 0:
        return "partial"
    try:
        date(y, mo, dy)
        return "valid"
    except ValueError:
        return "impossible"


def dedup_key(extraction, event):
    """A best-effort event identity: (primary flooded location, earliest flood date).

    Articles about the same flood usually agree on where and when, so this clusters
    most retellings together. It is a heuristic - it will over-merge two floods in
    the same town on the same day, and under-merge when outlets name the place
    differently ('Bozkurt' vs 'Bozkurt district'). Returns None when neither a
    location nor a date is available, i.e. the mention cannot be clustered."""
    locs = extraction.get("flooded_locations") or []
    loc = None
    for l in locs:
        if isinstance(l, str) and l.strip():
            loc = l.strip().lower()
            break
    if loc is None:
        for f in PLACE_FIELDS:
            v = event.get(f)
            if isinstance(v, str) and v.strip():
                loc = v.strip().lower()
                break

    day = None
    for d in extraction.get("flood_dates") or []:
        if date_quality(d) == "valid" and (day is None or d < day):
            day = d
    if day is None:
        ev = event.get("event_date")
        if isinstance(ev, str) and date_quality(ev) == "valid":
            day = ev

    if loc is None and day is None:
        return None
    return (loc or "?", day or "?")


def pct(n, d):
    return f"{100 * n / d:.1f}%" if d else "0.0%"


def median(vals):
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def bar(n, d, width=28):
    filled = round(width * n / d) if d else 0
    return "█" * filled + "·" * (width - filled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument(
        "--dedup",
        action="store_true",
        help="add a deduplicated impact estimate: cluster article-mentions by "
        "(primary flooded location, earliest flood date), take the max value per "
        "cluster, then sum across clusters",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    field = cfg["output"]["extraction_field"]
    root = Path(cfg["paths"]["output_root"])
    files = sorted(p for p in root.glob("*.jsonl") if not p.name.endswith(".failed.jsonl"))
    failed = sum(1 for _ in iter_records(sorted(root.glob("*.failed.jsonl"))))
    out = args.output or root / "analytics_report.md"

    # accumulators
    total = flood = verifiable = 0
    with_date = with_loc = 0
    by_month_all = Counter()        # articles per input month
    by_month_flood = Counter()      # flood articles per input month
    flood_by_pubmonth = Counter()   # flood articles by publish_date YYYY-MM
    flood_by_datemonth = Counter()  # by flood_dates YYYY-MM
    date_q = Counter()
    places = {f: Counter() for f in PLACE_FIELDS}
    waters = {f: Counter() for f in WATER_FIELDS}
    flooded_locs = Counter()
    causes = Counter()
    triggers = Counter()
    agencies = Counter()
    field_use = Counter()
    count_stats = {f: {"n": 0, "clean": 0, "prose": 0, "sum": 0.0, "max": 0.0,
                       "max_where": None, "vals": []} for f in COUNT_FIELDS}
    # implausible magnitudes: (field, value, article_id, title) for a sanity table
    IMPLAUSIBLE = {"deaths": 1000, "affected_people": 5_000_000, "evacuated": 2_000_000,
                   "displaced": 2_000_000, "houses_destroyed": 500_000}
    outliers = []
    type_units = defaultdict(Counter)   # e.g. deaths_type -> {"people": n}
    events_per_flood = Counter()
    # dedup: key -> {field: max clean value in the cluster}
    groups = {}
    ungroupable = 0
    grouped_mentions = 0
    n_date_values = 0

    for rec in iter_records(files):
        total += 1
        e = rec.get(field) or {}
        month = rec.get("month", "")
        by_month_all[month] += 1

        if not e.get("contains_flood_event"):
            continue
        flood += 1
        by_month_flood[month] += 1
        if e.get("is_verifiable_flood"):
            verifiable += 1

        dates = e.get("flood_dates") or []
        locs = e.get("flooded_locations") or []
        if dates:
            with_date += 1
        if locs:
            with_loc += 1
        for l in locs:
            if isinstance(l, str):
                flooded_locs[l.strip()] += 1

        for d in dates:
            n_date_values += 1
            q = date_quality(d)
            date_q[q] += 1
            if q in ("valid", "partial"):
                flood_by_datemonth[str(d)[:7]] += 1

        pub = rec.get("publish_date")
        if isinstance(pub, str) and len(pub) >= 7:
            flood_by_pubmonth[pub[:7]] += 1

        evs = events_of(e)
        events_per_flood[len(evs)] += 1
        for ev in evs:
            for k, v in ev.items():
                field_use[k] += 1
                if k in PLACE_FIELDS and isinstance(v, str):
                    places[k][v.strip()] += 1
                if k in WATER_FIELDS and isinstance(v, str):
                    waters[k][v.strip()] += 1
                if k == "cause" and isinstance(v, str):
                    causes[v.strip().lower()] += 1
                if k == "trigger" and isinstance(v, str):
                    triggers[v.strip().lower()] += 1
                if k == "response_agencies":
                    for a in (v if isinstance(v, list) else [v]):
                        if isinstance(a, str):
                            agencies[a.strip()] += 1
                if k.endswith("_type") and isinstance(v, str):
                    type_units[k][v.strip().lower()] += 1
                if k in count_stats:
                    num, clean = as_number(v)
                    st = count_stats[k]
                    st["n"] += 1
                    if num is not None:
                        st["clean" if clean else "prose"] += 1
                        st["sum"] += num
                        st["vals"].append(num)
                        if num > st["max"]:
                            st["max"], st["max_where"] = num, rec.get("article_id")
                        if num >= IMPLAUSIBLE.get(k, float("inf")):
                            outliers.append((k, num, rec.get("article_id"),
                                             rec.get("translated_title", "")[:70]))

        if args.dedup:
            key = dedup_key(e, evs[0] if evs else {})
            if key is None:
                ungroupable += 1
            else:
                grouped_mentions += 1
                bucket = groups.setdefault(key, {})
                for ev in evs:
                    for k, v in ev.items():
                        if k in count_stats:
                            num, clean = as_number(v)
                            if num is not None and clean:
                                # keep the largest reported figure for this event
                                bucket[k] = max(bucket.get(k, 0.0), num)

    # ---- write report ----
    L = []
    A = L.append
    A("# Flood extraction - analytics report")
    A("")
    A(f"Source: `{root}`  |  {len(files)} monthly file(s)  |  generated by analytics.py")
    A("")
    A("> Descriptive statistics over model-extracted fields. The corpus is **not "
      "deduplicated** - one flood covered by many outlets is counted many times - so "
      "totals below are article-mentions, not real-world counts. Every value depends "
      "on the model's extraction accuracy.")
    A("")

    A("## 1. Corpus")
    A("")
    A("| metric | value |")
    A("|---|---:|")
    A(f"| articles processed | {total:,} |")
    A(f"| parse failures (separate) | {failed:,} |")
    A(f"| describe a flood event | {flood:,} ({pct(flood, total)}) |")
    A(f"| not a flood event | {total - flood:,} ({pct(total - flood, total)}) |")
    A(f"| verifiable (date + location) | {verifiable:,} ({pct(verifiable, total)} of all, "
      f"{pct(verifiable, flood)} of floods) |")
    A("")
    A("Among flood articles, what is present:")
    A("")
    A("| | count | share of floods |")
    A("|---|---:|---:|")
    A(f"| has >=1 flood date | {with_date:,} | {pct(with_date, flood)} |")
    A(f"| has >=1 flooded location | {with_loc:,} | {pct(with_loc, flood)} |")
    A("")
    only_loc = with_loc - verifiable
    A(f"The gap between flood articles and verifiable ones is driven by **missing dates**: "
      f"{only_loc:,} articles name a flooded place but no usable date. Most articles write "
      f"a weekday and month without a year.")
    A("")

    A("## 2. When (flood articles by month)")
    A("")
    A("By input batch month:")
    A("")
    mx = max(by_month_flood.values()) if by_month_flood else 1
    for m in sorted(by_month_flood):
        n = by_month_flood[m]
        A(f"`{m}` {bar(n, mx)} {n:,}  ({pct(n, by_month_all[m])} of that month)")
    A("")
    if flood_by_datemonth:
        A("By the actual flood date the model extracted (top periods):")
        A("")
        A("| flood month (YYYY-MM) | article-mentions |")
        A("|---|---:|")
        for m, n in sorted(flood_by_datemonth.items(), key=lambda kv: -kv[1])[:12]:
            A(f"| {m} | {n:,} |")
        A("")

    A("## 3. Where")
    A("")
    for label, ctr in [("Countries", places["country"]), ("Cities", places["city"]),
                       ("Districts", places["district"]), ("States/provinces",
                       places["state"] + places["province"])]:
        if not ctr:
            continue
        A(f"**{label}** (top {args.top}, by article-mentions)")
        A("")
        for name, n in ctr.most_common(args.top):
            A(f"- {name} - {n:,}")
        A("")
    if flooded_locs:
        A(f"**Most-cited flooded locations** (top {args.top})")
        A("")
        for name, n in flooded_locs.most_common(args.top):
            A(f"- {name} - {n:,}")
        A("")
    rivers = waters["river"]
    if rivers:
        A(f"**Rivers** (top {args.top})")
        A("")
        for name, n in rivers.most_common(args.top):
            A(f"- {name} - {n:,}")
        A("")

    A("## 4. Human impact (article-mention sums - NOT deduplicated)")
    A("")
    A("> These sum a value across every article that reported it. Because outlets "
      "repeat the same event, and because some values were prose ('more than 200'), "
      "read these as scale indicators, not tallies. 'clean' = value arrived as a plain "
      "number; 'prose' = a number was scraped out of text.")
    A("")
    A("**Median is the trustworthy column** - it is unaffected by the handful of "
      "extraction errors that dominate the sums (see the sanity check below).")
    A("")
    A("| field | articles w/ value | clean | prose | median | sum* | largest single | in article |")
    A("|---|---:|---:|---:|---:|---:|---:|---|")
    for f in COUNT_FIELDS:
        st = count_stats[f]
        if not st["n"]:
            continue
        A(f"| {f} | {st['n']:,} | {st['clean']:,} | {st['prose']:,} | "
          f"{median(st['vals']):,.0f} | {st['sum']:,.0f} | {st['max']:,.0f} | "
          f"{st['max_where'] or ''} |")
    A("")
    A("*Sum mixes units where the article specified them (people vs families vs "
      "households) - see the unit breakdown below before trusting any total.")
    A("")
    if outliers:
        A("**Sanity check - largest values, and why the sums are unusable.** These "
          "pass a magnitude threshold most floods never reach. Spot-checking shows two "
          "different causes mixed together: (1) genuine mega-events - the Henan floods "
          "really did affect ~14.5M people, Bihar ~3.7M ('37.39 lakh') - but each is "
          "reported by dozens of outlets and the corpus is not deduplicated, so every "
          "retelling adds to the sum; and (2) real extraction errors - "
          "`affected_people`=1.33 billion is China's population, and `evacuated`=161 "
          "million came from an article about two deaths. Both inflate the totals, "
          "which is why median, not sum, is the honest central number.")
        A("")
        A("| field | value | article | title |")
        A("|---|---:|---|---|")
        for f, v, aid, title in sorted(outliers, key=lambda x: -x[1])[:15]:
            A(f"| {f} | {v:,.0f} | {aid} | {title} |")
        A("")
    for tf in ["deaths_type", "displaced_type", "evacuated_type", "affected_people_type"]:
        if type_units.get(tf):
            top = ", ".join(f"{u}: {c:,}" for u, c in type_units[tf].most_common(5))
            A(f"- `{tf}` -> {top}")
    A("")

    if args.dedup:
        n_groups = len(groups)
        collapse = grouped_mentions / n_groups if n_groups else 0
        A("## 4b. Deduplicated impact estimate")
        A("")
        A(f"Clustering by (primary flooded location, earliest flood date) collapsed "
          f"**{grouped_mentions:,} clusterable mentions into {n_groups:,} distinct "
          f"events** ({collapse:.1f} mentions each on average). A further "
          f"{ungroupable:,} flood articles had neither a location nor a date and could "
          f"not be clustered - they are excluded here.")
        A("")
        A("> Heuristic. Same place + same day is treated as one event, so this "
          "**over-merges** distinct floods that coincide and **under-merges** when "
          "outlets spell a place differently. Per cluster the **largest** reported "
          "figure is kept (tolls climb as a disaster unfolds; the max is closest to "
          "final). Extraction errors like the 1.33-billion outlier survive clustering, "
          "so median remains the safer number.")
        A("")
        A("| field | events w/ value | median (per event) | sum across events |")
        A("|---|---:|---:|---:|")
        for f in COUNT_FIELDS:
            vals = [b[f] for b in groups.values() if f in b]
            if not vals:
                continue
            A(f"| {f} | {len(vals):,} | {median(vals):,.0f} | {sum(vals):,.0f} |")
        A("")
        # show what the biggest clusters actually are
        top_events = sorted(
            groups.items(),
            key=lambda kv: -kv[1].get("deaths", kv[1].get("affected_people", 0)),
        )[:10]
        A("**Largest deduplicated events** (by deaths, else affected):")
        A("")
        A("| location | date | deaths | affected | displaced | evacuated |")
        A("|---|---|---:|---:|---:|---:|")
        for (loc, day), b in top_events:
            A(f"| {loc} | {day} | {b.get('deaths', 0):,.0f} | "
              f"{b.get('affected_people', 0):,.0f} | {b.get('displaced', 0):,.0f} | "
              f"{b.get('evacuated', 0):,.0f} |")
        A("")

    A("## 5. Causes and triggers")
    A("")
    if causes:
        A(f"**cause** (top {args.top})")
        A("")
        for name, n in causes.most_common(args.top):
            A(f"- {name} - {n:,}")
        A("")
    if triggers:
        A(f"**trigger** (top {args.top})")
        A("")
        for name, n in triggers.most_common(args.top):
            A(f"- {name} - {n:,}")
        A("")

    if agencies:
        A("## 6. Response agencies")
        A("")
        for name, n in agencies.most_common(args.top):
            A(f"- {name} - {n:,}")
        A("")

    A("## 7. Data quality")
    A("")
    A(f"**flood_dates format** ({n_date_values:,} date values)")
    A("")
    A("| quality | count | share |")
    A("|---|---:|---:|")
    for q in ["valid", "partial", "not-iso", "impossible", "non-string"]:
        if date_q.get(q):
            A(f"| {q} | {date_q[q]:,} | {pct(date_q[q], n_date_values)} |")
    A("")
    multi = sum(v for k, v in events_per_flood.items() if k > 1)
    A(f"- Events per flood article: " +
      ", ".join(f"{k}->{v:,}" for k, v in sorted(events_per_flood.items())))
    A(f"- The prompt requires EXACTLY ONE event, but **{multi:,} articles returned 2+** "
      f"(up to {max(events_per_flood)}). The single-event gate is not strictly enforced "
      f"by the model.")
    A(f"- Distinct event field names seen: **{len(field_use):,}** "
      f"(prompt sanctions ~50; the rest are model-invented synonyms/variants).")
    prose_total = sum(count_stats[f]["prose"] for f in COUNT_FIELDS)
    clean_total = sum(count_stats[f]["clean"] for f in COUNT_FIELDS)
    if clean_total + prose_total:
        A(f"- Numeric count fields that were prose rather than plain numbers: "
          f"**{pct(prose_total, clean_total + prose_total)}** "
          f"({prose_total:,} of {clean_total + prose_total:,}).")
    A("")
    A("---")
    A(f"_One row per flood article unless noted. Generated over {total:,} records._")

    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}  ({len(L)} lines)")
    print(f"  articles {total:,} | floods {flood:,} ({pct(flood, total)}) | "
          f"verifiable {verifiable:,} ({pct(verifiable, flood)} of floods)")


if __name__ == "__main__":
    main()
