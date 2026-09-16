"""Compute deck-ready metrics for 2021, 2022 and 2023.

Reads the extraction JSONL trees (article-level truth) plus the upstream
stage artefacts, and writes one markdown report per year block.
"""
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"C:\darsh\pipeline")
OUT = Path(r"C:\Users\DNK\AppData\Local\Temp\claude\c--darsh-pipeline\05957676-323a-4620-a828-22c520f766fe\scratchpad\year_metrics.md")

YEARS = ["2021", "2022", "2023"]
ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")

# Raw crawl size, measured: 2021 documented in PIPELINE_FLOW.md; 2022 from the
# 4_clean_dedup report over H:\2022; 2023 counted on disk in data\ssd\2023.
RAW = {"2021": 1383600, "2022": 1935415, "2023": 2950352}
# Where clean+dedup ran. 2021 ran it at stage 4 (post-filter); 2022 ran it up
# front on the raw crawl; 2023 never ran it at all.
PRECLEAN = {"2022": 785911}

TITLES = {
    "2021": ROOT / "data" / "translated titles" / "translated_titles.jsonl",
    "2022": ROOT / "data" / "translated titles" / "2022" / "translated_titles.jsonl",
    "2023": ROOT / "data" / "translated titles" / "2023" / "translated_titles.jsonl",
}
ONLY_ENG_REPORT = {
    "2022": ROOT / "data" / "only eng" / "_reports" / "2022_report.json",
    "2023": ROOT / "data" / "only eng" / "_reports" / "2023_report.json",
}


def count_lines(path):
    if not path.exists():
        return None
    n = 0
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 22)
            if not b:
                break
            n += b.count(b"\n")
    return n


def events_of(ex):
    evs = ex.get("events")
    if not evs and isinstance(ex.get("event_details"), dict):
        evs = [ex["event_details"]] if ex["event_details"] else []
    return [e for e in (evs or []) if isinstance(e, dict)]


def filled(ev):
    return {k: v for k, v in ev.items() if v not in (None, "", [], {}, "null", "N/A")}


def scan_year(year):
    d = ROOT / "data" / "extracted" / year
    files = sorted(d.glob(f"{year}_??.jsonl"))
    s = {
        "articles": 0, "flood": 0, "verifiable": 0,
        "has_date": 0, "has_loc": 0, "events": 0, "ver_events": 0,
        "multi_event": 0, "zero_event_flood": 0,
    }
    by_month = Counter()             # flood articles per batch month
    month_total = Counter()          # all articles per batch month
    ver_by_month = Counter()         # verifiable articles per batch month
    flood_date_month = Counter()     # by extracted flood date
    field_counts = Counter()         # events containing field
    distinct_fields = set()
    countries = Counter()
    locations = Counter()
    rivers = Counter()
    causes = Counter()
    date_bad = 0
    date_total = 0
    richest = []                     # (nfields, article_id, month, title, event)
    seen = set()                     # (month, article_id) - resume can double-write
    dup_lines = 0

    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (r.get("month"), r.get("article_id"))
                if key in seen:
                    dup_lines += 1
                    continue
                seen.add(key)
                s["articles"] += 1
                m = r.get("month") or path.stem
                month_total[m] += 1
                ex = r.get("extraction") or {}
                if not ex.get("contains_flood_event"):
                    continue
                s["flood"] += 1
                by_month[m] += 1

                fdates = ex.get("flood_dates") or []
                flocs = ex.get("flooded_locations") or []
                if fdates:
                    s["has_date"] += 1
                if flocs:
                    s["has_loc"] += 1
                ver = bool(ex.get("is_verifiable_flood"))
                if ver:
                    s["verifiable"] += 1
                    ver_by_month[m] += 1
                for dv in fdates:
                    date_total += 1
                    mt = ISO.match(str(dv))
                    if mt:
                        flood_date_month[f"{mt.group(1)}-{mt.group(2)}"] += 1
                    else:
                        date_bad += 1
                for lv in flocs:
                    if isinstance(lv, str) and lv.strip():
                        locations[lv.strip()] += 1

                evs = events_of(ex)
                s["events"] += len(evs)
                if ver:
                    s["ver_events"] += len(evs)
                if len(evs) == 0:
                    s["zero_event_flood"] += 1
                if len(evs) >= 2:
                    s["multi_event"] += 1
                for ev in evs:
                    f = filled(ev)
                    distinct_fields.update(ev.keys())
                    for k in f:
                        field_counts[k] += 1
                    if isinstance(f.get("country"), str):
                        countries[f["country"].strip()] += 1
                    if isinstance(f.get("river"), str):
                        rivers[f["river"].strip()] += 1
                    if isinstance(f.get("cause"), str):
                        causes[f["cause"].strip().lower()] += 1
                    richest.append((len(f), r.get("article_id"), m,
                                    r.get("translated_title", ""), ev))
                    richest.sort(key=lambda t: -t[0])
                    del richest[8:]

    s["date_total"] = date_total
    s["date_bad"] = date_bad
    s["dup_lines"] = dup_lines
    return dict(s=s, by_month=by_month, month_total=month_total,
                ver_by_month=ver_by_month, flood_date_month=flood_date_month,
                field_counts=field_counts, distinct=len(distinct_fields),
                countries=countries, locations=locations, rivers=rivers,
                causes=causes, richest=richest)


def bar(n, mx, width=26):
    if mx <= 0:
        return ""
    k = max(1, round(n / mx * width)) if n else 0
    return "█" * k + "·" * (width - k)


def main():
    out = []
    w = out.append
    w("# Flood pipeline — output metrics by year (2021 · 2022 · 2023)\n")

    results = {}
    funnel = {}
    for y in YEARS:
        print(f"scanning {y} ...", flush=True)
        results[y] = scan_year(y)
        titles = count_lines(TITLES[y])
        fe = sum(count_lines(p) or 0
                 for p in sorted((ROOT / "data" / "flood_events" / y).glob("*.jsonl")))
        oe = None
        if y in ONLY_ENG_REPORT and ONLY_ENG_REPORT[y].exists():
            oe = json.load(open(ONLY_ENG_REPORT[y], encoding="utf-8"))["summary"]["written_files"]
        elif y == "2021":
            oe = 171742
        funnel[y] = {"titles": titles, "flood_events": fe, "only_eng": oe}
        print(f"  {y}: titles={titles} flood_events={fe} only_eng={oe}", flush=True)

    # ---------- cross-year funnel ----------
    w("## 1. The funnel — all three years side by side\n")
    w("| Stage | 2021 | 2022 | 2023 |")
    w("|---|---:|---:|---:|")
    rows = [
        ("0 · GDELT crawl (raw articles)", lambda y: RAW[y]),
        ("0b · Clean + dedup (pre-filter)", lambda y: PRECLEAN.get(y)),
        ("1 · Titles translated (NLLB)", lambda y: funnel[y]["titles"]),
        ("2 · Lexical title filter", lambda y: funnel[y]["flood_events"]),
        ("3 · Article translation", lambda y: funnel[y]["flood_events"]),
        ("4 · Clean+dedup / project (LLM input)", lambda y: funnel[y]["only_eng"]),
        ("5 · Articles LLM-extracted", lambda y: results[y]["s"]["articles"]),
        ("5a · Describe a flood event", lambda y: results[y]["s"]["flood"]),
        ("5b · VERIFIABLE (date + location)", lambda y: results[y]["s"]["verifiable"]),
        ("6 · Event objects in CSV", lambda y: results[y]["s"]["events"]),
    ]
    for label, fn in rows:
        cells = []
        for y in YEARS:
            v = fn(y)
            cells.append(f"{v:,}" if isinstance(v, int) else "—")
        w(f"| {label} | " + " | ".join(cells) + " |")
    w("")

    w("**Retention at each gate**\n")
    w("| Gate | 2021 | 2022 | 2023 |")
    w("|---|---:|---:|---:|")

    def pct(a, b):
        return f"{100*a/b:.2f} %" if a and b else "—"
    w("| title filter (stage 2 / titles in) | " + " | ".join(
        pct(funnel[y]["flood_events"], funnel[y]["titles"]) for y in YEARS) + " |")
    w("| clean+dedup (wherever it ran) | " + " | ".join(
        (pct(PRECLEAN[y], RAW[y]) if y in PRECLEAN
         else pct(funnel[y]["only_eng"], funnel[y]["flood_events"]) if y == "2021"
         else "not run") for y in YEARS) + " |")
    w("| LLM flood gate (5a / 5) | " + " | ".join(
        pct(results[y]["s"]["flood"], results[y]["s"]["articles"]) for y in YEARS) + " |")
    w("| verifiability gate (5b / 5a) | " + " | ".join(
        pct(results[y]["s"]["verifiable"], results[y]["s"]["flood"]) for y in YEARS) + " |")
    w("| **end-to-end (5b / raw crawl)** | " + " | ".join(
        pct(results[y]["s"]["verifiable"], RAW[y]) for y in YEARS) + " |")
    w("| compression factor (raw → verifiable) | " + " | ".join(
        (f"{RAW[y]/results[y]['s']['verifiable']:.0f}×"
         if results[y]["s"]["verifiable"] else "—") for y in YEARS) + " |")
    w("")
    w("> 2021 ran clean+dedup **after** the title filter (stage 4); 2022 moved it **in front** "
      "of title translation (1,935,415 → 785,911, −59.4 %); **2023 never ran it** — the title "
      "translator read the raw crawl directly, so 2023 counts still contain exact duplicates "
      "and sub-100-word stubs. This is the single biggest reason 2023's rates sit below the "
      "other two years.\n")
    dl = {y: results[y]["s"].get("dup_lines", 0) for y in YEARS}
    w("Duplicate output lines dropped while counting (extraction resume double-writes): " +
      ", ".join(f"{y} {dl[y]:,}" for y in YEARS) + "\n")

    # ---------- per year ----------
    for y in YEARS:
        R = results[y]
        s = R["s"]
        w(f"\n---\n\n# {y}\n")
        w("## Corpus\n")
        w("| metric | value |")
        w("|---|---:|")
        w(f"| articles LLM-extracted | {s['articles']:,} |")
        w(f"| describe a flood event | {s['flood']:,} ({100*s['flood']/s['articles']:.1f} %) |")
        w(f"| not a flood event | {s['articles']-s['flood']:,} ({100*(s['articles']-s['flood'])/s['articles']:.1f} %) |")
        w(f"| **verifiable (date + location)** | **{s['verifiable']:,}** "
          f"({100*s['verifiable']/s['articles']:.1f} % of all, {100*s['verifiable']/s['flood']:.1f} % of floods) |")
        w(f"| has >=1 flood date | {s['has_date']:,} ({100*s['has_date']/s['flood']:.1f} % of floods) |")
        w(f"| has >=1 flooded location | {s['has_loc']:,} ({100*s['has_loc']/s['flood']:.1f} % of floods) |")
        w(f"| event objects emitted | {s['events']:,} |")
        w(f"| flood articles with 0 events | {s['zero_event_flood']:,} |")
        w(f"| flood articles with 2+ events (prompt says exactly 1) | {s['multi_event']:,} |")
        w(f"| distinct event field names emitted | {R['distinct']:,} |")
        loc_only = s['has_loc'] - s['verifiable']
        w(f"| named a place but no usable date | {loc_only:,} |")
        w("")

        w("## Flood articles by batch month\n```")
        mx = max(R["by_month"].values()) if R["by_month"] else 0
        for m in sorted(R["month_total"]):
            n = R["by_month"][m]
            tot = R["month_total"][m]
            w(f"{m}  {bar(n, mx)} {n:>6,}  ({100*n/tot:.1f}% of month, {tot:,} in)")
        w("```\n")

        w("## Verifiable articles by batch month\n```")
        mxv = max(R["ver_by_month"].values()) if R["ver_by_month"] else 0
        for m in sorted(R["month_total"]):
            n = R["ver_by_month"][m]
            w(f"{m}  {bar(n, mxv)} {n:>6,}")
        w("```\n")

        w(f"## Extracted flood dates by month (in-year only, {s['date_total']:,} date values)\n```")
        inyear = {k: v for k, v in R["flood_date_month"].items() if k.startswith(y)}
        mxd = max(inyear.values()) if inyear else 0
        for k in sorted(inyear):
            w(f"{k}  {bar(inyear[k], mxd)} {inyear[k]:>6,}")
        w("```")
        out_of_year = sum(v for k, v in R["flood_date_month"].items() if not k.startswith(y))
        w(f"\nOut-of-year date values: **{out_of_year:,}** "
          f"({100*out_of_year/max(1,s['date_total']):.1f} % of all dates) — "
          f"unparseable/partial: {s['date_bad']:,}\n")
        top_out = [(k, v) for k, v in R["flood_date_month"].most_common() if not k.startswith(y)][:8]
        w("Largest out-of-year buckets: " + ", ".join(f"`{k}` {v:,}" for k, v in top_out) + "\n")

        w("## Field fill rates (share of the year's event objects)\n")
        w("| field | events | fill | field | events | fill |")
        w("|---|---:|---:|---|---:|---:|")
        top = R["field_counts"].most_common(30)
        ev = max(1, s["events"])
        for i in range(0, min(30, len(top)), 2):
            a = top[i]
            b = top[i+1] if i+1 < len(top) else ("", 0)
            bc = f"`{b[0]}` | {b[1]:,} | {100*b[1]/ev:.1f} %" if b[0] else " | | "
            w(f"| `{a[0]}` | {a[1]:,} | {100*a[1]/ev:.1f} % | {bc} |")
        w("")

        w("## Top flooded locations (unsupervised — nothing was seeded)\n")
        w("| location | mentions | | country | mentions |")
        w("|---|---:|---|---|---:|")
        L = R["locations"].most_common(15)
        C = R["countries"].most_common(15)
        for i in range(15):
            l = L[i] if i < len(L) else ("", 0)
            c = C[i] if i < len(C) else ("", 0)
            w(f"| {l[0]} | {l[1]:,} | | {c[0]} | {c[1]:,} |")
        w("")

        w("**Top rivers:** " + ", ".join(f"{k} ({v})" for k, v in R["rivers"].most_common(10) if k) + "\n")
        w("**Top causes:** " + ", ".join(f"{k} ({v})" for k, v in R["causes"].most_common(10) if k) + "\n")

        w("## Richest extracted records — the pipeline at its best\n")
        w("These are the event objects with the most non-empty fields in the year. "
          "Nothing was hand-picked beyond 'most fields filled'.\n")
        for rank, (n, aid, mon, title, ev) in enumerate(R["richest"][:3], 1):
            w(f"### {y} · #{rank} — {n} fields · `{aid}` ({mon})\n")
            w(f"> {title}\n")
            w("```json")
            w(json.dumps(filled(ev), indent=2, ensure_ascii=False))
            w("```\n")
        w("Runner-up field counts: " + ", ".join(str(t[0]) for t in R["richest"][3:8]) + "\n")

    OUT.write_text("\n".join(out), encoding="utf-8")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
