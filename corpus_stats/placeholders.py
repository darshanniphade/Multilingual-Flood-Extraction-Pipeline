"""Measure 'not specified'-style placeholder values, then re-rank records without them.

The prompt forbids nulls and empty strings, so the model routes around it by
writing prose placeholders ("not specified", "not explicitly stated"). Those are
counted as filled by any naive fill-rate calculation, which inflates every
coverage number - most visibly in 2022 and 2023.
"""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(r"C:\darsh\pipeline")
OUT = Path(r"C:\Users\DNK\AppData\Local\Temp\claude\c--darsh-pipeline\05957676-323a-4620-a828-22c520f766fe\scratchpad\placeholders.md")
YEARS = ["2021", "2022", "2023"]

CANON = set("""country state province district city village location river stream lake dam water_body
event_date start_date end_date cause trigger rainfall_mm water_depth water_level water_height
water_unit flood_duration duration_unit affected_people evacuated displaced deaths injured missing
affected_people_type evacuated_type displaced_type deaths_type injured_type missing_type
houses_damaged houses_destroyed roads_damaged bridges_damaged schools_damaged hospitals_damaged
crop_damage crop_damage_unit livestock_loss livestock_loss_type economic_loss
economic_loss_currency response_agencies summary""".split())

PLACEHOLDER = re.compile(
    r"^(not\s+(specified|explicitly\s+\w+|mentioned|reported|available|provided|applicable|"
    r"stated|given|determined|quantified|known|listed|detailed|indicated|disclosed)"
    r"(\s+in\s+the\s+article)?|none|n/?a|unknown|unspecified|no\s+data|null|nil|"
    r"undisclosed|indeterminate|-{1,3})\.?$", re.I)


def is_placeholder(v):
    return isinstance(v, str) and bool(PLACEHOLDER.match(v.strip()))


def events_of(ex):
    evs = ex.get("events")
    if not evs and isinstance(ex.get("event_details"), dict):
        evs = [ex["event_details"]] if ex["event_details"] else []
    return [e for e in (evs or []) if isinstance(e, dict)]


def real(v):
    """A value carrying actual extracted information."""
    if v in (None, "", [], {}):
        return False
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)) and v == 0:
        return False
    if isinstance(v, str):
        s = v.strip()
        if not s or is_placeholder(s) or s == "0":
            return False
    return True


def main():
    out = []
    w = out.append
    w("# Placeholder values — the hidden hole in every fill rate\n")
    w("`system_prompt.txt` forbids nulls, empty strings and empty arrays. The model complies "
      "literally and writes prose instead: `\"deaths\": \"not specified\"`. Any fill-rate "
      "computed by 'is the key present and non-empty' counts those as data.\n")

    rows = []
    per_year_fields = {}
    best_by_year = {}

    for y in YEARS:
        d = ROOT / "data" / "extracted" / y
        seen = set()
        n_vals = n_ph = n_zero = n_bool = 0
        ph_by_field = Counter()
        ph_text = Counter()
        naive_fill = Counter()
        real_fill = Counter()
        events = 0
        best = []

        for path in sorted(d.glob(f"{y}_??.jsonl")):
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
                        continue
                    seen.add(key)
                    ex = r.get("extraction") or {}
                    if not ex.get("contains_flood_event"):
                        continue
                    ver = bool(ex.get("is_verifiable_flood"))
                    mon = (r.get("month") or "").replace("_", "-")
                    dates = [str(x) for x in (ex.get("flood_dates") or [])]
                    in_month = any(x.startswith(mon) for x in dates)

                    for ev in events_of(ex):
                        events += 1
                        clean = {}
                        for k, v in ev.items():
                            if v in (None, "", [], {}):
                                continue
                            n_vals += 1
                            naive_fill[k] += 1
                            if is_placeholder(v):
                                n_ph += 1
                                ph_by_field[k] += 1
                                ph_text[str(v).strip().lower()] += 1
                            elif isinstance(v, bool):
                                n_bool += 1
                            elif isinstance(v, (int, float)) and v == 0:
                                n_zero += 1
                            elif real(v):
                                real_fill[k] += 1
                                clean[k] = v
                        if ver and in_month:
                            nc = sum(1 for k in clean if k in CANON)
                            best.append((nc, len(clean), r.get("article_id"),
                                         r.get("month"), r.get("translated_title", ""),
                                         ex, clean))
            best.sort(key=lambda t: (-t[0], -t[1]))
            del best[6:]

        rows.append((y, events, n_vals, n_ph, n_zero, n_bool))
        per_year_fields[y] = (naive_fill, real_fill, events, ph_by_field, ph_text)
        best_by_year[y] = best

    w("## How much of the 'filled' data is not data\n")
    w("| year | event objects | non-empty values | `not specified`-type | zeros | bare booleans | **real values** |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for y, ev, nv, ph, z, b in rows:
        realv = nv - ph - z - b
        w(f"| {y} | {ev:,} | {nv:,} | {ph:,} ({100*ph/nv:.1f} %) | {z:,} ({100*z/nv:.1f} %) | "
          f"{b:,} ({100*b/nv:.1f} %) | **{realv:,} ({100*realv/nv:.1f} %)** |")
    w("")

    for y in YEARS:
        naive, realf, events, phf, pht = per_year_fields[y]
        w(f"\n### {y} — fill rate, naive vs real\n")
        w("| field | naive fill | real fill | inflation |")
        w("|---|---:|---:|---:|")
        for k, _ in naive.most_common(18):
            a = 100 * naive[k] / events
            b = 100 * realf[k] / events
            w(f"| `{k}` | {a:.1f} % | **{b:.1f} %** | {a-b:+.1f} pp |")
        w("")
        w("Most common placeholder strings: " +
          ", ".join(f'`{t}` {c:,}' for t, c in pht.most_common(6)) + "\n")
        w("Fields most often placeholdered: " +
          ", ".join(f"`{k}` {c:,}" for k, c in phf.most_common(8)) + "\n")

    w("\n---\n\n# The strongest real record of each year\n")
    w("Verifiable, flood date inside the article's own batch month, ranked by canonical "
      "fields holding a **real** value (placeholders, zeros and bare booleans excluded).\n")
    for y in YEARS:
        w(f"\n## {y}\n")
        for rank, (nc, tot, aid, mon, title, ex, clean) in enumerate(best_by_year[y][:2], 1):
            canon = {k: v for k, v in clean.items() if k in CANON}
            extra = {k: v for k, v in clean.items() if k not in CANON}
            w(f"### {y} · #{rank} — {nc} real canonical fields · `{aid}` · {mon}\n")
            w(f"**Headline:** {title}\n")
            locs = ex.get("flooded_locations") or []
            w(f"**Gate:** dates `{ex.get('flood_dates')}` · {len(locs)} locations "
              f"`{locs[:6]}`{' …' if len(locs) > 6 else ''}\n")
            w("```json")
            w(json.dumps(canon, indent=2, ensure_ascii=False))
            w("```")
            if extra:
                w(f"\n*Off-schema keys ({len(extra)}):* " +
                  ", ".join(f"`{k}`" for k in list(extra)[:12]) + "\n")
            w("")

    OUT.write_text("\n".join(out), encoding="utf-8")
    print("wrote", OUT)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
