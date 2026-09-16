"""Pick the single strongest record per year for the deck.

Scores only *informative* canonical fields: a value that is 0, False or a bare
True carries no extracted information (the prompt forbids guessing, so a 0 is
usually the model filling a blank). Also requires the article's own publish
month to agree with the extracted flood date, which rules out the historical
retrospectives that otherwise dominate the top of the ranking.
"""
import json
from pathlib import Path

ROOT = Path(r"C:\darsh\pipeline")
OUT = Path(r"C:\Users\DNK\AppData\Local\Temp\claude\c--darsh-pipeline\05957676-323a-4620-a828-22c520f766fe\scratchpad\best_records.md")
YEARS = ["2021", "2022", "2023"]

CANON = set("""country state province district city village location river stream lake dam water_body
event_date start_date end_date cause trigger rainfall_mm water_depth water_level water_height
water_unit flood_duration duration_unit affected_people evacuated displaced deaths injured missing
affected_people_type evacuated_type displaced_type deaths_type injured_type missing_type
houses_damaged houses_destroyed roads_damaged bridges_damaged schools_damaged hospitals_damaged
crop_damage crop_damage_unit livestock_loss livestock_loss_type economic_loss
economic_loss_currency response_agencies summary""".split())


def events_of(ex):
    evs = ex.get("events")
    if not evs and isinstance(ex.get("event_details"), dict):
        evs = [ex["event_details"]] if ex["event_details"] else []
    return [e for e in (evs or []) if isinstance(e, dict)]


def informative(v):
    if v in (None, "", [], {}, "null", "N/A"):
        return False
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)) and v == 0:
        return False
    if isinstance(v, str) and v.strip().lower() in ("0", "none", "unknown", "n/a"):
        return False
    return True


def main():
    out = []
    w = out.append
    w("# The strongest extracted record of each year\n")
    w("Ranked by **informative canonical fields** — the 50 fields the prompt names, "
      "counting only values that actually carry information (a `0`, `false` or bare `true` "
      "does not). Restricted to articles that passed the verifiability gate *and* whose "
      "extracted flood date falls in the article's own batch month, which excludes "
      "historical retrospectives.\n")

    for y in YEARS:
        d = ROOT / "data" / "extracted" / y
        seen = set()
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
                    if not ex.get("contains_flood_event") or not ex.get("is_verifiable_flood"):
                        continue
                    mon = (r.get("month") or "").replace("_", "-")
                    dates = [str(x) for x in (ex.get("flood_dates") or [])]
                    if not any(x.startswith(mon) for x in dates):
                        continue
                    for ev in events_of(ex):
                        f = {k: v for k, v in ev.items() if informative(v)}
                        nc = sum(1 for k in f if k in CANON)
                        best.append((nc, len(f), r.get("article_id"), r.get("month"),
                                     r.get("translated_title", ""), ex, f))
            best.sort(key=lambda t: (-t[0], -t[1]))
            del best[6:]

        w(f"\n---\n\n## {y}\n")
        for rank, (nc, tot, aid, mon, title, ex, f) in enumerate(best[:3], 1):
            canon = {k: v for k, v in f.items() if k in CANON}
            extra = {k: v for k, v in f.items() if k not in CANON}
            w(f"### {y} · #{rank} — {nc} informative canonical fields · `{aid}` · {mon}\n")
            w(f"**Headline:** {title}\n")
            w(f"**Gate passed:** dates `{ex.get('flood_dates')}` · "
              f"locations `{(ex.get('flooded_locations') or [])[:8]}`"
              f"{' …' if len(ex.get('flooded_locations') or []) > 8 else ''}\n")
            w("```json")
            w(json.dumps(canon, indent=2, ensure_ascii=False))
            w("```")
            if extra:
                w(f"\n*Off-schema keys ({len(extra)}):* " +
                  ", ".join(f"`{k}`" for k in list(extra)[:12]) + "\n")
            w("")

    OUT.write_text("\n".join(out), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
