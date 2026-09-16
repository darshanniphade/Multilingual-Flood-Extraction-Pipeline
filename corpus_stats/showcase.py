"""Pick the best final-output exemplars per year, and measure schema drift.

Two rankings:
  * canonical-rich - most filled fields drawn from the 50 the prompt names,
    restricted to verifiable articles. This is the "pipeline working" exhibit.
  * absolute-rich  - most filled fields of any name. This is the schema-drift
    exhibit (invented keys inflate the count).
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(r"C:\darsh\pipeline")
OUT = Path(r"C:\Users\DNK\AppData\Local\Temp\claude\c--darsh-pipeline\05957676-323a-4620-a828-22c520f766fe\scratchpad\showcase.md")
YEARS = ["2021", "2022", "2023"]

CANON = set("""country state province district city village location river stream lake dam water_body
event_date start_date end_date cause trigger rainfall_mm water_depth water_level water_height
water_unit flood_duration duration_unit affected_people evacuated displaced deaths injured missing
affected_people_type evacuated_type displaced_type deaths_type injured_type missing_type
houses_damaged houses_destroyed roads_damaged bridges_damaged schools_damaged hospitals_damaged
crop_damage crop_damage_unit livestock_loss livestock_loss_type economic_loss
economic_loss_currency response_agencies summary""".split())

MARQUEE = {
    "2021": ("Chamoli / Ida / Henan", ("chamoli", "zhengzhou", "henan", "ida", "dhauliganga")),
    "2022": ("Pakistan / Petropolis / Durban", ("pakistan", "petropolis", "petrópolis", "durban", "sindh")),
    "2023": ("Derna / Kakhovka / Sikkim", ("derna", "libya", "kherson", "kakhovka", "teesta", "sikkim")),
}


def events_of(ex):
    evs = ex.get("events")
    if not evs and isinstance(ex.get("event_details"), dict):
        evs = [ex["event_details"]] if ex["event_details"] else []
    return [e for e in (evs or []) if isinstance(e, dict)]


def filled(ev):
    return {k: v for k, v in ev.items() if v not in (None, "", [], {}, "null", "N/A")}


def blob(r, ev):
    parts = [str(r.get("translated_title", ""))]
    ex = r.get("extraction") or {}
    parts += [str(x) for x in (ex.get("flooded_locations") or [])]
    parts += [str(v) for v in ev.values() if isinstance(v, (str, int, float))]
    return " ".join(parts).lower()


def main():
    out = []
    w = out.append
    w("# Final-output exemplars + schema drift (2021 · 2022 · 2023)\n")
    w("Scored two ways. **Canonical** counts only the 50 fields `system_prompt.txt` names, "
      "and only for articles that passed the verifiability gate — this is the honest "
      "'here is what one good record looks like' exhibit. **Absolute** counts every key the "
      "model emitted, which is the schema-drift exhibit.\n")

    drift_rows = []

    for y in YEARS:
        d = ROOT / "data" / "extracted" / y
        seen = set()
        canon_best, abs_best, marquee_best = [], [], []
        inst_canon = inst_total = 0
        names = Counter()
        mq_label, mq_keys = MARQUEE[y]

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
                    for ev in events_of(ex):
                        f = filled(ev)
                        nc = sum(1 for k in f if k in CANON)
                        inst_total += len(f)
                        inst_canon += nc
                        for k in f:
                            names[k] += 1
                        rec = (r.get("article_id"), r.get("month"),
                               r.get("translated_title", ""), ex, ev, nc, len(f))
                        if ver:
                            canon_best.append((nc, rec))
                            canon_best.sort(key=lambda t: -t[0])
                            del canon_best[6:]
                            if any(k in blob(r, ev) for k in mq_keys):
                                marquee_best.append((nc, rec))
                                marquee_best.sort(key=lambda t: -t[0])
                                del marquee_best[4:]
                        abs_best.append((len(f), rec))
                        abs_best.sort(key=lambda t: -t[0])
                        del abs_best[4:]

        invented = [k for k in names if k not in CANON]
        drift_rows.append((y, len(names), len(invented),
                           100 * inst_canon / max(1, inst_total), inst_total))

        w(f"\n---\n\n## {y}\n")

        def dump(tag, items, n=2):
            for rank, (score, (aid, mon, title, ex, ev, nc, tot)) in enumerate(items[:n], 1):
                f = filled(ev)
                w(f"### {tag} #{rank} — **{nc} of 50 canonical** fields "
                  f"({tot} total) · `{aid}` · {mon}\n")
                w(f"**Headline:** {title}\n")
                w(f"**Gate:** `is_verifiable_flood = {ex.get('is_verifiable_flood')}` · "
                  f"dates `{ex.get('flood_dates')}` · "
                  f"{len(ex.get('flooded_locations') or [])} flooded locations\n")
                canon = {k: v for k, v in f.items() if k in CANON}
                extra = [k for k in f if k not in CANON]
                w("```json")
                w(json.dumps(canon, indent=2, ensure_ascii=False))
                w("```")
                if extra:
                    w(f"\n*Plus {len(extra)} off-schema keys the model invented:* "
                      + ", ".join(f"`{k}`" for k in extra[:14])
                      + (" …" if len(extra) > 14 else "") + "\n")
                else:
                    w("\n*No off-schema keys — this record is fully on-schema.*\n")

        w("### A. Best verifiable record (canonical-scored)\n")
        dump(f"{y} canonical", canon_best, 2)
        w(f"### B. Best record on the year's marquee event — {mq_label}\n")
        if marquee_best:
            dump(f"{y} {mq_label}", marquee_best, 1)
        else:
            w("_none matched_\n")
        w("### C. Most fields of any name (schema-drift exhibit)\n")
        dump(f"{y} absolute", abs_best, 1)

    w("\n---\n\n## Schema drift, measured\n")
    w("| year | distinct field names | of which invented | canonical share of filled values | filled values |")
    w("|---|---:|---:|---:|---:|")
    for y, nn, ni, share, tot in drift_rows:
        w(f"| {y} | {nn:,} | {ni:,} ({100*ni/nn:.1f} %) | **{share:.1f} %** | {tot:,} |")
    w("\nRead this as: the *long tail of names* is huge, but the *mass of data* still lands on "
      "the sanctioned schema. The invented keys are mostly one-off singletons.\n")

    OUT.write_text("\n".join(out), encoding="utf-8")
    print("wrote", OUT)
    for row in drift_rows:
        print(row)


if __name__ == "__main__":
    main()
