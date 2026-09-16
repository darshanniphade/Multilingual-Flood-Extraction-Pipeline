"""Build the annotation samples that every quality number in this project needs.

Nothing in this repository reports accuracy, F1 or nDCG unless it was measured
against human labels. This script produces the files those labels go into: one
JSONL per task, each row carrying what the system predicted plus an empty `gold`
block for the annotator to fill. `evaluation/score.py` then compares the two and
refuses to score a file nobody has annotated.

Sampling is stratified and seeded (runtime.seed), so the same command produces
the same sample and a second annotator can be given exactly the same rows.

Tasks:

    detection      is this article about one real, already-occurred flood?
                   Stratified across stage 5's three outcomes so the sample
                   contains positives, negatives and the borderline band -
                   sampling only positives measures nothing about recall.
    flood_type     which hydrological type? Includes rows the rule model
                   abstained on, or the abstention rate can never be judged.
    ner            entity spans. Pre-filled with the system's spans so the
                   annotator corrects rather than types.
    extraction     field-by-field: date, locations, rivers, counts, cause.
    clustering     pairs of articles: same real-world flood or not? Drawn from
                   inside clusters AND from near-miss pairs just below the
                   linking threshold, which is where the errors live.
    retrieval      query-document relevance, pooled over hybrid, BM25-only and
                   dense-only runs so the pool is not biased to one retriever.

    python make_annotation_sample.py --task detection --n 200
    python make_annotation_sample.py --task flood_type --n 150
    python make_annotation_sample.py --task ner --n 60
    python make_annotation_sample.py --task clustering --n 200
    python make_annotation_sample.py --task retrieval --queries queries.txt

Writes evaluation/gold/<task>_sample.jsonl and evaluation/gold/<task>_INSTRUCTIONS.md.
Existing annotated files are never overwritten without --force.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "7_semantic_layer"))

from common import (  # noqa: E402
    DEFAULT_CONFIG,
    article_text,
    iter_events,
    load_article,
    load_config,
    load_sidecar,
    output_root,
    read_jsonl,
    setup_logging,
    write_jsonl,
)

log = logging.getLogger("eval.sample")

GOLD_DIR = HERE / "gold"

FLOOD_TYPES = ["River Flood", "Flash Flood", "Urban Flood", "Coastal Flood", "Pluvial Flood", "Dam/Reservoir Flood", "Other", "Unknown"]


def stratum_of(ev) -> str:
    if ev.verifiable:
        return "verifiable"
    if ev.contains_flood:
        return "flood_not_verifiable"
    return "not_flood"


def stratified_sample(cfg: dict, n: int, strata: list[str], years: list[str] | None, seed: int) -> list:
    """Reservoir-sample each stratum in one pass over the corpus."""
    per = max(n // max(len(strata), 1), 1)
    rng = random.Random(seed)
    buckets: dict[str, list] = {s: [] for s in strata}
    seen: Counter = Counter()
    for ev in iter_events(cfg, years, select="all"):
        stratum = stratum_of(ev)
        if stratum not in buckets:
            continue
        seen[stratum] += 1
        bucket = buckets[stratum]
        if len(bucket) < per:
            bucket.append(ev)
        else:
            j = rng.randint(0, seen[stratum] - 1)
            if j < per:
                bucket[j] = ev
    log.info("stratum sizes seen: %s", dict(seen))
    out = [ev for bucket in buckets.values() for ev in bucket]
    rng.shuffle(out)
    return out


def preview(cfg: dict, ev, chars: int = 1500) -> str:
    return article_text(load_article(cfg, ev.year, ev.month, ev.article_id))[:chars]


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------


def task_detection(cfg: dict, args, seed: int) -> tuple[list[dict], list[str]]:
    strata = cfg["eval"].get("strata", ["verifiable", "flood_not_verifiable", "not_flood"])
    events = stratified_sample(cfg, args.n, strata, args.years, seed)[: args.n]
    rows = [
        {
            "task": "detection",
            "uid": ev.uid,
            "year": int(ev.year),
            "title": ev.title,
            "text": preview(cfg, ev),
            "stratum": stratum_of(ev),
            "system": {
                "contains_flood_event": ev.contains_flood,
                "is_verifiable_flood": ev.verifiable,
            },
            "gold": {"is_flood_event": None, "is_verifiable": None, "notes": ""},
        }
        for ev in events
    ]
    instructions = [
        "## detection",
        "",
        "For each row set `gold.is_flood_event` to true/false and, when true, `gold.is_verifiable` to true/false.",
        "",
        "- **is_flood_event**: the article's main subject is ONE real flood that has already happened or is",
        "  happening. False for forecasts, warnings, risk studies, funding announcements, anniversaries, and",
        "  for articles covering several unrelated floods.",
        "- **is_verifiable**: the article states BOTH a specific date the flooding occurred and at least one",
        "  named place that flooded. This is the stage-5b gate, so judge it literally.",
        "- Judge from `text` only; ignore what `system` says. If `text` is empty the body was missing - set",
        "  both to null and note it.",
    ]
    return rows, instructions


def task_flood_type(cfg: dict, args, seed: int) -> tuple[list[dict], list[str]]:
    root = output_root(cfg)
    predictions = load_sidecar(root / "flood_type.jsonl")
    if not predictions:
        raise SystemExit("no flood_type.jsonl - run 7_semantic_layer/flood_type.py first")
    rng = random.Random(seed)
    # Half from the typed rows (spread across labels), half from abstentions, so
    # both precision on the typed set and the cost of abstaining are measurable.
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in predictions.values():
        by_label[row.get("flood_type") or "Unknown"].append(row)
    typed = [row for label, rows in by_label.items() if label != "Unknown" for row in rows]
    unknown = by_label.get("Unknown", [])
    rng.shuffle(typed)
    rng.shuffle(unknown)
    half = args.n // 2
    chosen = typed[:half] + unknown[: args.n - half]
    rng.shuffle(chosen)

    wanted = {row["uid"] for row in chosen}
    events = {ev.uid: ev for ev in iter_events(cfg, args.years, select="all") if ev.uid in wanted}
    rows = []
    for prediction in chosen:
        ev = events.get(prediction["uid"])
        if ev is None:
            continue
        rows.append(
            {
                "task": "flood_type",
                "uid": ev.uid,
                "title": ev.title,
                "text": preview(cfg, ev),
                "system": {
                    "flood_type": prediction.get("flood_type"),
                    "confidence": prediction.get("confidence"),
                    "cue": prediction.get("cue"),
                    "method": prediction.get("method"),
                },
                "gold": {"flood_type": None, "notes": ""},
            }
        )
    instructions = [
        "## flood_type",
        "",
        f"Set `gold.flood_type` to exactly one of: {', '.join(FLOOD_TYPES)}.",
        "",
        "- **River Flood**: a watercourse overtopped or breached its banks/embankment.",
        "- **Flash Flood**: sudden onset, hours or less - cloudburst, hill torrent, GLOF.",
        "- **Urban Flood**: rainfall exceeded a built-up area's drainage; waterlogged streets.",
        "- **Coastal Flood**: seawater driven inland - storm surge, tide.",
        "- **Pluvial Flood**: rain ponding on the surface away from any channel or drainage system.",
        "- **Dam/Reservoir Flood**: a release, breach or failure of built water infrastructure.",
        "- **Other**: a real flood of a different mechanism (tsunami, burst main, ice jam).",
        "- **Unknown**: the article genuinely does not say. Use it freely - the system is scored on when it",
        "  abstains as well as on what it labels.",
    ]
    return rows, instructions


def task_ner(cfg: dict, args, seed: int) -> tuple[list[dict], list[str]]:
    root = output_root(cfg)
    tagged = load_sidecar(root / "ner.jsonl")
    if not tagged:
        raise SystemExit("no ner.jsonl - run 7_semantic_layer/ner.py first")
    rng = random.Random(seed)
    uids = list(tagged)
    rng.shuffle(uids)
    chosen = uids[: args.n]
    wanted = set(chosen)
    events = {ev.uid: ev for ev in iter_events(cfg, args.years, select="all") if ev.uid in wanted}

    max_chars = int(cfg["ner"]["max_chars"])
    rows = []
    for uid in chosen:
        ev = events.get(uid)
        if ev is None:
            continue
        body = article_text(load_article(cfg, ev.year, ev.month, ev.article_id))
        text = ((ev.title.strip() + "\n") if cfg["ner"].get("use_title", True) else "") + body
        text = text[:max_chars]
        system_entities = [
            {k: e[k] for k in ("text", "label", "start", "end", "confidence", "source") if k in e}
            for e in tagged[uid].get("entities", [])
            if e.get("start", -1) >= 0
        ]
        rows.append(
            {
                "task": "ner",
                "uid": uid,
                "text": text,
                "system_entities": system_entities,
                "gold_entities": [],
                "gold": {"reviewed": False, "notes": ""},
            }
        )
    instructions = [
        "## ner",
        "",
        "Copy `system_entities` into `gold_entities`, then correct it: delete wrong spans, fix boundaries and",
        "labels, add what was missed. Set `gold.reviewed` to true when the row is finished - rows left false",
        "are ignored by the scorer.",
        "",
        "Each gold entity is `{\"text\": ..., \"label\": ..., \"start\": ..., \"end\": ...}` with `start`/`end`",
        "as character offsets into this row's `text` exactly as given (do not reflow it).",
        "",
        "Labels: PERSON, ORGANIZATION, LOCATION, DATE, NUMBER, RIVER, FLOOD_TYPE, WEATHER_EVENT,",
        "INFRASTRUCTURE, CASUALTY, EVACUATION, DAMAGE.",
        "",
        "- A named watercourse is RIVER, not LOCATION.",
        "- CASUALTY / EVACUATION / DAMAGE mark the trigger word or phrase ('killed', 'evacuated', 'washed",
        "  away'), not the number next to it - the number is its own NUMBER span.",
        "- DATE covers explicit dates only, not bare weekdays.",
    ]
    return rows, instructions


def task_extraction(cfg: dict, args, seed: int) -> tuple[list[dict], list[str]]:
    rng = random.Random(seed)
    pool = []
    for i, ev in enumerate(iter_events(cfg, args.years, select="verifiable")):
        if len(pool) < args.n:
            pool.append(ev)
        else:
            j = rng.randint(0, i)
            if j < args.n:
                pool[j] = ev
    flood_types = load_sidecar(output_root(cfg) / "flood_type.jsonl")
    rows = []
    for ev in pool:
        counts = ev.counts()
        rows.append(
            {
                "task": "extraction",
                "uid": ev.uid,
                "title": ev.title,
                "text": preview(cfg, ev, 4000),
                "system": {
                    "event_dates": ev.all_dates(),
                    "country": ev.country,
                    "locations": ev.places(),
                    "rivers": ev.rivers(),
                    "causes": ev.causes(),
                    "flood_type": (flood_types.get(ev.uid) or {}).get("flood_type", "Unknown"),
                    "deaths": counts.get("deaths"),
                    "injured": counts.get("injured"),
                    "affected_people": counts.get("affected_people"),
                    "displaced": counts.get("displaced"),
                    "evacuated": counts.get("evacuated"),
                },
                "gold": {
                    "event_dates": None,
                    "country": None,
                    "locations": None,
                    "rivers": None,
                    "causes": None,
                    "flood_type": None,
                    "deaths": None,
                    "injured": None,
                    "affected_people": None,
                    "displaced": None,
                    "evacuated": None,
                    "notes": "",
                },
            }
        )
    instructions = [
        "## extraction",
        "",
        "Fill each `gold` field from `text` alone.",
        "",
        "- Lists (`event_dates`, `locations`, `rivers`, `causes`): everything the article states, `[]` if none.",
        "  Dates as YYYY-MM-DD; only dates the flooding occurred on, never the publication date.",
        "- Counts: the number the article states, or null. Do not add numbers across paragraphs; if the",
        "  article gives both a district figure and a total, record the total.",
        "- A field left null is treated as 'not annotated' and skipped by the scorer, so write 0 or [] when",
        "  the correct answer is genuinely 'none'.",
    ]
    return rows, instructions


def task_clustering(cfg: dict, args, seed: int) -> tuple[list[dict], list[str]]:
    root = output_root(cfg)
    clusters_path = root / "clusters.jsonl"
    if not clusters_path.exists():
        raise SystemExit("no clusters.jsonl - run 7_semantic_layer/cluster_events.py first")
    meta = {row["uid"]: row for row in read_jsonl(root / "events_meta.jsonl")}
    rng = random.Random(seed)

    linked_pairs: list[tuple[str, str]] = []
    for row in read_jsonl(clusters_path):
        members = row.get("members", [])
        if len(members) < 2:
            continue
        for _ in range(min(len(members) - 1, 3)):
            a, b = rng.sample(members, 2)
            linked_pairs.append((a, b))
    rng.shuffle(linked_pairs)

    # Near misses: highest-cosine pairs the linker refused. These are where a
    # coreference system is actually wrong, and a sample of random non-pairs
    # (which are trivially different) would hide that.
    import numpy as np

    matrix = np.load(root / "embeddings.npy")
    ids = json.loads((root / "ids.json").read_text(encoding="utf-8"))
    cluster_of = {}
    for row in read_jsonl(clusters_path):
        for uid in row.get("members", []):
            cluster_of[uid] = row["event_id"]
    idx = rng.sample(range(len(ids)), min(4000, len(ids)))
    sub = matrix[idx]
    sims = sub @ sub.T
    np.fill_diagonal(sims, -1)
    near: list[tuple[float, str, str]] = []
    for i in range(len(idx)):
        j = int(np.argmax(sims[i]))
        a, b = ids[idx[i]], ids[idx[j]]
        if cluster_of.get(a) != cluster_of.get(b):
            near.append((float(sims[i][j]), a, b))
    near.sort(reverse=True)

    half = args.n // 2
    pairs = [(a, b, True) for a, b in linked_pairs[:half]]
    pairs += [(a, b, False) for _s, a, b in near[: args.n - half]]
    rng.shuffle(pairs)

    rows = []
    for a, b, same in pairs:
        ma, mb = meta.get(a, {}), meta.get(b, {})
        rows.append(
            {
                "task": "clustering",
                "uid_a": a,
                "uid_b": b,
                "a": {"title": ma.get("title"), "date": ma.get("date_min"), "country": ma.get("country"), "locations": ma.get("locations")},
                "b": {"title": mb.get("title"), "date": mb.get("date_min"), "country": mb.get("country"), "locations": mb.get("locations")},
                "system": {"same_event": same, "event_id_a": cluster_of.get(a), "event_id_b": cluster_of.get(b)},
                "gold": {"same_event": None, "notes": ""},
            }
        )
    instructions = [
        "## clustering (event coreference)",
        "",
        "Set `gold.same_event` to true when A and B describe THE SAME real-world flood.",
        "",
        "- Same flood, different day of coverage, different outlet, different death toll: **true**.",
        "- Same river and same week but two distinct flood episodes: **false**.",
        "- Same country and same month but different districts, with no shared event: **false**.",
        "- A national round-up covering many floods versus one of them: **false**.",
        "",
        "Half these pairs were linked by the system and half are its highest-similarity refusals, so expect",
        "roughly half to be true. Do not use `system` to decide.",
    ]
    return rows, instructions


def task_retrieval(cfg: dict, args, seed: int) -> tuple[list[dict], list[str]]:
    from search import BENCHMARK_QUERIES, HybridSearch

    queries = BENCHMARK_QUERIES
    if args.queries:
        queries = [q.strip() for q in Path(args.queries).read_text(encoding="utf-8").splitlines() if q.strip()]
    depth = int(args.pool_depth)
    engine = HybridSearch(cfg, load_dense=True)

    rows = []
    for query in queries:
        pooled: dict[str, dict] = {}
        # Pool over three runs so the judgements are not biased to one retriever.
        for run_name in ("hybrid", "bm25", "dense"):
            if run_name == "bm25":
                hits = [uid for uid, _ in engine.bm25.search(query, depth)]
            elif run_name == "dense":
                result = engine.retrieve(query, {}, depth)
                hits = [c["uid"] for c in sorted(result["all_candidates"], key=lambda c: -c["cosine"])[:depth]]
            else:
                result = engine.retrieve(query, {}, depth)
                hits = [c["uid"] for c in result["all_candidates"][:depth]]
            for rank, uid in enumerate(hits, 1):
                entry = pooled.setdefault(uid, {"uid": uid, "runs": {}})
                entry["runs"][run_name] = rank
        for uid, entry in pooled.items():
            row_meta = engine.meta.get(uid, {})
            rows.append(
                {
                    "task": "retrieval",
                    "query": query,
                    "uid": uid,
                    "event_id": engine.uid_to_event.get(uid),
                    "title": row_meta.get("title"),
                    "date": row_meta.get("date_min"),
                    "country": row_meta.get("country"),
                    "locations": row_meta.get("locations"),
                    "pooled_from": entry["runs"],
                    "gold": {"relevance": None, "notes": ""},
                }
            )
    instructions = [
        "## retrieval",
        "",
        "Set `gold.relevance` to a graded judgement of how well the article answers the query:",
        "",
        "- **2** - fully on target: this is the event the query asks for.",
        "- **1** - related: right region or right phenomenon, wrong event or wrong period.",
        "- **0** - not relevant.",
        "",
        "Judge the query, not the article's quality. Every pooled row must be judged, including the ones that",
        "look obviously irrelevant - unjudged rows are treated as unjudged, not as zero, and that biases",
        "Recall@K.",
    ]
    return rows, instructions


TASKS = {
    "detection": task_detection,
    "flood_type": task_flood_type,
    "ner": task_ner,
    "extraction": task_extraction,
    "clustering": task_clustering,
    "retrieval": task_retrieval,
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Build stratified annotation samples for evaluation")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--n", type=int, help="rows to sample (default eval.sample_size)")
    ap.add_argument("--years", nargs="*")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--queries", help="file of queries, one per line (retrieval task)")
    ap.add_argument("--pool-depth", type=int, default=10, help="depth per run when pooling (retrieval task)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--force", action="store_true", help="overwrite an existing sample file")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    args.n = args.n or int(cfg["eval"].get("sample_size", 200))
    seed = args.seed if args.seed is not None else int(cfg["runtime"]["seed"])

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (GOLD_DIR / f"{args.task}_sample.jsonl")
    if out_path.exists() and not args.force:
        annotated = sum(
            1
            for row in read_jsonl(out_path)
            if any(v not in (None, "", [], False) for k, v in (row.get("gold") or {}).items() if k != "notes")
            or row.get("gold_entities")
        )
        raise SystemExit(
            f"{out_path} already exists with {annotated} annotated row(s). "
            "Use --force to overwrite (this destroys annotation) or --out for a new file."
        )

    rows, instructions = TASKS[args.task](cfg, args, seed)
    if not rows:
        raise SystemExit("sampled nothing - check --years and that the upstream stage has run")
    write_jsonl(out_path, rows)

    header = [
        f"# Annotation instructions - {args.task}",
        "",
        f"File: `{out_path.name}` ({len(rows)} rows), sampled with seed {seed}.",
        "",
        "Edit the `gold` block of each JSONL line in place and keep one JSON object per line. Do not reorder,",
        "delete or add rows - `evaluation/score.py` matches on the identifier fields. Leave a row's gold fields null",
        "if you cannot judge it; the scorer skips unjudged rows and reports how many it skipped.",
        "",
    ]
    (GOLD_DIR / f"{args.task}_INSTRUCTIONS.md").write_text("\n".join(header + instructions) + "\n", encoding="utf-8")
    log.info("wrote %d rows -> %s", len(rows), out_path)
    log.info("annotation guide -> %s", GOLD_DIR / f"{args.task}_INSTRUCTIONS.md")
    print(f"\nNext: annotate {out_path}, then\n  python evaluation/score.py --task {args.task}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
