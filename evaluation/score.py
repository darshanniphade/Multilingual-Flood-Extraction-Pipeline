"""Score the system against human annotation. The only source of quality numbers.

Every metric in this repository comes from here, and every one of them needs an
annotated file produced by `make_annotation_sample.py`. If a file has no gold
labels this script refuses to score it rather than inventing a baseline; if some
rows are unjudged it reports how many it skipped, in the same table as the
result.

    python score.py --task detection
    python score.py --task flood_type
    python score.py --task ner
    python score.py --task extraction
    python score.py --task clustering
    python score.py --task retrieval          # re-runs retrieval, needs the index

Metrics per task:

    detection    precision / recall / F1 / accuracy, confusion matrix, for both
                 the "is a flood event" and the "is verifiable" decisions
    flood_type   macro and weighted F1, per-class P/R/F1, confusion matrix,
                 and what abstention costs (accuracy on typed rows vs coverage)
    ner          entity-level P/R/F1, strict (exact span and label) and relaxed
                 (any character overlap, same label), overall and per label
    extraction   per-field: exact-match accuracy for scalars, set P/R/F1 for
                 lists, plus how often the system emits a value at all
    clustering   pairwise coreference precision / recall / F1 over judged pairs
    retrieval    Recall@K, Precision@K, MRR and nDCG@K for hybrid, BM25-only and
                 dense-only, over the pooled judgements

Writes evaluation/reports/<task>_eval.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "7_semantic_layer"))

from common import (  # noqa: E402
    DEFAULT_CONFIG,
    load_config,
    md_table,
    normalize_text,
    read_jsonl,
    setup_logging,
    to_number,
    write_report,
)

log = logging.getLogger("eval.score")

GOLD_DIR = HERE / "gold"
REPORT_DIR = HERE / "reports"


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def require_annotations(rows: list[dict], task: str, judged: int) -> None:
    if judged == 0:
        raise SystemExit(
            f"{task}: none of the {len(rows)} rows are annotated. "
            f"Fill the `gold` block in evaluation/gold/{task}_sample.jsonl first - "
            "this script will not report a metric that was not measured."
        )


def binary_table(name: str, tp: int, fp: int, fn: int, tn: int) -> list[str]:
    precision, recall, f1 = prf(tp, fp, fn)
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    return (
        md_table(
            ["metric", "value"],
            [
                ["precision", f"{precision:.4f}"],
                ["recall", f"{recall:.4f}"],
                ["F1", f"{f1:.4f}"],
                ["accuracy", f"{accuracy:.4f}"],
                ["judged rows", total],
            ],
            align=["---", "---:"],
        )
        + [
            "",
            f"Confusion matrix for **{name}** (rows = human, columns = system):",
            "",
            "| | system false | system true |",
            "|---|---:|---:|",
            f"| human false | {tn} | {fp} |",
            f"| human true | {fn} | {tp} |",
        ]
    )


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------


def score_detection(cfg: dict, rows: list[dict]) -> list[tuple[str, list[str]]]:
    counts = {"flood": Counter(), "verifiable": Counter()}
    judged = {"flood": 0, "verifiable": 0}
    for row in rows:
        gold, system = row.get("gold") or {}, row.get("system") or {}
        if isinstance(gold.get("is_flood_event"), bool):
            judged["flood"] += 1
            key = ("tp" if system.get("contains_flood_event") else "fn") if gold["is_flood_event"] else ("fp" if system.get("contains_flood_event") else "tn")
            counts["flood"][key] += 1
        if isinstance(gold.get("is_verifiable"), bool):
            judged["verifiable"] += 1
            key = ("tp" if system.get("is_verifiable_flood") else "fn") if gold["is_verifiable"] else ("fp" if system.get("is_verifiable_flood") else "tn")
            counts["verifiable"][key] += 1
    require_annotations(rows, "detection", judged["flood"] + judged["verifiable"])

    sections = []
    for name, label in (("flood", "contains_flood_event"), ("verifiable", "is_verifiable_flood")):
        c = counts[name]
        if judged[name] == 0:
            sections.append((f"{label}", ["No rows judged for this decision."]))
            continue
        sections.append((f"{label} (stage-5 LLM vs human)", binary_table(label, c["tp"], c["fp"], c["fn"], c["tn"])))
    stratum = Counter(row.get("stratum") for row in rows)
    sections.append(
        (
            "Sample composition",
            md_table(["stratum", "rows"], stratum.most_common(), align=["---", "---:"])
            + [
                "",
                f"Rows: {len(rows)}; judged for is_flood_event: {judged['flood']}; for is_verifiable: {judged['verifiable']}.",
                "",
                "The sample is stratified across stage-5 outcomes, so these precision/recall figures are "
                "computed on a stratified sample, not on the corpus distribution. Read them as per-stratum "
                "behaviour; a corpus-level estimate requires reweighting by the true stratum sizes "
                "(1,555,628 total / 226,379 contains_flood / 74,447 verifiable as of the current run).",
            ],
        )
    )
    return sections


# --------------------------------------------------------------------------
# flood_type
# --------------------------------------------------------------------------


def score_flood_type(cfg: dict, rows: list[dict]) -> list[tuple[str, list[str]]]:
    pairs = [
        ((row.get("gold") or {}).get("flood_type"), (row.get("system") or {}).get("flood_type"))
        for row in rows
        if (row.get("gold") or {}).get("flood_type")
    ]
    require_annotations(rows, "flood_type", len(pairs))
    gold = [g for g, _ in pairs]
    pred = [p or "Unknown" for _, p in pairs]

    from sklearn.metrics import classification_report, confusion_matrix

    labels = sorted(set(gold) | set(pred))
    report = classification_report(gold, pred, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(gold, pred, labels=labels)

    per_class = [
        [
            label,
            f"{report[label]['precision']:.4f}",
            f"{report[label]['recall']:.4f}",
            f"{report[label]['f1-score']:.4f}",
            int(report[label]["support"]),
        ]
        for label in labels
        if label in report
    ]
    typed = [(g, p) for g, p in zip(gold, pred) if p != "Unknown"]
    typed_correct = sum(1 for g, p in typed if g == p)
    gold_typed = sum(1 for g in gold if g != "Unknown")
    missed = sum(1 for g, p in zip(gold, pred) if p == "Unknown" and g != "Unknown")

    header = ["human \\ system"] + labels
    matrix_rows = [[labels[i]] + [int(v) for v in matrix[i]] for i in range(len(labels))]
    return [
        (
            "Headline",
            md_table(
                ["metric", "value"],
                [
                    ["macro F1", f"{report['macro avg']['f1-score']:.4f}"],
                    ["weighted F1", f"{report['weighted avg']['f1-score']:.4f}"],
                    ["accuracy (all rows, Unknown counted as a class)", f"{report['accuracy']:.4f}"],
                    ["judged rows", len(pairs)],
                ],
                align=["---", "---:"],
            ),
        ),
        ("Per class", md_table(["class", "precision", "recall", "F1", "support"], per_class, align=["---", "---:", "---:", "---:", "---:"])),
        (
            "Abstention",
            md_table(
                ["", "value"],
                [
                    ["rows the system typed", len(typed)],
                    ["of those, correct", f"{typed_correct} ({100 * typed_correct / max(len(typed), 1):.1f}%)"],
                    ["coverage (typed / all judged)", f"{100 * len(typed) / max(len(pairs), 1):.1f}%"],
                    ["rows a human could type but the system left Unknown", missed],
                    ["recall of typeable rows", f"{100 * (gold_typed - missed) / max(gold_typed, 1):.1f}%"],
                ],
                align=["---", "---:"],
            )
            + [
                "",
                "Accuracy on typed rows is the number that matters for downstream use; coverage is what it "
                "cost. Raising `flood_type.min_score` moves the first up and the second down.",
            ],
        ),
        ("Confusion matrix", md_table(header, matrix_rows, align=["---"] + ["---:"] * len(labels))),
    ]


# --------------------------------------------------------------------------
# ner
# --------------------------------------------------------------------------


def score_ner(cfg: dict, rows: list[dict]) -> list[tuple[str, list[str]]]:
    reviewed = [r for r in rows if (r.get("gold") or {}).get("reviewed")]
    require_annotations(rows, "ner", len(reviewed))

    strict = Counter()
    relaxed = Counter()
    per_label_strict: dict[str, Counter] = defaultdict(Counter)
    for row in reviewed:
        gold = [(int(e["start"]), int(e["end"]), e["label"]) for e in row.get("gold_entities", [])]
        pred = [
            (int(e["start"]), int(e["end"]), e["label"])
            for e in row.get("system_entities", [])
            if int(e.get("start", -1)) >= 0
        ]
        gold_set, pred_set = set(gold), set(pred)
        for span in pred_set & gold_set:
            strict["tp"] += 1
            per_label_strict[span[2]]["tp"] += 1
        for span in pred_set - gold_set:
            strict["fp"] += 1
            per_label_strict[span[2]]["fp"] += 1
        for span in gold_set - pred_set:
            strict["fn"] += 1
            per_label_strict[span[2]]["fn"] += 1

        unmatched = list(gold)
        for p_start, p_end, p_label in pred:
            hit = next(
                (g for g in unmatched if g[2] == p_label and p_start < g[1] and g[0] < p_end),
                None,
            )
            if hit:
                unmatched.remove(hit)
                relaxed["tp"] += 1
            else:
                relaxed["fp"] += 1
        relaxed["fn"] += len(unmatched)

    sp, sr, sf = prf(strict["tp"], strict["fp"], strict["fn"])
    rp, rr, rf = prf(relaxed["tp"], relaxed["fp"], relaxed["fn"])
    per_label_rows = []
    for label, c in sorted(per_label_strict.items(), key=lambda kv: -(kv[1]["tp"] + kv[1]["fn"])):
        p, r, f = prf(c["tp"], c["fp"], c["fn"])
        per_label_rows.append([label, c["tp"] + c["fn"], f"{p:.4f}", f"{r:.4f}", f"{f:.4f}"])

    return [
        (
            "Entity-level scores",
            md_table(
                ["matching", "precision", "recall", "F1", "TP", "FP", "FN"],
                [
                    ["strict (exact span + label)", f"{sp:.4f}", f"{sr:.4f}", f"{sf:.4f}", strict["tp"], strict["fp"], strict["fn"]],
                    ["relaxed (overlap + label)", f"{rp:.4f}", f"{rr:.4f}", f"{rf:.4f}", relaxed["tp"], relaxed["fp"], relaxed["fn"]],
                ],
                align=["---", "---:", "---:", "---:", "---:", "---:", "---:"],
            )
            + [
                "",
                f"Reviewed documents: {len(reviewed)} of {len(rows)} sampled.",
                "",
                "Strict is the standard CoNLL criterion. The gap to relaxed is boundary error - the tagger "
                "found the entity but disagreed about where it ends, which for the rule-generated labels is "
                "usually the phrase length.",
            ],
        ),
        (
            "Per label (strict)",
            md_table(["label", "gold spans", "precision", "recall", "F1"], per_label_rows, align=["---", "---:", "---:", "---:", "---:"])
            + [
                "",
                "PERSON / ORGANIZATION / LOCATION / MISC come from the pretrained CoNLL-2003 model; RIVER, "
                "FLOOD_TYPE, WEATHER_EVENT, INFRASTRUCTURE, CASUALTY, EVACUATION, DAMAGE and the regex DATE / "
                "NUMBER come from the domain layer. They are different systems and should be read separately.",
            ],
        ),
    ]


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

LIST_FIELDS = ["event_dates", "locations", "rivers", "causes"]
SCALAR_FIELDS = ["country", "flood_type", "deaths", "injured", "affected_people", "displaced", "evacuated"]


def _norm_list(values) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {normalize_text(v) for v in values if str(v).strip()}


def score_extraction(cfg: dict, rows: list[dict]) -> list[tuple[str, list[str]]]:
    judged_rows = [r for r in rows if any(v is not None for k, v in (r.get("gold") or {}).items() if k != "notes")]
    require_annotations(rows, "extraction", len(judged_rows))

    list_stats: dict[str, Counter] = defaultdict(Counter)
    scalar_stats: dict[str, Counter] = defaultdict(Counter)
    for row in judged_rows:
        gold, system = row.get("gold") or {}, row.get("system") or {}
        for field in LIST_FIELDS:
            if gold.get(field) is None:
                continue
            g, p = _norm_list(gold[field]), _norm_list(system.get(field))
            list_stats[field]["tp"] += len(g & p)
            list_stats[field]["fp"] += len(p - g)
            list_stats[field]["fn"] += len(g - p)
            list_stats[field]["rows"] += 1
        for field in SCALAR_FIELDS:
            if gold.get(field) is None:
                continue
            stats = scalar_stats[field]
            stats["rows"] += 1
            g, p = gold[field], system.get(field)
            if field in {"deaths", "injured", "affected_people", "displaced", "evacuated"}:
                g, p = to_number(g), to_number(p)
            else:
                g = normalize_text(g) if g is not None else None
                p = normalize_text(p) if p is not None else None
            if p is not None:
                stats["predicted"] += 1
            if g is not None:
                stats["gold_present"] += 1
            if g == p:
                stats["exact"] += 1
            if g is not None and p is not None:
                stats["both"] += 1
                if g == p:
                    stats["both_correct"] += 1

    list_rows = []
    for field, c in list_stats.items():
        p, r, f = prf(c["tp"], c["fp"], c["fn"])
        list_rows.append([field, c["rows"], c["tp"], c["fp"], c["fn"], f"{p:.4f}", f"{r:.4f}", f"{f:.4f}"])
    scalar_rows = []
    for field, c in scalar_stats.items():
        scalar_rows.append(
            [
                field,
                c["rows"],
                f"{c['exact'] / max(c['rows'], 1):.4f}",
                f"{c['both_correct'] / max(c['both'], 1):.4f}",
                f"{c['predicted'] / max(c['rows'], 1):.2f}",
                f"{c['gold_present'] / max(c['rows'], 1):.2f}",
            ]
        )
    return [
        (
            "List fields (micro P/R/F1 over normalised strings)",
            md_table(
                ["field", "rows", "TP", "FP", "FN", "precision", "recall", "F1"],
                list_rows,
                align=["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
            ),
        ),
        (
            "Scalar fields",
            md_table(
                ["field", "rows", "exact match (incl. both-null)", "accuracy when both present", "system fill rate", "gold fill rate"],
                scalar_rows,
                align=["---", "---:", "---:", "---:", "---:", "---:"],
            )
            + [
                "",
                "'exact match' counts agreeing that a value is absent as correct, which is the behaviour a "
                "disaster database wants. 'accuracy when both present' isolates value errors from "
                "omission/hallucination, which the two fill rates describe.",
                "",
                f"Judged rows: {len(judged_rows)} of {len(rows)} sampled.",
            ],
        ),
    ]


# --------------------------------------------------------------------------
# clustering
# --------------------------------------------------------------------------


def score_clustering(cfg: dict, rows: list[dict]) -> list[tuple[str, list[str]]]:
    judged = [r for r in rows if isinstance((r.get("gold") or {}).get("same_event"), bool)]
    require_annotations(rows, "clustering", len(judged))
    counts = Counter()
    for row in judged:
        gold = row["gold"]["same_event"]
        system = bool((row.get("system") or {}).get("same_event"))
        key = ("tp" if system else "fn") if gold else ("fp" if system else "tn")
        counts[key] += 1
    return [
        (
            "Pairwise coreference",
            binary_table("same real-world flood", counts["tp"], counts["fp"], counts["fn"], counts["tn"])
            + [
                "",
                f"Judged pairs: {len(judged)} of {len(rows)} sampled.",
                "",
                "The sample is half system-linked pairs and half its highest-similarity refusals, so these "
                "are pairwise scores on a deliberately hard, non-random pair sample - not the pairwise "
                "precision over all C(n,2) pairs, which would be dominated by trivially-different pairs.",
            ],
        )
    ]


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------


def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def score_retrieval(cfg: dict, rows: list[dict]) -> list[tuple[str, list[str]]]:
    judged = [r for r in rows if (r.get("gold") or {}).get("relevance") is not None]
    require_annotations(rows, "retrieval", len(judged))
    relevance: dict[tuple[str, str], int] = {
        (r["query"], r["uid"]): int(r["gold"]["relevance"]) for r in judged
    }
    queries = sorted({q for q, _ in relevance})

    from search import HybridSearch

    engine = HybridSearch(cfg, load_dense=True)
    ks = [int(k) for k in cfg["eval"].get("retrieval_k", [1, 5, 10, 20])]
    max_k = max(ks)

    runs: dict[str, dict[str, list[str]]] = {"hybrid": {}, "bm25": {}, "dense": {}}
    for query in queries:
        result = engine.retrieve(query, {}, max_k)
        runs["hybrid"][query] = [c["uid"] for c in result["all_candidates"][:max_k]]
        runs["dense"][query] = [c["uid"] for c in sorted(result["all_candidates"], key=lambda c: -c["cosine"])[:max_k]]
        runs["bm25"][query] = [uid for uid, _ in engine.bm25.search(query, max_k)]

    table_rows = []
    for run_name, run in runs.items():
        for k in ks:
            recalls, precisions, ndcgs, rrs = [], [], [], []
            for query in queries:
                judged_for_query = {uid: rel for (q, uid), rel in relevance.items() if q == query}
                total_relevant = sum(1 for rel in judged_for_query.values() if rel > 0)
                ranked = run.get(query, [])[:k]
                gains = [judged_for_query.get(uid, 0) for uid in ranked]
                hits = sum(1 for g in gains if g > 0)
                recalls.append(hits / total_relevant if total_relevant else 0.0)
                precisions.append(hits / k)
                ideal = sorted(judged_for_query.values(), reverse=True)[:k]
                ndcgs.append(dcg(gains) / dcg(ideal) if dcg(ideal) else 0.0)
                rank = next((i + 1 for i, g in enumerate(gains) if g > 0), None)
                rrs.append(1 / rank if rank else 0.0)
            table_rows.append(
                [
                    run_name,
                    k,
                    f"{sum(recalls) / len(queries):.4f}",
                    f"{sum(precisions) / len(queries):.4f}",
                    f"{sum(ndcgs) / len(queries):.4f}",
                    f"{sum(rrs) / len(queries):.4f}",
                ]
            )
    return [
        (
            "Retrieval quality over pooled judgements",
            md_table(
                ["run", "K", "Recall@K", "Precision@K", "nDCG@K", "MRR@K"],
                table_rows,
                align=["---", "---:", "---:", "---:", "---:", "---:"],
            )
            + [
                "",
                f"Queries: {len(queries)}; judged query-document pairs: {len(judged)}; "
                f"reranker: {'on' if engine.reranker.available else 'off (' + engine.reranker.reason + ')'}.",
                "",
                "Recall@K is over the POOLED judged set, not the whole corpus - the standard caveat for "
                "pooled evaluation. An unjudged document counts as non-relevant, which is why every pooled "
                "row must be annotated.",
            ],
        )
    ]


TASKS = {
    "detection": score_detection,
    "flood_type": score_flood_type,
    "ner": score_ner,
    "extraction": score_extraction,
    "clustering": score_clustering,
    "retrieval": score_retrieval,
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Score the system against human annotation")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--gold", type=Path, help="annotated JSONL (default evaluation/gold/<task>_sample.jsonl)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    gold_path = args.gold or (GOLD_DIR / f"{args.task}_sample.jsonl")
    if not gold_path.exists():
        raise SystemExit(
            f"{gold_path} not found. Create it with:\n"
            f"  python evaluation/make_annotation_sample.py --task {args.task}"
        )
    rows = list(read_jsonl(gold_path))
    log.info("%s: %d rows from %s", args.task, len(rows), gold_path.name)

    sections = TASKS[args.task](cfg, rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (REPORT_DIR / f"{args.task}_eval.md")
    write_report(
        out_path,
        f"Evaluation - {args.task}",
        sections,
        preamble=(
            f"Produced by `evaluation/score.py --task {args.task}` against `{gold_path.name}`. "
            "Every figure here is measured against human annotation; nothing is estimated."
        ),
    )
    print(out_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
