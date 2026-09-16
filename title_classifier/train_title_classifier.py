"""Distantly-supervised title classifier: is this headline about a real flood?

Why this exists. Stage 5 sends 1.55M articles through a 14B model to find the
226k that describe a flood event, at roughly 0.5 article/s/worker. Most of that
compute is spent rejecting headlines a linear model can reject: sports results,
"flood of applications", policy debates. A calibrated title classifier in front
of the LLM turns a fixed compute budget into a much larger corpus.

Supervision. The labels are stage 5's own `contains_flood_event` verdicts -
distant supervision, not ground truth. Everything this script reports is
therefore AGREEMENT WITH THE LLM, and is labelled as such. To measure real
accuracy, annotate a sample with `evaluation/make_annotation_sample.py` and score with
`evaluation/score.py`; the LLM's own error rate is then the ceiling on what agreement
means.

Model. TF-IDF (word 1-2 grams + character 3-5 grams, which carry transliterated
place names) into logistic regression. Linear on purpose: every decision can be
traced to weighted n-grams, it trains on a CPU in minutes, and it gives a
probability that can be thresholded for the recall the pipeline needs.

    python train_title_classifier.py --max-samples 300000
    python train_title_classifier.py --years 2021 2022 --target verifiable
    python train_title_classifier.py --predict "Heavy rains flood Assam villages"

Outputs (default under data/events/classifier/):

    title_classifier.joblib
    title_classifier_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "7_semantic_layer"))

from common import (  # noqa: E402  (path set above)
    DEFAULT_CONFIG,
    extraction_files,
    fmt_duration,
    load_config,
    make_uid,
    md_table,
    setup_logging,
    write_report,
)

log = logging.getLogger("classifier.title")


def load_titles(cfg: dict, years: list[str] | None, target: str, max_samples: int, seed: int):
    """Stream every stage-5 record and reservoir-sample titles with labels.

    target: "flood" -> label is contains_flood_event
            "verifiable" -> label is is_verifiable_flood
    """
    key = "contains_flood_event" if target == "flood" else "is_verifiable_flood"
    rng = random.Random(seed)
    reservoir: list[tuple[str, int]] = []
    seen_uids: set[str] = set()
    n_seen = 0
    counts = Counter()
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
                title = str(record.get("translated_title") or "").strip()
                if not title:
                    counts["no_title"] += 1
                    continue
                uid = make_uid(year, str(record.get("month", "")), str(record.get("article_id", "")))
                if uid in seen_uids:
                    counts["repeat_uid"] += 1
                    continue
                seen_uids.add(uid)
                label = int(bool((record.get("extraction") or {}).get(key)))
                counts[f"label_{label}"] += 1
                item = (title, label)
                if len(reservoir) < max_samples:
                    reservoir.append(item)
                else:
                    j = rng.randint(0, n_seen)
                    if j < max_samples:
                        reservoir[j] = item
                n_seen += 1
        log.info("  read %s: %d records so far", path.name, n_seen)
    return reservoir, n_seen, counts


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a title-level flood classifier on stage-5 labels")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--years", nargs="*")
    ap.add_argument("--target", choices=["flood", "verifiable"], default="flood")
    ap.add_argument("--max-samples", type=int, default=300000)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--out", type=Path, help="model path (default <output_root>/classifier/title_classifier.joblib)")
    ap.add_argument("--predict", help="load the trained model and score one title")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    seed = int(cfg["runtime"]["seed"])
    out_dir = Path(cfg["paths"]["output_root"]) / "classifier"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out or (out_dir / "title_classifier.joblib")

    import joblib

    if args.predict:
        if not model_path.exists():
            raise SystemExit(f"no model at {model_path}; train one first")
        pipe = joblib.load(model_path)
        proba = float(pipe.predict_proba([args.predict])[0][1])
        print(json.dumps({"title": args.predict, "p_flood": round(proba, 4), "label": int(proba >= 0.5)}))
        return 0

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        classification_report,
        confusion_matrix,
        precision_recall_curve,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import FeatureUnion, Pipeline

    started = time.time()
    data, n_seen, counts = load_titles(cfg, args.years, args.target, args.max_samples, seed)
    if not data:
        raise SystemExit("no titles found")
    texts = [t for t, _ in data]
    labels = [y for _, y in data]
    log.info("sampled %d titles of %d records; positives %d (%.2f%%)", len(data), n_seen, sum(labels), 100 * sum(labels) / len(labels))

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=args.test_size, random_state=seed, stratify=labels
    )
    pipe = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, strip_accents="unicode")),
                        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, sublinear_tf=True)),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced")),
        ]
    )
    log.info("fitting on %d titles ...", len(x_train))
    pipe.fit(x_train, y_train)
    joblib.dump(pipe, model_path)

    proba = pipe.predict_proba(x_test)[:, 1]
    predicted = (proba >= 0.5).astype(int)
    report = classification_report(y_test, predicted, digits=4, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predicted)
    roc = roc_auc_score(y_test, proba)
    ap_score = average_precision_score(y_test, proba)
    precision, recall, thresholds = precision_recall_curve(y_test, proba)

    # The operating question for a prefilter is: at what threshold do we keep
    # 99%/95%/90% of the LLM's positives, and how much of the corpus survives?
    operating_rows = []
    for want in (0.99, 0.98, 0.95, 0.90):
        best = None
        for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
            if r >= want and (best is None or t > best[2]):
                best = (p, r, t)
        if best:
            kept = float((proba >= best[2]).mean())
            operating_rows.append(
                [f"{want:.0%}", f"{best[2]:.4f}", f"{best[0]:.4f}", f"{best[1]:.4f}", f"{100 * kept:.1f}%", f"{100 * (1 - kept):.1f}%"]
            )

    # Most informative n-grams (word features only - char n-grams are unreadable)
    names = pipe.named_steps["features"].transformer_list[0][1].get_feature_names_out()
    weights = pipe.named_steps["clf"].coef_[0][: len(names)]
    order = weights.argsort()
    top_pos = [[names[i], f"{weights[i]:+.3f}"] for i in order[::-1][:20]]
    top_neg = [[names[i], f"{weights[i]:+.3f}"] for i in order[:20]]

    target_name = "contains_flood_event" if args.target == "flood" else "is_verifiable_flood"
    sections = [
        (
            "What is being measured",
            [
                f"The labels are stage 5's `{target_name}` verdicts from qwen3:14b - **distant supervision, "
                "not ground truth**. Every score below is agreement with the LLM on held-out titles. It is "
                "the right metric for the intended use (a prefilter that must not drop what the LLM would "
                "have kept) and the wrong one for 'is this article really about a flood', which needs human "
                "annotation via `eval/`.",
            ],
        ),
        (
            "Data",
            md_table(
                ["", "count"],
                [
                    ["stage-5 records streamed", n_seen],
                    ["records with no title", counts["no_title"]],
                    ["repeated uids dropped", counts["repeat_uid"]],
                    ["titles sampled", len(data)],
                    ["positives in the sample", f"{sum(labels)} ({100 * sum(labels) / len(labels):.2f}%)"],
                    ["train / test", f"{len(x_train)} / {len(x_test)}"],
                    ["years", " ".join(args.years) if args.years else "all"],
                    ["fit time", fmt_duration(time.time() - started)],
                ],
                align=["---", "---:"],
            ),
        ),
        (
            "Held-out agreement with the stage-5 LLM (threshold 0.5)",
            md_table(
                ["class", "precision", "recall", "f1", "support"],
                [
                    ["not flood (0)", f"{report['0']['precision']:.4f}", f"{report['0']['recall']:.4f}", f"{report['0']['f1-score']:.4f}", int(report["0"]["support"])],
                    ["flood (1)", f"{report['1']['precision']:.4f}", f"{report['1']['recall']:.4f}", f"{report['1']['f1-score']:.4f}", int(report["1"]["support"])],
                    ["macro avg", f"{report['macro avg']['precision']:.4f}", f"{report['macro avg']['recall']:.4f}", f"{report['macro avg']['f1-score']:.4f}", int(report["macro avg"]["support"])],
                    ["weighted avg", f"{report['weighted avg']['precision']:.4f}", f"{report['weighted avg']['recall']:.4f}", f"{report['weighted avg']['f1-score']:.4f}", int(report["weighted avg"]["support"])],
                ],
                align=["---", "---:", "---:", "---:", "---:"],
            )
            + [
                "",
                f"accuracy {report['accuracy']:.4f} | ROC AUC {roc:.4f} | average precision {ap_score:.4f}",
                "",
                "Confusion matrix (rows = LLM label, columns = classifier):",
                "",
                "| | pred 0 | pred 1 |",
                "|---|---:|---:|",
                f"| true 0 | {matrix[0][0]} | {matrix[0][1]} |",
                f"| true 1 | {matrix[1][0]} | {matrix[1][1]} |",
            ],
        ),
        (
            "Operating points for use as an LLM prefilter",
            md_table(
                ["recall target", "threshold", "precision", "recall", "corpus kept", "LLM calls saved"],
                operating_rows,
                align=["---", "---:", "---:", "---:", "---:", "---:"],
            )
            + [
                "",
                "'corpus kept' is the share of held-out titles scoring above the threshold - i.e. how much "
                "of the crawl would still reach the 14B model. The saving is real compute; the cost is the "
                "recall you gave up, which is stated in the same row.",
            ],
        ),
        (
            "Most informative word n-grams",
            md_table(["towards flood", "weight", "towards not-flood", "weight"],
                     [[p[0], p[1], n[0], n[1]] for p, n in zip(top_pos, top_neg)],
                     align=["---", "---:", "---", "---:"]),
        ),
    ]
    path = write_report(
        out_dir / "title_classifier_report.md",
        "Title classifier (distant supervision from stage 5)",
        sections,
        preamble=f"Produced by `title_classifier/train_title_classifier.py`. Model saved to `{model_path}`.",
    )
    log.info(
        "macro F1 %.4f, ROC AUC %.4f -> %s (%s)",
        report["macro avg"]["f1-score"], roc, path.name, fmt_duration(time.time() - started),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
