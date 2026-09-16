"""Stage 7c - flood-type classification.

A flood is not one phenomenon. A river topping its embankment, a cloudburst
emptying a hillside in ten minutes, a storm surge pushing seawater inland and a
dam releasing on schedule all produce "flooding" in a headline and require
completely different reading. Stage 5 does not label the type, so this does.

Three classifiers over the same text, in decreasing order of how much you should
trust them without a gold set:

  rules   A transparent weighted-lexicon model. Every cue phrase and its weight
          lives in config.flood_type.lexicon, so a wrong call is inspectable and
          fixable. Abstains (returns Unknown) unless the winning score clears
          `min_score` and beats the runner-up by `margin`.
  llm     Zero-shot through the local Qwen3-14B. Reads the same text, must
          return one of the labels plus a verbatim quote from the text. No
          probability is available, so `confidence` is null - it is not
          reported as a number the model did not produce.
  model   An optional supervised TF-IDF + logistic-regression classifier. Only
          usable once a labelled set exists; `--train` builds it from an
          annotated JSONL and refuses to run on weak labels.

`--method agree` runs the rules first and only asks the LLM about the ones the
rules abstained on, which is where the cost is worth paying.

    python flood_type.py --sample 2000
    python flood_type.py --years 2021
    python flood_type.py --method agree
    python flood_type.py --train ../evaluation/gold/flood_type_sample.jsonl

Output under paths.output_root:

    flood_type.jsonl        uid, flood_type, confidence, evidence, method, scores
    flood_type_report.md    measured label distribution, abstention rate, cues

build_index.py folds this file into the vector metadata (run it again with
--refresh-metadata afterwards), and cluster_events.py uses type agreement as one
of its coreference signals.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

from common import (
    DEFAULT_CONFIG,
    FLOOD_TYPE_UNKNOWN,
    Event,
    JsonlAppender,
    LLMUnavailable,
    fmt_duration,
    iter_bodies,
    iter_events,
    load_config,
    md_table,
    ollama_available,
    ollama_client,
    ollama_json,
    output_root,
    read_jsonl,
    seed_everything,
    setup_logging,
    write_report,
)

log = logging.getLogger("events.flood_type")


# --------------------------------------------------------------------------
# Text the classifiers read
# --------------------------------------------------------------------------


def classification_text(cfg: dict, ev: Event, body: str) -> str:
    """Title and summary first - they state the type when it is stated at all -
    then the extracted cause slots, then a bounded head of the body."""
    ft = cfg["flood_type"]
    parts = [ev.title, ev.summary]
    parts += ev.causes()
    if ft.get("use_body", True) and body:
        parts.append(body[: int(ft.get("max_body_chars", 1500))])
    return "\n".join(p for p in parts if p and str(p).strip())


# --------------------------------------------------------------------------
# Rule classifier
# --------------------------------------------------------------------------


class RuleClassifier:
    """Weighted lexicon. Each distinct cue phrase contributes its weight once,
    however many times it occurs: one article repeating "flash flood" nine times
    is one piece of evidence, not nine."""

    def __init__(self, cfg: dict):
        ft = cfg["flood_type"]
        self.min_score = float(ft.get("min_score", 1.0))
        self.margin = float(ft.get("margin", 0.5))
        self.patterns: dict[str, list[tuple[str, float, re.Pattern]]] = {}
        for label, cues in ft["lexicon"].items():
            self.patterns[label] = [
                (phrase, float(weight), re.compile(r"\b" + re.escape(phrase.lower()) + r"\b"))
                for phrase, weight in cues.items()
            ]

    def classify(self, text: str) -> dict:
        lowered = (text or "").lower()
        scores: dict[str, float] = {}
        hits: dict[str, list[tuple[str, float, int]]] = {}
        for label, cues in self.patterns.items():
            total = 0.0
            matched: list[tuple[str, float, int]] = []
            for phrase, weight, pattern in cues:
                m = pattern.search(lowered)
                if m:
                    total += weight
                    matched.append((phrase, weight, m.start()))
            if total:
                scores[label] = round(total, 3)
                hits[label] = matched

        if not scores:
            return {"flood_type": FLOOD_TYPE_UNKNOWN, "confidence": 0.0, "evidence": "", "scores": {}, "reason": "no cue matched"}

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best_label, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        if best < self.min_score:
            return {"flood_type": FLOOD_TYPE_UNKNOWN, "confidence": 0.0, "evidence": "", "scores": scores, "reason": f"top score {best} below min_score {self.min_score}"}
        if best - second < self.margin:
            return {"flood_type": FLOOD_TYPE_UNKNOWN, "confidence": 0.0, "evidence": "", "scores": scores, "reason": f"margin {best - second:.2f} below {self.margin}"}

        phrase, _weight, pos = max(hits[best_label], key=lambda h: h[1])
        window = text[max(0, pos - 60) : pos + len(phrase) + 60].replace("\n", " ").strip()
        return {
            "flood_type": best_label,
            # Separability of the top two classes, NOT a calibrated probability.
            "confidence": round((best - second) / (best + second), 4) if (best + second) else 0.0,
            "evidence": window,
            "cue": phrase,
            "scores": scores,
            "reason": "",
        }


# --------------------------------------------------------------------------
# LLM classifier
# --------------------------------------------------------------------------

LLM_SYSTEM = (
    "You classify the type of a flood described in a news article.\n"
    "Choose exactly one label from the list you are given, based only on what the text says.\n"
    "If the text does not say which kind of flood it was, answer Unknown. Do not guess.\n"
    'Return only JSON: {"flood_type": "<label>", "quote": "<verbatim sentence from the text that '
    'justifies it, or empty string>"}'
)


def llm_classify(cfg: dict, text: str, client=None) -> dict:
    labels = cfg["flood_type"]["labels"]
    max_chars = int(cfg["flood_type"]["llm"].get("max_chars", 4000))
    user = "Labels: " + ", ".join(labels) + "\n\nArticle:\n" + text[:max_chars]
    obj = ollama_json(cfg, LLM_SYSTEM, user, client=client)
    label = str(obj.get("flood_type") or FLOOD_TYPE_UNKNOWN).strip()
    if label not in labels:
        label = FLOOD_TYPE_UNKNOWN
    quote = str(obj.get("quote") or "").strip()
    # An unquoted or hallucinated justification is not evidence.
    grounded = bool(quote) and quote.lower() in text.lower()
    return {
        "flood_type": label,
        "confidence": None,  # the model reports no probability; none is invented here
        "evidence": quote if grounded else "",
        "evidence_grounded": grounded,
        "scores": {},
        "reason": "" if grounded or not quote else "quote not found in source text",
    }


# --------------------------------------------------------------------------
# Optional supervised classifier
# --------------------------------------------------------------------------


def train_supervised(cfg: dict, gold_path: Path, out_path: Path) -> dict:
    """TF-IDF + logistic regression over an ANNOTATED set.

    gold_path is JSONL with {"uid": ..., "flood_type": ...} and optionally
    {"text": ...}; missing text is rebuilt from the corpus. Refuses to train on
    anything whose labels came from the rule model - a classifier fitted on its
    own weak labels measures nothing.
    """
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    rows = [r for r in read_jsonl(gold_path) if r.get("flood_type")]
    if not rows:
        raise SystemExit(f"no labelled rows in {gold_path}")
    weak = [r for r in rows if r.get("label_source") in {"rules", "llm", "weak"}]
    if weak:
        raise SystemExit(
            f"{len(weak)}/{len(rows)} rows are machine-labelled (label_source in rules/llm/weak). "
            "Train on human annotation only."
        )
    texts, labels = [], []
    by_uid = {r["uid"]: r for r in rows if "uid" in r}
    missing = 0
    for uid, row in by_uid.items():
        text = row.get("text")
        if not text:
            missing += 1
            continue
        texts.append(text)
        labels.append(row["flood_type"])
    if missing:
        log.warning("%d gold rows carry no text and were skipped", missing)
    if len(set(labels)) < 2:
        raise SystemExit("need at least two distinct labels to train")

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=cfg["runtime"]["seed"], stratify=labels if min(Counter(labels).values()) > 1 else None
    )
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    pipe.fit(x_train, y_train)
    report = classification_report(y_test, pipe.predict(x_test), output_dict=True, zero_division=0)
    joblib.dump(pipe, out_path)
    log.info("trained on %d, held out %d, saved -> %s", len(x_train), len(x_test), out_path)
    return {"n_train": len(x_train), "n_test": len(x_test), "report": report}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def reservoir(stream, k: int, rng: random.Random) -> list:
    out: list = []
    for i, item in enumerate(stream):
        if i < k:
            out.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                out[j] = item
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Classify the hydrological type of each extracted flood")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--years", nargs="*")
    ap.add_argument("--select", choices=["verifiable", "flood", "all"], default="verifiable")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sample", type=int)
    ap.add_argument("--method", choices=["rules", "llm", "agree", "model"])
    ap.add_argument("--model-path", type=Path, help="supervised classifier to load (method=model)")
    ap.add_argument("--train", type=Path, help="train the supervised classifier from this annotated JSONL and exit")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    seed = seed_everything(cfg)
    root = output_root(cfg)

    if args.train:
        out = args.model_path or (root / "flood_type_clf.joblib")
        result = train_supervised(cfg, args.train, out)
        macro = result["report"].get("macro avg", {})
        log.info("held-out macro F1 %.3f on %d examples", macro.get("f1-score", 0.0), result["n_test"])
        return 0

    method = args.method or cfg["flood_type"].get("method", "rules")
    out_path = root / "flood_type.jsonl"
    appender = JsonlAppender(out_path)
    done = set() if args.force else appender.done_ids()
    if args.force and out_path.exists():
        out_path.unlink()
    log.info("method=%s already classified=%d", method, len(done))

    rules = RuleClassifier(cfg)
    supervised = None
    if method == "model":
        import joblib

        path = args.model_path or (root / "flood_type_clf.joblib")
        if not Path(path).exists():
            raise SystemExit(
                f"no supervised classifier at {path}. Train one first: "
                f"python flood_type.py --train <annotated.jsonl>"
            )
        supervised = joblib.load(path)

    client = None
    if method in {"llm", "agree"}:
        ok, message = ollama_available(cfg)
        if not ok:
            raise SystemExit(f"{message}. Start the server (`ollama serve`) or use --method rules.")
        log.info(message)
        client = ollama_client(cfg)

    stream = iter_events(cfg, args.years, select=args.select, limit=args.limit, skip_uids=done)
    if args.sample:
        events = reservoir(stream, args.sample, random.Random(seed))
        log.info("sampled %d extractions (seed=%d)", len(events), seed)
    else:
        events = stream

    label_counts: Counter = Counter()
    method_counts: Counter = Counter()
    cue_counts: Counter = Counter()
    abstain_reasons: Counter = Counter()
    stats = Counter()
    started = time.time()

    with appender:
        for ev, body in iter_bodies(cfg, events, workers=int(cfg["grounding"].get("io_workers", 16))):
            if not body:
                stats["body_missing"] += 1
            text = classification_text(cfg, ev, body)
            if not text.strip():
                stats["empty"] += 1
                continue

            used = method
            if method == "rules":
                result = rules.classify(text)
            elif method == "llm":
                result = llm_classify(cfg, text, client=client)
            elif method == "agree":
                result = rules.classify(text)
                used = "rules"
                if result["flood_type"] == FLOOD_TYPE_UNKNOWN:
                    try:
                        result = llm_classify(cfg, text, client=client)
                        used = "llm"
                    except LLMUnavailable as exc:
                        log.error("LLM went away mid-run: %s", exc)
                        raise
            else:  # model
                label = str(supervised.predict([text])[0])
                proba = max(supervised.predict_proba([text])[0])
                result = {"flood_type": label, "confidence": round(float(proba), 4), "evidence": "", "scores": {}, "reason": ""}

            row = {
                "uid": ev.uid,
                "year": int(ev.year),
                "flood_type": result["flood_type"],
                "confidence": result["confidence"],
                "evidence": result.get("evidence", ""),
                "cue": result.get("cue", ""),
                "method": used,
                "scores": result.get("scores", {}),
            }
            if result.get("reason"):
                row["abstain_reason"] = result["reason"]
                abstain_reasons[result["reason"].split(" below")[0]] += 1
            appender.write(row)
            stats["classified"] += 1
            label_counts[row["flood_type"]] += 1
            method_counts[used] += 1
            if row["cue"]:
                cue_counts[row["cue"]] += 1
            if stats["classified"] % 5000 == 0:
                log.info("  %d classified (%.0f/s)", stats["classified"], stats["classified"] / max(time.time() - started, 1e-6))

    seconds = time.time() - started
    # Report over the whole file, not just this run: a resumed run would
    # otherwise describe a fragment of the corpus as if it were the corpus.
    label_counts = Counter()
    method_counts = Counter()
    cue_counts = Counter()
    abstain_reasons = Counter()
    n_total = 0
    for row in read_jsonl(out_path):
        n_total += 1
        label_counts[row.get("flood_type") or FLOOD_TYPE_UNKNOWN] += 1
        method_counts[row.get("method") or "?"] += 1
        if row.get("cue"):
            cue_counts[row["cue"]] += 1
        if row.get("abstain_reason"):
            abstain_reasons[str(row["abstain_reason"]).split(" below")[0]] += 1
    total = max(n_total, 1)
    known = total - label_counts[FLOOD_TYPE_UNKNOWN]
    sections = [
        (
            "Run",
            md_table(
                ["", "value"],
                [
                    ["method", method],
                    ["select", args.select],
                    ["years", " ".join(args.years) if args.years else "all"],
                    ["sample", args.sample or "-"],
                    ["already classified before this run", len(done)],
                    ["classified this run", stats["classified"]],
                    ["rows in flood_type.jsonl (what the tables below describe)", n_total],
                    ["bodies missing on disk", stats["body_missing"]],
                    ["skipped: no text", stats["empty"]],
                    ["min_score / margin", f"{cfg['flood_type']['min_score']} / {cfg['flood_type']['margin']}"],
                    ["wall clock", fmt_duration(seconds)],
                    ["throughput", f"{stats['classified'] / max(seconds, 1e-6):.0f} doc/s"],
                ],
                align=["---", "---:"],
            ),
        ),
        (
            "Label distribution",
            md_table(
                ["flood_type", "count", "share"],
                [[label, n, f"{100 * n / total:.1f}%"] for label, n in label_counts.most_common()],
                align=["---", "---:", "---:"],
            )
            + [
                "",
                f"Typed: {known}/{total} ({100 * known / total:.1f}%). "
                "`Unknown` is an abstention, not a class: the text carried no cue strong enough to "
                "separate the top two candidates. It is deliberately large - guessing a type to fill "
                "the column would corrupt every downstream count.",
            ],
        ),
        (
            "Why the classifier abstained",
            md_table(["reason", "documents"], abstain_reasons.most_common(), align=["---", "---:"])
            or ["(no abstentions)"],
        ),
        (
            "Most frequent decisive cues",
            md_table(["cue phrase", "documents it decided"], cue_counts.most_common(25), align=["---", "---:"]),
        ),
        (
            "Which classifier answered",
            md_table(["method", "documents"], method_counts.most_common(), align=["---", "---:"]),
        ),
    ]
    path = write_report(
        root / "flood_type_report.md",
        "Flood-type classification report",
        sections,
        preamble=(
            "Produced by `7_semantic_layer/flood_type.py`. The rule model is a weighted lexicon defined in "
            "`config.json`; its `confidence` is the separability of the top two classes, not a "
            "calibrated probability. Accuracy against human labels is only measurable through "
            "`evaluation/score.py` on an annotated sample."
        ),
    )
    log.info("wrote %d rows to %s, report -> %s (%s)", stats["classified"], out_path.name, path.name, fmt_duration(seconds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
