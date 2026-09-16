"""Stage 7d - event coreference: collapse many articles into one real flood.

Stage 5 produces one extraction per article. Fifty outlets covering the same
Assam flood therefore yield fifty "events". This script decides which
extractions are the same real-world flood and merges them into one record.

The decision is a weighted score over five agreement signals plus two hard
gates, all configurable and all ablatable by setting a weight to zero:

    score = w_semantic  * cosine(e_i, e_j)          dense embedding agreement
          + w_date      * date proximity            1 at same day, decaying to 0
          + w_location  * location overlap          exact or token-Jaccard
          + w_river     * watercourse overlap       normalised river names
          + w_flood_type* type agreement            River/Flash/Urban/...
          + w_country   * country agreement

    hard gates (checked first, veto regardless of score):
      - date gap beyond clustering.date_window_days
      - two different countries, when both were extracted
      - a merge that would stretch the cluster past max_cluster_span_days

Missing evidence never scores as agreement OR as disagreement: an absent field
contributes `unknown_field_score`, so two articles that simply do not mention a
river are not rewarded for it. Candidates come from exact top-k cosine kNN over
the exported embedding matrix (FAISS if installed, otherwise a blocked matmul in
torch), and accepted links are joined with union-find.

Merging keeps disagreement rather than averaging it: a death toll that climbs
from 12 to 31 across reports is stored as min/max/reports/conflict, and the
scalar `deaths` is the maximum. See event_schema.py.

    python cluster_events.py
    python cluster_events.py --threshold 0.85 --window 2
    python cluster_events.py --years 2023
    python cluster_events.py --ablate river flood_type

Outputs under paths.output_root:

    consolidated_events.jsonl / .csv    one record per real-world flood event
    clusters.jsonl                      event_id -> [uid, ...]
    clustering_report.md                measured counts, signal statistics
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from common import (
    DEFAULT_CONFIG,
    NUMERIC_SLOTS,
    fmt_duration,
    load_config,
    location_similarity,
    md_table,
    normalize_text,
    output_root,
    parse_iso,
    read_jsonl,
    river_similarity,
    setup_logging,
    write_jsonl,
    write_report,
)
from event_schema import CSV_COLUMNS, build_event, to_csv_row

log = logging.getLogger("events.cluster")


# --------------------------------------------------------------------------
# k-NN
# --------------------------------------------------------------------------


def knn(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Exact top-k inner-product neighbours (vectors are unit-norm, so this is
    cosine). Returns (scores, indices, backend), each array (n, k), self
    excluded."""
    n = matrix.shape[0]
    k = min(k, max(n - 1, 1))
    try:
        import faiss

        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        scores, idx = index.search(matrix, k + 1)
        backend = "faiss (CPU, exact)"
    except Exception:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        mat = torch.from_numpy(matrix).to(device)
        scores = np.zeros((n, k + 1), dtype=np.float32)
        idx = np.zeros((n, k + 1), dtype=np.int64)
        block = 4096
        for start in range(0, n, block):
            sims = mat[start : start + block] @ mat.T
            top = torch.topk(sims, min(k + 1, n), dim=1)
            scores[start : start + block, : top.values.shape[1]] = top.values.cpu().numpy()
            idx[start : start + block, : top.indices.shape[1]] = top.indices.cpu().numpy()
        backend = f"torch/{device} (exact, blocked)"
    out_scores = np.zeros((n, k), dtype=np.float32)
    out_idx = np.full((n, k), -1, dtype=np.int64)
    for i in range(n):
        keep = [(s, j) for s, j in zip(scores[i], idx[i]) if j != i][:k]
        for col, (s, j) in enumerate(keep):
            out_scores[i, col], out_idx[i, col] = s, j
    log.info("kNN backend=%s n=%d k=%d", backend, n, k)
    return out_scores, out_idx, backend


# --------------------------------------------------------------------------
# Union-find with per-component date spans
# --------------------------------------------------------------------------


class UnionFind:
    """Union-find that also tracks each component's [min, max] flood date, so a
    merge that would stretch an event past max_cluster_span_days can be
    refused. Without it, transitive chaining walks a monsoon season into one
    12,000-article 'event'."""

    def __init__(self, n: int, spans: list[tuple[date | None, date | None]]):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.lo = [s[0] for s in spans]
        self.hi = [s[1] for s in spans]

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def span_days(self, ra: int, rb: int) -> int | None:
        los = [d for d in (self.lo[ra], self.lo[rb]) if d]
        his = [d for d in (self.hi[ra], self.hi[rb]) if d]
        if not los or not his:
            return None
        return (max(his) - min(los)).days

    def union(self, a: int, b: int, max_span_days: int | None = None) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if max_span_days is not None:
            span = self.span_days(ra, rb)
            if span is not None and span > max_span_days:
                return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        los = [d for d in (self.lo[ra], self.lo[rb]) if d]
        his = [d for d in (self.hi[ra], self.hi[rb]) if d]
        self.lo[ra] = min(los) if los else None
        self.hi[ra] = max(his) if his else None
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------


def _dates(meta: dict) -> tuple[date | None, date | None]:
    return parse_iso(meta.get("date_min")), parse_iso(meta.get("date_max"))


def date_gap_days(a: dict, b: dict) -> int | None:
    """Gap in days between two closed date intervals; 0 when they overlap,
    None when either side has no usable date."""
    a0, a1 = _dates(a)
    b0, b1 = _dates(b)
    if not (a0 and a1 and b0 and b1):
        return None
    if a1 < b0:
        return (b0 - a1).days
    if b1 < a0:
        return (a0 - b1).days
    return 0


def _split_semi(value) -> list[str]:
    if not isinstance(value, str):
        return []
    return [x.strip() for x in value.split(";") if x.strip()]


def pair_signals(a: dict, b: dict, sim: float, cl: dict) -> dict[str, float]:
    """Each signal in [0, 1]. Unknown on either side yields
    clustering.unknown_field_score, which is neither agreement nor conflict."""
    unknown = float(cl.get("unknown_field_score", 0.5))

    gap = date_gap_days(a, b)
    if gap is None:
        date_score = float(cl.get("unknown_date_score", 0.35))
    else:
        decay = max(float(cl.get("date_decay_days", 7)), 1e-6)
        date_score = max(0.0, 1.0 - gap / decay)

    loc_a, loc_b = _split_semi(a.get("locations")), _split_semi(b.get("locations"))
    loc_score = location_similarity(loc_a, loc_b) if (loc_a and loc_b) else unknown

    riv_a, riv_b = _split_semi(a.get("rivers")), _split_semi(b.get("rivers"))
    riv_score = river_similarity(riv_a, riv_b) if (riv_a and riv_b) else unknown

    ta, tb = a.get("flood_type") or "Unknown", b.get("flood_type") or "Unknown"
    if ta == "Unknown" or tb == "Unknown":
        type_score = unknown
    else:
        type_score = 1.0 if ta == tb else 0.0

    ca, cb = normalize_text(a.get("country") or ""), normalize_text(b.get("country") or "")
    if not ca or not cb:
        country_score = unknown
    else:
        country_score = 1.0 if ca == cb else 0.0

    return {
        "semantic": float(min(max(sim, 0.0), 1.0)),
        "date": date_score,
        "location": loc_score,
        "river": riv_score,
        "flood_type": type_score,
        "country": country_score,
    }


def hard_veto(a: dict, b: dict, cl: dict) -> str | None:
    """Returns the name of the gate that refused the pair, or None."""
    if cl.get("hard_date_gate", True):
        gap = date_gap_days(a, b)
        if gap is not None and gap > int(cl["date_window_days"]):
            return "date_window"
    if cl.get("require_country_agreement", True):
        ca, cb = normalize_text(a.get("country") or ""), normalize_text(b.get("country") or "")
        if ca and cb and ca != cb:
            return "country_disagreement"
    return None


def score_pair(signals: dict[str, float], weights: dict[str, float]) -> float:
    return sum(weights.get(name, 0.0) * value for name, value in signals.items())


# --------------------------------------------------------------------------
# Merging
# --------------------------------------------------------------------------


def medoid(members: list[int], matrix: np.ndarray, rng: np.random.Generator) -> int:
    """The member with the highest mean similarity to the rest: the article
    that reads most like the consensus account of the event."""
    if len(members) == 1:
        return members[0]
    sub = matrix[members]
    if len(members) > 2000:  # cap the all-pairs cost on pathological clusters
        sample = rng.choice(len(members), 2000, replace=False)
        sims = sub @ sub[sample].T
    else:
        sims = sub @ sub.T
    return members[int(np.argmax(sims.mean(axis=1)))]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Cluster article-level extractions into real-world flood events")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--years", nargs="*", help="restrict to these years")
    ap.add_argument("--threshold", type=float, help="override clustering.link_threshold")
    ap.add_argument("--window", type=int, help="override clustering.date_window_days")
    ap.add_argument("--top-k", type=int, help="override clustering.top_k")
    ap.add_argument("--ablate", nargs="*", default=[], help="zero these signal weights (semantic date location river flood_type country)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    cl = cfg["clustering"]
    if args.threshold is not None:
        cl["link_threshold"] = args.threshold
    if args.window is not None:
        cl["date_window_days"] = args.window
    if args.top_k is not None:
        cl["top_k"] = args.top_k
    weights = dict(cl["weights"])
    for name in args.ablate:
        if name not in weights:
            raise SystemExit(f"unknown signal {name!r}; known: {', '.join(weights)}")
        weights[name] = 0.0
        log.warning("ablation: weight[%s] = 0", name)

    root = output_root(cfg)
    matrix_path = root / "embeddings.npy"
    if not matrix_path.exists():
        raise SystemExit(f"{matrix_path} not found - run build_index.py first")
    matrix = np.load(matrix_path)
    ids = json.loads((root / "ids.json").read_text(encoding="utf-8"))
    meta_by_uid = {row["uid"]: row for row in read_jsonl(root / "events_meta.jsonl")}
    meta = [meta_by_uid[uid] for uid in ids]

    if args.years:
        keep = [i for i, m in enumerate(meta) if str(m["year"]) in set(args.years)]
        matrix, meta = matrix[keep], [meta[i] for i in keep]
    n = len(meta)
    if n == 0:
        raise SystemExit("nothing to cluster")
    log.info("%d extractions, dim %d", n, matrix.shape[1])

    started = time.time()
    sims, nbrs, backend = knn(matrix, int(cl["top_k"]))
    log.info("kNN done in %s", fmt_duration(time.time() - started))

    spans = [_dates(m) for m in meta]
    uf = UnionFind(n, spans)
    min_cos = float(cl.get("candidate_min_cosine", 0.0))
    threshold = float(cl["link_threshold"])
    max_span = cl.get("max_cluster_span_days")

    accepted_signals: list[dict[str, float]] = []
    rejected_scores: list[float] = []
    veto_counts: Counter = Counter()
    stats = Counter()
    for i in range(n):
        for sim, j in zip(sims[i], nbrs[i]):
            j = int(j)
            if j < 0 or j <= i:  # each unordered pair once
                continue
            stats["candidate_pairs"] += 1
            if sim < min_cos:
                stats["below_candidate_cosine"] += 1
                continue
            veto = hard_veto(meta[i], meta[j], cl)
            if veto:
                veto_counts[veto] += 1
                continue
            signals = pair_signals(meta[i], meta[j], float(sim), cl)
            score = score_pair(signals, weights)
            if score < threshold:
                rejected_scores.append(score)
                continue
            if uf.union(i, j, max_span_days=max_span):
                stats["links"] += 1
                accepted_signals.append({**signals, "score": score})
            else:
                veto_counts["max_cluster_span"] += 1

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    log.info("%d links -> %d events", stats["links"], len(groups))

    rng = np.random.default_rng(int(cfg["runtime"]["seed"]))
    prefix = cfg["event"].get("id_prefix", "E")
    bands = cfg["event"].get("severity_bands")
    events = []
    clusters_out = []
    for cid, members in enumerate(sorted(groups.values(), key=lambda g: (-len(g), g[0]))):
        rows = [meta[i] for i in members]
        rep = meta[medoid(members, matrix, rng)]
        first_year = min((int(r["year"]) for r in rows), default=0)
        event_id = f"{prefix}-{first_year}-{cid:06d}"
        event = build_event(event_id, rows, rep, severity_bands=bands)
        problems = event.validate()
        if problems:
            stats["invalid_events"] += 1
            log.warning("%s: %s", event_id, "; ".join(problems))
        events.append(event)
        clusters_out.append({"event_id": event_id, "n_articles": len(members), "members": event.source_uids})

    write_jsonl(root / "consolidated_events.jsonl", (e.to_dict() for e in events))
    write_jsonl(root / "clusters.jsonl", clusters_out)
    csv_path = root / "consolidated_events.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for event in events:
            writer.writerow(to_csv_row(event))

    write_clustering_report(
        root / "clustering_report.md",
        cfg, weights, meta, events, sims, stats, veto_counts,
        accepted_signals, rejected_scores, backend, time.time() - started,
    )
    log.info(
        "wrote consolidated_events.{jsonl,csv}, clusters.jsonl, clustering_report.md in %s",
        fmt_duration(time.time() - started),
    )
    return 0


def write_clustering_report(
    path: Path,
    cfg: dict,
    weights: dict,
    meta: list[dict],
    events: list,
    sims: np.ndarray,
    stats: Counter,
    veto_counts: Counter,
    accepted_signals: list[dict],
    rejected_scores: list[float],
    backend: str,
    seconds: float,
) -> Path:
    cl = cfg["clustering"]
    n = len(meta)
    sizes = Counter(e.n_articles for e in events)
    singles = sizes.get(1, 0)
    hist = np.histogram(sims[:, 0], bins=[0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0001])[0]

    sections: list[tuple[str, list[str]]] = []
    sections.append(
        (
            "Scoring function",
            md_table(
                ["signal", "weight", "what it measures"],
                [
                    ["semantic", weights["semantic"], "cosine between the two event embeddings"],
                    ["date", weights["date"], f"1 at same-day, linear decay to 0 over {cl['date_decay_days']} days"],
                    ["location", weights["location"], "best-pair location match: exact after normalisation, else token Jaccard"],
                    ["river", weights["river"], "shared watercourse after stripping 'river'/'nadi'/... suffixes"],
                    ["flood_type", weights["flood_type"], "same flood-type label (Unknown on either side scores as unknown)"],
                    ["country", weights["country"], "same country"],
                ],
                align=["---", "---:", "---"],
            )
            + [
                "",
                f"- link threshold **{cl['link_threshold']}**, candidate floor cosine {cl.get('candidate_min_cosine')}, top-k {cl['top_k']}",
                f"- unknown field scores {cl['unknown_field_score']}, unknown date scores {cl['unknown_date_score']}",
                f"- hard gates: date window {cl['date_window_days']} days ({'on' if cl.get('hard_date_gate', True) else 'off'}), "
                f"country agreement {'required' if cl.get('require_country_agreement', True) else 'not required'}, "
                f"max cluster span {cl.get('max_cluster_span_days')} days",
                f"- kNN backend: {backend}",
            ],
        )
    )
    sections.append(
        (
            "Result",
            md_table(
                ["", "count"],
                [
                    ["article-level extractions in", n],
                    ["candidate pairs examined", stats["candidate_pairs"]],
                    ["below candidate cosine floor", stats["below_candidate_cosine"]],
                    ["vetoed by a hard gate", sum(veto_counts.values())],
                    ["scored below the link threshold", len(rejected_scores)],
                    ["links accepted", stats["links"]],
                    ["consolidated events out", len(events)],
                    ["singleton events (one article)", singles],
                    ["multi-article events", len(events) - singles],
                    ["articles absorbed into a multi-article event", n - len(events)],
                    ["reduction", f"{100 * (1 - len(events) / max(n, 1)):.1f}%"],
                    ["records failing schema validation", stats["invalid_events"]],
                    ["wall clock", fmt_duration(seconds)],
                ],
                align=["---", "---:"],
            ),
        )
    )
    sections.append(
        (
            "Hard-gate vetoes",
            md_table(["gate", "pairs refused"], veto_counts.most_common(), align=["---", "---:"]) or ["(none)"],
        )
    )
    if accepted_signals:
        names = ["semantic", "date", "location", "river", "flood_type", "country", "score"]
        sections.append(
            (
                "Signal values on accepted links",
                md_table(
                    ["signal", "mean", "p10", "median", "p90"],
                    [
                        [
                            name,
                            f"{np.mean([s[name] for s in accepted_signals]):.3f}",
                            f"{np.percentile([s[name] for s in accepted_signals], 10):.3f}",
                            f"{np.median([s[name] for s in accepted_signals]):.3f}",
                            f"{np.percentile([s[name] for s in accepted_signals], 90):.3f}",
                        ]
                        for name in names
                    ],
                    align=["---", "---:", "---:", "---:", "---:"],
                )
                + [
                    "",
                    f"Rejected pairs (scored, but below threshold): {len(rejected_scores)}, "
                    f"mean score {np.mean(rejected_scores):.3f}, max {np.max(rejected_scores):.3f}"
                    if rejected_scores
                    else "",
                ],
            )
        )
    buckets = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 25), (26, 100), (101, 10**9)]
    sections.append(
        (
            "Cluster size distribution",
            md_table(
                ["articles per event", "events"],
                [
                    [
                        f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"),
                        sum(v for k, v in sizes.items() if lo <= k <= hi),
                    ]
                    for lo, hi in buckets
                ],
                align=["---:", "---:"],
            ),
        )
    )
    edges = ["<0.5", "0.5-0.6", "0.6-0.7", "0.7-0.75", "0.75-0.8", "0.8-0.85", "0.85-0.9", "0.9-0.95", ">=0.95"]
    sections.append(
        (
            "Nearest-neighbour cosine (top-1 per extraction)",
            md_table(["bin", "extractions"], list(zip(edges, (int(x) for x in hist))), align=["---", "---:"]),
        )
    )
    conflict_rows = []
    for slot in NUMERIC_SLOTS:
        reported = sum(1 for e in events if e.counts.get(slot, {}).get("reports", 0) >= 2)
        conflicting = sum(
            1 for e in events if e.counts.get(slot, {}).get("reports", 0) >= 2 and e.counts[slot]["conflict"]
        )
        conflict_rows.append([slot, reported, conflicting, f"{100 * conflicting / reported:.0f}%" if reported else "-"])
    sections.append(
        (
            "Count disagreement across reports of the same event",
            md_table(
                ["slot", "events with >= 2 reports", "conflicting", "conflict rate"],
                conflict_rows,
                align=["---", "---:", "---:", "---:"],
            )
            + [
                "",
                "Disagreement is preserved, never averaged: `counts[slot]` keeps min, max and the "
                "number of articles that stated it. The scalar field is the maximum reported value.",
            ],
        )
    )
    type_hist = Counter(e.flood_type for e in events)
    sections.append(
        (
            "Consolidated events by flood type",
            md_table(
                ["flood_type", "events", "share"],
                [[t, c, f"{100 * c / max(len(events), 1):.1f}%"] for t, c in type_hist.most_common()],
                align=["---", "---:", "---:"],
            ),
        )
    )
    sections.append(
        (
            "Largest events",
            md_table(
                ["articles", "date span", "country", "flood type", "title"],
                [
                    [
                        e.n_articles,
                        f"{e.date_start} to {e.date_end}" if e.date_start else "-",
                        e.country or "-",
                        e.flood_type,
                        (e.title or "").replace("|", "/")[:100],
                    ]
                    for e in sorted(events, key=lambda e: -e.n_articles)[:25]
                ],
                align=["---:", "---", "---", "---", "---"],
            ),
        )
    )
    return write_report(
        path,
        "Event coreference report",
        sections,
        preamble=(
            f"Produced by `7_semantic_layer/cluster_events.py` from `events_meta.jsonl` ({n} indexed "
            "extractions). Every number measured by the run that wrote this file."
        ),
    )


if __name__ == "__main__":
    sys.exit(main())
