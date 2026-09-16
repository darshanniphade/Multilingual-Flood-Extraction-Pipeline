"""Stage 7e - duplicate detection, lexical and semantic.

Two passes over the same corpus, answering two different questions.

LEXICAL (MinHash + LSH, the default)
    "Is this literally the same story?" Stage 4 removed byte-identical articles
    with SHA-256. It cannot see a wire story republished with a new byline, a
    trimmed syndication copy, or the same report with a paragraph appended -
    which is the bulk of duplication in a news crawl. MinHash estimates Jaccard
    over w-word shingles and LSH finds the candidate pairs, so the cost is
    linear in the corpus rather than quadratic. Jaccard >= 0.8 means the same
    story; those are safe to collapse.

SEMANTIC (--semantic, embeddings)
    "Is this the same story rewritten?" Two desks writing up the same press
    release share almost no 5-word shingles:

        A: "Heavy rainfall caused severe flooding across three districts"
        B: "Intense rains triggered widespread inundation in three districts"

    MinHash scores that pair near zero; the embeddings put it above 0.95. This
    pass reports those as CANDIDATES and stops there. Nothing is merged on
    embedding similarity alone - high cosine also fires on two genuinely
    different floods in the same place a week apart, and collapsing those would
    destroy exactly the events this project exists to count. Merging is the job
    of cluster_events.py, which additionally requires date, location and
    country agreement.

    python near_dup.py                        # lexical, articles the model accepted
    python near_dup.py --semantic             # both passes
    python near_dup.py --years 2023 --threshold 0.7
    python near_dup.py --select all           # every article stage 5 saw (slow)

Outputs under paths.output_root:

    near_dup_pairs.jsonl            (uid_a, uid_b, jaccard) above threshold
    near_dup_clusters.jsonl         connected components: which to keep, which drop
    semantic_dup_candidates.jsonl   high cosine + low lexical overlap, for review
    near_dup_report.md              measured counts and worked examples
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from common import (
    DEFAULT_CONFIG,
    article_text,
    fmt_duration,
    iter_events,
    load_article,
    load_config,
    md_table,
    output_root,
    read_jsonl,
    setup_logging,
    write_jsonl,
    write_report,
)

log = logging.getLogger("events.near_dup")

_TOKEN = re.compile(r"[a-z0-9]+")


def shingles(text: str, w: int) -> set[str]:
    toks = _TOKEN.findall(text.lower())
    if len(toks) < w:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + w]) for i in range(len(toks) - w + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# --------------------------------------------------------------------------
# Semantic pass
# --------------------------------------------------------------------------


def semantic_candidates(cfg: dict, root: Path, allowed: set[str], bodies: dict[str, str]) -> tuple[list[dict], dict]:
    """Pairs whose embeddings nearly coincide. Exact lexical Jaccard is then
    computed for each pair so the report can separate "MinHash already had this"
    from "only the embeddings see this"."""
    sem = cfg["near_dup"]["semantic"]
    matrix_path = root / "embeddings.npy"
    if not matrix_path.exists():
        log.warning("no embeddings.npy - run build_index.py first; skipping the semantic pass")
        return [], {"skipped": True}

    matrix = np.load(matrix_path)
    ids = json.loads((root / "ids.json").read_text(encoding="utf-8"))
    keep = [i for i, uid in enumerate(ids) if uid in allowed] if allowed else list(range(len(ids)))
    if not keep:
        log.warning("no indexed uid overlaps this selection; skipping the semantic pass")
        return [], {"skipped": True}
    matrix = np.ascontiguousarray(matrix[keep])
    ids = [ids[i] for i in keep]
    log.info("semantic pass over %d indexed articles", len(ids))

    k = min(int(sem.get("top_k", 10)) + 1, len(ids))
    try:
        import faiss

        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        scores, nbrs = index.search(matrix, k)
        backend = "faiss (CPU, exact)"
    except Exception:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        mat = torch.from_numpy(matrix).to(device)
        scores = np.zeros((len(ids), k), dtype=np.float32)
        nbrs = np.zeros((len(ids), k), dtype=np.int64)
        for start in range(0, len(ids), 4096):
            top = torch.topk(mat[start : start + 4096] @ mat.T, k, dim=1)
            scores[start : start + 4096] = top.values.cpu().numpy()
            nbrs[start : start + 4096] = top.indices.cpu().numpy()
        backend = f"torch/{device} (exact, blocked)"

    threshold = float(sem.get("cosine_threshold", 0.95))
    ceiling = float(sem.get("lexical_ceiling", 0.5))
    max_pairs = int(sem.get("max_pairs", 200000))
    w = int(cfg["near_dup"]["shingle_words"])

    shingle_cache: dict[str, set[str]] = {}

    def get_shingles(uid: str) -> set[str]:
        if uid not in shingle_cache:
            shingle_cache[uid] = shingles(bodies.get(uid, ""), w)
        return shingle_cache[uid]

    out: list[dict] = []
    stats = {"pairs_above_cosine": 0, "lexically_caught": 0, "semantic_only": 0, "backend": backend}
    seen: set[tuple[str, str]] = set()
    for i in range(len(ids)):
        for score, j in zip(scores[i], nbrs[i]):
            j = int(j)
            if j == i or score < threshold:
                continue
            key = (ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i])
            if key in seen:
                continue
            seen.add(key)
            stats["pairs_above_cosine"] += 1
            lex = jaccard(get_shingles(key[0]), get_shingles(key[1]))
            if lex >= ceiling:
                stats["lexically_caught"] += 1
                continue
            stats["semantic_only"] += 1
            out.append(
                {
                    "uid_a": key[0],
                    "uid_b": key[1],
                    "cosine": round(float(score), 4),
                    "lexical_jaccard": round(lex, 4),
                    "status": "candidate",
                    "note": "high embedding similarity, low shingle overlap - review, do not auto-merge",
                }
            )
            if len(out) >= max_pairs:
                log.warning("hit semantic.max_pairs=%d, stopping", max_pairs)
                return out, stats
    return out, stats


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="MinHash-LSH and embedding near-duplicate detection")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--years", nargs="*")
    ap.add_argument("--select", choices=["verifiable", "flood", "all"])
    ap.add_argument("--threshold", type=float, help="override near_dup.jaccard_threshold")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--semantic", action="store_true", help="also run the embedding-based pass")
    ap.add_argument("--semantic-only", action="store_true", help="skip MinHash, run only the embedding pass")
    args = ap.parse_args()

    from datasketch import MinHash, MinHashLSH

    cfg = load_config(args.config)
    setup_logging(cfg)
    nd = cfg["near_dup"]
    select = args.select or nd.get("select", "flood")
    threshold = args.threshold or nd["jaccard_threshold"]
    w, perm = int(nd["shingle_words"]), int(nd["num_perm"])
    root = output_root(cfg)

    events = list(iter_events(cfg, args.years, select=select, limit=args.limit))
    log.info("%d articles (select=%s); reading bodies with %d threads", len(events), select, nd["io_workers"])

    def load(ev):
        art = load_article(cfg, ev.year, ev.month, ev.article_id)
        if not art:
            return ev.uid, None, None
        return ev.uid, str(art.get("translated_title") or art.get("title") or ev.title), article_text(art)

    started = time.time()
    lsh = MinHashLSH(threshold=threshold, num_perm=perm)
    sketches: dict[str, MinHash] = {}
    titles: dict[str, str] = {}
    words: dict[str, int] = {}
    bodies: dict[str, str] = {}
    missing = short = 0
    with ThreadPoolExecutor(max_workers=int(nd["io_workers"])) as pool:
        for i, (uid, title, text) in enumerate(pool.map(load, events, chunksize=64), 1):
            if text is None:
                missing += 1
                continue
            titles[uid] = title
            n_words = len(text.split())
            words[uid] = n_words
            if args.semantic or args.semantic_only:
                bodies[uid] = text
            if n_words < int(nd["min_words"]):
                short += 1
                continue
            if not args.semantic_only:
                m = MinHash(num_perm=perm)
                for sh in shingles(text, w):
                    m.update(sh.encode("utf-8"))
                sketches[uid] = m
                lsh.insert(uid, m)
            if i % 10000 == 0:
                log.info("  %d/%d read, %s", i, len(events), fmt_duration(time.time() - started))

    pairs: list[dict] = []
    clusters: list[dict] = []
    if not args.semantic_only:
        log.info("sketched %d (missing %d, short %d); querying LSH", len(sketches), missing, short)
        uf = UnionFind()
        seen: set[tuple[str, str]] = set()
        for uid, m in sketches.items():
            for other in lsh.query(m):
                if other == uid:
                    continue
                key = (uid, other) if uid < other else (other, uid)
                if key in seen:
                    continue
                seen.add(key)
                j = m.jaccard(sketches[other])
                if j >= threshold:
                    pairs.append({"uid_a": key[0], "uid_b": key[1], "jaccard": round(float(j), 4)})
                    uf.union(*key)

        groups: dict[str, list[str]] = defaultdict(list)
        for uid in sketches:
            groups[uf.find(uid)].append(uid)
        for members in sorted(groups.values(), key=lambda g: (-len(g), g[0])):
            if len(members) < 2:
                continue
            # keep the longest body: syndication copies are trimmed, not padded
            keep = max(members, key=lambda u: words[u])
            clusters.append(
                {"keep": keep, "drop": [u for u in members if u != keep], "n": len(members), "title": titles[keep]}
            )
        write_jsonl(root / "near_dup_pairs.jsonl", pairs)
        write_jsonl(root / "near_dup_clusters.jsonl", clusters)

    sem_pairs: list[dict] = []
    sem_stats: dict = {}
    if args.semantic or args.semantic_only:
        sem_pairs, sem_stats = semantic_candidates(cfg, root, set(bodies), bodies)
        write_jsonl(root / "semantic_dup_candidates.jsonl", sem_pairs)

    removable = sum(len(c["drop"]) for c in clusters)
    sections: list[tuple[str, list[str]]] = [
        (
            "Lexical pass (MinHash-LSH)",
            md_table(
                ["", "count"],
                [
                    ["articles considered", len(events)],
                    ["bodies missing on disk", missing],
                    [f"below {nd['min_words']} words (skipped)", short],
                    ["sketched", len(sketches)],
                    ["near-duplicate pairs", len(pairs)],
                    ["near-duplicate groups", len(clusters)],
                    ["articles removable (keep one per group)", removable],
                    ["near-duplicate rate", f"{100 * removable / max(len(sketches), 1):.2f}%"],
                    ["wall clock", fmt_duration(time.time() - started)],
                ],
                align=["---", "---:"],
            )
            + [
                "",
                f"select={select}, years={' '.join(args.years) if args.years else 'all'}, "
                f"shingle w={w}, num_perm={perm}, Jaccard >= {threshold}.",
                "Stage 4 already removed byte-identical copies; every pair here survived SHA-256.",
            ]
            if not args.semantic_only
            else ["(skipped: --semantic-only)"],
        )
    ]
    if clusters:
        sections.append(
            (
                "Largest lexical groups",
                md_table(
                    ["copies", "kept article", "title"],
                    [[c["n"], c["keep"], c["title"].replace("|", "/")[:100]] for c in clusters[:20]],
                    align=["---:", "---", "---"],
                ),
            )
        )
    if pairs:
        sections.append(
            (
                "Lexical pairs nearest the threshold",
                md_table(
                    ["jaccard", "a", "b"],
                    [
                        [f"{p['jaccard']:.3f}", titles.get(p["uid_a"], "")[:55], titles.get(p["uid_b"], "")[:55]]
                        for p in sorted(pairs, key=lambda p: p["jaccard"])[:15]
                    ],
                    align=["---:", "---", "---"],
                ),
            )
        )
    if args.semantic or args.semantic_only:
        sem = cfg["near_dup"]["semantic"]
        body = (
            ["The semantic pass was skipped (no embeddings.npy, or no overlap with the selection)."]
            if sem_stats.get("skipped")
            else md_table(
                ["", "count"],
                [
                    [f"pairs with cosine >= {sem['cosine_threshold']}", sem_stats.get("pairs_above_cosine", 0)],
                    [f"of those, lexical Jaccard >= {sem['lexical_ceiling']} (MinHash already had them)", sem_stats.get("lexically_caught", 0)],
                    ["**candidates only the embeddings found**", sem_stats.get("semantic_only", 0)],
                    ["kNN backend", sem_stats.get("backend", "-")],
                ],
                align=["---", "---:"],
            )
            + [
                "",
                "These pairs are **not merged**. High cosine with low lexical overlap means either a "
                "rewrite of the same report or two different floods described in near-identical "
                "language; only date, location and country agreement can tell those apart, which is "
                "what `cluster_events.py` does.",
            ]
        )
        sections.append(("Semantic pass (embeddings)", body))
        if sem_pairs:
            sections.append(
                (
                    "Examples: same story, different words",
                    md_table(
                        ["cosine", "lexical J", "a", "b"],
                        [
                            [
                                f"{p['cosine']:.3f}",
                                f"{p['lexical_jaccard']:.3f}",
                                titles.get(p["uid_a"], "")[:50],
                                titles.get(p["uid_b"], "")[:50],
                            ]
                            for p in sorted(sem_pairs, key=lambda p: p["lexical_jaccard"])[:15]
                        ],
                        align=["---:", "---:", "---", "---"],
                    ),
                )
            )
    path = write_report(
        root / "near_dup_report.md",
        "Near-duplicate report",
        sections,
        preamble="Produced by `7_semantic_layer/near_dup.py`. All counts measured on this run.",
    )
    log.info(
        "%d lexical pairs, %d groups, %d removable, %d semantic candidates, %s -> %s",
        len(pairs), len(clusters), removable, len(sem_pairs), fmt_duration(time.time() - started), path.name,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
