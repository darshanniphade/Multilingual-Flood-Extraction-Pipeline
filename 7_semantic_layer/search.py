"""Stage 7i - hybrid retrieval, and the LLM answer step on top of it.

    Query
      |
      +-----------------+
      v                 v
    BM25            Dense (BGE + Chroma)
      |                 |
      +--------+--------+
               v
    Reciprocal rank fusion
               v
        metadata filters
               v
      cross-encoder rerank (optional, offline-only)
               v
       top-K articles -> the events they belong to
               v
      Qwen3-14B over the ORIGINAL TEXT of those articles   (--llm)
               v
       structured JSON with quoted evidence

Why hybrid. Dense retrieval matches meaning: "flash floods caused by extreme
rainfall" finds cloudburst reports that never use the word "flash". It is also
the one that fails on a rare proper noun - "Kishtwar" is one token the model has
barely seen, while BM25 treats it as the sharpest signal in the query. Fusing
the two ranked lists by RRF (Cormack et al., 2009) needs no score calibration
between them, which is exactly what makes it robust here.

What is NOT sent to the LLM: embeddings, similarity scores, or metadata rows.
The vector index only decides WHICH articles the model reads; the model then
reads their natural-language text and nothing else.

    python search.py "Assam floods June 2021"
    python search.py "flash floods caused by extreme rainfall" --k 20
    python search.py "major floods affecting villages in India" --country India --year 2021
    python search.py "dam release flooding downstream" --level article --json
    python search.py "how many died in the Kishtwar cloudburst" --llm
    python search.py --benchmark          # timed run over the built-in query set

Reads: chroma/, events_meta.jsonl, and - when they exist - clusters.jsonl and
consolidated_events.jsonl for event-level answers.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from build_index import embed_query, load_embedder, open_collection
from common import (
    DEFAULT_CONFIG,
    LLMUnavailable,
    article_text,
    fmt_duration,
    load_article,
    load_config,
    md_table,
    output_root,
    parse_uid,
    read_jsonl,
    setup_logging,
    ollama_available,
    ollama_json,
    write_report,
)
from event_schema import LLM_EVENT_SCHEMA, LLM_SYSTEM_PROMPT
from rerank import Reranker

log = logging.getLogger("events.search")

_TOKEN = re.compile(r"[a-z0-9]+")

BENCHMARK_QUERIES = [
    "Assam floods June 2021",
    "flash floods caused by extreme rainfall",
    "major floods affecting villages in India",
    "storm surge coastal flooding",
    "dam release flooded downstream villages",
    "urban waterlogging brought traffic to a halt",
    "monsoon floods displaced thousands in Bangladesh",
    "cloudburst swept away houses in the Himalayas",
]


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


# --------------------------------------------------------------------------
# Lexical index
# --------------------------------------------------------------------------


class BM25Index:
    """Okapi BM25 over the same event strings the embeddings were built from, so
    the two retrievers see exactly the same documents."""

    def __init__(self, cfg: dict, root: Path):
        from rank_bm25 import BM25Okapi

        rc = cfg["retrieval"]
        meta_path = root / "events_meta.jsonl"
        if not meta_path.exists():
            raise SystemExit(f"{meta_path} not found - run build_index.py first")
        cache_path = root / "bm25_cache.pkl"
        signature = (meta_path.stat().st_mtime_ns, meta_path.stat().st_size)

        tokens = None
        if rc.get("cache_bm25", True) and cache_path.exists():
            try:
                with cache_path.open("rb") as fh:
                    blob = pickle.load(fh)
                if tuple(blob.get("signature", ())) == signature:
                    tokens, self.uids, self.texts = blob["tokens"], blob["uids"], blob["texts"]
                    log.info("BM25 cache hit (%d documents)", len(self.uids))
            except Exception as exc:
                log.warning("BM25 cache unusable (%s), rebuilding", exc)

        if tokens is None:
            started = time.time()
            self.uids, self.texts, tokens = [], [], []
            for row in read_jsonl(meta_path):
                self.uids.append(row["uid"])
                self.texts.append(row.get("text", ""))
                tokens.append(tokenize(row.get("text", "")))
            log.info("tokenised %d documents in %s", len(self.uids), fmt_duration(time.time() - started))
            if rc.get("cache_bm25", True):
                with cache_path.open("wb") as fh:
                    pickle.dump({"signature": signature, "tokens": tokens, "uids": self.uids, "texts": self.texts}, fh, protocol=5)

        self.bm25 = BM25Okapi(tokens, k1=float(rc.get("bm25_k1", 1.5)), b=float(rc.get("bm25_b", 0.75)))
        self.position = {uid: i for i, uid in enumerate(self.uids)}

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        if len(scores) == 0:
            return []
        import numpy as np

        top = np.argsort(-scores)[:k]
        return [(self.uids[i], float(scores[i])) for i in top if scores[i] > 0]


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------


def reciprocal_rank_fusion(runs: dict[str, list[str]], k: int) -> list[tuple[str, float, dict]]:
    """RRF: score(d) = sum over runs of 1 / (k + rank(d)). No score
    normalisation between retrievers, which is the point - BM25 scores and
    cosine similarities are not comparable quantities."""
    fused: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict] = defaultdict(dict)
    for run_name, uids in runs.items():
        for rank, uid in enumerate(uids, 1):
            fused[uid] += 1.0 / (k + rank)
            ranks[uid][run_name] = rank
    ordered = sorted(fused.items(), key=lambda kv: -kv[1])
    return [(uid, score, ranks[uid]) for uid, score in ordered]


# --------------------------------------------------------------------------
# Retriever
# --------------------------------------------------------------------------


class HybridSearch:
    def __init__(self, cfg: dict, load_dense: bool = True):
        self.cfg = cfg
        self.root = output_root(cfg)
        self.rc = cfg["retrieval"]
        self.meta: dict[str, dict] = {row["uid"]: row for row in read_jsonl(self.root / "events_meta.jsonl")}
        self.bm25 = BM25Index(cfg, self.root)
        self.collection = open_collection(cfg) if load_dense else None
        self.model = load_embedder(cfg) if load_dense else None
        self.reranker = Reranker(cfg)

        self.uid_to_event: dict[str, str] = {}
        self.events: dict[str, dict] = {}
        clusters = self.root / "clusters.jsonl"
        if clusters.exists():
            for row in read_jsonl(clusters):
                for uid in row.get("members", []):
                    self.uid_to_event[uid] = row["event_id"]
        events_path = self.root / "consolidated_events.jsonl"
        if events_path.exists():
            self.events = {row["event_id"]: row for row in read_jsonl(events_path)}
        log.info(
            "index: %d indexed extractions, %d consolidated events, reranker=%s",
            len(self.meta), len(self.events), "on" if self.reranker.available else f"off ({self.reranker.reason})",
        )

    # -- passages -----------------------------------------------------------

    def _chunk_collection(self):
        """The optional body-passage index, opened lazily and only if it has
        been built. Its absence is normal, not an error."""
        if not hasattr(self, "_chunks"):
            self._chunks = None
            name = self.cfg["chunking"].get("collection", "flood_chunks")
            try:
                import chromadb

                client = chromadb.PersistentClient(path=self.cfg["paths"]["chroma_dir"])
                collection = client.get_collection(name)
                if collection.count():
                    self._chunks = collection
                    log.info("passage index %s: %d chunks", name, collection.count())
            except Exception:
                pass  # not built - the LLM step falls back to the article head
        return self._chunks

    def best_chunks(self, uid: str, query: str, max_chars: int, top_n: int = 3) -> str:
        collection = self._chunk_collection()
        if collection is None or self.model is None:
            return ""
        vector = embed_query(self.model, self.cfg, query)
        try:
            got = collection.query(
                query_embeddings=[vector.tolist()],
                n_results=top_n,
                where={"uid": uid},
                include=["documents", "metadatas"],
            )
        except Exception:
            return ""
        docs = got.get("documents") or [[]]
        metas = got.get("metadatas") or [[]]
        if not docs or not docs[0]:
            return ""
        ordered = sorted(zip(docs[0], metas[0]), key=lambda pair: pair[1].get("char_start", 0))
        out = "\n...\n".join(doc for doc, _ in ordered)
        return out[:max_chars]

    # -- filters ------------------------------------------------------------

    def matches(self, row: dict, filters: dict) -> bool:
        if filters.get("year") and str(row.get("year")) not in filters["year"]:
            return False
        if filters.get("country") and filters["country"].lower() not in str(row.get("country", "")).lower():
            return False
        if filters.get("flood_type") and row.get("flood_type") != filters["flood_type"]:
            return False
        if filters.get("date_from") and (row.get("date_max") or "") < filters["date_from"]:
            return False
        if filters.get("date_to") and (row.get("date_min") or "9999") > filters["date_to"]:
            return False
        if filters.get("min_deaths") is not None and (row.get("deaths") or -1) < filters["min_deaths"]:
            return False
        return True

    # -- retrieval ----------------------------------------------------------

    def chroma_where(self, filters: dict) -> dict | None:
        """The part of the filter the vector store can evaluate itself. Chroma
        compares strings only for equality, so a date RANGE cannot go here and
        is applied in Python below."""
        clauses: list[dict] = []
        if filters.get("year"):
            clauses.append({"year": {"$in": [int(y) for y in filters["year"]]}})
        if filters.get("flood_type"):
            clauses.append({"flood_type": {"$eq": filters["flood_type"]}})
        if filters.get("min_deaths") is not None:
            clauses.append({"deaths": {"$gte": int(filters["min_deaths"])}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def retrieve(self, query: str, filters: dict | None = None, k: int | None = None) -> dict:
        filters = filters or {}
        k = k or int(self.rc.get("final_top_k", 10))
        timings: dict[str, float] = {}

        # A filter applied after retrieval starves the result set: the top 100
        # fused candidates for "Assam floods" contain almost nothing inside a
        # one-month date window. Whatever the vector store cannot filter itself
        # is compensated for by retrieving deeper.
        where = self.chroma_where(filters)
        python_side = bool(filters.get("country") or filters.get("date_from") or filters.get("date_to"))
        depth = int(self.rc.get("filter_depth_multiplier", 10)) if python_side else 1

        t0 = time.time()
        lexical = self.bm25.search(query, int(self.rc.get("bm25_top_k", 100)) * depth)
        timings["bm25_ms"] = (time.time() - t0) * 1000

        dense: list[tuple[str, float]] = []
        if self.collection is not None:
            t0 = time.time()
            vector = embed_query(self.model, self.cfg, query)
            timings["embed_ms"] = (time.time() - t0) * 1000
            t0 = time.time()
            query_args = {
                "query_embeddings": [vector.tolist()],
                "n_results": min(int(self.rc.get("dense_top_k", 100)) * depth, len(self.meta)),
                "include": ["distances"],
            }
            if where:
                query_args["where"] = where
            got = self.collection.query(**query_args)
            dense = [
                (uid, 1.0 - float(dist))  # Chroma returns cosine distance
                for uid, dist in zip(got["ids"][0], got["distances"][0])
            ]
            timings["dense_ms"] = (time.time() - t0) * 1000

        t0 = time.time()
        fused = reciprocal_rank_fusion(
            {"bm25": [uid for uid, _ in lexical], "dense": [uid for uid, _ in dense]},
            int(self.rc.get("rrf_k", 60)),
        )
        timings["fuse_ms"] = (time.time() - t0) * 1000

        dense_scores = dict(dense)
        lexical_scores = dict(lexical)
        candidates: list[dict] = []
        for uid, score, ranks in fused:
            row = self.meta.get(uid)
            if row is None or not self.matches(row, filters):
                continue
            candidates.append(
                {
                    "uid": uid,
                    "rrf_score": round(score, 6),
                    "bm25_rank": ranks.get("bm25"),
                    "dense_rank": ranks.get("dense"),
                    "bm25_score": round(lexical_scores.get(uid, 0.0), 4),
                    "cosine": round(dense_scores.get(uid, 0.0), 4),
                    "text": row.get("text", ""),
                    "meta": row,
                }
            )

        if self.reranker.available and candidates:
            t0 = time.time()
            candidates = self.reranker.rerank(query, candidates)
            timings["rerank_ms"] = (time.time() - t0) * 1000

        return {
            "query": query,
            "filters": filters,
            "n_bm25": len(lexical),
            "n_dense": len(dense),
            "n_after_filters": len(candidates),
            "reranked": self.reranker.available,
            "timings_ms": {name: round(value, 1) for name, value in timings.items()},
            "articles": candidates[:k],
            "all_candidates": candidates,
        }

    def to_events(self, result: dict, k: int) -> list[dict]:
        """Fold article hits onto the real-world events they belong to.

        An event scores its BEST-matching article, with a small logarithmic
        bonus for how many of its articles matched:

            score = best · (1 + support_weight · ln(1 + n_matching))

        Summing the articles instead was the obvious first choice and is wrong:
        the Yalta 2021 cluster has 90 articles, so 90 weak matches outscored a
        directly on-topic Assam report and it ranked first for "Assam floods
        June 2021". Corroboration should break ties, not decide the ranking.
        """
        support_weight = float(self.rc.get("event_support_weight", 0.05))
        grouped: dict[str, dict] = {}
        for rank, candidate in enumerate(result["all_candidates"], 1):
            event_id = self.uid_to_event.get(candidate["uid"])
            if event_id is None:
                event_id = f"(unclustered) {candidate['uid']}"
            bucket = grouped.setdefault(
                event_id,
                {
                    "event_id": event_id,
                    "best": float("-inf"),
                    "n_hits": 0,
                    "best_rank": rank,
                    "sources": [],
                    "event": self.events.get(event_id),
                    "first_meta": candidate["meta"],
                },
            )
            bucket["best"] = max(bucket["best"], candidate.get("rerank_score", candidate["rrf_score"]))
            bucket["n_hits"] += 1
            bucket["best_rank"] = min(bucket["best_rank"], rank)
            bucket["sources"].append(
                {
                    "uid": candidate["uid"],
                    "title": candidate["meta"].get("title", ""),
                    "rrf_score": candidate["rrf_score"],
                    "cosine": candidate["cosine"],
                    "bm25_score": candidate["bm25_score"],
                }
            )
        for bucket in grouped.values():
            bucket["score"] = bucket["best"] * (1.0 + support_weight * math.log1p(bucket["n_hits"]))
        ordered = sorted(grouped.values(), key=lambda b: (-b["score"], b["best_rank"]))
        out = []
        for bucket in ordered[:k]:
            event = bucket["event"] or {}
            # An article that coreference has not seen yet still answers the
            # query; fall back to its own metadata rather than printing blanks.
            meta = bucket["first_meta"]
            deaths = event.get("deaths")
            if deaths is None and isinstance(meta.get("deaths"), (int, float)) and meta["deaths"] >= 0:
                deaths = meta["deaths"]
            out.append(
                {
                    "event_id": bucket["event_id"],
                    "clustered": bucket["event"] is not None,
                    "score": round(bucket["score"], 6),
                    "matching_articles": bucket["n_hits"],
                    "date_start": event.get("date_start") or meta.get("date_min") or None,
                    "date_end": event.get("date_end") or meta.get("date_max") or None,
                    "country": event.get("country") or meta.get("country") or None,
                    "locations": (event.get("locations") or [x for x in str(meta.get("locations") or "").split("; ") if x])[:8],
                    "rivers": (event.get("rivers") or [x for x in str(meta.get("rivers") or "").split("; ") if x])[:5],
                    "flood_type": event.get("flood_type") or meta.get("flood_type") or "Unknown",
                    "deaths": deaths,
                    "affected_population": event.get("affected_population"),
                    "severity": event.get("severity"),
                    "title": event.get("title") or bucket["sources"][0]["title"],
                    "n_articles": event.get("n_articles", len(bucket["sources"])),
                    "sources": bucket["sources"][:10],
                }
            )
        return out


# --------------------------------------------------------------------------
# LLM answer step
# --------------------------------------------------------------------------


def gather_evidence(cfg: dict, candidates: list[dict], engine: "HybridSearch | None" = None, query: str = "") -> list[dict]:
    """The passages the model is allowed to read: ORIGINAL article text, never
    anything derived from a vector.

    When the passage index exists (build_index.py --chunks), the passages are
    the query-relevant chunks of each article rather than its opening
    `max_evidence_chars` characters - the fact that answers the question is
    often in paragraph nine. Without that index, the head of the article is
    used and the run says so.
    """
    llm = cfg["llm"]
    max_docs = int(llm.get("max_evidence_docs", 5))
    max_chars = int(llm.get("max_evidence_chars", 2500))
    evidence = []
    for candidate in candidates[:max_docs]:
        year, month, article_id = parse_uid(candidate["uid"])
        body = article_text(load_article(cfg, year, month, article_id))
        text, source = (body.strip() or candidate["text"])[:max_chars], "article head"
        if engine is not None and query:
            passages = engine.best_chunks(candidate["uid"], query, max_chars)
            if passages:
                text, source = passages, "retrieved passages"
        evidence.append(
            {
                "n": len(evidence) + 1,
                "uid": candidate["uid"],
                "title": candidate["meta"].get("title", ""),
                "publish_date": candidate["meta"].get("publish_date", ""),
                "text": text,
                "source": source,
            }
        )
    return evidence


def llm_answer(cfg: dict, query: str, evidence: list[dict]) -> dict:
    passages = "\n\n".join(
        f"[{e['n']}] (published {e['publish_date'] or 'unknown'}) {e['title']}\n{e['text']}" for e in evidence
    )
    user = (
        f"Question: {query}\n\n"
        f"Schema (return exactly these keys):\n{json.dumps(LLM_EVENT_SCHEMA, indent=1)}\n\n"
        f"Evidence passages:\n{passages}"
    )
    answer = ollama_json(cfg, LLM_SYSTEM_PROMPT, user)
    # Verify each quote is really in the evidence. An unverifiable quote is
    # dropped and flagged rather than shown as a citation.
    haystack = "\n".join(e["text"] for e in evidence).lower()
    checked = []
    for item in answer.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        quote = str(item.get("quote", "")).strip()
        item["verified"] = bool(quote) and quote.lower() in haystack
        checked.append(item)
    answer["evidence"] = checked
    answer["_evidence_uids"] = [e["uid"] for e in evidence]
    answer["_quotes_verified"] = sum(1 for c in checked if c["verified"])
    answer["_quotes_total"] = len(checked)
    return answer


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def print_events(events: list[dict]) -> None:
    for i, event in enumerate(events, 1):
        span = f"{event['date_start']} to {event['date_end']}" if event.get("date_start") else "date unknown"
        head = f"{i:2d}. [{event['score']:.4f}] {event['event_id']}  {span}"
        print(head)
        print(f"    {event['title'][:110]}")
        bits = [
            event.get("country") or "country unknown",
            event.get("flood_type", "Unknown"),
            f"{event.get('n_articles', 0)} article(s)",
        ]
        if event.get("deaths") is not None:
            bits.append(f"deaths {event['deaths']}")
        if event.get("severity"):
            bits.append(event["severity"])
        print("    " + " | ".join(str(b) for b in bits))
        if event.get("locations"):
            print("    locations: " + "; ".join(event["locations"]))
        print("    sources: " + ", ".join(s["uid"] for s in event["sources"][:5]))
        print()


def print_articles(candidates: list[dict]) -> None:
    for i, candidate in enumerate(candidates, 1):
        meta = candidate["meta"]
        score = candidate.get("rerank_score")
        score_str = f"rerank {score:.4f}" if score is not None else f"rrf {candidate['rrf_score']:.4f}"
        print(f"{i:2d}. [{score_str}] {candidate['uid']}  {meta.get('date_min') or '-'}  {meta.get('country') or '-'}")
        print(f"    {meta.get('title', '')[:110]}")
        print(
            f"    bm25 rank {candidate['bm25_rank']} score {candidate['bm25_score']} | "
            f"dense rank {candidate['dense_rank']} cosine {candidate['cosine']} | "
            f"flood_type {meta.get('flood_type')}"
        )
        print()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Hybrid BM25 + dense retrieval over the flood event index")
    ap.add_argument("query", nargs="?", help="natural-language query")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--k", type=int, help="results to return (default retrieval.final_top_k)")
    ap.add_argument("--level", choices=["event", "article"], default="event")
    ap.add_argument("--year", nargs="*", help="restrict to these years")
    ap.add_argument("--country")
    ap.add_argument("--flood-type")
    ap.add_argument("--date-from")
    ap.add_argument("--date-to")
    ap.add_argument("--min-deaths", type=int)
    ap.add_argument("--no-dense", action="store_true", help="BM25 only (ablation)")
    ap.add_argument("--no-bm25", action="store_true", help="dense only (ablation)")
    ap.add_argument("--llm", action="store_true", help="run Qwen3 structured extraction over the retrieved text")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--benchmark", action="store_true", help="time the built-in query set and write search_report.md")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    if not args.query and not args.benchmark:
        ap.error("give a query, or --benchmark")

    engine = HybridSearch(cfg, load_dense=not args.no_dense)
    if args.no_bm25:
        engine.bm25.search = lambda query, k: []  # ablation switch, keeps the plumbing identical

    filters = {
        "year": set(args.year) if args.year else None,
        "country": args.country,
        "flood_type": args.flood_type,
        "date_from": args.date_from,
        "date_to": args.date_to,
        "min_deaths": args.min_deaths,
    }
    filters = {key: value for key, value in filters.items() if value is not None}
    k = args.k or int(cfg["retrieval"].get("final_top_k", 10))

    if args.benchmark:
        return run_benchmark(cfg, engine, k)

    started = time.time()
    result = engine.retrieve(args.query, filters, k)
    elapsed = (time.time() - started) * 1000

    payload: dict = {
        "query": args.query,
        "filters": {key: sorted(value) if isinstance(value, set) else value for key, value in filters.items()},
        "latency_ms": round(elapsed, 1),
        "stage_timings_ms": result["timings_ms"],
        "bm25_hits": result["n_bm25"],
        "dense_hits": result["n_dense"],
        "candidates_after_filters": result["n_after_filters"],
        "reranked": result["reranked"],
        "reranker_note": engine.reranker.reason,
    }
    if args.level == "event":
        payload["events"] = engine.to_events(result, k)
    else:
        payload["articles"] = [
            {key: value for key, value in candidate.items() if key != "meta"} | {"title": candidate["meta"].get("title", "")}
            for candidate in result["articles"]
        ]

    if args.llm:
        ok, message = ollama_available(cfg)
        if not ok:
            payload["llm_error"] = message
            log.error("%s - retrieval results are still below", message)
        else:
            evidence = gather_evidence(cfg, result["all_candidates"], engine, args.query)
            try:
                started = time.time()
                payload["llm"] = llm_answer(cfg, args.query, evidence)
                payload["llm_latency_ms"] = round((time.time() - started) * 1000, 1)
            except (LLMUnavailable, ValueError) as exc:
                payload["llm_error"] = str(exc)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"\nquery: {args.query!r}")
    print(
        f"bm25 {payload['bm25_hits']} hits, dense {payload['dense_hits']} hits, "
        f"{payload['candidates_after_filters']} after filters, {payload['latency_ms']:.0f} ms "
        f"({', '.join(f'{name} {value}' for name, value in payload['stage_timings_ms'].items())})"
    )
    print(f"reranker: {'on' if payload['reranked'] else 'off - ' + payload['reranker_note']}\n")
    if args.level == "event":
        print_events(payload["events"])
    else:
        print_articles(result["articles"])
    if "llm" in payload:
        print("--- Qwen3-14B structured extraction over the retrieved article text ---")
        print(json.dumps(payload["llm"], ensure_ascii=False, indent=2))
        print(
            f"\nquotes verified against the evidence: "
            f"{payload['llm']['_quotes_verified']}/{payload['llm']['_quotes_total']}"
        )
    elif "llm_error" in payload:
        print(f"[llm] {payload['llm_error']}")
    return 0


def run_benchmark(cfg: dict, engine: HybridSearch, k: int) -> int:
    rows = []
    for query in BENCHMARK_QUERIES:
        started = time.time()
        result = engine.retrieve(query, {}, k)
        total = (time.time() - started) * 1000
        events = engine.to_events(result, k)
        rows.append(
            {
                "query": query,
                "total_ms": total,
                "timings": result["timings_ms"],
                "candidates": result["n_after_filters"],
                "events": len(events),
                "top": events[0]["title"][:70] if events else "-",
            }
        )
        log.info("%-45s %6.0f ms  %3d candidates", query[:45], total, result["n_after_filters"])

    mean_total = sum(r["total_ms"] for r in rows) / len(rows)
    sections = [
        (
            "Configuration",
            md_table(
                ["", "value"],
                [
                    ["indexed extractions", len(engine.meta)],
                    ["consolidated events", len(engine.events) or "not built"],
                    ["dense top-k", cfg["retrieval"]["dense_top_k"]],
                    ["bm25 top-k", cfg["retrieval"]["bm25_top_k"]],
                    ["rrf k", cfg["retrieval"]["rrf_k"]],
                    ["final top-k", k],
                    ["reranker", "on" if engine.reranker.available else f"off ({engine.reranker.reason})"],
                ],
                align=["---", "---:"],
            ),
        ),
        (
            "Latency by query (single process, warm caches)",
            md_table(
                ["query", "total ms", "embed", "dense", "bm25", "fuse", "candidates", "top result"],
                [
                    [
                        r["query"][:40],
                        f"{r['total_ms']:.0f}",
                        r["timings"].get("embed_ms", "-"),
                        r["timings"].get("dense_ms", "-"),
                        r["timings"].get("bm25_ms", "-"),
                        r["timings"].get("fuse_ms", "-"),
                        r["candidates"],
                        r["top"].replace("|", "/"),
                    ]
                    for r in rows
                ],
                align=["---", "---:", "---:", "---:", "---:", "---:", "---:", "---"],
            )
            + ["", f"Mean end-to-end latency: **{mean_total:.0f} ms** over {len(rows)} queries."],
        ),
        (
            "What this does and does not measure",
            [
                "These are latencies, not retrieval quality. Recall@K, MRR and nDCG require a gold set "
                "of query-to-event judgements; build one with `evaluation/make_annotation_sample.py --task retrieval` "
                "and score it with `evaluation/score.py --task retrieval`. No quality number is reported here "
                "because none has been measured.",
            ],
        ),
    ]
    path = write_report(
        output_root(cfg) / "search_report.md",
        "Hybrid retrieval benchmark",
        sections,
        preamble="Produced by `7_semantic_layer/search.py --benchmark`.",
    )
    print(f"mean {mean_total:.0f} ms over {len(rows)} queries -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
