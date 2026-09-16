"""Stage 7a - transformer embeddings and the vector index.

Every flood extraction that survived stage 5 is turned into a dense vector with
BAAI/bge-base-en-v1.5 and stored in a persistent Chroma collection alongside the
flat metadata the rest of the stage filters on. This is the representation
layer: coreference, near-duplicate detection and semantic search all read it.

    python build_index.py --limit 300 --years 2021    # smoke test
    python build_index.py                             # all years, verifiable
    python build_index.py --select flood              # every accepted article
    python build_index.py --chunks                    # + passage index over bodies
    python build_index.py --refresh-metadata          # re-attach sidecars, no re-embed
    python build_index.py --rebuild                   # drop the collection first

Artefacts under paths.output_root:

    chroma/                     persistent Chroma collection (HNSW, cosine)
    events_meta.jsonl           one line per indexed extraction: uid, text, metadata
    embeddings.npy + ids.json   the same vectors as a dense matrix, row-aligned
                                with ids.json, for the all-pairs scripts
    build_index_report.md       measured counts, throughput and distributions

Resumable: uids already in the collection are skipped, so a killed run costs
only the time already spent. Nothing is held in memory but the current batch.

If ner.py / flood_type.py have already run, their sidecars are merged into the
embedded text and the stored metadata. They are optional - the index builds
without them and `--refresh-metadata` folds them in later without re-embedding.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from common import (
    DEFAULT_CONFIG,
    Event,
    apply_offline_env,
    article_text,
    chunk_text,
    count_events,
    fmt_duration,
    gpu_note,
    iter_events,
    load_article,
    load_config,
    load_sidecar,
    md_table,
    output_root,
    setup_logging,
    write_jsonl,
    write_report,
)

log = logging.getLogger("events.index")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def load_embedder(cfg: dict, max_seq_length: int | None = None):
    apply_offline_env(cfg)
    emb = cfg["embedding"]
    from sentence_transformers import SentenceTransformer  # heavy import, keep local

    model = SentenceTransformer(emb["model"], device=emb.get("device", "cuda"))
    model.max_seq_length = max_seq_length or emb.get("max_seq_length", 256)
    log.info(
        "embedder %s on %s, max_seq_length=%d",
        emb["model"], emb.get("device", "cuda"), model.max_seq_length,
    )
    return model


def embed_texts(model, cfg: dict, texts: list[str]) -> np.ndarray:
    emb = cfg["embedding"]
    return model.encode(
        texts,
        batch_size=emb.get("batch_size", 128),
        normalize_embeddings=emb.get("normalize", True),
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)


def embed_query(model, cfg: dict, query: str) -> np.ndarray:
    """Queries get bge's documented retrieval instruction prefix; documents
    deliberately do not. Mixing the two costs a few points of nDCG."""
    prefix = cfg["embedding"].get("query_instruction", "")
    return embed_texts(model, cfg, [prefix + query])[0]


# --------------------------------------------------------------------------
# Chroma
# --------------------------------------------------------------------------


def open_collection(cfg: dict, name: str | None = None, rebuild: bool = False):
    import chromadb

    client = chromadb.PersistentClient(path=cfg["paths"]["chroma_dir"])
    name = name or cfg["index"]["collection"]
    if rebuild:
        try:
            client.delete_collection(name)
            log.warning("dropped existing collection %s", name)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=name, metadata={"hnsw:space": cfg["index"].get("space", "cosine")}
    )


def existing_ids(collection, page: int = 5000) -> set[str]:
    total = collection.count()
    ids: set[str] = set()
    for offset in range(0, total, page):
        got = collection.get(limit=page, offset=offset, include=[])
        ids.update(got["ids"])
    return ids


def export_matrix(collection, root: Path, page: int = 5000) -> tuple[np.ndarray, list[str], list[dict]]:
    """Re-export the whole collection as a dense matrix + metadata sidecar, so
    the all-pairs scripts (clustering, semantic dedup) see exactly what is
    indexed rather than only what this run added."""
    total = collection.count()
    ids: list[str] = []
    meta_rows: list[dict] = []
    matrix: np.ndarray | None = None
    filled = 0
    for offset in range(0, total, page):
        got = collection.get(limit=page, offset=offset, include=["embeddings", "documents", "metadatas"])
        vecs = np.asarray(got["embeddings"], dtype=np.float32)
        if matrix is None:
            matrix = np.zeros((total, vecs.shape[1]), dtype=np.float32)
        matrix[filled : filled + len(vecs)] = vecs
        filled += len(vecs)
        ids.extend(got["ids"])
        for uid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
            meta_rows.append({"uid": uid, "text": doc, **meta})
        log.info("  exported %d/%d", filled, total)
    if matrix is None:
        matrix = np.zeros((0, 0), dtype=np.float32)
    np.save(root / "embeddings.npy", matrix[:filled])
    (root / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
    write_jsonl(root / "events_meta.jsonl", meta_rows)
    return matrix[:filled], ids, meta_rows


# --------------------------------------------------------------------------
# Sidecars
# --------------------------------------------------------------------------


def load_enrichment(root: Path) -> dict[str, dict]:
    """flood_type.jsonl and ner.jsonl if earlier stages produced them."""
    enrich: dict[str, dict] = {}
    ft = load_sidecar(root / "flood_type.jsonl")
    for uid, row in ft.items():
        enrich.setdefault(uid, {}).update(
            {
                "flood_type": row.get("flood_type"),
                "flood_type_confidence": row.get("confidence"),
                "flood_type_evidence": row.get("evidence"),
            }
        )
    ner = load_sidecar(root / "ner.jsonl")
    for uid, row in ner.items():
        rivers = [e["text"] for e in row.get("entities", []) if e.get("label") == "RIVER"]
        if rivers:
            enrich.setdefault(uid, {}).setdefault("rivers", []).extend(rivers)
    if enrich:
        log.info("enrichment sidecars: flood_type=%d ner=%d", len(ft), len(ner))
    return enrich


# --------------------------------------------------------------------------
# Event index
# --------------------------------------------------------------------------


def index_events(cfg: dict, collection, model, select: str, years, limit, enrich) -> dict:
    """Stream, embed and upsert. Returns measured counters."""
    done = existing_ids(collection)
    log.info("collection=%s existing=%d select=%s", collection.name, len(done), select)

    batch_size = int(cfg["index"].get("upsert_batch", 2000))
    stats = Counter()
    buf: list[Event] = []
    started = time.time()

    def flush() -> None:
        if not buf:
            return
        texts = [ev.text_for_embedding() for ev in buf]
        vectors = embed_texts(model, cfg, texts)
        collection.upsert(
            ids=[ev.uid for ev in buf],
            embeddings=vectors.tolist(),
            documents=texts,
            metadatas=[ev.metadata() for ev in buf],
        )
        stats["embedded"] += len(buf)
        elapsed = max(time.time() - started, 1e-6)
        log.info("  upserted %d  (%.0f ev/s)", stats["embedded"], stats["embedded"] / elapsed)
        buf.clear()

    for ev in iter_events(cfg, years, select=select, limit=limit, skip_uids=done, enrich=enrich):
        stats["candidates"] += 1
        if not ev.text_for_embedding().strip():
            stats["empty_text"] += 1
            continue
        stats["flood_type_" + ev.flood_type] += 1
        stats["year_" + ev.year] += 1
        buf.append(ev)
        if len(buf) >= batch_size:
            flush()
    flush()
    stats["skipped_already_indexed"] = len(done)
    stats["seconds"] = time.time() - started
    return dict(stats)


def refresh_metadata(cfg: dict, collection, select: str, years, enrich) -> dict:
    """Re-attach sidecar metadata to rows that are already embedded. Only the
    metadata is written; the vectors are left untouched, which is why this is
    seconds rather than minutes."""
    done = existing_ids(collection)
    batch = int(cfg["index"].get("upsert_batch", 2000))
    ids: list[str] = []
    metas: list[dict] = []
    stats = Counter()
    for ev in iter_events(cfg, years, select=select, enrich=enrich):
        if ev.uid not in done:
            continue
        ids.append(ev.uid)
        metas.append(ev.metadata())
        stats["updated"] += 1
        if len(ids) >= batch:
            collection.update(ids=ids, metadatas=metas)
            log.info("  refreshed %d", stats["updated"])
            ids, metas = [], []
    if ids:
        collection.update(ids=ids, metadatas=metas)
    log.info("refreshed metadata on %d rows", stats["updated"])
    return dict(stats)


# --------------------------------------------------------------------------
# Chunk (passage) index
# --------------------------------------------------------------------------


def index_chunks(cfg: dict, model, select: str, years, limit, rebuild: bool) -> dict:
    """Second collection over article BODIES, cut into overlapping passages.

    The event collection answers "which flood is this"; the passage collection
    answers "which sentence says so", which is what the RAG answer step and any
    quote-level evidence need. Off by default - see chunking.enabled.
    """
    ch = cfg["chunking"]
    collection = open_collection(cfg, name=ch["collection"], rebuild=rebuild)
    done_uids = {i.split("#", 1)[0] for i in existing_ids(collection)}
    log.info("chunk collection=%s existing_articles=%d", collection.name, len(done_uids))

    stats = Counter()
    started = time.time()
    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    batch = int(cfg["index"].get("upsert_batch", 2000))

    def flush() -> None:
        if not ids:
            return
        vectors = embed_texts(model, cfg, docs)
        collection.upsert(ids=ids, embeddings=vectors.tolist(), documents=docs, metadatas=metas)
        stats["chunks"] += len(ids)
        elapsed = max(time.time() - started, 1e-6)
        log.info("  upserted %d chunks (%.0f chunk/s)", stats["chunks"], stats["chunks"] / elapsed)
        ids.clear()
        docs.clear()
        metas.clear()

    for ev in iter_events(cfg, years, select=select, limit=limit, skip_uids=done_uids):
        text = article_text(load_article(cfg, ev.year, ev.month, ev.article_id))
        if not text.strip():
            stats["body_missing"] += 1
            continue
        pieces = chunk_text(
            text,
            target_chars=int(ch["target_chars"]),
            overlap_chars=int(ch["overlap_chars"]),
            min_chars=int(ch["min_chars"]),
            max_chunks=int(ch["max_chunks_per_article"]),
        )
        stats["articles"] += 1
        for i, (start, end, piece) in enumerate(pieces):
            ids.append(f"{ev.uid}#{i}")
            docs.append(piece)
            metas.append(
                {
                    "uid": ev.uid,
                    "year": int(ev.year),
                    "month": ev.month,
                    "article_id": ev.article_id,
                    "chunk_index": i,
                    "char_start": start,
                    "char_end": end,
                    "title": ev.title[:300],
                }
            )
        if len(ids) >= batch:
            flush()
    flush()
    stats["seconds"] = time.time() - started
    stats["total_in_collection"] = collection.count()
    return dict(stats)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def build_report(
    cfg: dict,
    root: Path,
    stats: dict,
    chunk_stats: dict | None,
    meta_rows: list[dict],
    matrix: np.ndarray,
    select: str,
    years,
    corpus_total: int | None,
) -> Path:
    emb = cfg["embedding"]
    years_hist = Counter(str(r.get("year")) for r in meta_rows)
    type_hist = Counter(r.get("flood_type") or "Unknown" for r in meta_rows)
    dated = sum(1 for r in meta_rows if r.get("date_min"))
    located = sum(1 for r in meta_rows if r.get("locations"))
    rivers = sum(1 for r in meta_rows if r.get("rivers"))
    multi = sum(1 for r in meta_rows if (r.get("n_events") or 0) > 1)
    seconds = float(stats.get("seconds", 0.0))
    embedded = int(stats.get("embedded", 0))

    sections: list[tuple[str, list[str]]] = []
    sections.append(
        (
            "Run",
            md_table(
                ["setting", "value"],
                [
                    ["model", emb["model"]],
                    ["device", gpu_note(emb.get("device", "cuda"))],
                    ["max_seq_length", emb.get("max_seq_length")],
                    ["batch_size", emb.get("batch_size")],
                    ["normalised", emb.get("normalize", True)],
                    ["select", select],
                    ["years", " ".join(years) if years else "all"],
                    ["collection", cfg["index"]["collection"]],
                    ["distance", cfg["index"].get("space", "cosine")],
                ],
                align=["---", "---"],
            ),
        )
    )
    throughput = f"{embedded / seconds:.0f} events/s" if seconds > 0 and embedded else "-"
    counts_rows = [
        ["stage-5 records matching select", corpus_total if corpus_total is not None else "not counted"],
        ["already in the collection at start", stats.get("skipped_already_indexed", 0)],
        ["streamed this run", stats.get("candidates", 0)],
        ["skipped: empty representation", stats.get("empty_text", 0)],
        ["embedded this run", embedded],
        ["vectors in the collection now", len(meta_rows)],
        ["embedding dimension", matrix.shape[1] if matrix.size else 0],
        ["wall clock", fmt_duration(seconds)],
        ["throughput", throughput],
    ]
    sections.append(("Counts", md_table(["", "count"], counts_rows, align=["---", "---:"])))

    sections.append(
        (
            "Indexed records by year",
            md_table(["year", "vectors"], sorted(years_hist.items()), align=["---", "---:"]),
        )
    )
    sections.append(
        (
            "Coverage of the indexed representation",
            md_table(
                ["field", "records", "share"],
                [
                    [name, n, f"{100 * n / max(len(meta_rows), 1):.1f}%"]
                    for name, n in [
                        ["has a flood date", dated],
                        ["has a location", located],
                        ["has a river/water body", rivers],
                        ["article listed >1 sub-event", multi],
                    ]
                ],
                align=["---", "---:", "---:"],
            ),
        )
    )
    sections.append(
        (
            "Flood-type distribution in the index",
            md_table(
                ["flood_type", "records", "share"],
                [
                    [t, n, f"{100 * n / max(len(meta_rows), 1):.1f}%"]
                    for t, n in type_hist.most_common()
                ],
                align=["---", "---:", "---:"],
            )
            + (
                []
                if type_hist and set(type_hist) != {"Unknown"}
                else ["", "flood_type.py has not been run yet, so every record carries the default `Unknown`."]
            ),
        )
    )
    if chunk_stats:
        sections.append(
            (
                "Passage index (article bodies)",
                md_table(
                    ["", "count"],
                    [
                        ["articles chunked this run", chunk_stats.get("articles", 0)],
                        ["bodies missing on disk", chunk_stats.get("body_missing", 0)],
                        ["chunks embedded this run", chunk_stats.get("chunks", 0)],
                        ["chunks in the collection now", chunk_stats.get("total_in_collection", 0)],
                        ["wall clock", fmt_duration(float(chunk_stats.get("seconds", 0)))],
                    ],
                    align=["---", "---:"],
                ),
            )
        )
    return write_report(
        root / "build_index_report.md",
        "Embedding and vector index report",
        sections,
        preamble=(
            "Produced by `7_semantic_layer/build_index.py`. Every number below was measured by the run "
            "that wrote this file."
        ),
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Embed flood extractions into a Chroma collection")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--years", nargs="*")
    ap.add_argument("--select", choices=["verifiable", "flood", "all"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--rebuild", action="store_true", help="drop the collection before indexing")
    ap.add_argument("--force", action="store_true", help="alias for --rebuild")
    ap.add_argument("--refresh-metadata", action="store_true", help="re-attach sidecar metadata without re-embedding")
    ap.add_argument("--chunks", action="store_true", help="also build the passage index over article bodies")
    ap.add_argument("--no-count", action="store_true", help="skip the corpus pre-count (saves one pass)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    select = args.select or cfg["index"].get("select", "verifiable")
    root = output_root(cfg)
    rebuild = args.rebuild or args.force

    collection = open_collection(cfg, rebuild=rebuild)
    enrich = load_enrichment(root)

    corpus_total = None
    if not args.no_count and not args.limit:
        corpus_total = count_events(cfg, args.years, select=select)
        log.info("stage-5 records matching select=%s: %d", select, corpus_total)

    model = None
    if args.refresh_metadata:
        stats = refresh_metadata(cfg, collection, select, args.years, enrich)
        stats.setdefault("seconds", 0.0)
    else:
        model = load_embedder(cfg)
        stats = index_events(cfg, collection, model, select, args.years, args.limit, enrich)

    chunk_stats = None
    if args.chunks or cfg["chunking"].get("enabled"):
        if model is None:
            model = load_embedder(cfg, max_seq_length=cfg["chunking"].get("max_seq_length"))
        else:
            model.max_seq_length = cfg["chunking"].get("max_seq_length", 512)
        chunk_stats = index_chunks(cfg, model, select, args.years, args.limit, rebuild)

    matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
    meta_rows: list[dict] = []
    if cfg["index"].get("export_matrix", True):
        log.info("exporting dense matrix + events_meta.jsonl from the collection")
        matrix, _ids, meta_rows = export_matrix(collection, root)

    path = build_report(cfg, root, stats, chunk_stats, meta_rows, matrix, select, args.years, corpus_total)
    log.info(
        "done: %d vectors in %s, dim=%d, report -> %s",
        collection.count(), collection.name, matrix.shape[1] if matrix.size else 0, path.name,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
