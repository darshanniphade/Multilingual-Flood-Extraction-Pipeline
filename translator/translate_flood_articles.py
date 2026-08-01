"""
translate_flood_articles.py
===========================

Entry point: translate every article referenced by the flood-event list into
English with NLLB-200-distilled-1.3B, fully offline, on a single CUDA GPU.

Pipeline
--------
::

    flood_events/*.jsonl ──► article ids (+ pre-translated titles)
                             │
                             ▼
                    article_id -> path index  (built once, cached)
                             │
                             ▼
      ProcessPool workers ──► load JSON, detect language, split into
      (N CPU cores)           sentence-level chunks, count tokens
                             │  (prefetched: overlaps with GPU)
                             ▼
      GPU (fp16, CUDA)  ───► group by language, sort by length, dynamic
                             token-budget batches, greedy decode
                             │
                             ▼
                    reassemble ──► write JSON (all original fields + translated_*)
                             │
                             ▼
                    checkpoint journal (resume-safe)

Run ``python translate_flood_articles.py --help`` for options.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from checkpoint import CheckpointManager
from dataset import (
    ArticleLoader,
    PreparedArticle,
    build_index,
    build_tasks,
    chunks_from_block,
    load_event_ids,
)
from logger import ArticleRecord, RecordLogger, setup_logging
from utils import (
    atomic_write_json,
    human_count,
    human_time,
    load_config,
    validate_language_tables,
)

log = logging.getLogger("translator")

_INTERRUPTED = False


def _install_signal_handler() -> None:
    """Turn Ctrl+C into a graceful stop at the next block boundary."""

    def handler(signum: int, frame: Any) -> None:
        global _INTERRUPTED
        if _INTERRUPTED:
            log.warning("second interrupt -- exiting immediately")
            sys.exit(130)
        _INTERRUPTED = True
        log.warning("interrupt received -- finishing current block, then checkpointing. "
                    "Press Ctrl+C again to abort now.")

    try:
        signal.signal(signal.SIGINT, handler)
    except (ValueError, OSError):  # pragma: no cover - non-main thread
        pass


# --------------------------------------------------------------------------- #
# Output assembly
# --------------------------------------------------------------------------- #


def assemble(art: PreparedArticle, translations: dict[tuple[str, int], str],
             cfg: dict[str, Any]) -> dict[str, Any]:
    """Build the output record: every original field, plus translated fields.

    The source dict is copied and only *added to* -- no original key is ever
    removed or overwritten, which is what "preserve every original field"
    requires.
    """
    out = dict(art.data)  # preserve everything, in original order

    text_key = cfg["output"].get("translated_text_field", "translated_text")
    title_key = cfg["output"].get("translated_title_field", "translated_title")
    sep = cfg["output"].get("chunk_join", " ")

    if art.status == "skipped_english":
        # Already English: copy through verbatim rather than round-tripping it
        # through the model, which costs GPU time and can only degrade the text.
        out[text_key] = art.data.get(cfg["text_field"], "") or ""
        out[title_key] = art.translated_title or art.data.get(cfg["title_field"], "") or ""
    elif art.status == "failed":
        # Parsed fine but untranslatable (e.g. language undetermined). The file
        # is still written so the output mirrors the input set exactly; the
        # empty translated_text plus translation_meta.status makes it greppable
        # rather than silently missing.
        out[text_key] = ""
        out[title_key] = art.translated_title or ""
    else:
        parts = [translations.get(("text", i), "") for i in range(len(art.text_chunks))]
        out[text_key] = sep.join(p for p in parts if p).strip()
        if art.translated_title:
            out[title_key] = art.translated_title
        elif art.title_chunks:
            tparts = [translations.get(("title", i), "") for i in range(len(art.title_chunks))]
            out[title_key] = sep.join(p for p in tparts if p).strip()
        else:
            out[title_key] = art.data.get(cfg["title_field"], "") or ""

    if cfg["output"].get("add_metadata", True):
        meta: dict[str, Any] = {
            "source_language": art.lang,
            "language_source": art.lang_source,
            "target_language": "eng_Latn",
            "model": cfg["model"].get("name", "facebook/nllb-200-distilled-1.3B"),
            "chunks": len(art.text_chunks),
            "status": art.status,
            "title_source": "event_file" if art.translated_title else "translated",
        }
        if art.error:
            meta["error"] = art.error
        out[cfg["output"].get("metadata_field", "translation_meta")] = meta
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def run(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Execute the full translation run. Returns a process exit code."""
    t_start = time.time()
    paths = cfg["paths"]
    state_dir = Path(paths["state_dir"])
    state_dir.mkdir(parents=True, exist_ok=True)

    # Fail loudly on language-table regressions: a missing entry does not crash,
    # it silently marks every article of that language untranslatable.
    for problem in validate_language_tables():
        log.error("language table: %s", problem)

    # ---- 1. event ids ------------------------------------------------------ #
    ids_to_titles = load_event_ids(paths["events_dir"])
    if args.limit:
        ids_to_titles = dict(list(sorted(ids_to_titles.items()))[: args.limit])
        log.info("--limit: restricted to %d article(s)", len(ids_to_titles))

    # ---- 2. index (built once, cached) ------------------------------------- #
    index, lang_hints = build_index(
        list(ids_to_titles),
        paths["data_dir"],
        cache_path=state_dir / "article_index.json",
        articles_subdir=cfg["dataset"].get("articles_subdir", "articles"),
        workers=int(cfg["cpu"].get("index_workers", 16)),
        rebuild=args.rebuild_index,
        with_languages=cfg["dataset"].get("sort_by_language", True),
    )

    # ---- 3. checkpoint / resume ------------------------------------------- #
    cp = CheckpointManager(
        state_dir / "checkpoint.jsonl",
        interval=int(cfg["checkpoint"].get("every_n_articles", 500)),
    )
    done = set() if args.no_resume else cp.load()
    if args.no_resume:
        log.warning("--no-resume: ignoring existing checkpoint")

    tasks, unresolved = build_tasks(
        ids_to_titles, index, paths["output_dir"], paths["data_dir"], done,
        lang_hints=lang_hints if cfg["dataset"].get("sort_by_language", True) else None,
    )
    if unresolved:
        log.warning("%d event id(s) have no article file and will be skipped", len(unresolved))
        try:
            (state_dir / "unresolved_ids.json").write_text(
                json.dumps(unresolved, indent=2), encoding="utf-8")
        except OSError:
            pass

    total = len(tasks)
    if not total:
        log.info("nothing to do: all %d article(s) already translated", len(ids_to_titles))
        cp.close()
        return 0

    log.info("plan: %s article(s) to translate (%s already done, %s total in event list)",
             human_count(total), human_count(len(done)), human_count(len(ids_to_titles)))

    if args.dry_run:
        log.info("--dry-run: stopping before model load")
        cp.close()
        return 0

    # ---- 4. model ---------------------------------------------------------- #
    from translator import NLLBTranslator  # imported late: keeps --dry-run torch-free

    gpu_cfg = {**cfg["gpu"], **cfg["generation"],
               "chunk_max_tokens": cfg["chunking"]["max_tokens"]}
    tr = NLLBTranslator(cfg["model"]["dir"], gpu_cfg)
    tr.warmup()

    # ---- 5. loader --------------------------------------------------------- #
    worker_cfg = {
        "text_field": cfg["text_field"],
        "title_field": cfg["title_field"],
        "language_field": cfg["language_field"],
        "skip_english": cfg["dataset"].get("skip_english", True),
        "trust_metadata_language": cfg["dataset"].get("trust_metadata_language", True),
        "repair_mojibake": cfg["dataset"].get("repair_mojibake", True),
        "chunk_target_tokens": cfg["chunking"]["target_tokens"],
        "chunk_max_tokens": cfg["chunking"]["max_tokens"],
        "max_article_chars": cfg["dataset"].get("max_article_chars", 0),
    }
    loader = ArticleLoader(
        tasks, cfg["model"]["dir"], worker_cfg,
        block_size=int(cfg["cpu"].get("block_size", 2000)),
        workers=int(cfg["cpu"].get("workers", 12)),
        prefetch=int(cfg["cpu"].get("prefetch_blocks", 2)),
    )

    reclog = RecordLogger(state_dir / "translation_records.jsonl")
    _install_signal_handler()

    # ---- 6. main loop ------------------------------------------------------ #
    import psutil  # local: only needed for live metrics
    proc = psutil.Process()

    done_articles = skipped = failed = corrupt = 0
    src_tok_total = out_tok_total = 0
    gpu_seconds = 0.0
    lang_counter: Counter[str] = Counter()
    written = 0
    # Tracked explicitly rather than read from tqdm: a disabled bar (--no-progress
    # or a non-tty) never advances its own counter, which would zero the ETA.
    processed = 0

    bar = tqdm(total=total, unit="art", dynamic_ncols=True, smoothing=0.05,
               desc="translate", disable=args.no_progress)
    try:
        for block in loader:
            block_t0 = time.time()
            chunks = chunks_from_block(block)
            batches = tr.plan(chunks)

            if batches:
                # Batch width is the single best predictor of throughput on this
                # model, and it is driven by how many languages the block spans.
                # Logging it makes a slow run diagnosable instead of mysterious.
                sizes = sorted(len(b) for b in batches)
                log.info(
                    "block plan: %s chunk(s) | %d langs | %d batches | "
                    "size min/med/mean/max %d/%d/%.0f/%d",
                    human_count(len(chunks)), len({c.lang for c in chunks}), len(batches),
                    sizes[0], sizes[len(sizes) // 2], sum(sizes) / len(sizes), sizes[-1],
                )

            # chunk-key -> translated text, per article
            results: dict[int, dict[tuple[str, int], str]] = {}
            block_src = block_out = 0
            last_bs = 0
            last_lang = "-"

            for batch in batches:
                if not batch:
                    continue
                res = tr.translate_batch(batch)
                gpu_seconds += res.seconds
                block_src += res.src_tokens
                block_out += res.out_tokens
                last_bs = res.batch_size
                last_lang = batch[0].lang
                for ch, txt in zip(batch, res.texts):
                    results.setdefault(ch.article_idx, {})[(ch.field, ch.order)] = txt

                alloc, peak = tr.vram()
                bar.set_postfix_str(
                    f"{last_lang} bs={last_bs} kv={tr.slot_budget // 1000}k "
                    f"vram={alloc:.1f}/{peak:.1f}G "
                    f"tok/s={human_count((out_tok_total + block_out) / max(1e-6, gpu_seconds))} "
                    f"ram={proc.memory_info().rss / 1024 ** 3:.1f}G",
                    refresh=False,
                )

            # ---- write out ------------------------------------------------- #
            recs: list[ArticleRecord] = []
            marks: list[tuple[str, str]] = []
            for i, art in enumerate(block):
                rec = ArticleRecord(
                    article_id=art.article_id, language=art.lang,
                    lang_source=art.lang_source, status=art.status,
                    chunks=len(art.text_chunks) + len(art.title_chunks),
                    src_tokens=art.n_src_tokens, batch_size=last_bs,
                    gpu_mem_gib=round(tr.vram()[1], 2), error=art.error,
                )
                # A corrupt file could not be parsed at all -- there is nothing to
                # write. Everything else (including untranslatable articles) is
                # written so the output set mirrors the input set.
                if art.status == "corrupt":
                    corrupt += 1
                    processed += 1
                    log.debug("skip %s: corrupt (%s)", art.article_id, art.error)
                    recs.append(rec)
                    marks.append((art.article_id, art.status))
                    bar.update(1)
                    continue

                try:
                    payload = assemble(art, results.get(i, {}), cfg)
                    atomic_write_json(art.out_path, payload,
                                      indent=cfg["output"].get("json_indent", 2))
                    written += 1
                except OSError as exc:
                    rec.status, rec.error = "failed", f"write failed: {exc}"
                    failed += 1
                    processed += 1
                    log.error("write failed for %s: %s", art.article_id, exc)
                    recs.append(rec)
                    marks.append((art.article_id, "failed"))
                    bar.update(1)
                    continue

                if art.status == "skipped_english":
                    skipped += 1
                elif art.status == "failed":
                    failed += 1
                    log.debug("untranslated %s: %s", art.article_id, art.error)
                else:
                    done_articles += 1
                    lang_counter[art.lang or "?"] += 1
                processed += 1
                recs.append(rec)
                marks.append((art.article_id, art.status))
                bar.update(1)

            reclog.log_many(recs)
            cp.mark_many(marks)
            cp.stats.src_tokens += block_src
            cp.stats.out_tokens += block_out
            cp.stats.gpu_seconds = gpu_seconds
            if cp.maybe_flush():
                reclog.flush()

            src_tok_total += block_src
            out_tok_total += block_out
            elapsed = time.time() - t_start
            rate = processed / max(1e-6, elapsed)
            log.info(
                "block done: %d article(s) in %.1fs | %s src-tok, %s out-tok | "
                "%d/%d | %.1f art/s | %s out-tok/s | ETA %s | oom=%d",
                len(block), time.time() - block_t0, human_count(block_src),
                human_count(block_out), processed, total, rate,
                human_count(out_tok_total / max(1e-6, gpu_seconds)),
                human_time((total - processed) / max(1e-6, rate)), tr.oom_count,
            )

            if _INTERRUPTED:
                log.warning("stopping early at user request")
                break
    finally:
        bar.close()
        loader.close()
        cp.maybe_flush(force=True)
        cp.close()
        reclog.close()
        try:
            tr.close()
        except Exception:  # pragma: no cover
            pass

    # ---- 7. summary -------------------------------------------------------- #
    elapsed = time.time() - t_start
    log.info("=" * 72)
    log.info("run complete in %s", human_time(elapsed))
    log.info("  translated        : %s", human_count(done_articles))
    log.info("  copied (English)  : %s", human_count(skipped))
    log.info("  failed            : %s", human_count(failed))
    log.info("  corrupt/skipped   : %s", human_count(corrupt))
    log.info("  files written     : %s", human_count(written))
    log.info("  source tokens     : %s", human_count(src_tok_total))
    log.info("  output tokens     : %s", human_count(out_tok_total))
    if gpu_seconds > 0:
        log.info("  gpu throughput    : %s out-tok/s (%.0f%% of wall clock on GPU)",
                 human_count(out_tok_total / gpu_seconds), 100 * gpu_seconds / max(1e-6, elapsed))
    if lang_counter:
        top = ", ".join(f"{k}={v}" for k, v in lang_counter.most_common(8))
        log.info("  top languages     : %s", top)
    log.info("  oom events        : %d (all recovered)", tr.oom_count)
    log.info("  output dir        : %s", paths["output_dir"])
    log.info("=" * 72)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        prog="translate_flood_articles",
        description="Translate flood-event articles to English with NLLB-200 (offline, CUDA).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-c", "--config", default=str(Path(__file__).parent / "config.json"),
                   help="path to config.json")
    p.add_argument("--limit", type=int, default=0,
                   help="translate at most N articles (smoke test)")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve ids and build the index, then stop before loading the model")
    p.add_argument("--no-resume", action="store_true",
                   help="ignore the checkpoint and re-translate everything")
    p.add_argument("--rebuild-index", action="store_true",
                   help="force a rebuild of the article_id -> path index")
    p.add_argument("--no-progress", action="store_true", help="disable the progress bar")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    # Convenience overrides for the knobs most worth tuning per machine.
    p.add_argument("--kv-slot-budget", type=int,
                   help="override gpu.kv_slot_budget (batch x (src+generated) tokens)")
    p.add_argument("--workers", type=int, help="override cpu.workers")
    p.add_argument("--block-size", type=int, help="override cpu.block_size")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    cfg = load_config(args.config)

    if args.kv_slot_budget:
        cfg["gpu"]["kv_slot_budget"] = args.kv_slot_budget
    if args.workers:
        cfg["cpu"]["workers"] = args.workers
    if args.block_size:
        cfg["cpu"]["block_size"] = args.block_size

    Path(cfg["paths"]["state_dir"]).mkdir(parents=True, exist_ok=True)
    setup_logging(cfg["paths"].get("log_file", "translation.log"), level=args.log_level)

    log.info("=" * 72)
    log.info("flood article translator | %s -> English", cfg["model"].get("name"))
    log.info("  events : %s", cfg["paths"]["events_dir"])
    log.info("  data   : %s", cfg["paths"]["data_dir"])
    log.info("  output : %s", cfg["paths"]["output_dir"])
    log.info("  model  : %s", cfg["model"]["dir"])
    log.info("=" * 72)

    try:
        return run(cfg, args)
    except KeyboardInterrupt:  # pragma: no cover
        log.warning("interrupted -- progress is checkpointed; rerun to resume")
        return 130
    except Exception as exc:
        log.exception("fatal: %s", exc)
        return 1


if __name__ == "__main__":
    # Required on Windows: worker processes re-import this module under spawn.
    import multiprocessing

    multiprocessing.freeze_support()
    sys.exit(main())
