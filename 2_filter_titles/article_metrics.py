"""Metrics over the translated_articles stage (the flood articles, fetched and
fully translated). Scans every article JSON and aggregates status, language and
translation coverage. Totals are exact, not sampled.

    C:\\darsh\\AI_MODELS\\translator_env\\Scripts\\python.exe article_metrics.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(r"C:\darsh\pipeline\translated_articles")


def all_files():
    for month_dir in sorted(ROOT.glob("2021_*")):
        yield month_dir.name, sorted((month_dir / "articles").glob("*.json"))


def main() -> None:
    started = time.perf_counter()

    per_month = Counter()
    fetch_status = Counter()        # top-level "status"
    http_status = Counter()
    trans_status = Counter()        # translation_meta.status
    src_lang = Counter()
    lang_source = Counter()
    title_source = Counter()
    chunks_hist = Counter()

    total = 0
    unreadable = 0
    have_translated_text = 0
    empty_translated_text = 0
    have_translated_title = 0
    chunks_sum = 0
    text_chars = 0
    trans_chars = 0

    file_lists = list(all_files())
    grand = sum(len(f) for _, f in file_lists)

    with tqdm(total=grand, unit=" files", unit_scale=True, desc="scanning") as bar:
        for month, files in file_lists:
            for path in files:
                bar.update(1)
                total += 1
                per_month[month] += 1
                try:
                    rec = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                except (ValueError, OSError):
                    unreadable += 1
                    continue

                fetch_status[rec.get("status", "<none>")] += 1
                http_status[rec.get("http_status", "<none>")] += 1

                tt = rec.get("translated_text")
                if isinstance(tt, str) and tt.strip():
                    have_translated_text += 1
                    trans_chars += len(tt)
                else:
                    empty_translated_text += 1

                if isinstance(rec.get("translated_title"), str) and rec["translated_title"].strip():
                    have_translated_title += 1

                txt = rec.get("text")
                if isinstance(txt, str):
                    text_chars += len(txt)

                meta = rec.get("translation_meta") or {}
                if isinstance(meta, dict):
                    trans_status[meta.get("status", "<none>")] += 1
                    src_lang[meta.get("source_language", "<none>")] += 1
                    lang_source[meta.get("language_source", "<none>")] += 1
                    title_source[meta.get("title_source", "<none>")] += 1
                    ch = meta.get("chunks")
                    if isinstance(ch, int):
                        chunks_sum += ch
                        bucket = ("1" if ch <= 1 else "2-5" if ch <= 5 else
                                  "6-15" if ch <= 15 else "16-40" if ch <= 40 else "40+")
                        chunks_hist[bucket] += 1
                else:
                    trans_status["<no-meta>"] += 1

    elapsed = time.perf_counter() - started

    def block(title, counter, top=None, total_for_pct=None):
        print("\n" + title)
        print("-" * len(title))
        base = total_for_pct or sum(counter.values()) or 1
        items = counter.most_common(top) if top else sorted(
            counter.items(), key=lambda kv: (-kv[1], str(kv[0])))
        for k, v in items:
            print("  %-28s %9d  %6.2f%%" % (str(k), v, v / base * 100))

    print("\n" + "=" * 60)
    print("TRANSLATED_ARTICLES METRICS")
    print("=" * 60)
    print("Articles (files)             : %d" % total)
    print("Unreadable / bad JSON        : %d" % unreadable)
    print("With non-empty translated_text: %d  (%.2f%%)" % (
        have_translated_text, have_translated_text / total * 100 if total else 0))
    print("Empty/missing translated_text : %d" % empty_translated_text)
    print("With translated_title         : %d  (%.2f%%)" % (
        have_translated_title, have_translated_title / total * 100 if total else 0))
    print("Avg source-text length        : %d chars" % (text_chars // total if total else 0))
    print("Avg translated-text length    : %d chars" % (
        trans_chars // have_translated_text if have_translated_text else 0))
    print("Total translation chunks      : %d  (avg %.1f/article)" % (
        chunks_sum, chunks_sum / total if total else 0))
    print("Scan time                     : %.1fs" % elapsed)

    block("Articles per month", per_month, total_for_pct=total)
    block("Fetch status (top-level 'status')", fetch_status, total_for_pct=total)
    block("HTTP status", http_status, top=12, total_for_pct=total)
    block("Translation status (meta.status)", trans_status, total_for_pct=total)
    block("Source language (top 20)", src_lang, top=20, total_for_pct=total)
    block("Language source", lang_source, total_for_pct=total)
    block("Title source", title_source, total_for_pct=total)
    block("Chunks per article", chunks_hist, total_for_pct=total)
    print("=" * 60)


if __name__ == "__main__":
    main()
