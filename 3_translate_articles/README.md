# Flood Article Translator

Translates every article referenced by the flood-event list into English using
**facebook/nllb-200-distilled-1.3B**, fully offline, on a single CUDA GPU.

Built and calibrated against this exact machine: **RTX 4060 Ti (16 GB), 26-core
CPU, 512 GB RAM, Windows 10 / WDDM, Python 3.14, torch 2.11+cu128,
transformers 5.13**.

---

## Quick start

```bat
run.bat                    :: translate everything (auto-resumes)
run.bat --limit 200        :: smoke test on 200 articles
run.bat --dry-run          :: resolve ids + build the index, no model load
run.bat --no-resume        :: ignore the checkpoint and start over
```

Ctrl+C is safe: the current block finishes, progress is checkpointed, and the
next run resumes exactly where it stopped.

---

## What this actually does

| Stage | Where | Notes |
|---|---|---|
| Read flood events | `dataset.load_event_ids` | 12 JSONL files → **233,153 unique article ids** |
| Resolve id → path | `dataset.build_index` | derived from the id, **cached**; no dataset walk |
| Language hint scan | `dataset.scan_languages` | byte-regex, cached; drives batch ordering |
| Load / detect / chunk | `dataset.prepare_article` | in a process pool, prefetched |
| Translate | `translator.NLLBTranslator` | fp16, CUDA, KV-budgeted dynamic batches |
| Write | `translate_flood_articles.assemble` | all original fields + `translated_*` |
| Checkpoint | `checkpoint.CheckpointManager` | append-only journal, crash-safe |

### Input → output

Input `C:\darsh\pipeline\1_translate_titles\data\2021_01\articles\article_000000027.json`
→ output `C:\darsh\pipeline\translated_articles\2021_01\articles\article_000000027.json`

Folder structure and filenames are identical. **Every original field is
preserved byte-for-byte**; the pipeline only *adds* keys:

```jsonc
{
  "article_id": "article_000000027",   // ← all original fields, untouched
  "url": "...", "title": "...", "author": "...", "publish_date": "...",
  "language": "es", "text": "...", "status": "success", "http_status": 200,
  "final_url": "...",

  "translated_title": "...",           // ← added
  "translated_text": "...",            // ← added
  "translation_meta": {                // ← added (set add_metadata:false to omit)
    "source_language": "spa_Latn",
    "language_source": "metadata",     // metadata | langdetect
    "target_language": "eng_Latn",
    "model": "facebook/nllb-200-distilled-1.3B",
    "chunks": 12,
    "status": "ok",                    // ok | skipped_english | failed
    "title_source": "event_file"       // event_file | translated
  }
}
```

Writes are atomic (temp file + `os.replace`), so a kill mid-write can never
leave a half-written article.

---

## Design decisions worth knowing

These are the non-obvious ones. Each was driven by a measurement on this
machine, and several contradict the intuitive approach.

### 1. Filling VRAM makes it *slower*, not faster

The brief asked to "use nearly all available VRAM". **On this machine that is
actively harmful**, so the pipeline deliberately does not.

Under Windows' WDDM driver model, an oversized allocation does **not** OOM — the
NVIDIA driver silently spills it into host RAM over PCIe. Measured:

| Batch | Peak "VRAM" | Throughput |
|---|---|---|
| 512 × 76 tok | 8.97 GiB | **5,509 out-tok/s** |
| 1024 × 76 tok | 15.38 GiB | 2,911 out-tok/s |
| 1536 × 76 tok | 21.78 GiB (on a 16 GB card) | **617 out-tok/s** — 9× slower |

Nothing crashed. It just quietly ran at a fraction of the speed. So the target
is the **throughput knee (~7.5–9.5 GiB peak)**, not maximum occupancy.

### 2. `set_per_process_memory_fraction` is what makes OOM recovery real

Because the driver spills instead of raising, "reduce batch size on OOM" cannot
work by default — there is no OOM to catch. Capping the allocator at
`memory_fraction: 0.88` restores a genuine, catchable `OutOfMemoryError`.
Verified: after a caught OOM the model still produces correct output, so the
pipeline halves the batch, empties the cache, and continues.

### 3. The batch budget counts KV slots, not source tokens

This is the subtle one. Peak memory on an encoder-decoder is dominated by the
attention KV cache, which scales with `batch × (src_len + generated_len)` —
**not** by source length. Budgeting on source tokens alone lets a wide batch of
short chunks look cheap while reserving an enormous decode cache. That is
exactly what produced **19 OOM events** in an early 3,000-article run.

The fix (`utils.batch_kv_slots`) budgets on:

```
slots     = batch_size × (max_src_tokens + projected_max_new_tokens)
peak_GiB ≈ 2.56 (fp16 weights) + slots / 16384
```

Fitted against 15 measured runs spanning 4.2–12.6 GiB: **mean absolute error
0.54 GiB**. Throughput peaks near 72–95k slots and does *not* improve past it
(6,883 tok/s at 143k slots vs 6,935 at 72k), so `max_kv_slot_budget` sits at the
knee, not at the VRAM limit.

### 4. Articles are ordered by language — this is a throughput feature

NLLB encodes the source language as a prefix token, so **a batch cannot mix
languages**. This corpus spans ~55 languages. Processing in id order shatters
every block into dozens of tiny per-language batches, and small batches are
catastrophic here:

| Chunks per batch | Throughput |
|---|---|
| ~22 (id order, 200-article blocks) | 550 out-tok/s |
| ~500 (language-sorted) | ~6,900 out-tok/s |

So `scan_languages` reads each article's declared `language` with a byte-level
regex (no full JSON parse), caches it in the index, and the work list is sorted
by language. The hint only affects *ordering* — real detection still runs per
article, so a wrong hint costs a little batch efficiency, never correctness.

### 5. English articles are copied, not translated

**~39% of this corpus is already English.** Round-tripping English through NLLB
would cost hours of GPU time and could only degrade the text, so those articles
are copied through verbatim (`skip_english: true`) and marked
`status: "skipped_english"`.

That skip rests on the corpus's own `language` field, so the field was
validated rather than assumed. Over a 4,000-article sample, metadata and
`langdetect` agree at the FLORES level **91.4%** of the time, and on the
decision that actually matters — "is this English?" — they disagree on only
**2.38%** of English-tagged articles. **28 of those 35 disagreements are
`langdetect` reporting Afrikaans on English text**, a well-known false positive
of that library. Genuine disagreement is therefore ~0.5%, so
`trust_metadata_language: true` is the default and detection runs only when the
field is missing or unmappable (~1.4% of the corpus). Set it to `false` to force
`langdetect` on every article.

### 6. Titles come from the event files for free

Every one of the 233,153 event records already carries a `translated_title` from
the upstream title pipeline. Those are reused; a title is only translated when
the event file has none (`title_source` records which happened).

### 7. Chunking is sentence-level, and that is both a quality and a speed win

NLLB is a *sentence-level* model whose quality degrades on long inputs, and short
sequences also batch far better: **48-token chunks run ~1.8× faster per token
than 192-token chunks** (6,935 vs 3,880 out-tok/s). Text is split on sentence
boundaries (multilingual terminators: `.!?` `。！？` `؟` `۔` `।` …, with an
abbreviation guard), packed to ~64 tokens, and reassembled after translation.

### 8. Model input is repaired for mojibake; stored fields are not

Parts of the corpus contain UTF-8-decoded-as-CP1252 artefacts (`itâ€™s`). Those
waste tokens and hurt translation, so the text *fed to the model* is repaired —
only when the repair round-trips cleanly. The **preserved original fields are
never modified**.

### 9. The language tables are validated against the checkpoint, not from memory

The FLORES code list was extracted from the local tokenizer, because the
published FLORES-200 list and this checkpoint genuinely disagree (it has
`als_Latn`/`sat_Beng`; it lacks `arb_Latn`/`min_Arab`/`sat_Olck`).
`validate_language_tables()` runs at startup and asserts that every code
`langdetect` can emit is mappable — a missing `"es"` entry once silently marked
**~15,700 Spanish articles** untranslatable, which is precisely the failure mode
that check exists to prevent.

---

## Files

| File | Role |
|---|---|
| `translate_flood_articles.py` | entry point + orchestration |
| `translator.py` | model load, KV-budgeted batching, OOM recovery |
| `dataset.py` | events, index, language scan, workers, prefetch loader |
| `utils.py` | language tables, chunking, batching math, helpers |
| `checkpoint.py` | append-only resume journal |
| `logger.py` | `translation.log` + per-article JSONL records |
| `config.json` | all tuning knobs |
| `run.bat` | Windows launcher (sets offline env vars) |

## Outputs

| Path | Contents |
|---|---|
| `C:\darsh\pipeline\translated_articles\` | translated articles, mirrored structure |
| `translation.log` | human-readable log (rotating) |
| `state\translation_records.jsonl` | per-article: id, language, time, retries, GPU mem, batch size, errors |
| `state\checkpoint.jsonl` | resume journal (completed ids) |
| `state\checkpoint.json` | live summary counters |
| `state\article_index.json` | cached id → path + language hints |
| `state\unresolved_ids.json` | event ids with no article file |

Progress bar shows: articles/s, ETA, out-tok/s, current language, batch size, KV
budget, VRAM (live/peak) and process RAM.

---

## Tuning

Most users should change nothing. If you do:

| Knob | Default | Effect |
|---|---|---|
| `gpu.kv_slot_budget` | 90000 | main throughput/VRAM dial. ~`2.56 + slots/16384` GiB peak |
| `gpu.max_kv_slot_budget` | 120000 | ceiling for adaptive growth — at the measured knee |
| `gpu.memory_fraction` | 0.88 | allocator cap; **keep < 1.0** or OOM recovery silently breaks (§2) |
| `gpu.max_batch_size` | 512 | hard cap on sequences per batch |
| `cpu.block_size` | 6000 | batching pool size — bigger ⇒ fuller batches |
| `cpu.workers` | 16 | preprocessing processes (each loads a tokenizer) |
| `chunking.target_tokens` | 64 | shorter ⇒ faster + better NLLB quality |
| `generation.num_beams` | 1 | greedy. `4` is much slower for modest gain |
| `dataset.skip_english` | true | set false to force English through the model |

**On a different GPU**, re-run the calibration: peak VRAM ≈ `2.56 + slots/16384`
holds for NLLB-1.3B fp16, so scale `kv_slot_budget` to
`(target_peak_GiB − 2.56) × 16384` and confirm the throughput knee empirically.

---

## Error handling

Nothing here is fatal to the run:

| Situation | Behaviour |
|---|---|
| Corrupt / unparseable article | logged, skipped, marked `corrupt`; no output file |
| Language undetermined | file **is** written with all fields + empty `translated_text`, `status: "failed"` |
| CUDA OOM | batch halved, cache cleared, retried down to a single chunk |
| Single chunk that will not fit | that chunk yields `""`; the article still writes |
| Event id with no article file | recorded in `state\unresolved_ids.json` |
| Worker pool death | falls back to in-process preprocessing |
| Ctrl+C | finishes the block, checkpoints, exits 130 |

## Requirements

The target env (`C:\darsh\AI_MODELS\translator_env`) already satisfies
everything. `requirements.txt` documents the calibrated versions — **do not
`pip install -r` it blindly**, as a plain `torch` install would replace the CUDA
build with the CPU wheel and the pipeline would refuse to start (CUDA is
required by design).

The model must exist at `C:\darsh\AI_MODELS\model`. Nothing is ever downloaded:
`local_files_only=True` is forced, `HF_HUB_OFFLINE=1` is set, and a missing
checkpoint raises immediately with a clear message.
