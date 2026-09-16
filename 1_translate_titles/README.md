# Title Translator

Translates **only** the `title` field of an article dataset into English using a
local NLLB model (GPU if available, else CPU) and writes three fields per
article to `translated_titles.jsonl`:

```json
{
  "article_id": "2022_01/article_000094047",
  "original_title": "Nhật Bản xây dựng các khu bảo tồn chim ngay trong thành phố",
  "translated_title": "Japan builds bird sanctuaries right in the city"
}
```

No keyword matching, filtering, or classification happens here — that is left
to a later stage which must search **only** on `translated_title`.

> **Note on `article_id`:** the raw `article_id` restarts numbering in every
> month folder (so `article_000000002` names a *different* article in every
> month). It is therefore prefixed with the dataset year and the month folder —
> `2022_01/article_000000002` — so every record is unique and traceable back to
> its source file, and so resume works directly from the output file.

## Configuration

All settings live in [config.json](config.json) next to the program. Point it at
a dataset and an output folder and run — no code changes needed.

```json
"paths": {
  "input_dir":       "C:/darsh/pipeline/data/ssd/2023",
  "output_dir":      "C:/darsh/pipeline/data/translated titles/2023",
  "output_file":     "translated_titles.jsonl",
  "articles_subdir": "articles"
}
```

| Section      | Key                         | Default                             | Meaning                                                         |
| ------------ | --------------------------- | ----------------------------------- | --------------------------------------------------------------- |
| `paths`      | `input_dir`                 | `.../data/ssd/2023`                 | Dataset root, scanned recursively for `.json` / `.jsonl`         |
|              | `output_dir`                | `.../translated titles/2023`        | Created if missing                                               |
|              | `output_file`               | `translated_titles.jsonl`           | Output JSONL — also the resume checkpoint                        |
|              | `articles_subdir`           | `articles`                          | Only read files under folders of this name (`""` = scan all)      |
| `article_id` | `prefix`                    | `auto`                              | `auto` = the input folder name when it is a year (`2023`)        |
|              | `separator`                 | `_`                                 | Joins prefix and month folder → `2022_01`                        |
| `fields`     | `id` / `title` / `language` | `article_id` / `title` / `language` | Record keys read from each source file                           |
| `language`   | `detect_when_missing`       | `true`                              | Detect the source language from the title when none is declared  |
|              | `detect_min_chars`          | `10`                                | Shorter titles are copied through instead of guessed at          |
| `model`      | `name`                      | `facebook/nllb-200-distilled-1.3B`  | Local translation model                                          |
|              | `target_language`           | `eng_Latn`                          | FLORES-200 target code                                           |
| `gpu`        | `kv_slot_budget`            | `110000`                            | VRAM budget in KV-cache slots — the real batch-size knob         |
|              | `max_batch_size`            | `1536`                              | Fragmentation guard on the logits buffer, not a throughput knob  |
|              | `attn_implementation`       | `sdpa`                              | Fused attention kernels                                          |
| `cpu`        | `readers`                   | `24`                                | Threads parsing the (tiny, numerous) json files                  |
|              | `detect_workers`            | `20`                                | **Processes** running language detection                         |
|              | `detect_chunk`              | `4096`                              | Titles per detection round trip                                  |
| `runtime`    | `translate_chunk`           | `16384`                             | Titles pooled and length-sorted before a translation pass        |
|              | `buffer_queue`              | `3`                                 | Ready pools kept queued for the GPU thread                       |
|              | `limit`                     | `null`                              | Stop after ~N new articles (quick test run)                      |

### Tuning for your GPU

Batches are sized by **KV-cache budget**, not by a fixed count. One slot is one
cached token = `decoder_layers × 2 × d_model × 2` bytes = **96 KiB** for the
1.3B model, so peak VRAM ≈ `2.6 GB (weights) + kv_slot_budget × 96 KiB`. The
default 110000 targets ~10.5 GB of cache on a 16 GB card. Because the pool is
length-sorted first, a batch of 20-token headlines ends up several times wider
than a batch of 128-token ones while both peak at the same VRAM.

- Peak VRAM sitting well below ~14 GB → raise `kv_slot_budget`.
- `oom batch splits` showing up in the summary → lower it.

## Run

```powershell
cd C:\darsh\pipeline\1_translate_titles
C:\darsh\AI_MODELS\translator_env\Scripts\python.exe translate_titles.py
```

The env already has every dependency (including CUDA torch). To rebuild it:

```powershell
C:\darsh\AI_MODELS\translator_env\Scripts\python.exe -m pip install -r requirements.txt
```

### One-off overrides

Flags override `config.json` for a single run; everything else still comes from
the file.

```powershell
translate_titles.py --input-dir "C:\darsh\pipeline\data\ssd\2023" `
                    --output-dir "C:\darsh\pipeline\data\translated titles\2023" --limit 500
```

`--config --input-dir --output-dir --output-file --articles-subdir --model
--kv-budget --batch-size --readers --detect-workers --limit --no-detect`

## The model

`facebook/nllb-200-distilled-1.3B` — Meta's **NLLB-200** ("No Language Left
Behind"), a 1.3B-parameter model covering 200 languages, distilled for speed.
~5.5 GB on disk, ~2.6 GB VRAM. It is loaded strictly from the local HuggingFace
cache (offline), so there are no network round-trips at run time.

## Behaviour

- Reads every `.json` / `.jsonl` file recursively (BOM-tolerant), restricted to
  the `articles_subdir` folders — the ssd month folders also carry crawler
  bookkeeping (`progress.json`, `dns_cache.json`, `metrics/`, `logs/`) which is
  json but not article data. A `.json` file may hold one article or a list;
  `.jsonl` is one article per line.
- Month folders that already carry the year (`2023/2023_01`) are not prefixed
  twice; ids stay `2023_01/article_000000001` as in the 2022 run.
- Only `title` is translated. All other fields are ignored.
- **Source language:** the record's `language` field is trusted first, mapped to
  NLLB's FLORES-200 codes. The `ssd/2022` export carries **no** `language` field
  (records hold only `article_id`, `title`, `text`), so the language is detected
  from the title text itself via `langdetect` — without this every title would
  be copied through untranslated.
- Titles that are already English, empty, too short to detect, or in an
  unrecognised language are copied through unchanged (kept searchable).
- **GPU auto-detect:** CUDA with bf16/fp16 when available, else CPU/fp32.
- **Fault tolerant:** unreadable files and malformed records are logged and
  skipped; a failed translation batch falls back to the original titles.

## Performance design

The goal is a permanently busy GPU, so every stage that is *not* the GPU runs
somewhere else:

- **Reading** on a pool of threads — the files are tiny and numerous, so this
  stage is disk-bound and threads are enough.
- **Language detection in separate processes.** This is the CPU-heavy stage and
  `langdetect` is pure Python: run on threads the GIL serialises it and starves
  the card (measured at ~5% GPU utilisation). A `ProcessPoolExecutor` sidesteps
  the GIL and puts every core to work; readers hand over `detect_chunk` titles
  per round trip, and block on the result — which releases the GIL.
- **Translation on its own thread**, fed by a small queue of ready buffers, so
  the GPU keeps working while the next buffer is being assembled.
- **Cross-language batching:** titles of different source languages are packed,
  length-sorted, into one padded batch and translated in a single pass (NLLB
  shares one English target). This roughly doubles throughput versus one batch
  per language.
- **KV-budget batching** (see *Tuning* above) keeps VRAM full whether titles are
  short or long, with automatic splitting on out-of-memory as the safety net.
- **Background writer thread:** finished records are handed off through a queue
  and written on a separate thread, so translation never stalls to save.

`torch` and `transformers` are imported lazily, inside the parent process only —
on Windows every detection worker re-imports this module, and keeping the heavy
imports out of module scope means those workers start in milliseconds and never
touch CUDA.

## Checkpointing & resume

- The output JSONL *is* the checkpoint. Records are flushed to the OS often and
  `fsync`-ed to disk every 50000, by the writer thread.
- On restart the program reads the existing `translated_titles.jsonl`, collects
  the `article_id`s already done, and skips them — so an interrupted run resumes
  automatically. Just re-run the same command.

> Note: if a previous run used an older/buggy version, delete or rename the
> existing `translated_titles.jsonl` before running, so resume does not preserve
> stale records.
