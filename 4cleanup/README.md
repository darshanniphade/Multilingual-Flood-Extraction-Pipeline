# Flood News Corpus Cleaning Pipeline

A parallel, streaming preprocessing pipeline that turns a tree of scraped and
machine-translated flood-news articles into a clean, deduplicated,
three-field corpus suitable for downstream NLP.

```
C:\darsh\pipeline\data\translated_articles   ->   C:\darsh\pipeline\data\only_english
```

The output mirrors the input directory structure exactly. Every output
document contains **only** three fields — the identifier, the title and the
body, each keeping the name it had in the source:

```json
{
  "article_id": "article_000000027",
  "translated_title": "IN PHOTOS: Heavy rain causes flash floods in Negros Occidental",
  "translated_text": "Heavy rain triggered flash floods in various areas of ..."
}
```

Cleaning a raw scrape tree instead (`--title-field title --text-field text`)
writes `article_id` / `title` / `text`. No field is ever renamed and nothing
is translated here — translation is stage 3.

---

## Quick start

```powershell
pip install -r requirements.txt
python clean_articles.py
```

That is the whole run. Defaults point at the paths above and use every CPU
core. Two artefacts are written next to the output tree:

| File | Contents |
| --- | --- |
| `processing_report.json` | Machine-readable run statistics |
| `processing.log` | Full run log, including one line per skipped/duplicate file |

### Common variations

```powershell
python clean_articles.py --limit 5000            # smoke test on a subset
python clean_articles.py --workers 16            # cap parallelism
python clean_articles.py --min-words 150         # stricter length floor
python clean_articles.py --no-dedup              # keep duplicates
python clean_articles.py --compact               # unindented output JSON
python clean_articles.py --overwrite             # write into a non-empty output dir
python clean_articles.py --verbose               # per-file decisions on the console
python clean_articles.py --input D:\a --output D:\b
```

Run `python clean_articles.py --help` for the full list.

The pipeline refuses to write into an output directory that already contains
JSON files unless `--overwrite` is given, so a half-finished run cannot be
silently mixed with a new one.

---

## Running a whole year (`run_year.py`)

`clean_articles.py` cleans one tree. `run_year.py` wraps it so a calendar year
— or several — can be cleaned with one command, each into its own output tree
with its own report and log.

```powershell
python run_year.py --list              # which years exist, which are done
python run_year.py 2022                # one year
python run_year.py 2022 2023 2024      # several
python run_year.py 2019-2024           # an inclusive range
python run_year.py 2022 --limit 3000   # smoke test
python run_year.py 2022 --overwrite    # re-clean a finished year
```

Adding a year requires **no edit to any file** — a new directory under the
source root is enough.

```
H:\2022\<MM>\articles\*.json                         (source)
  -> C:\darsh\pipeline\data\ssd\2022\<MM>\articles\*.json
     C:\darsh\pipeline\data\ssd\_reports\2022_report.json
     C:\darsh\pipeline\data\ssd\_reports\2022.log
```

Per-year report and log paths matter: the plain `clean_articles.py` defaults
put both beside the output root, so a second year would overwrite the first
year's report.

### Profiles — where a year's text comes from

The corpus exists in two shapes and the field names differ, so the source root
and the fields it reads are selected together by `--profile`:

| Profile | Source root | Title / text fields | Content |
| --- | --- | --- | --- |
| `raw` (default) | `H:\<year>` | `title` / `text` | Stage-0 GDELT scrape — **multilingual** |
| `translated` | `data\translated_articles\<year>` | `translated_title` / `translated_text` | Stage-3 translator output — English |

**Field names are preserved.** This stage cleans text; it never renames. A
cleaned raw tree is written back as `article_id` / `title` / `text`, a cleaned
translated tree as `article_id` / `translated_title` / `translated_text`. Only
these three fields survive — everything else in the source is dropped.

Downstream, `extract/config.json` selects the fields it reads
(`"title_field"`, `"text_field"`), so point it at whichever pair the tree
you're feeding it actually uses.

Override either half when needed:

```powershell
python run_year.py 2022 --profile translated
python run_year.py 2022 --source-root D:\corpus --text-field body
python run_year.py 2022 --dest-root E:\clean
```

`clean_articles.py` takes the same `--title-field` / `--text-field` flags.

### What `run_year.py` guarantees

- **Idempotent.** A year whose output tree already holds JSON is skipped with
  a message; `--overwrite` re-cleans it. Re-running the same command after
  adding a year only processes the new one.
- **A bad year does not abort the batch.** Missing source or an occupied
  output tree is recorded and the run moves to the next year.
- **Deduplication is per year.** Each year builds a fresh SHA-256 index, so
  runs are independent and order does not matter. A wire story republished
  across a year boundary therefore survives in both years.
- **A cross-year summary** is printed at the end: input, written, skipped,
  duplicates, errors and runtime per year, plus a total.

> **Caveat when cleaning `raw`:** nothing is translated. The text is cleaned in
> its original language. The 100-word floor is also counted on whitespace, so
> unspaced scripts (Chinese, Japanese, Thai) score far below their true length
> and are dropped as `fewer_than_min_words`. Lower `--min-words` or run the
> `translated` profile if that matters.

---

## Project layout

| Module | Responsibility |
| --- | --- |
| `run_year.py` | Per-year / multi-year driver, profiles, cross-year summary |
| `clean_articles.py` | CLI, process pool, result aggregation, reporting |
| `cleaner.py` | Text normalisation stages |
| `validators.py` | Validation rules and rejection reasons |
| `deduplicator.py` | SHA-256 exact-duplicate index |
| `config.py` | All tunables and boilerplate patterns |
| `utils.py` | Filesystem streaming, JSON I/O, logging, formatting |

Boilerplate rules live in `config.py` as data, not code. Adding a pattern is a
one-line edit to a list; no pipeline logic changes.

---

## Cleaning stages

`cleaner.clean_text` applies nine stages in a fixed order. The order is
load-bearing:

1. **Decode HTML entities** (twice when double-encoded) so `&lt;p&gt;` becomes
   a real tag for stage 2.
2. **Strip HTML** — tags, comments, CDATA, and `<script>`/`<style>` bodies.
   Tags become spaces so `a<br>b` does not become `ab`.
3. **Unicode normalisation** — NFKC.
4. **Remove invisible characters** — zero-width, bidi controls, soft hyphen,
   BOM, C0/C1 controls. NFKC does not remove these, so this must follow it.
5. **Fold quotes and dashes** — curly quotes to `'`/`"`, all dash variants to
   `-`.
6. **Remove URLs** — `http(s)://`, `ftp://`, bare `www.` hosts.
7. **Strip boilerplate** — see below.
8. **Repair punctuation** — `,,` -> `,`, `..` -> `.`, `;;` -> `;`, plus debris
   created by stage 7.
9. **Normalise whitespace** — collapse spaces, at most one blank line, trim
   every line, drop punctuation-only lines.

`clean_text` is **idempotent**: `clean_text(clean_text(x)) == clean_text(x)`,
verified across a 4,000-article random sample.

Titles go through `cleaner.clean_title`, which runs every stage **except**
boilerplate stripping. Line-level rules match whole lines, and a title is a
single line — a legitimate headline like `"READ: Dam overflows"` would
otherwise be erased rather than cleaned.

### Boilerplate removal

Rules are grouped and applied in order. The grouping reflects a decision made
after inspecting real occurrences in the corpus:

**Markers are line-scoped, not token-scoped.** In this data, `READ:` and
`Related:` introduce a cross-promotion whose *entire line* is boilerplate:

```
...closed pending cleanup. But things were rough down in the city as well.
READ: Another round of flooding possible as new flood watch issued for noon
With more than two inches of rain...
```

Deleting only the token `READ:` would strand `Another round of flooding
possible as new flood watch issued for noon` in the middle of the article — a
headline fragment with no relationship to the surrounding prose. The rule
therefore removes the marker **through end of line**.

| Group | Behaviour | Members |
| --- | --- | --- |
| Standalone lines | Drop the whole line | `Advertisement`, `ADVERTISEMENT`, `Sponsored`, `Article continues below this ad`, `Continue Reading`, `Read More`, `Loading...`, `Video Player`, `SUMMARY`, `Share this story`, `Cookie Policy`, `Privacy Policy`, `Terms of Service`, `Follow us`, `Follow Rappler`, `Photo Credit`, `Image Credit` |
| Cross-promotions | Marker through end of line, plus bracketed inline form | `READ:`, `ALSO READ:`, `MUST READ:`, `SEE ALSO:`, `READ MORE:`, `RELATED:`, `Related:` |
| Calls to action | Match through end of line, anchored to line start or sentence start | `Subscribe`, `Sign up`, `Follow us`, `Follow Rappler`, `Click here`, `Share this story`, `Continue Reading`, `Read More` |
| Legal notices | Remove the carrying sentence or line | `All rights reserved`, `©`, `™`, `Copyright <year> ...`, `This material may not be published, broadcast, rewritten...` |
| Inline phrases | Remove wherever they occur | `This is AI generated summarization, which may have errors.`, `For context, always refer to the full article.`, `How does this make you feel?`, `[Advertisement]`, `Advertisement`, `Sponsored`, `Cookie Policy`, `Privacy Policy`, `Terms of Service`, `Photo Credit`, `Image Credit`, `Video Player`, `Loading...` |

Two precision safeguards, both driven by observed data:

- **CTA rules are anchored** to line start or sentence start, so ordinary prose
  survives: `"Residents had to sign up for flood alerts, officials said."` is
  left untouched, while a line beginning `"Sign up for our FREE morning
  newsletter..."` is removed.
- **The bracketed cross-promotion rule is case-sensitive.** It matches
  `(READ: ...)` and `[Related: ...]`, but deliberately not the lowercase
  rhetorical idiom `(read: in other words)`.

### Additions beyond the requested list

Three patterns were added because the corpus showed them sitting directly
against the requested ones, and removing one without the other leaves debris:

- `ALSO READ:` / `MUST READ:` / `SEE ALSO:` — observed variants of `READ:`.
- `Article continues below this ad` — always adjacent to `Advertisement`.
- `This material may not be published, broadcast, rewritten or redistributed.`
  — the wire-service sentence that always trails `All rights reserved`.

---

## Validation

An article is skipped when any of these hold:

| Reason code | Condition |
| --- | --- |
| `missing_translated_text` | Field absent or not a string |
| `empty_translated_text` | Field empty or whitespace only |
| `missing_translated_title` | Field absent or not a string |
| `empty_translated_title` | Title empty after cleaning |
| `empty_translated_text_after_cleaning` | Body was entirely boilerplate |
| `fewer_than_min_words` | Fewer than 100 words after cleaning |
| `malformed_json` | File is not valid JSON |
| `not_a_json_object` | Top-level value is not an object |
| `unreadable_file` | I/O error reading the source |
| `write_failed` | I/O error writing the output |

**Validation runs after cleaning.** This is deliberate: an article padded to
120 words by advertising chrome is not a 120-word article, so the length floor
is measured against the text that actually reaches disk.

`article_id` falls back to the source filename stem when the field is missing,
so a usable article is never dropped over a missing identifier.

### Optional: garbled syndication text

A small fraction of wire-service articles are stored ROT47-encoded and are
unusable (`kAm`, `k^Am`, `E96 ...`). A detector exists in
`validators.is_garbled` and is wired to the reason code `garbled_text`, but it
is **disabled by default** because it is not part of the required
specification. Measured prevalence: ~0.25% of the corpus. Enable it with:

```python
# config.py
ENABLE_GARBLED_TEXT_CHECK = True
```

---

## Deduplication

Exact matching only — SHA-256 over the **cleaned** text. No fuzzy or
near-duplicate logic is involved.

Workers hash their own cleaned text in parallel and return only the 32-byte
digest; the parent process owns the single authoritative index and makes every
keep/drop decision. Cleaned text never crosses the process boundary, so IPC
volume scales with the *number* of articles rather than their size.

**First occurrence wins, deterministically.** The filesystem walk sorts
directory and file names at every level, and the driver consumes worker
results in submission order (`Pool.imap`, not `imap_unordered`). The same
corpus therefore always yields the same survivor.

Duplicates are written by the worker and unlinked by the parent once the
collision is known. This keeps the run single-pass over the corpus; the cost
is one redundant write for the duplicate fraction, far cheaper than a second
full read pass. Duplicates are counted and logged with the path of the article
they duplicate.

---

## Performance and memory

- **All cores by default** via `multiprocessing.Pool` (spawn context).
- **Batched IPC.** Files are handed to workers 200 at a time. At corpus scale,
  per-file IPC round-trips dominate runtime; batching amortises them away.
- **Bounded submission.** A semaphore gates how many batches are in flight, so
  `Pool.imap` cannot race ahead and materialise the entire path list.
- **Streaming traversal.** Directories are walked with `os.scandir` through a
  generator. Only the current directory's entries are resident.
- **No corpus in RAM.** One article is held per worker at a time.
- **Cached `mkdir`.** Each output directory is created once per worker rather
  than once per file.
- **tqdm progress** with live kept / duplicate / skipped counters.

Memory is flat with respect to corpus size, with one exception by design: the
deduplication index holds one 32-byte digest plus one relative path per unique
article (roughly 150-200 MB per million unique articles). Set
`Deduplicator(track_originals=False)` to drop the paths and roughly halve it,
at the cost of not naming the original in the duplicate log.

### Measured on this corpus

52 workers, 233,153 input files, 1.47 GB:

| Metric | Value |
| --- | --- |
| Runtime | 62.9 s |
| Throughput | 3,704 files/sec |
| Valid after validation | 218,739 |
| Skipped | 14,414 (14,392 under 100 words, 22 boilerplate-only) |
| Exact duplicates removed | 46,997 (21.5% of valid) |
| **Written** | **171,742** (432 MB, 69.8M words, 406 words/article avg) |
| Errors | 0 |

### Reproducibility

Output is deterministic. Verified by running the same 4,000-file subset twice
under different parallelism — 52 workers / batch 200 versus 7 workers /
batch 37 — and comparing a SHA-256 fingerprint over every output path and its
bytes. The trees were **byte-identical**, including which member of each
duplicate group survived.

---

## Error handling

The pipeline does not crash on bad input. Every file is processed inside a
per-file guard in the worker; malformed JSON, unreadable files, non-object
payloads and write failures are each captured, counted under a reason code,
sampled into the report, and the run continues.

`Ctrl+C` terminates workers, then still writes the report and log covering
everything processed so far.

---

## The report

`processing_report.json`:

```json
{
  "summary": {
    "total_files":   233153,
    "valid_files":   ...,
    "skipped_files": ...,
    "duplicate_files": ...,
    "output_files":  ...,
    "error_files":   ...,
    "runtime_seconds": ...,
    "runtime_human": "...",
    "average_files_per_second": ...
  },
  "skip_reasons": { "<code>": { "count": N, "description": "...", "examples": [...] } },
  "duplicates":   { "hash_algorithm": "sha256", "matching": "exact", "count": N,
                    "unique_texts": N, "examples": [{"duplicate": "...", "original": "..."}] },
  "output":       { "bytes_written": N, "megabytes_written": N,
                    "total_words": N, "average_words_per_article": N },
  "configuration": { "...": "every setting the run used" }
}
```

`configuration` records the exact settings of the run, so a report is
sufficient to reproduce it.

---

## Tuning

Everything adjustable lives in `config.py`:

| Setting | Default | Notes |
| --- | --- | --- |
| `MIN_WORD_COUNT` | `100` | Post-cleaning floor |
| `SOURCE_TITLE_FIELD` | `"translated_title"` | Input field read as the title |
| `SOURCE_TEXT_FIELD` | `"translated_text"` | Input field read as the body |
| `BATCH_SIZE` | `200` | Files per IPC round-trip |
| `WORKER_COUNT` | `None` | `None` means all cores |
| `QUEUE_DEPTH_BATCHES` | `64` | In-flight batches per worker |
| `UNICODE_NORMAL_FORM` | `"NFKC"` | Any form `unicodedata` accepts |
| `ELLIPSIS_POLICY` | `"keep"` | `"keep"` normalises runs of dots to `...`; `"collapse"` reduces them to `.` |
| `PRETTY_OUTPUT` | `True` | Indented output JSON |
| `ENABLE_GARBLED_TEXT_CHECK` | `False` | ROT47 detector |
| `MAX_LOGGED_EXAMPLES` | `50` | Samples retained per reason |

`ELLIPSIS_POLICY` defaults to `"keep"` because ellipses are semantically
meaningful inside quoted speech, which appears in 7.4% of articles in this
corpus. Runs of four or more dots are still normalised to exactly three.

---

## Requirements

- Python 3.12+
- `orjson`, `regex`, `tqdm` (see `requirements.txt`)

All modules are fully type-hinted and documented.
