# Flood Title Filter

Filters the translated 2021 titles down to an index of flood-related articles.
No translation, no summarisation, no article fetching — this stage only decides
*which* `article_id`s are worth retrieving later.

```powershell
cd C:\darsh\pipeline\ttilte_filter
C:\darsh\AI_MODELS\translator_env\Scripts\python.exe filter_flood_titles.py
```

Last full run: **1,383,600 titles scanned → 233,153 flood-related (16.85%) in 45s.**

## Input — note the deviation from the original spec

The task specified `C:\darsh\pipeline\title_tranlator\data` as the input. That
directory is **not** the translated titles: it holds the ~1.2M *raw* article
files (`2021_01/articles/article_000000002.json`), which carry an untranslated
`title` and no `translated_title` field at all, so it cannot satisfy "match on
`translated_title` only".

The real input is the single 329 MB file the translator stage emits:

```
C:\darsh\pipeline\ttilte_filter\translated_titles.jsonl   (a copy of
C:\darsh\pipeline\title_tranlator\translated_titles.jsonl)
```

`--input` also accepts a directory, which is scanned recursively for `*.jsonl`.

## Output

One file per month in `C:\darsh\pipeline\flood_events`, named from the month
prefix of `article_id` (`2021_07/article_...` → `2021_07.jsonl`) — which is what
the spec's example filenames and records call for:

```json
{"article_id":"2021_07/article_000101304","translated_title":"Monsoon rains flood Philippine villages, thousands evacuate"}
```

Only `article_id` and `translated_title`; titles are copied byte-for-byte.

## Why 16.86% and not ~2%

This corpus is a flood/disaster crawl, not general news. Measured directly on
the raw titles, ignoring the lexicon entirely:

| literal substring        | share of all titles |
| ------------------------ | ------------------- |
| `flood`                  | 11.90%              |
| `rain`                   | 5.56%               |
| `cyclone`/`hurricane`/`typhoon`| 2.06%        |
| any of the above         | 17.89%              |

So a ~17% hit rate is the expected order of magnitude. The filter lands slightly
*below* the naive substring rate because it removes metaphors and homonyms,
while adding recall the substrings miss (waterlogging, in spate, submerged,
cloudburst, glacier burst).

## How matching works

`flood_lexicon.py`, three stages over a normalised title:

1. **Guards** — metaphors and homonyms are *masked out of the text* before
   matching, rather than rejecting the title outright. So "Flood of criticism"
   dies, but "Flood of donations as floods hit Assam" keeps its real signal.
   Covers: `flood of migrants`, `floodgates`, `floodlights`, `landslide
   victory`, `Iowa State Cyclones`, `rescue package`, `rescue dog`, `Great
   Depression`, `perfect storm`, `Monsoon Session`, `wet market`, winter
   weather (`winter storm`, `cold wave`, `blizzard`), and more.
2. **Strong** — sufficient alone: the flood family, qualified/destructive rain,
   waterlogging, inundation, submersion, cloudburst, `in spate`, dam and
   embankment failure or discharge, storm surge, named cyclones.
3. **Weak + context** — impact and response terms that are common in unrelated
   news (`rescue`, `evacuation`, `death toll`, `red alert`, `landslide`, `power
   outage`) only count when the title *also* carries an independent water/rain
   word that is not part of the weak match itself.

Stage 3 is what enforces "flooding must be the primary subject":

| title | verdict |
| ----- | ------- |
| `Landslide buries five in Idukki` | no — no water signal |
| `Rain-triggered landslide buries five in Idukki` | yes |
| `Rescue dog finds forever home` | no — guarded |
| `Rescue operations underway as rains flood Pune` | yes |

Bare `rain` is **context only**, never strong — otherwise every forecast
matches. Word boundaries matter: `\brain` correctly ignores "B**rain**",
"T**rain**ing", "Bah**rain**".

## Tests

```powershell
C:\darsh\AI_MODELS\translator_env\Scripts\python.exe test_lexicon.py   # 81/81
```

81 cases, including 18 regressions harvested from real corpus output (the
`Flashflood` solid compound, Texas winter-storm false positives, `wave of COVID`
context leaks).

## Tuning

```powershell
cd C:\darsh\pipeline\ttilte_filter
C:\darsh\AI_MODELS\translator_env\Scripts\python.exe audit_matches.py --limit 300000 --top 30 --samples 3
```

Prints which rule fired for each match, ranked by frequency, with example
titles — the fastest way to spot a rule that is too greedy. Edit the `GUARDS`,
`STRONG`, `WEAK` or `CONTEXT` lists in `flood_lexicon.py` and re-run the tests.

### Known residual noise

Precision is good but not perfect; the main remaining false positives are:

- **Weather forecasts that are not flood events** — "Freezing rain warning
  issued", "Snow and rain to bring disruption". Driven by the `weather
  warning`/`alert issued` weak terms plus a `rain` context. Drop those entries
  from `WEAK` if you want a stricter index.
- **Negated or meta mentions** — "Icebreaking operation, no threat of flooding",
  "Whether flood damage is covered depends on the case". Negation is not modelled.

Both are cheap to filter at the next stage, where the article body is available.

## Semantic matching

Matching is lexical: a large tiered phrase lexicon with morphological variants,
context gating and metaphor masking — not embeddings. A true embedding pass was
considered and **not** built, because:

- `C:\darsh\AI_MODELS\model` is `M2M100ForConditionalGeneration`
  (NLLB-200-distilled-1.3B) — a **translation** model. Its encoder has no
  trained sentence pooling, so it is a poor similarity model, and this stage is
  explicitly not allowed to translate.
- `sentence_transformers` is not installed in either env and no embedding model
  is in the local HF cache, so it would need a network download.

To add one later, install `sentence-transformers`, embed the non-matching
titles against flood prototype phrases, and union the results — the lexicon
already works as stage 1 of that design.
