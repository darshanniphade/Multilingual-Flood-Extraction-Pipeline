# Stage 7 — semantic layer: representation, event understanding, retrieval

This directory turns 74k article-level flood extractions into a queryable
knowledge base of real-world flood events. It is the NLP half of the pipeline:
transformer embeddings, a vector index, NER, flood-type classification, event
coreference, duplicate detection, source grounding, geospatial linking, hybrid
retrieval, and an LLM answer step over retrieved text.

Everything runs offline. Every stage is resumable, config-driven, and writes a
report whose numbers were measured by the run that wrote it.

---

## The two things this stage keeps apart

| | key | what it is |
|---|---|---|
| **mention** | `uid = "<year>/<month>/<article_id>"` | one article's extraction |
| **event** | `event_id = "E-<year>-<n>"` | one real-world flood, cited by 1..n uids |

`article_id` alone is not a key — each year's crawl restarts its numbering.
Neither is `<year>/<article_id>`: measured over the 74,447 verifiable
extractions, **2,075 article_ids occur in two different months of the same
year, and they are different articles**:

```
2021/2021_01/article_000001527   "At least 96 killed ... as quake, floods hit Indonesia"
2021/2021_05/article_000001527   "Parapat City Paralyzed, Hit by Floods and Landslides"
```

So the month — which is also what addresses the article on disk — is part of the
key, and a uid maps 1:1 onto exactly one article file. `common.parse_uid()` also
accepts the older two-part form.

---

## Pipeline

```
                       stage 5 output (data/extracted/<year>/<YYYY_MM>.jsonl)
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
              flood_type.py           ner.py           build_index.py
           weighted lexicon /   dslim/bert-base-NER    BGE embeddings +
           zero-shot Qwen3      + domain rule layer    Chroma (HNSW cosine)
                    │                   │                   │
                    └─────────┬─────────┘                   │
                              ▼                             ▼
                  build_index.py --refresh-metadata   embeddings.npy
                              │                       events_meta.jsonl
                              ▼                             │
                     cluster_events.py  ◄──────────────────┘
              weighted coreference over kNN candidates
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
      consolidated_    near_dup.py         geocode.py
      events.jsonl   MinHash-LSH +      geonamescache
      (the event KB)  embedding pass     entity linking
                              │
                              ▼
                          ground.py
              every field checked against source text
                              │
                              ▼
                          search.py
        BM25 + dense → RRF → filters → rerank.py → top-K
                              │
                              ▼
                    qwen3:14b over the retrieved TEXT
                     (--llm; embeddings never sent)
```

Dependency order for a cold start:

```powershell
cd 7_semantic_layer
python build_index.py --limit 300 --years 2021   # smoke test first
python build_index.py                            # ~6 min, 74k vectors
python flood_type.py                             # ~6 min
python ner.py                                    # ~40 min (GPU)
python build_index.py --refresh-metadata         # fold both into the index
python cluster_events.py
python near_dup.py --semantic
python geocode.py
python ground.py --sample 5000
python search.py "Assam floods June 2021"
python search.py --benchmark
```

`flood_type.py` and `ner.py` can run before the index; `--refresh-metadata`
attaches their output to the vectors without re-embedding anything.

---

## The scripts

### `common.py`
No model is imported at module scope, so every script is cheap to load. Holds
the `Event` record, uid handling, streaming readers (`iter_events`,
`iter_bodies` with threaded read-ahead), text normalisation, number surface
forms, sentence-aware chunking, resumable JSONL I/O, the Ollama helpers and the
report writer.

`iter_events` streams; nothing ever loads the corpus. `iter_bodies` prefetches
article files on a thread pool with a bounded window — on this corpus that took
`flood_type.py` from 34 to 212 documents/second and `ner.py` from ~9 to ~32,
because both were waiting on millions of small file reads rather than on a
model.

### `event_schema.py`
`FloodEvent`, the canonical record for one real-world flood, with
`validate()`, CSV flattening, and the schema handed to the LLM.

Counts are the awkward part of merging. Fifty outlets report a toll climbing
from 7 to 31; averaging is meaningless and taking the first is arbitrary. So
every count keeps `min`, `max`, `reports`, `conflict` in `counts[slot]`, and the
scalar beside it is the maximum reported value. Disagreement is data.

### `build_index.py` — embeddings and the vector store
`BAAI/bge-base-en-v1.5` (109M params, 768-d) over a composed event string:
title, summary, flooded locations, rivers, country, flood type, dates, cause.
Normalised vectors into a persistent Chroma collection with HNSW over cosine.

Resumable (uids already in the collection are skipped), streaming, batched,
`--limit` / `--years` / `--select` / `--rebuild` / `--refresh-metadata`.
`--chunks` additionally builds a sentence-aware passage index over article
bodies for quote-level evidence (off by default; see `chunking` in the config).

Also exports `embeddings.npy` + `ids.json` + `events_meta.jsonl` re-read from
the collection, so the all-pairs scripts always see what is actually indexed.

### `ner.py` — generic NER **plus a flood-domain normalisation layer**
Read this honestly, because the distinction is in the output:

- `source="model"` — `dslim/bert-base-NER`, BERT-base fine-tuned on CoNLL-2003.
  It knows PER / ORG / LOC / MISC and **nothing about floods**. Carries the
  model's own confidence.
- `source="rule"` — the domain layer written for this corpus: lexicons and
  patterns producing RIVER, FLOOD_TYPE, WEATHER_EVENT, INFRASTRUCTURE,
  CASUALTY, EVACUATION, DAMAGE, plus regex DATE and NUMBER. Fixed confidence
  from the config, not a learned probability.
- `source="slot"` — values stage 5 already extracted, re-anchored at their
  character offsets where they literally occur (`in_text: false` and
  `start: -1` when they do not).

**This is not a fine-tuned flood NER model.** Training one needs a span-annotated
flood corpus; `evaluation/make_annotation_sample.py --task ner` produces the sample to
start one, and `evaluation/score.py --task ner` scores whatever replaces this.

One domain rule worth calling out: a LOCATION the tagger found is promoted to
RIVER when the following word is a watercourse noun ("Brahmaputra" + "river").
The reverse patterns are deliberately case-sensitive — matching case-insensitively
turned "river overflowed" and "river basin" into river names.

### `flood_type.py` — flood-type classification
Labels: River, Flash, Urban, Coastal, Pluvial, Dam/Reservoir, Other, Unknown.

- `rules` (default) — weighted lexicon; every cue and weight is in
  `config.json`. Weights encode evidence strength: 4 = the label is stated, 3 =
  a mechanism only that type produces, 2 = a strong associate, 1 = context
  worthless alone. `min_score` 2.0 is what stops a single mention of the word
  "river" from typing an article as a River Flood.
- `llm` — zero-shot Qwen3-14B, which must return a verbatim quote; an unfound
  quote is discarded. `confidence` is `null`, because the model reports no
  probability and inventing one would be a lie.
- `agree` — rules first, LLM only on the abstentions.
- `model` — an optional supervised TF-IDF + logistic-regression classifier.
  `--train` refuses to fit on machine-generated labels.

`Unknown` is an abstention, not a class. On the full corpus the rule model types
33.7% of extractions and abstains on the rest; that is the intended behaviour.

### `cluster_events.py` — event coreference
Candidates from exact top-k cosine kNN (FAISS, else a blocked torch matmul),
then a weighted score:

```
score = w_semantic·cos + w_date·proximity + w_location·overlap
      + w_river·overlap + w_flood_type·agreement + w_country·agreement
```

with hard gates that veto regardless of score: date gap beyond the window,
disagreeing countries, and any merge that would stretch a cluster past
`max_cluster_span_days` (without which transitive chaining walks a whole monsoon
season into one "event"). Missing evidence scores `unknown_field_score` —
neither agreement nor conflict. All weights are in the config; `--ablate river
flood_type` zeroes any of them.

Outputs `consolidated_events.{jsonl,csv}`, `clusters.jsonl` and a report with
the signal statistics of accepted links.

### `near_dup.py` — two duplicate questions
- **Lexical** (default): MinHash + LSH over 5-word shingles, Jaccard ≥ 0.8.
  "Is this literally the same story?" Safe to collapse.
- **Semantic** (`--semantic`): embedding cosine ≥ 0.95 with low shingle overlap.
  "Is this the same story rewritten?" Reported as **candidates only**. High
  cosine also fires on two genuinely different floods described in near-identical
  language; merging on embedding similarity alone would destroy exactly the
  events this project counts.

### `ground.py` — evidence verification
Every extracted field is checked back against the article it came from: dates
against every surface form a newsroom writes, counts against digit, separated,
scaled and spelled-out forms, places and rivers against exact then content-token
matching, flood type against the cue that fired. Locations are additionally
cross-checked against what the NER model reads as a place.

`grounded = true` means *the article says this*, not *this is true*. That is the
distinction that matters when an LLM produced the value.

### `geocode.py` — geospatial entity linking
Offline `geonamescache` (~34k cities, all countries). Four progressively more
aggressive readings of a mention, filtered by the extracted country, ranked by
population — but only accepted when the top candidate is `ambiguity_ratio` times
larger than the runner-up. Otherwise `status="ambiguous"` with every candidate
listed. GeoNames' alternate names are what resolve Bombay → Mumbai and
Bangalore → Bengaluru; no alias table is hand-maintained.

### `search.py` — hybrid retrieval + the LLM step
BM25 (rank_bm25, cached) and dense (BGE + Chroma) over the same documents, fused
by RRF (k=60), metadata-filtered, optionally reranked, folded onto events.

Dense retrieval matches meaning — "flash floods caused by extreme rainfall"
finds cloudburst reports that never use the word "flash". It is also the one
that fails on a rare proper noun, where BM25 is sharpest. RRF needs no score
calibration between them, which is why it is used here.

`--llm` then sends the **original article text** of the top hits to qwen3:14b
with a strict schema, temperature 0, `format=json`, `think=false`, and verifies
every returned quote is a real substring of the evidence. Embeddings are never
sent to the model; the index only decides what it reads.

Ablation switches: `--no-dense`, `--no-bm25`.

### `rerank.py` — optional cross-encoder
Offline-enforced: a reranker is used only if `rerank.model` names a local
directory or a model already in the HF cache. Otherwise `available` is False,
search keeps the RRF order and says so. Nothing is ever downloaded.
`python rerank.py --check` reports the state.

---

## Configuration

Everything tunable is in `config.json`, with a `_comment` per block explaining
what the numbers mean and what moves them. Nothing in there is a research
constant — they are starting points to sweep. Notable ones:

| key | default | why |
|---|---|---|
| `embedding.batch_size` | 256 | measured 219 events/s end-to-end on a 4060 Ti |
| `flood_type.min_score` | 2.0 | below this a single weight-1 cue could decide a label |
| `clustering.link_threshold` | 0.82 | with the weights below it, swept per corpus |
| `clustering.max_cluster_span_days` | 45 | stops seasonal chaining |
| `near_dup.jaccard_threshold` | 0.8 | same story, not same event |
| `near_dup.semantic.cosine_threshold` | 0.95 | candidate generation only |
| `geocode.ambiguity_ratio` | 5.0 | below this the mention stays ambiguous |
| `retrieval.rrf_k` | 60 | the value from Cormack et al. 2009 |

---

## Reports

Each script writes a `*_report.md` next to its data under
`data/events/`. They contain only measured numbers: input/processed/skipped/
failed counts, throughput, device, distributions, and worked examples. No
benchmark figure is written that the run did not observe, and no quality metric
appears in any of them — those live in `evaluation/` and only exist once something has
been annotated.

## Measured on the full corpus

One pass of the whole stage over the 74,281 verifiable extractions, on an RTX
4060 Ti (16 GB):

| stage | output | wall clock |
|---|---|---:|
| `build_index.py` | 74,281 × 768-d vectors, 219 events/s, 2.50 GiB peak VRAM | 5m 40s |
| `flood_type.py` | 25,056 typed (33.7%), 49,225 abstentions, 212 docs/s | 5m 50s |
| `ner.py` | 3,795,927 entities (1,683,926 model · 1,692,468 rule · 419,533 slot) | 38m 24s |
| `cluster_events.py` | 32,413 events from 74,281 mentions, 41,868 links (56.4% reduction) | 36s |
| `near_dup.py --semantic` | 994,372 lexical pairs / 49,272 removable (21.9%); 8,226 embedding-only candidates | 23m 21s |
| `ground.py` | 602,917 field assertions, 90.5% supported, 51.7% of articles fully | 1m 1s |
| `geocode.py` | 26,097 of 130,402 mentions linked; 57.8% of extractions mappable | 11s |
| `search.py --benchmark` | 249 ms mean end-to-end over 8 queries | — |

Largest consolidated events: Texas July 2025 (4,871 articles), Derna/Libya
September 2023 (2,974), Valencia October 2024 (1,768), Kakhovka dam May 2023
(837, correctly typed Dam/Reservoir Flood), Uttarakhand February 2021 (836).

None of these are quality metrics. They are throughput and volume.

## Evaluation

`evaluation/make_annotation_sample.py` builds stratified, seeded annotation samples
(detection, flood_type, ner, extraction, clustering, retrieval).
`evaluation/score.py` scores them and **refuses to run on an unannotated file**.
