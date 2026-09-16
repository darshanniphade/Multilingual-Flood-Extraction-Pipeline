# Annotation instructions - retrieval

File: `retrieval_sample.jsonl` (173 rows), sampled with seed 0.

Edit the `gold` block of each JSONL line in place and keep one JSON object per line. Do not reorder,
delete or add rows - `evaluation/score.py` matches on the identifier fields. Leave a row's gold fields null
if you cannot judge it; the scorer skips unjudged rows and reports how many it skipped.

## retrieval

Set `gold.relevance` to a graded judgement of how well the article answers the query:

- **2** - fully on target: this is the event the query asks for.
- **1** - related: right region or right phenomenon, wrong event or wrong period.
- **0** - not relevant.

Judge the query, not the article's quality. Every pooled row must be judged, including the ones that
look obviously irrelevant - unjudged rows are treated as unjudged, not as zero, and that biases
Recall@K.
