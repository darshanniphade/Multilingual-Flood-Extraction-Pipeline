# Annotation instructions - detection

File: `detection_sample.jsonl` (198 rows), sampled with seed 0.

Edit the `gold` block of each JSONL line in place and keep one JSON object per line. Do not reorder,
delete or add rows - `evaluation/score.py` matches on the identifier fields. Leave a row's gold fields null
if you cannot judge it; the scorer skips unjudged rows and reports how many it skipped.

## detection

For each row set `gold.is_flood_event` to true/false and, when true, `gold.is_verifiable` to true/false.

- **is_flood_event**: the article's main subject is ONE real flood that has already happened or is
  happening. False for forecasts, warnings, risk studies, funding announcements, anniversaries, and
  for articles covering several unrelated floods.
- **is_verifiable**: the article states BOTH a specific date the flooding occurred and at least one
  named place that flooded. This is the stage-5b gate, so judge it literally.
- Judge from `text` only; ignore what `system` says. If `text` is empty the body was missing - set
  both to null and note it.
