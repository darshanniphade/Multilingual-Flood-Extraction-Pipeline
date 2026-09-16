# Annotation instructions - extraction

File: `extraction_sample.jsonl` (100 rows), sampled with seed 0.

Edit the `gold` block of each JSONL line in place and keep one JSON object per line. Do not reorder,
delete or add rows - `evaluation/score.py` matches on the identifier fields. Leave a row's gold fields null
if you cannot judge it; the scorer skips unjudged rows and reports how many it skipped.

## extraction

Fill each `gold` field from `text` alone.

- Lists (`event_dates`, `locations`, `rivers`, `causes`): everything the article states, `[]` if none.
  Dates as YYYY-MM-DD; only dates the flooding occurred on, never the publication date.
- Counts: the number the article states, or null. Do not add numbers across paragraphs; if the
  article gives both a district figure and a total, record the total.
- A field left null is treated as 'not annotated' and skipped by the scorer, so write 0 or [] when
  the correct answer is genuinely 'none'.
