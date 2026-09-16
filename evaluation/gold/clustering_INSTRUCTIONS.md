# Annotation instructions - clustering

File: `clustering_sample.jsonl` (200 rows), sampled with seed 0.

Edit the `gold` block of each JSONL line in place and keep one JSON object per line. Do not reorder,
delete or add rows - `evaluation/score.py` matches on the identifier fields. Leave a row's gold fields null
if you cannot judge it; the scorer skips unjudged rows and reports how many it skipped.

## clustering (event coreference)

Set `gold.same_event` to true when A and B describe THE SAME real-world flood.

- Same flood, different day of coverage, different outlet, different death toll: **true**.
- Same river and same week but two distinct flood episodes: **false**.
- Same country and same month but different districts, with no shared event: **false**.
- A national round-up covering many floods versus one of them: **false**.

Half these pairs were linked by the system and half are its highest-similarity refusals, so expect
roughly half to be true. Do not use `system` to decide.
