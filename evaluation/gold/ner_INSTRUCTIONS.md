# Annotation instructions - ner

File: `ner_sample.jsonl` (60 rows), sampled with seed 0.

Edit the `gold` block of each JSONL line in place and keep one JSON object per line. Do not reorder,
delete or add rows - `evaluation/score.py` matches on the identifier fields. Leave a row's gold fields null
if you cannot judge it; the scorer skips unjudged rows and reports how many it skipped.

## ner

Copy `system_entities` into `gold_entities`, then correct it: delete wrong spans, fix boundaries and
labels, add what was missed. Set `gold.reviewed` to true when the row is finished - rows left false
are ignored by the scorer.

Each gold entity is `{"text": ..., "label": ..., "start": ..., "end": ...}` with `start`/`end`
as character offsets into this row's `text` exactly as given (do not reflow it).

Labels: PERSON, ORGANIZATION, LOCATION, DATE, NUMBER, RIVER, FLOOD_TYPE, WEATHER_EVENT,
INFRASTRUCTURE, CASUALTY, EVACUATION, DAMAGE.

- A named watercourse is RIVER, not LOCATION.
- CASUALTY / EVACUATION / DAMAGE mark the trigger word or phrase ('killed', 'evacuated', 'washed
  away'), not the number next to it - the number is its own NUMBER span.
- DATE covers explicit dates only, not bare weekdays.
