# Annotation instructions - flood_type

File: `flood_type_sample.jsonl` (150 rows), sampled with seed 0.

Edit the `gold` block of each JSONL line in place and keep one JSON object per line. Do not reorder,
delete or add rows - `evaluation/score.py` matches on the identifier fields. Leave a row's gold fields null
if you cannot judge it; the scorer skips unjudged rows and reports how many it skipped.

## flood_type

Set `gold.flood_type` to exactly one of: River Flood, Flash Flood, Urban Flood, Coastal Flood, Pluvial Flood, Dam/Reservoir Flood, Other, Unknown.

- **River Flood**: a watercourse overtopped or breached its banks/embankment.
- **Flash Flood**: sudden onset, hours or less - cloudburst, hill torrent, GLOF.
- **Urban Flood**: rainfall exceeded a built-up area's drainage; waterlogged streets.
- **Coastal Flood**: seawater driven inland - storm surge, tide.
- **Pluvial Flood**: rain ponding on the surface away from any channel or drainage system.
- **Dam/Reservoir Flood**: a release, breach or failure of built water infrastructure.
- **Other**: a real flood of a different mechanism (tsunami, burst main, ice jam).
- **Unknown**: the article genuinely does not say. Use it freely - the system is scored on when it
  abstains as well as on what it labels.
