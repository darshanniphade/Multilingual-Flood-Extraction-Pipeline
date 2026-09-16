# Flood pipeline — output metrics by year (2021 · 2022 · 2023)

## 1. The funnel — all three years side by side

| Stage | 2021 | 2022 | 2023 |
|---|---:|---:|---:|
| 0 · GDELT crawl (raw articles) | 1,383,600 | 1,935,415 | 2,950,352 |
| 0b · Clean + dedup (pre-filter) | — | 785,911 | — |
| 1 · Titles translated (NLLB) | 1,383,600 | 785,911 | 2,950,347 |
| 2 · Lexical title filter | 233,153 | 151,308 | 381,560 |
| 3 · Article translation | 233,153 | 151,308 | 381,560 |
| 4 · Clean+dedup / project (LLM input) | 171,742 | 151,308 | 381,560 |
| 5 · Articles LLM-extracted | 171,725 | 151,280 | 381,411 |
| 5a · Describe a flood event | 35,367 | 28,748 | 53,909 |
| 5b · VERIFIABLE (date + location) | 15,644 | 7,756 | 17,009 |
| 6 · Event objects in CSV | 33,029 | 24,060 | 46,184 |

**Retention at each gate**

| Gate | 2021 | 2022 | 2023 |
|---|---:|---:|---:|
| title filter (stage 2 / titles in) | 16.85 % | 19.25 % | 12.93 % |
| clean+dedup (wherever it ran) | 73.66 % | 40.61 % | not run |
| LLM flood gate (5a / 5) | 20.60 % | 19.00 % | 14.13 % |
| verifiability gate (5b / 5a) | 44.23 % | 26.98 % | 31.55 % |
| **end-to-end (5b / raw crawl)** | 1.13 % | 0.40 % | 0.58 % |
| compression factor (raw → verifiable) | 88× | 250× | 173× |

> 2021 ran clean+dedup **after** the title filter (stage 4); 2022 moved it **in front** of title translation (1,935,415 → 785,911, −59.4 %); **2023 never ran it** — the title translator read the raw crawl directly, so 2023 counts still contain exact duplicates and sub-100-word stubs. This is the single biggest reason 2023's rates sit below the other two years.

Duplicate output lines dropped while counting (extraction resume double-writes): 2021 0, 2022 0, 2023 5,275


---

# 2021

## Corpus

| metric | value |
|---|---:|
| articles LLM-extracted | 171,725 |
| describe a flood event | 35,367 (20.6 %) |
| not a flood event | 136,358 (79.4 %) |
| **verifiable (date + location)** | **15,644** (9.1 % of all, 44.2 % of floods) |
| has >=1 flood date | 15,814 (44.7 % of floods) |
| has >=1 flooded location | 25,748 (72.8 % of floods) |
| event objects emitted | 33,029 |
| flood articles with 0 events | 7,510 |
| flood articles with 2+ events (prompt says exactly 1) | 1,927 |
| distinct event field names emitted | 2,697 |
| named a place but no usable date | 10,104 |

## Flood articles by batch month
```
2021_01  ████████████████████······  4,596  (39.9% of month, 11,516 in)
2021_02  ██████████████████████████  5,878  (45.0% of month, 13,067 in)
2021_03  █████████·················  2,117  (29.1% of month, 7,281 in)
2021_04  ███████████···············  2,434  (36.4% of month, 6,688 in)
2021_05  █████·····················  1,165  (13.8% of month, 8,462 in)
2021_06  ███████···················  1,615  (14.2% of month, 11,354 in)
2021_07  █████████████████████·····  4,848  (15.2% of month, 31,997 in)
2021_08  ██████████████████········  4,020  (17.9% of month, 22,520 in)
2021_09  ██████████················  2,209  (12.3% of month, 18,003 in)
2021_10  ████████··················  1,698  (11.5% of month, 14,703 in)
2021_11  ███████████···············  2,524  (19.0% of month, 13,309 in)
2021_12  ██████████················  2,263  (17.6% of month, 12,825 in)
```

## Verifiable articles by batch month
```
2021_01  █████████████████·········  2,570
2021_02  ██████████████████████████  3,936
2021_03  ████████··················  1,270
2021_04  █████████·················  1,404
2021_05  ███·······················    513
2021_06  ███·······················    501
2021_07  ████████··················  1,153
2021_08  ████████··················  1,227
2021_09  █████·····················    780
2021_10  ████······················    636
2021_11  ███████···················  1,019
2021_12  ████······················    635
```

## Extracted flood dates by month (in-year only, 17,875 date values)
```
2021-01  ██████████████████········  2,887
2021-02  ██████████████████████████  4,127
2021-03  ████████··················  1,258
2021-04  █████████·················  1,416
2021-05  ███·······················    544
2021-06  ████······················    568
2021-07  ███████████···············  1,699
2021-08  █████████·················  1,351
2021-09  ████······················    659
2021-10  ████······················    657
2021-11  ██████····················    986
2021-12  ████······················    566
```

Out-of-year date values: **1,110** (6.2 % of all dates) — unparseable/partial: 47

Largest out-of-year buckets: `2020-12` 122, `2020-10` 119, `2020-11` 61, `2026-07` 54, `2020-01` 50, `2020-08` 37, `2020-05` 36, `2020-07` 30

## Field fill rates (share of the year's event objects)

| field | events | fill | field | events | fill |
|---|---:|---:|---|---:|---:|
| `summary` | 17,010 | 51.5 % | `affected_people` | 16,332 | 49.4 % |
| `location` | 13,759 | 41.7 % | `flooded_locations` | 9,636 | 29.2 % |
| `event_date` | 9,569 | 29.0 % | `response_agencies` | 8,408 | 25.5 % |
| `affected_people_type` | 8,254 | 25.0 % | `deaths` | 6,462 | 19.6 % |
| `houses_damaged` | 6,149 | 18.6 % | `evacuated` | 5,101 | 15.4 % |
| `deaths_type` | 4,503 | 13.6 % | `cause` | 4,499 | 13.6 % |
| `country` | 3,725 | 11.3 % | `missing` | 3,500 | 10.6 % |
| `water_unit` | 3,408 | 10.3 % | `evacuated_type` | 3,215 | 9.7 % |
| `trigger` | 3,048 | 9.2 % | `river` | 2,798 | 8.5 % |
| `water_depth` | 2,798 | 8.5 % | `missing_type` | 2,473 | 7.5 % |
| `displaced` | 2,377 | 7.2 % | `houses_destroyed` | 2,306 | 7.0 % |
| `roads_damaged` | 2,292 | 6.9 % | `bridges_damaged` | 1,753 | 5.3 % |
| `state` | 1,650 | 5.0 % | `city` | 1,623 | 4.9 % |
| `crop_damage` | 1,545 | 4.7 % | `economic_loss` | 1,530 | 4.6 % |
| `injured` | 1,406 | 4.3 % | `flood_dates` | 1,187 | 3.6 % |

## Top flooded locations (unsupervised — nothing was seeded)

| location | mentions | | country | mentions |
|---|---:|---|---|---:|
| Zhengzhou | 363 | | Indonesia | 714 |
| Sinop | 299 | | India | 411 |
| Kastamonu | 261 | | China | 358 |
| Bozkurt | 252 | | United States | 351 |
| Chamoli district | 236 | | Turkey | 295 |
| Waverly | 228 | | Germany | 282 |
| Bartin | 220 | | Belgium | 101 |
| Bozkurt district of Kastamonu | 205 | | Brazil | 98 |
| Jakarta | 190 | | Australia | 72 |
| Uttarakhand | 178 | | Malaysia | 62 |
| Yalta | 163 | | Afghanistan | 61 |
| Dhauliganga river | 154 | | Mexico | 59 |
| Dili | 147 | | Russia | 46 |
| Humphreys County | 142 | | France | 45 |
| Raini village | 140 | | US | 38 |

**Top rivers:** Ezine River (73), Garonne (50), Ahr (38), Dhauliganga (35), Dhauliganga River (32), Alaknanda (29), Ahr River (27), Tula River (26), Ebro (26), Ciliwung River (25)

**Top causes:** heavy rains (486), heavy rainfall (265), heavy rain (182), glacier burst (100), flooding (52), tropical cyclone seroja (47), flood (31), torrential rains (28), high rainfall (26), glacier collapse (21)

## Richest extracted records — the pipeline at its best

These are the event objects with the most non-empty fields in the year. Nothing was hand-picked beyond 'most fields filled'.

### 2021 · #1 — 39 fields · `article_000056052` (2021_09)

> New York, New Jersey, where Hurricane Ida, the "death wreck", caused massive flooding, killing at least 45 people in the southeastern United States.

```json
{
  "country": "United States",
  "state": "New Jersey",
  "city": "Elizabeth",
  "location": "New York Free International Airport",
  "event_date": "2021-09-01",
  "start_date": "2021-09-01",
  "end_date": "2021-09-02",
  "cause": "Hurricane Ida",
  "trigger": "Remnants of Hurricane Ida",
  "rainfall_mm": 215,
  "water_depth": 2.5,
  "water_unit": "m",
  "flood_duration": 1,
  "duration_unit": "day",
  "affected_people": 45,
  "affected_people_type": "people",
  "deaths": 45,
  "deaths_type": "people",
  "evacuated": 600,
  "evacuated_type": "people",
  "displaced": 15000,
  "displaced_type": "people",
  "injured": 82,
  "injured_type": "people",
  "missing": 5,
  "missing_type": "people",
  "houses_damaged": 1000,
  "roads_damaged": 500,
  "bridges_damaged": 200,
  "schools_damaged": 150,
  "hospitals_damaged": 80,
  "crop_damage": 500000,
  "crop_damage_unit": "USD",
  "livestock_loss": 350,
  "livestock_loss_type": "cattle",
  "economic_loss": 2000000,
  "economic_loss_currency": "USD",
  "response_agencies": "US Federal Meteorological Service, New York City Department of Disaster Management, President Biden, New York Mayor Bill de Blasio, Governor Kathy Hochul",
  "summary": "Hurricane Ida caused massive flooding in New York and New Jersey, resulting in at least 45 deaths, significant infrastructure damage, and widespread disruption. The storm brought record rainfall, overwhelmed drainage systems, and led to flooding in subway stations, airports, and residential areas."
}
```

### 2021 · #2 — 37 fields · `article_000145665` (2021_05)

> Washed away by the waters... north of Castile.

```json
{
  "country": "Spain",
  "province": "Vallisoletana",
  "city": "Valladolid",
  "location": "Plaza del Poniente",
  "event_date": "1962-01-03",
  "start_date": "1962-01-03",
  "end_date": "1962-01-04",
  "flood_duration": 1,
  "duration_unit": "days",
  "water_depth": 7.88,
  "water_unit": "m",
  "affected_people": 37,
  "affected_people_type": "families",
  "evacuated": 37,
  "evacuated_type": "families",
  "displaced": 37,
  "displaced_type": "families",
  "deaths": 2,
  "deaths_type": "people",
  "injured": 0,
  "injured_type": "people",
  "missing": 0,
  "missing_type": "people",
  "houses_damaged": 34,
  "houses_destroyed": 34,
  "roads_damaged": 5,
  "bridges_damaged": 0,
  "schools_damaged": 0,
  "hospitals_damaged": 0,
  "crop_damage": 500,
  "crop_damage_unit": "hectares",
  "livestock_loss": 1000,
  "livestock_loss_type": "hens",
  "economic_loss": 6000000,
  "economic_loss_currency": "pesetas",
  "response_agencies": [
    "Army",
    "Civil Guard",
    "Spanish University Union",
    "Youth Front",
    "volunteers",
    "firefighters"
  ],
  "summary": "Floodwaters washed in on January 3, 1962, causing the worst flood of the 20th century in Valladolid, flooding the Plaza del Poniente and devastating 25 villages. The flood was preceded by 15 days of rains and caused a Pisuerga surge of 7.88 meters above normal. The situation was particularly severe in Tierra de Campos, where rivers Cea, Valderaduey, and Sequillo were threatened with overflow. The flood caused significant damage, including the destruction of 34 buildings, the evacuation of 37 families, and the flooding of 500 hectares. The devastation was so severe that some villages were described as 'missing' due to the flood."
}
```

### 2021 · #3 — 37 fields · `article_000065697` (2021_06)

> Many in Berks County were surprised by the flooding 15 years ago

```json
{
  "country": "United States",
  "state": "Pennsylvania",
  "city": "Reading",
  "location": "Berks County",
  "river": "Schuylkill River",
  "water_body": "Blue Marsh Lake",
  "event_date": "2006-06-28",
  "water_level": "23.75",
  "water_unit": "feet",
  "flood_duration": "unknown",
  "duration_unit": "unknown",
  "affected_people": 0,
  "affected_people_type": "unknown",
  "evacuated": 0,
  "evacuated_type": "unknown",
  "displaced": 0,
  "displaced_type": "unknown",
  "deaths": 0,
  "deaths_type": "unknown",
  "injured": 0,
  "injured_type": "unknown",
  "missing": 0,
  "missing_type": "unknown",
  "houses_damaged": 0,
  "houses_destroyed": 0,
  "roads_damaged": 0,
  "bridges_damaged": 0,
  "schools_damaged": 0,
  "hospitals_damaged": 0,
  "crop_damage": 0,
  "crop_damage_unit": "unknown",
  "livestock_loss": 0,
  "livestock_loss_type": "unknown",
  "economic_loss": 0,
  "economic_loss_currency": "unknown",
  "response_agencies": "National Weather Service, Reading Fire Department, Fort Indiantown Gap, U.S. Army Corps of Engineers",
  "summary": "A severe flood occurred on June 28, 2006, in Berks County, Pennsylvania, caused by heavy rainfall leading to the Schuylkill River cresting at 23.75 feet. The flood inundated areas including the Reading Area Community College campus, several blocks near the riverfront, and parts of Perry Township. The Blue Marsh Lake spillway overflowed for the first time, releasing a large volume of water. Emergency responders conducted rescue operations, including a dramatic helicopter rescue. The flood was the worst in Berks County since the remnants of Tropical Storm Agnes in 1972, though it did not cause the same level of damage."
}
```

Runner-up field counts: 36, 34, 33, 33, 33


---

# 2022

## Corpus

| metric | value |
|---|---:|
| articles LLM-extracted | 151,280 |
| describe a flood event | 28,748 (19.0 %) |
| not a flood event | 122,532 (81.0 %) |
| **verifiable (date + location)** | **7,756** (5.1 % of all, 27.0 % of floods) |
| has >=1 flood date | 7,913 (27.5 % of floods) |
| has >=1 flooded location | 18,515 (64.4 % of floods) |
| event objects emitted | 24,060 |
| flood articles with 0 events | 8,555 |
| flood articles with 2+ events (prompt says exactly 1) | 1,521 |
| distinct event field names emitted | 1,249 |
| named a place but no usable date | 10,759 |

## Flood articles by batch month
```
2022_01  █████·····················    779  (21.3% of month, 3,651 in)
2022_02  ██████████████············  2,207  (25.5% of month, 8,647 in)
2022_03  ███████···················  1,166  (15.8% of month, 7,368 in)
2022_04  ████████████··············  1,985  (25.8% of month, 7,705 in)
2022_05  ██████████················  1,685  (18.7% of month, 9,011 in)
2022_06  ██████████████████········  2,992  (21.1% of month, 14,194 in)
2022_07  ███████████████████████···  3,708  (23.2% of month, 15,991 in)
2022_08  ██████████████████████████  4,243  (20.2% of month, 21,038 in)
2022_09  █████████████████████·····  3,353  (12.6% of month, 26,717 in)
2022_10  █████████████████████·····  3,406  (17.5% of month, 19,420 in)
2022_11  ████████··················  1,351  (14.8% of month, 9,113 in)
2022_12  ███████████···············  1,873  (22.2% of month, 8,425 in)
```

## Verifiable articles by batch month
```
2022_01  ██████····················    269
2022_02  ███████████████···········    676
2022_03  ███████████···············    509
2022_04  █████████·················    412
2022_05  ██████████················    438
2022_06  █████████████·············    587
2022_07  ██████████████████████····    998
2022_08  ███████████████████·······    857
2022_09  █████████████████·········    792
2022_10  ██████████████████████████  1,196
2022_11  ██████████················    465
2022_12  ████████████··············    557
```

## Extracted flood dates by month (in-year only, 8,825 date values)
```
2022-01  ██████····················    157
2022-02  ███████████████···········    374
2022-03  █████████████·············    338
2022-04  ██████████················    245
2022-05  ██████████················    244
2022-06  ██████████████············    340
2022-07  ████████████████████████··    603
2022-08  ████████████████··········    409
2022-09  ██████████████············    362
2022-10  ██████████████████████████    653
2022-11  ███████████···············    281
2022-12  █████████████·············    332
```

Out-of-year date values: **4,487** (50.8 % of all dates) — unparseable/partial: 0

Largest out-of-year buckets: `2023-10` 955, `2023-09` 429, `2023-08` 388, `2023-06` 321, `2023-07` 287, `2023-04` 231, `2021-07` 216, `2023-05` 146

## Field fill rates (share of the year's event objects)

| field | events | fill | field | events | fill |
|---|---:|---:|---|---:|---:|
| `location` | 13,847 | 57.6 % | `affected_people` | 12,063 | 50.1 % |
| `affected_people_type` | 11,786 | 49.0 % | `summary` | 11,776 | 48.9 % |
| `flooded_locations` | 9,588 | 39.9 % | `event_date` | 8,070 | 33.5 % |
| `deaths` | 7,352 | 30.6 % | `deaths_type` | 7,144 | 29.7 % |
| `houses_damaged` | 4,590 | 19.1 % | `response_agencies` | 4,348 | 18.1 % |
| `evacuated` | 4,260 | 17.7 % | `water_unit` | 4,076 | 16.9 % |
| `evacuated_type` | 3,941 | 16.4 % | `missing` | 3,344 | 13.9 % |
| `water_depth` | 3,280 | 13.6 % | `missing_type` | 3,115 | 12.9 % |
| `country` | 2,863 | 11.9 % | `houses_destroyed` | 2,192 | 9.1 % |
| `injured` | 1,928 | 8.0 % | `roads_damaged` | 1,878 | 7.8 % |
| `river` | 1,827 | 7.6 % | `injured_type` | 1,718 | 7.1 % |
| `displaced` | 1,699 | 7.1 % | `displaced_type` | 1,551 | 6.4 % |
| `economic_loss` | 1,465 | 6.1 % | `bridges_damaged` | 1,461 | 6.1 % |
| `cause` | 1,407 | 5.8 % | `crop_damage` | 1,324 | 5.5 % |
| `city` | 1,267 | 5.3 % | `economic_loss_currency` | 1,243 | 5.2 % |

## Top flooded locations (unsupervised — nothing was seeded)

| location | mentions | | country | mentions |
|---|---:|---|---|---:|
| Durban | 355 | | Brazil | 428 |
| Petropolis | 169 | | Pakistan | 276 |
| Petrópolis | 151 | | South Africa | 266 |
| Seoul | 136 | | India | 169 |
| KwaZulu-Natal | 135 | | Philippines | 139 |
| KwaZulu-Natal province | 123 | | Italy | 129 |
| Senigallia | 122 | | United States | 100 |
| Sylhet | 116 | | Australia | 92 |
| Lismore | 111 | | Afghanistan | 87 |
| Yellowstone National Park | 109 | | Bangladesh | 80 |
| Sunamganj | 104 | | Sudan | 75 |
| Kinshasa | 100 | | South Korea | 74 |
| Sindh | 93 | | China | 72 |
| Assam | 85 | | Indonesia | 72 |
| Gangnam district | 80 | | Iran | 71 |

**Top rivers:** Yellowstone River (61), Godavari (19), river (18), Indus River (15), Ahr (13), Misa (12), Elbe (12), Peace River (11), Bitoulet (10), Brahmaputra River (10)

**Top causes:** heavy rains (158), heavy rainfall (78), heavy rain (45), cloudburst (29), high rainfall (15), flood (9), heavy downpour (8), luapan kali ciliwung (8), heavy rains and clogged drains (8), torrential rains (7)

## Richest extracted records — the pipeline at its best

These are the event objects with the most non-empty fields in the year. Nothing was hand-picked beyond 'most fields filled'.

### 2022 · #1 — 39 fields · `article_000025550` (2022_06)

> Ten Years Later: Looking Back At The June 2012 Northland Flood

```json
{
  "country": "United States",
  "state": "Minnesota",
  "city": "Duluth",
  "location": "Northland",
  "event_date": "2012-06-19",
  "start_date": "2012-06-19",
  "end_date": "2012-06-20",
  "rainfall_mm": 104.5,
  "rainfall_mm_type": "millimeters",
  "water_depth": 2,
  "water_unit": "feet",
  "water_level": 16.62,
  "flood_duration": 1,
  "duration_unit": "day",
  "affected_people": 15000,
  "affected_people_type": "people",
  "evacuated": 600,
  "evacuated_type": "people",
  "displaced": 450,
  "displaced_type": "families",
  "deaths": 14,
  "deaths_type": "people",
  "injured": 82,
  "injured_type": "people",
  "missing": 5,
  "missing_type": "people",
  "houses_damaged": 1200,
  "roads_damaged": 50,
  "bridges_damaged": 3,
  "schools_damaged": 2,
  "hospitals_damaged": 1,
  "crop_damage": 1500000,
  "crop_damage_unit": "dollars",
  "livestock_loss": 350,
  "livestock_loss_type": "cattle",
  "economic_loss": 100000000,
  "economic_loss_currency": "USD",
  "response_agencies": [
    "Minnesota DNR",
    "National Weather Service",
    "Duluth office of the National Weather Service"
  ],
  "summary": "The June 2012 Northland Flood was the most damaging flood in Duluth's history, surpassing the 1972 flood. It was caused by heavy rainfall over two days, leading to widespread flooding, road damage, and evacuations. The flood resulted in significant economic losses, property damage, and impacts on infrastructure, including Jay Cooke State Park and the St. Louis River."
}
```

### 2022 · #2 — 38 fields · `article_000201258` (2022_08)

> Flash flooding in Dallas area takes residents by surprise as rescue crews respond to hundreds of calls for help

```json
{
  "location": "Dallas-Fort Worth area",
  "event_date": "2023-09-18",
  "rainfall_mm": 228.6,
  "rainfall_mm_type": "millimeters",
  "water_depth": 12,
  "water_unit": "inches",
  "water_depth_type": "flooding depth at heavy equipment rental business after one hour",
  "affected_people": 15000,
  "affected_people_type": "people",
  "evacuated": 600,
  "evacuated_type": "people",
  "displaced": 15000,
  "displaced_type": "people",
  "deaths": 1,
  "deaths_type": "people",
  "injured": 0,
  "injured_type": "people",
  "missing": 0,
  "missing_type": "people",
  "houses_damaged": 1,
  "houses_damaged_type": "heavy equipment rental business",
  "roads_damaged": 0,
  "roads_damaged_type": "roads",
  "bridges_damaged": 0,
  "bridges_damaged_type": "bridges",
  "schools_damaged": 0,
  "schools_damaged_type": "schools",
  "hospitals_damaged": 0,
  "hospitals_damaged_type": "hospitals",
  "crop_damage": 0,
  "crop_damage_unit": "units",
  "crop_damage_type": "crops",
  "livestock_loss": 0,
  "livestock_loss_type": "livestock",
  "economic_loss": 0,
  "economic_loss_currency": "USD",
  "response_agencies": [
    "Fort Worth Fire Department",
    "Dallas Fire Rescue",
    "Dallas Police Department",
    "Dallas Water Utilities Department"
  ],
  "summary": "Flash flooding in the Dallas-Fort Worth area caused significant damage, with over 9 inches of rainfall recorded at Dallas Forth Worth Airport over a 24-hour period. Rescue crews responded to hundreds of calls for aid, and one person was killed when floodwaters swept away her vehicle. The city experienced flooded streets, homes, and highways, with hundreds of traffic accidents reported."
}
```

### 2022 · #3 — 36 fields · `article_000188164` (2022_08)

> Bihar Flood News: Water level rises in Kosi, flood waters enter dozens of villages in Suppal

```json
{
  "river": "Kosi",
  "water_level": "all time high",
  "flooded_locations": [
    "Suppal",
    "Balwa",
    "Piparakurd",
    "Ghurran",
    "Basbitty",
    "Gopalpur Sireh",
    "Telwa Panchayat of Sadr block",
    "Nirmali Anandal area",
    "Sisauni",
    "Goharria",
    "Dighiya Panchayats of the Maruna block",
    "Nirmali block"
  ],
  "event_date": "Tuesday",
  "cause": "continuous rainfall in the lowlands and water catchment area of the Kosi River in Nepal",
  "trigger": "rising water level of Kosi",
  "water_depth": "not explicitly stated",
  "water_unit": "not explicitly stated",
  "flood_duration": "not explicitly stated",
  "duration_unit": "not explicitly stated",
  "affected_people": "not explicitly stated",
  "affected_people_type": "not explicitly stated",
  "evacuated": "not explicitly stated",
  "evacuated_type": "not explicitly stated",
  "displaced": "not explicitly stated",
  "displaced_type": "not explicitly stated",
  "deaths": "not explicitly stated",
  "deaths_type": "not explicitly stated",
  "injured": "not explicitly stated",
  "injured_type": "not explicitly stated",
  "missing": "not explicitly stated",
  "missing_type": "not explicitly stated",
  "houses_damaged": "not explicitly stated",
  "houses_destroyed": "not explicitly stated",
  "roads_damaged": "not explicitly stated",
  "bridges_damaged": "not explicitly stated",
  "schools_damaged": "not explicitly stated",
  "hospitals_damaged": "not explicitly stated",
  "crop_damage": "hundreds of acres of crops have been submerged",
  "crop_damage_unit": "acres",
  "livestock_loss": "not explicitly stated",
  "livestock_loss_type": "not explicitly stated",
  "economic_loss": "not explicitly stated",
  "economic_loss_currency": "not explicitly stated",
  "response_agencies": "not explicitly stated",
  "summary": "Flood water penetrated dozens of villages in Suppal due to rising water level of Kosi. The Kosi River is at an all time high on Tuesday, reaching its highest level this year due to continuous rainfall in the lowlands and water catchment area of the Kosi River in Nepal. Several villages have experienced rapid erosion of the river, which is causing people to be displaced. The floodwaters are spreading rapidly in several villages within the Kosi Dam. Floodwaters have flooded many homes, and hundreds of acres of crops have been submerged."
}
```

Runner-up field counts: 36, 36, 35, 35, 34


---

# 2023

## Corpus

| metric | value |
|---|---:|
| articles LLM-extracted | 381,411 |
| describe a flood event | 53,909 (14.1 %) |
| not a flood event | 327,502 (85.9 %) |
| **verifiable (date + location)** | **17,009** (4.5 % of all, 31.6 % of floods) |
| has >=1 flood date | 17,152 (31.8 % of floods) |
| has >=1 flooded location | 37,641 (69.8 % of floods) |
| event objects emitted | 46,184 |
| flood articles with 0 events | 13,800 |
| flood articles with 2+ events (prompt says exactly 1) | 2,477 |
| distinct event field names emitted | 1,613 |
| named a place but no usable date | 20,632 |

## Flood articles by batch month
```
2023_01  ███·······················  1,591  (9.6% of month, 16,498 in)
2023_02  ███·······················  1,643  (16.0% of month, 10,243 in)
2023_03  ████······················  2,177  (14.1% of month, 15,488 in)
2023_04  ██························  1,326  (8.1% of month, 16,338 in)
2023_05  █████████·················  4,834  (14.7% of month, 32,904 in)
2023_06  █████████████·············  6,992  (17.3% of month, 40,478 in)
2023_07  ████████████··············  6,577  (14.5% of month, 45,426 in)
2023_08  ████████··················  4,330  (7.8% of month, 55,530 in)
2023_09  ██████████████████████████ 14,220  (25.9% of month, 54,800 in)
2023_10  █████████·················  4,805  (12.6% of month, 38,180 in)
2023_11  █████·····················  2,797  (10.4% of month, 26,921 in)
2023_12  █████·····················  2,617  (9.1% of month, 28,605 in)
```

## Verifiable articles by batch month
```
2023_01  ████······················    634
2023_02  ███·······················    482
2023_03  ████······················    734
2023_04  ███·······················    480
2023_05  █████·····················    954
2023_06  █████████·················  1,654
2023_07  ██████████████············  2,445
2023_08  ████████··················  1,352
2023_09  ██████████████████████████  4,613
2023_10  ███████████···············  1,966
2023_11  █████·····················    837
2023_12  █████·····················    858
```

## Extracted flood dates by month (in-year only, 19,265 date values)
```
2023-01  ███·······················    628
2023-02  ███·······················    562
2023-03  ████······················    771
2023-04  ██························    451
2023-05  ██████····················  1,190
2023-06  █████████·················  1,703
2023-07  ██████████████············  2,730
2023-08  ███████···················  1,437
2023-09  ██████████████████████████  5,015
2023-10  █████████·················  1,827
2023-11  ████······················    815
2023-12  ████······················    777
```

Out-of-year date values: **1,356** (7.0 % of all dates) — unparseable/partial: 3

Largest out-of-year buckets: `2022-09` 161, `2022-12` 149, `2021-07` 116, `2026-07` 71, `2022-10` 51, `2022-06` 50, `2022-11` 49, `2022-07` 49

## Field fill rates (share of the year's event objects)

| field | events | fill | field | events | fill |
|---|---:|---:|---|---:|---:|
| `affected_people` | 26,426 | 57.2 % | `summary` | 25,983 | 56.3 % |
| `affected_people_type` | 25,961 | 56.2 % | `location` | 24,475 | 53.0 % |
| `deaths` | 19,204 | 41.6 % | `deaths_type` | 18,712 | 40.5 % |
| `event_date` | 17,248 | 37.3 % | `flooded_locations` | 17,212 | 37.3 % |
| `missing` | 13,875 | 30.0 % | `missing_type` | 13,173 | 28.5 % |
| `evacuated` | 10,798 | 23.4 % | `evacuated_type` | 10,240 | 22.2 % |
| `response_agencies` | 9,596 | 20.8 % | `houses_damaged` | 8,369 | 18.1 % |
| `country` | 7,616 | 16.5 % | `water_unit` | 6,041 | 13.1 % |
| `displaced` | 5,654 | 12.2 % | `displaced_type` | 5,344 | 11.6 % |
| `water_depth` | 4,845 | 10.5 % | `roads_damaged` | 4,060 | 8.8 % |
| `river` | 4,032 | 8.7 % | `injured` | 3,851 | 8.3 % |
| `city` | 3,804 | 8.2 % | `injured_type` | 3,503 | 7.6 % |
| `houses_destroyed` | 3,410 | 7.4 % | `cause` | 3,388 | 7.3 % |
| `economic_loss` | 3,263 | 7.1 % | `bridges_damaged` | 3,143 | 6.8 % |
| `economic_loss_currency` | 2,645 | 5.7 % | `crop_damage` | 2,292 | 5.0 % |

## Top flooded locations (unsupervised — nothing was seeded)

| location | mentions | | country | mentions |
|---|---:|---|---|---:|
| Derna | 7,033 | | Libya | 2,306 |
| Benghazi | 1,232 | | India | 830 |
| Susa | 1,149 | | Italy | 527 |
| Bayda | 1,118 | | Greece | 423 |
| Marj | 1,060 | | United States | 378 |
| Shahatt | 731 | | Brazil | 283 |
| Kherson | 676 | | Democratic Republic of Congo | 271 |
| Kherson region | 475 | | Ukraine | 243 |
| Nyamukubi | 453 | | China | 223 |
| Derne | 379 | | South Korea | 181 |
| Volos | 377 | | Turkey | 163 |
| Faenza | 375 | | Afghanistan | 111 |
| Wadi Derna | 366 | | Spain | 96 |
| Delaware River | 361 | | Haiti | 83 |
| Bushushu | 356 | | New Zealand | 78 |

**Top rivers:** Teesta River (259), Dnieper River (184), Teesta (143), River Rother (127), River South Esk (120), Mendenhall River (104), Yamuna (100), Wadi Derna (89), Dnipro River (68), Winooski River (55)

**Top causes:** heavy rains (403), collapse of two dams (175), storm daniel (112), heavy rainfall (93), cloudburst (69), mediterranean storm daniel (64), heavy rain (54), dam collapse (49), hurricane daniel (46), glacial lake overflow (40)

## Richest extracted records — the pipeline at its best

These are the event objects with the most non-empty fields in the year. Nothing was hand-picked beyond 'most fields filled'.

### 2023 · #1 — 50 fields · `article_000010428` (2023_06)

> : Floods in Serbia: Floodwaters in 53 municipalities, more than 300 people evacuated  Water over houses, damaged roads, water collapsed and bridge in Mrčajevski

```json
{
  "flooded_locations": [
    "Mrčajevski",
    "Adreni",
    "Kraljevo",
    "Prokoplje",
    "Draganija",
    "Kurshumlija",
    "Prokuplje",
    "Mala Guba",
    "Tarnavica River",
    "Josanica River",
    "Kucevo",
    "Jagodina",
    "Cerimidžice",
    "Trnava",
    "Junkovac",
    "Kragujevac",
    "Grosnica",
    "Doljevac",
    "Klece",
    "Lazarevac",
    "Rudovca",
    "Charlinca",
    "Shainovac",
    "Glibovac",
    "Privorica",
    "Palanka",
    "Kubrshnica"
  ],
  "flood_dates": [
    "2023-06-18"
  ],
  "water_level": "308 centimeters",
  "evacuated_people": 24,
  "evacuated_children": 7,
  "evacuated_pregnant_women": 1,
  "evacuated_people_from_other_areas": 10,
  "evacuated_children_from_other_areas": 5,
  "flooded_houses": 30,
  "flooded_courtyards": 30,
  "flooded_auxiliary_facilities": 30,
  "bridge_damage": "bridge collapse in Adrani",
  "bridge_repair": "new bridge construction announced",
  "bridge_construction_deadline": "four months",
  "bridge_construction_company": "Novi Bazar Road JSC",
  "pumps_installed": "large-capacity pump in Karadjordje",
  "pumps_discharge": "atmospheric and groundwater into Kubrznica stream",
  "embankment_damage": "river Kubrshnica broke through part of the embankment",
  "embankment_location": "between the village atars of Glibovac and Privorica",
  "emergency_measures": "emergency intervention of the emergency headquarters and the sector, MUD, reinforcement of ramparts, bringing more materials, reinforcing fortifications",
  "electricity_outage": "four substations in Lazarevac shut down, leaving 550 households without electricity for several hours",
  "electricity_restored": "all households in Lazarevac have electricity",
  "preschool_reopening": "preschool institutions in Rudovca expected to reopen on Monday",
  "military_intervention": "Serbian Army built a pontoon bridge across the river Trnavica in Klece",
  "elderly_home": "Home for the Elderly Disabled in Doljevac is not endangered",
  "elderly_home_users": 84,
  "elderly_home_basement": "flooded basement rooms",
  "bridge_age": "bridge in Mrčajevski built in the 1980's",
  "bridge_renovation": "state allocated 500 million dinars from the budget to rebuild the bridge, but this had not happened because the water was high",
  "bridge_new_cost": "probably twice or twice two and a half times the previous allocation",
  "bridge_new_construction": "new bridge will be built",
  "bridge_construction_start": "urgent diversion and demolition of the old bridge",
  "bridge_construction_completion": "ready for traffic in the next ten days",
  "river_level": "Toplica River level several days above the flood protection limit",
  "river_forecast": "heavy rainfall forecasted overnight",
  "river_defenses": "emergency flood defenses still in place",
  "river_water_level": "about 12 feet [306 cm]",
  "river_stabilization": "lake is at a standstill",
  "river_peak": "peak of the wave passing through Palanka on both rivers",
  "river_teams": "teams of the relevant services still on the ground and monitoring the situation",
  "river_water_discharge": "water being drawn from residential and commercial buildings and road infrastructure being renovated",
  "river_damage": "landslide in Grosnica activated, damaging part of the main traffic lane",
  "river_alternative_route": "alternative route through the village of Čava",
  "river_damage_compensation": "appeal to citizens to report damages to the telephone number 034/505-800 or to the Operational Center of the Ministry of Internal Affairs at the telephone number 034/1985",
  "river_emergency_declaration": "emergency situation declared in Kragujevac",
  "river_infrastructure_damage": "great damage to infrastructure and residential buildings",
  "river_repair_efforts": "emergency department, public utility workers, and firefighters on the ground working to repair the damage",
  "river_situation_control": "situation under control, river watercourses in their basins",
  "river_water_level_decline": "water level in decline",
  "river_water_supply": "two pumps that supply about 20 percent of the city's water closed or partially shut down, making them field cisterns"
}
```

### 2023 · #2 — 41 fields · `article_000251264` (2023_07)

> Jul 2023: Search ongoing for four missing after NS flooding

```json
{
  "country": "Canada",
  "province": "Nova Scotia",
  "city": "West Hants",
  "location": "flooded field",
  "event_date": "2023-07-16",
  "start_date": "2023-07-16",
  "end_date": "2023-07-19",
  "cause": "record rainfall",
  "trigger": "torrential rains",
  "rainfall_mm": 200,
  "water_depth": 2.5,
  "water_unit": "m",
  "flood_duration": 4,
  "duration_unit": "days",
  "affected_people": 500,
  "affected_people_type": "people",
  "evacuated": 600,
  "evacuated_type": "people",
  "displaced": 200,
  "displaced_type": "people",
  "missing": 4,
  "missing_type": "people",
  "deaths": 0,
  "deaths_type": "people",
  "injured": 0,
  "injured_type": "people",
  "houses_damaged": 0,
  "houses_destroyed": 0,
  "roads_damaged": 25,
  "bridges_damaged": 19,
  "bridges_destroyed": 6,
  "schools_damaged": 0,
  "hospitals_damaged": 0,
  "crop_damage": 0,
  "crop_damage_unit": "hectares",
  "livestock_loss": 0,
  "livestock_loss_type": "cattle",
  "economic_loss": 0,
  "economic_loss_currency": "CAD",
  "response_agencies": [
    "RCMP",
    "Halifax Regional Municipality",
    "Premier Tim Houston",
    "Emergency Preparedness Minister Bill Blair"
  ],
  "summary": "Search continues for four people missing after record rainfall in Nova Scotia caused floodwaters that submerged vehicles and led to the destruction of roads and bridges. The flood occurred in West Hants and affected areas including the Meander River, Crawford Brook, Hammonds Plains, Bedford, and Lower Sackville. The event lasted from July 16 to July 19, 2023, with over 200 mm of rainfall recorded in some areas."
}
```

### 2023 · #3 — 40 fields · `article_000000915` (2023_07)

> Tragedy in Crimea: Flooding in Krasnodar region on July 6, 2012

```json
{
  "country": "Russia",
  "region": "Krasnodar region",
  "city": "Krymsk",
  "location": "Adagum River basin",
  "river": "Adagum River",
  "event_date": "2012-07-06",
  "start_date": "2012-07-06",
  "end_date": "2012-07-07",
  "cause": "heavy rains",
  "trigger": "high-altitude cyclone",
  "rainfall_mm": 311,
  "water_depth": 1.5,
  "water_unit": "m",
  "water_level": 7.1,
  "flood_duration": 1,
  "duration_unit": "days",
  "affected_people": 53000,
  "affected_people_type": "people",
  "evacuated": 29000,
  "evacuated_type": "people",
  "displaced": 29000,
  "displaced_type": "people",
  "deaths": 171,
  "deaths_type": "people",
  "injured": 82,
  "injured_type": "people",
  "missing": 5,
  "missing_type": "people",
  "houses_damaged": 8000,
  "houses_destroyed": 1700,
  "roads_damaged": true,
  "bridges_damaged": true,
  "schools_damaged": true,
  "hospitals_damaged": true,
  "crop_damage": true,
  "livestock_loss": true,
  "livestock_loss_type": "birds",
  "economic_loss": true,
  "response_agencies": "Russian Hydromet, Investigative Committee of the Russian Federation",
  "summary": "A catastrophic flood occurred in the Krasnodar region of Russia on July 6, 2012, caused by extreme rainfall. The flood affected multiple locations, including Krymsk, Nizhny Bakansky, Gelendzhik, and Novorossiysk, with over 53,000 people affected, 171 deaths, and significant damage to infrastructure. The flood was exacerbated by poor maintenance of the Adagum River basin and inadequate emergency response."
}
```

Runner-up field counts: 39, 36, 36, 35, 34
