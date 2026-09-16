# Final-output exemplars + schema drift (2021 · 2022 · 2023)

Scored two ways. **Canonical** counts only the 50 fields `system_prompt.txt` names, and only for articles that passed the verifiability gate — this is the honest 'here is what one good record looks like' exhibit. **Absolute** counts every key the model emitted, which is the schema-drift exhibit.


---

## 2021

### A. Best verifiable record (canonical-scored)

### 2021 canonical #1 — **39 of 50 canonical** fields (39 total) · `article_000056052` · 2021_09

**Headline:** New York, New Jersey, where Hurricane Ida, the "death wreck", caused massive flooding, killing at least 45 people in the southeastern United States.

**Gate:** `is_verifiable_flood = True` · dates `['2021-09-01', '2021-09-02']` · 6 flooded locations

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

*No off-schema keys — this record is fully on-schema.*

### 2021 canonical #2 — **37 of 50 canonical** fields (37 total) · `article_000145665` · 2021_05

**Headline:** Washed away by the waters... north of Castile.

**Gate:** `is_verifiable_flood = True` · dates `['1962-01-03']` · 28 flooded locations

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

*No off-schema keys — this record is fully on-schema.*

### B. Best record on the year's marquee event — Chamoli / Ida / Henan

### 2021 Chamoli / Ida / Henan #1 — **39 of 50 canonical** fields (39 total) · `article_000056052` · 2021_09

**Headline:** New York, New Jersey, where Hurricane Ida, the "death wreck", caused massive flooding, killing at least 45 people in the southeastern United States.

**Gate:** `is_verifiable_flood = True` · dates `['2021-09-01', '2021-09-02']` · 6 flooded locations

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

*No off-schema keys — this record is fully on-schema.*

### C. Most fields of any name (schema-drift exhibit)

### 2021 absolute #1 — **39 of 50 canonical** fields (39 total) · `article_000056052` · 2021_09

**Headline:** New York, New Jersey, where Hurricane Ida, the "death wreck", caused massive flooding, killing at least 45 people in the southeastern United States.

**Gate:** `is_verifiable_flood = True` · dates `['2021-09-01', '2021-09-02']` · 6 flooded locations

```json
c
```

*No off-schema keys — this record is fully on-schema.*


---

## 2022

### A. Best verifiable record (canonical-scored)

### 2022 canonical #1 — **38 of 50 canonical** fields (39 total) · `article_000025550` · 2022_06

**Headline:** Ten Years Later: Looking Back At The June 2012 Northland Flood

**Gate:** `is_verifiable_flood = True` · dates `['2012-06-19', '2012-06-20']` · 18 flooded locations

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

*Plus 1 off-schema keys the model invented:* `rainfall_mm_type`

### 2022 canonical #2 — **36 of 50 canonical** fields (36 total) · `article_000199015` · 2022_08

**Headline:** Anniversary of flood disaster in Ayancik

**Gate:** `is_verifiable_flood = True` · dates `['2022-08-11']` · 4 flooded locations

```json
{
  "country": "Turkey",
  "district": "Ayancik",
  "city": "Sinop",
  "event_date": "2022-08-11",
  "cause": "heavy rainfall",
  "rainfall_mm": "240",
  "water_depth": "not specified",
  "water_unit": "not specified",
  "flood_duration": "not specified",
  "duration_unit": "not specified",
  "affected_people": "16",
  "affected_people_type": "people",
  "evacuated": "not specified",
  "evacuated_type": "not specified",
  "displaced": "not specified",
  "displaced_type": "not specified",
  "deaths": "16",
  "deaths_type": "people",
  "injured": "not specified",
  "injured_type": "not specified",
  "missing": "not specified",
  "missing_type": "not specified",
  "houses_damaged": "not specified",
  "houses_destroyed": "not specified",
  "roads_damaged": "not specified",
  "bridges_damaged": "not specified",
  "schools_damaged": "not specified",
  "hospitals_damaged": "not specified",
  "crop_damage": "not specified",
  "crop_damage_unit": "not specified",
  "livestock_loss": "not specified",
  "livestock_loss_type": "not specified",
  "economic_loss": "not specified",
  "economic_loss_currency": "not specified",
  "response_agencies": "not specified",
  "summary": "A flood disaster occurred in Ayancik district of Sinop on August 11, 2022, caused by heavy rainfall. The flood resulted in 16 deaths and caused significant damage to infrastructure, including bridges, roads, and art structures. The floodwaters submerged the district, and helicopters were deployed to rescue people trapped on rooftops."
}
```

*No off-schema keys — this record is fully on-schema.*

### B. Best record on the year's marquee event — Pakistan / Petropolis / Durban

### 2022 Pakistan / Petropolis / Durban #1 — **31 of 50 canonical** fields (31 total) · `article_000019424` · 2022_02

**Headline:** The death toll from the torrential rains in Rio de Janeiro is up to 94, with 35 still missing

**Gate:** `is_verifiable_flood = True` · dates `['2026-02-15']` · 3 flooded locations

```json
{
  "country": "Brazil",
  "state": "Rio de Janeiro",
  "city": "Petrópolis",
  "location": "Alto da Serra",
  "event_date": "2026-02-15",
  "affected_people": 94,
  "affected_people_type": "people",
  "missing": 35,
  "missing_type": "people",
  "evacuated": 377,
  "evacuated_type": "people",
  "deaths": 94,
  "deaths_type": "people",
  "injured": 0,
  "injured_type": "people",
  "displaced": 372,
  "displaced_type": "people",
  "houses_damaged": 80,
  "houses_destroyed": 55,
  "roads_damaged": 0,
  "bridges_damaged": 0,
  "schools_damaged": 0,
  "hospitals_damaged": 0,
  "crop_damage": 0,
  "crop_damage_unit": "hectares",
  "livestock_loss": 0,
  "livestock_loss_type": "animals",
  "economic_loss": 0,
  "economic_loss_currency": "BRL",
  "response_agencies": [
    "Brazilian Civil Defense Department",
    "Firefighters",
    "Police",
    "Forensic experts",
    "Civil Defense Department"
  ],
  "summary": "Torrential rains caused flooding and landslides in Petrópolis, Rio de Janeiro, leading to 94 deaths, 35 missing, and significant damage to homes and infrastructure. Rescue operations were ongoing with multiple agencies involved."
}
```

*No off-schema keys — this record is fully on-schema.*

### C. Most fields of any name (schema-drift exhibit)

### 2022 absolute #1 — **38 of 50 canonical** fields (39 total) · `article_000025550` · 2022_06

**Headline:** Ten Years Later: Looking Back At The June 2012 Northland Flood

**Gate:** `is_verifiable_flood = True` · dates `['2012-06-19', '2012-06-20']` · 18 flooded locations

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

*Plus 1 off-schema keys the model invented:* `rainfall_mm_type`


---

## 2023

### A. Best verifiable record (canonical-scored)

### 2023 canonical #1 — **40 of 50 canonical** fields (41 total) · `article_000251264` · 2023_07

**Headline:** Jul 2023: Search ongoing for four missing after NS flooding

**Gate:** `is_verifiable_flood = True` · dates `['2023-07-16', '2023-07-17', '2023-07-18', '2023-07-19']` · 7 flooded locations

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

*Plus 1 off-schema keys the model invented:* `bridges_destroyed`

### 2023 canonical #2 — **39 of 50 canonical** fields (40 total) · `article_000000915` · 2023_07

**Headline:** Tragedy in Crimea: Flooding in Krasnodar region on July 6, 2012

**Gate:** `is_verifiable_flood = True` · dates `['2012-07-06']` · 8 flooded locations

```json
{
  "country": "Russia",
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

*Plus 1 off-schema keys the model invented:* `region`

### B. Best record on the year's marquee event — Derna / Kakhovka / Sikkim

### 2023 Derna / Kakhovka / Sikkim #1 — **39 of 50 canonical** fields (39 total) · `article_000117943` · 2023_10

**Headline:** Sikkim floods: How glacial lake outburst flood (GLOF) led to massive devastation

**Gate:** `is_verifiable_flood = True` · dates `['2023-10-03', '2023-10-04']` · 10 flooded locations

```json
{
  "country": "India",
  "state": "Sikkim",
  "district": "North Sikkim",
  "location": "Lhonak Lake",
  "river": "Teesta River",
  "event_date": "2023-10-03",
  "start_date": "2023-10-03",
  "end_date": "2023-10-04",
  "cause": "glacial lake outburst flood (GLOF)",
  "trigger": "cloudburst",
  "rainfall_mm": 40.9,
  "water_depth": 2.5,
  "water_unit": "metres",
  "water_level": 298.4,
  "flood_duration": 1,
  "duration_unit": "day",
  "affected_people": 15000,
  "affected_people_type": "people",
  "evacuated": 600,
  "evacuated_type": "people",
  "displaced": 15,
  "displaced_type": "families",
  "deaths": 14,
  "deaths_type": "people",
  "injured": 82,
  "injured_type": "people",
  "missing": 102,
  "missing_type": "people",
  "houses_damaged": 15,
  "houses_destroyed": 15,
  "roads_damaged": 14,
  "bridges_damaged": 14,
  "schools_damaged": 0,
  "hospitals_damaged": 0,
  "crop_damage": 0,
  "livestock_loss": 0,
  "economic_loss": 0,
  "response_agencies": "Sikkim government, Sikkim Urja Limited, emergency services, India's army",
  "summary": "Catastrophic flooding caused by a glacial lake outburst flood (GLOF) led to massive devastation in Sikkim, India. The floods were triggered by a cloudburst over the Lhonak glacial lake, resulting in 14 deaths, 82 injured, and 102 people missing. The Teesta River overflowed, damaging infrastructure such as the Teesta-III dam, 14 bridges, and National Highway 10. Over 15 houses were washed away in Singtam, and the situation was described as a 'disaster waiting to happen' due to prior warnings about the vulnerability of the Lhonak Lake."
}
```

*No off-schema keys — this record is fully on-schema.*

### C. Most fields of any name (schema-drift exhibit)

### 2023 absolute #1 — **1 of 50 canonical** fields (50 total) · `article_000010428` · 2023_06

**Headline:** : Floods in Serbia: Floodwaters in 53 municipalities, more than 300 people evacuated  Water over houses, damaged roads, water collapsed and bridge in Mrčajevski

**Gate:** `is_verifiable_flood = True` · dates `['2023-06-18']` · 27 flooded locations

```json
{
  "water_level": "308 centimeters"
}
```

*Plus 49 off-schema keys the model invented:* `flooded_locations`, `flood_dates`, `evacuated_people`, `evacuated_children`, `evacuated_pregnant_women`, `evacuated_people_from_other_areas`, `evacuated_children_from_other_areas`, `flooded_houses`, `flooded_courtyards`, `flooded_auxiliary_facilities`, `bridge_damage`, `bridge_repair`, `bridge_construction_deadline`, `bridge_construction_company` …


---

## Schema drift, measured

| year | distinct field names | of which invented | canonical share of filled values | filled values |
|---|---:|---:|---:|---:|
| 2021 | 2,645 | 2,595 (98.1 %) | **88.3 %** | 173,065 |
| 2022 | 1,241 | 1,191 (96.0 %) | **90.3 %** | 151,107 |
| 2023 | 1,592 | 1,542 (96.9 %) | **92.3 %** | 336,068 |

Read this as: the *long tail of names* is huge, but the *mass of data* still lands on the sanctioned schema. The invented keys are mostly one-off singletons.
