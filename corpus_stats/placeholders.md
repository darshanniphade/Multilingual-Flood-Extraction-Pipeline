# Placeholder values — the hidden hole in every fill rate

`system_prompt.txt` forbids nulls, empty strings and empty arrays. The model complies literally and writes prose instead: `"deaths": "not specified"`. Any fill-rate computed by 'is the key present and non-empty' counts those as data.

## How much of the 'filled' data is not data

| year | event objects | non-empty values | `not specified`-type | zeros | bare booleans | **real values** |
|---|---:|---:|---:|---:|---:|---:|
| 2021 | 33,029 | 173,065 | 1,525 (0.9 %) | 4,989 (2.9 %) | 680 (0.4 %) | **165,871 (95.8 %)** |
| 2022 | 24,060 | 151,107 | 650 (0.4 %) | 5,868 (3.9 %) | 781 (0.5 %) | **143,808 (95.2 %)** |
| 2023 | 46,184 | 336,068 | 685 (0.2 %) | 12,385 (3.7 %) | 1,035 (0.3 %) | **321,963 (95.8 %)** |


### 2021 — fill rate, naive vs real

| field | naive fill | real fill | inflation |
|---|---:|---:|---:|
| `summary` | 51.5 % | **51.5 %** | +0.0 pp |
| `affected_people` | 49.4 % | **47.7 %** | +1.7 pp |
| `location` | 41.7 % | **41.7 %** | +0.0 pp |
| `flooded_locations` | 29.2 % | **29.2 %** | +0.0 pp |
| `event_date` | 29.0 % | **29.0 %** | +0.0 pp |
| `response_agencies` | 25.5 % | **25.4 %** | +0.0 pp |
| `affected_people_type` | 25.0 % | **25.0 %** | +0.0 pp |
| `deaths` | 19.6 % | **18.6 %** | +1.0 pp |
| `houses_damaged` | 18.6 % | **16.7 %** | +1.9 pp |
| `evacuated` | 15.4 % | **13.9 %** | +1.6 pp |
| `deaths_type` | 13.6 % | **13.6 %** | +0.0 pp |
| `cause` | 13.6 % | **13.6 %** | +0.0 pp |
| `country` | 11.3 % | **11.3 %** | +0.0 pp |
| `missing` | 10.6 % | **9.4 %** | +1.2 pp |
| `water_unit` | 10.3 % | **10.3 %** | +0.0 pp |
| `evacuated_type` | 9.7 % | **9.7 %** | +0.0 pp |
| `trigger` | 9.2 % | **9.2 %** | +0.0 pp |
| `river` | 8.5 % | **8.5 %** | +0.0 pp |

Most common placeholder strings: `not specified` 594, `not explicitly mentioned` 503, `unknown` 188, `not mentioned` 118, `not explicitly stated` 94, `unspecified` 15

Fields most often placeholdered: `houses_damaged` 190, `economic_loss` 125, `livestock_loss` 117, `bridges_damaged` 115, `crop_damage` 108, `displaced` 103, `affected_people` 101, `roads_damaged` 100


### 2022 — fill rate, naive vs real

| field | naive fill | real fill | inflation |
|---|---:|---:|---:|
| `location` | 57.6 % | **57.6 %** | +0.0 pp |
| `affected_people` | 50.1 % | **47.6 %** | +2.6 pp |
| `affected_people_type` | 49.0 % | **48.9 %** | +0.1 pp |
| `summary` | 48.9 % | **48.9 %** | +0.0 pp |
| `flooded_locations` | 39.9 % | **39.9 %** | +0.0 pp |
| `event_date` | 33.5 % | **33.5 %** | +0.0 pp |
| `deaths` | 30.6 % | **29.1 %** | +1.5 pp |
| `deaths_type` | 29.7 % | **29.6 %** | +0.1 pp |
| `houses_damaged` | 19.1 % | **16.8 %** | +2.2 pp |
| `response_agencies` | 18.1 % | **18.0 %** | +0.1 pp |
| `evacuated` | 17.7 % | **15.0 %** | +2.7 pp |
| `water_unit` | 16.9 % | **16.9 %** | +0.1 pp |
| `evacuated_type` | 16.4 % | **16.3 %** | +0.1 pp |
| `missing` | 13.9 % | **12.0 %** | +1.9 pp |
| `water_depth` | 13.6 % | **13.5 %** | +0.1 pp |
| `missing_type` | 12.9 % | **12.9 %** | +0.1 pp |
| `country` | 11.9 % | **11.9 %** | +0.0 pp |
| `houses_destroyed` | 9.1 % | **7.5 %** | +1.6 pp |

Most common placeholder strings: `not specified` 363, `not explicitly stated` 149, `unknown` 101, `not mentioned` 19, `not explicitly mentioned` 17, `unspecified` 1

Fields most often placeholdered: `economic_loss` 36, `bridges_damaged` 35, `missing` 34, `economic_loss_currency` 32, `houses_damaged` 30, `schools_damaged` 28, `evacuated` 28, `roads_damaged` 28


### 2023 — fill rate, naive vs real

| field | naive fill | real fill | inflation |
|---|---:|---:|---:|
| `affected_people` | 57.2 % | **55.0 %** | +2.2 pp |
| `summary` | 56.3 % | **56.3 %** | +0.0 pp |
| `affected_people_type` | 56.2 % | **56.2 %** | +0.0 pp |
| `location` | 53.0 % | **53.0 %** | +0.0 pp |
| `deaths` | 41.6 % | **40.0 %** | +1.6 pp |
| `deaths_type` | 40.5 % | **40.5 %** | +0.0 pp |
| `event_date` | 37.3 % | **37.3 %** | +0.0 pp |
| `flooded_locations` | 37.3 % | **37.3 %** | +0.0 pp |
| `missing` | 30.0 % | **28.5 %** | +1.6 pp |
| `missing_type` | 28.5 % | **28.5 %** | +0.0 pp |
| `evacuated` | 23.4 % | **21.0 %** | +2.4 pp |
| `evacuated_type` | 22.2 % | **22.2 %** | +0.0 pp |
| `response_agencies` | 20.8 % | **20.8 %** | +0.0 pp |
| `houses_damaged` | 18.1 % | **15.8 %** | +2.4 pp |
| `country` | 16.5 % | **16.5 %** | +0.0 pp |
| `water_unit` | 13.1 % | **13.0 %** | +0.0 pp |
| `displaced` | 12.2 % | **10.5 %** | +1.7 pp |
| `displaced_type` | 11.6 % | **11.6 %** | +0.0 pp |

Most common placeholder strings: `unknown` 265, `not specified` 195, `not explicitly stated` 140, `none` 38, `not mentioned` 23, `unspecified` 15

Fields most often placeholdered: `crop_damage_unit` 64, `economic_loss` 61, `economic_loss_currency` 60, `livestock_loss_type` 55, `bridges_damaged` 53, `houses_damaged` 41, `schools_damaged` 39, `livestock_loss` 35


---

# The strongest real record of each year

Verifiable, flood date inside the article's own batch month, ranked by canonical fields holding a **real** value (placeholders, zeros and bare booleans excluded).


## 2021

### 2021 · #1 — 39 real canonical fields · `article_000056052` · 2021_09

**Headline:** New York, New Jersey, where Hurricane Ida, the "death wreck", caused massive flooding, killing at least 45 people in the southeastern United States.

**Gate:** dates `['2021-09-01', '2021-09-02']` · 6 locations `['New York', 'New Jersey', 'New York City', 'Newark', 'Elizabeth', 'New York Free International Airport']`

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

### 2021 · #2 — 36 real canonical fields · `article_000116950` · 2021_12

**Headline:** The worst flooding in the dry zone

**Gate:** dates `['2021-12-18']` · 7 locations `['Jelebu', 'Chennah Village', 'Puom Village', 'Pah Village', 'Kampung Chennah', 'Kampung Puom']` …

```json
{
  "country": "Malaysia",
  "state": "Negeri Sembilan",
  "district": "Jelebu",
  "city": "Jelebu",
  "location": "Jelebu",
  "river": "Jelebu River",
  "event_date": "2021-12-18",
  "rainfall_mm": 199,
  "water_depth": 1,
  "water_unit": "m",
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
  "houses_destroyed": 300,
  "roads_damaged": 50,
  "bridges_damaged": 10,
  "schools_damaged": 8,
  "hospitals_damaged": 3,
  "crop_damage": 2000000,
  "crop_damage_unit": "ringgit",
  "livestock_loss": 350,
  "livestock_loss_type": "cattle",
  "economic_loss": 5000000,
  "economic_loss_currency": "ringgit",
  "response_agencies": "state and federal authorities, NGOs, communities from neighboring counties",
  "summary": "The floods that hit eight states including Negeri Sembilan on December 18, 2021, are being described as among the worst in the country in 50 years. The disaster occurred after heavy rainfall for more than five hours, causing severe flash flooding in some low-lying areas and flood risk in the Triang River basin. The damage to livestock and crops was immense, while some tourist attractions were also destroyed. Victim relief and cleanup efforts are now being actively undertaken by authorities - state and federal - assisted by NGOs and communities from neighboring counties."
}
```


## 2022

### 2022 · #1 — 32 real canonical fields · `article_000155574` · 2022_07

**Headline:** Deluge of woes for farmers

**Gate:** dates `['2022-07-16', '2022-07-17']` · 11 locations `['Khanapur village', 'Manthani mandal', 'Peddapalli district', 'Godavari river bank', 'Eklaspur', 'Surepalli']` …

```json
{
  "location": "Khanapur village",
  "river": "Godavari",
  "event_date": "2022-07-16",
  "flood_duration": 2,
  "duration_unit": "days",
  "water_depth": 2,
  "water_unit": "feet",
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
  "houses_damaged": 350,
  "roads_damaged": 1,
  "bridges_damaged": 3,
  "schools_damaged": 2,
  "hospitals_damaged": 1,
  "crop_damage": 200,
  "crop_damage_unit": "acres",
  "livestock_loss": 350,
  "livestock_loss_type": "cattle",
  "economic_loss": 1000000,
  "economic_loss_currency": "INR",
  "response_agencies": "NPDCL",
  "summary": "Monsoon rains caused severe flooding in Khanapur village, Peddapalli district, affecting thousands of farmers and their families. The flood occurred on July 16 and 17, 2022, with water depths reaching up to 2 feet. Significant damage was reported to agricultural fields, infrastructure, and public facilities. Over 15,000 people were affected, with 600 evacuated and 450 families displaced. The flood resulted in 14 deaths, 82 injuries, and 5 missing. Economic losses were estimated at 1,000,000 INR."
}
```

### 2022 · #2 — 31 real canonical fields · `article_000075358` · 2022_09

**Headline:** Fiona's floods devastated their homes. These residents are ready to start over

**Gate:** dates `['2022-09-18']` · 4 locations `['Villa Esperanza', 'La Playa neighborhood', 'Salinas', 'Rio Nigua']`

```json
{
  "location": "Villa Esperanza",
  "river": "Rio Nigua",
  "event_date": "2022-09-18",
  "water_depth": 4,
  "water_unit": "ft",
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
  "houses_damaged": 200,
  "houses_destroyed": 50,
  "roads_damaged": 15,
  "bridges_damaged": 3,
  "schools_damaged": 2,
  "hospitals_damaged": 1,
  "crop_damage": 1000000,
  "crop_damage_unit": "USD",
  "livestock_loss": 350,
  "livestock_loss_type": "cattle",
  "economic_loss": 5000000,
  "economic_loss_currency": "USD",
  "response_agencies": [
    "local government"
  ],
  "summary": "Hurricane Fiona caused severe flooding in Salinas, Puerto Rico, particularly in Villa Esperanza and La Playa neighborhood. Over 30 inches of rain led to the Rio Nigua river flooding, destroying homes and displacing residents. The flood was exacerbated by a bridge acting as a dam, trapping debris and forcing the river out of its banks. Many residents have since evacuated and are staying in a local high school shelter."
}
```


## 2023

### 2023 · #1 — 34 real canonical fields · `article_000117943` · 2023_10

**Headline:** Sikkim floods: How glacial lake outburst flood (GLOF) led to massive devastation

**Gate:** dates `['2023-10-03', '2023-10-04']` · 10 locations `['Mangan', 'Gangtok', 'Pakyong', 'Namchi Districts', 'Chunthang', 'Singtam']` …

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
  "response_agencies": "Sikkim government, Sikkim Urja Limited, emergency services, India's army",
  "summary": "Catastrophic flooding caused by a glacial lake outburst flood (GLOF) led to massive devastation in Sikkim, India. The floods were triggered by a cloudburst over the Lhonak glacial lake, resulting in 14 deaths, 82 injured, and 102 people missing. The Teesta River overflowed, damaging infrastructure such as the Teesta-III dam, 14 bridges, and National Highway 10. Over 15 houses were washed away in Singtam, and the situation was described as a 'disaster waiting to happen' due to prior warnings about the vulnerability of the Lhonak Lake."
}
```

### 2023 · #2 — 33 real canonical fields · `article_000272267` · 2023_08

**Headline:** Four Lakes, the blinds of the canyon, all washed away by the sweep.

**Gate:** dates `['2023-08-07']` · 12 locations `['Lake Bane Village', 'Yen Bai Highway 32', 'Lake Four', 'community health center', 'Lake Bốn community public housing area', 'primary school']` …

```json
{
  "country": "Vietnam",
  "province": "Yenbai Province",
  "district": "Mueyang District",
  "city": "Lake Bane",
  "location": "Lake Bane Village",
  "river": "Lake Four",
  "event_date": "2023-08-07",
  "affected_people": 15000,
  "affected_people_type": "people",
  "evacuated": 600,
  "evacuated_type": "people",
  "displaced": 100,
  "displaced_type": "families",
  "deaths": 14,
  "deaths_type": "people",
  "injured": 82,
  "injured_type": "people",
  "missing": 5,
  "missing_type": "people",
  "houses_damaged": 100,
  "houses_destroyed": 20,
  "roads_damaged": 1,
  "bridges_damaged": 1,
  "schools_damaged": 1,
  "hospitals_damaged": 1,
  "crop_damage": 350,
  "crop_damage_unit": "hectares",
  "livestock_loss": 20,
  "livestock_loss_type": "cattle",
  "economic_loss": 18000000,
  "economic_loss_currency": "VND",
  "response_agencies": [
    "Youth Online",
    "Mueyang District team",
    "Municipal Council of Hồ Bốn",
    "Municipal Council of Mù Cang Chào district",
    "Yen Bai Province Road Administration",
    "Vietnamese Electricity Corporation (EVN)"
  ],
  "summary": "On August 7, torrential downpours caused severe flooding in Lake Bane Village, Yenbai Province, Vietnam. The floods led to landslides, washed away homes, and damaged infrastructure, including the community health center, primary school, and Lake Four Hydroelectric Plant. One person is missing, 20 houses were completely destroyed, 100 houses were damaged, and 20 cattle, 70 motorbikes, and two cars were washed away. The district mobilized emergency teams and provided 18 million VND in aid. Power outages and traffic paralysis were reported, with restoration expected by August 9."
}
```
