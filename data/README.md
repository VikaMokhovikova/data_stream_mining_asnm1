# Dataset: hourly weather streams for Guimarães and Viana do Castelo

Raw input of PA04, by Khadija Belhamra (20230007, Guimarães) and Victoriya Mokhovikova (20230796, Viana do Castelo).

* **Business context:** textile storage facility (humid vs. dry regimes)
* **Locations:** Guimarães vs Viana do Castelo
* **Question:** How do humid and dry regimes differ between candidate warehouse locations?
* **Target:** T4, Humid vs Dry Regime

The two files are **snapshots**: they are collected once, never modified, and replayed as a stream by `src/`.

## Source

| Item | Value |
|---|---|
| Provider | [Open-Meteo](https://open-meteo.com/) **Historical Weather API** (`https://archive-api.open-meteo.com/v1/archive`) |
| Nature of the data | Reanalysis (gridded model output fitted to observations), **not** station measurements and **not** forecasts |
| Period | 2023-01-01 00:00 to 2025-12-31 23:00 (`config.yaml → data`) |
| Frequency | Hourly |
| Requested timezone | `Europe/Lisbon` |
| License | Open-Meteo data is CC BY 4.0. Cite Open-Meteo in the report |

## Files

| File | Content |
|---|---|
| `guimaraes_raw.json` | City A, raw API response (unchanged) |
| `viana_do_castelo_raw.json` | City B, raw API response (unchanged) |

The snapshots used for every reported result were retrieved no later than 2026-10-21, the date they were committed. The exact retrieval time was not recorded. Their SHA-256 checksums are:

| File | SHA-256 |
|---|---|
| `guimaraes_raw.json` | `08e1a925ed6f3d1b00f5a7a9e685736bafa9cfc790a3d717bdfa0f772a2e29b9` |
| `viana_do_castelo_raw.json` | `869cb6a1dbcc5780896732c1accfc9919bada2894149eb3e9b8b6e0c9098918d` |

No processed copy is stored. The clean streams are rebuilt in memory from these snapshots by `prepare_frames(cfg)` on every run.

Collection logic: `src/data.py → get_snapshot()`. A snapshot is reused if it exists. It is downloaded only if it is missing or `storage.force_download: true`.

## Locations

Open-Meteo snaps the requested coordinates to the nearest cell of its model grid. The results are based on the **grid point**, not the requested coordinates.

| Role | City | Requested (lat, lon) | Grid point used (lat, lon) | Grid elevation |
|---|---|---|---|---|
| City A | Guimarães | 41.4289, −8.3178 | 41.441, −8.288 | 170 m |
| City B | Viana do Castelo | 41.695, −8.82916 | 41.722, −8.857 | 15 m |

Guimarães is inland in the Ave valley. Viana do Castelo is on the Atlantic coast at the mouth of the Lima. The elevation and coast distance of each cell should be discussed in the report.

## JSON structure

```text
{
  "latitude", "longitude", "elevation",        # grid point actually used
  "timezone", "timezone_abbreviation", "utc_offset_seconds",
  "generationtime_ms",
  "hourly_units": {"time": "iso8601", "temperature_2m": "°C", ...},
  "hourly": {
    "time": ["2023-01-01T00:00", ...],          # 26 304 timestamps
    "temperature_2m": [...], "relative_humidity_2m": [...],
    "precipitation": [...], "wind_speed_10m": [...]
  }
}
```

## Variables

| Variable | Unit | Meaning | Role in the project |
|---|---|---|---|
| `temperature_2m` | °C | Air temperature at 2 m | Predictor (lag t−1 only) |
| `relative_humidity_2m` | % | Relative humidity at 2 m | **Defines the target** (H_t). Used as a predictor only at t−1 and in windows ending at t−1 |
| `precipitation` | mm | Total precipitation (rain + snow) over the preceding hour | Predictor (lag t−1 only) |
| `wind_speed_10m` | km/h | Wind speed at 10 m | Predictor (lag t−1 only) |

## Size and quality (both cities)

| | Guimarães | Viana do Castelo |
|---|---|---|
| Hourly rows | 26 304 | 26 304 |
| Duplicate timestamps / irregular steps | 0 / 0 | 0 / 0 |
| Missing values (any variable) | 0 | 0 |
| Values outside definitional ranges | 0 | 0 |
| Temperature min / mean / max (°C) | −1.5 / 15.6 / 39.8 | 1.6 / 15.4 / 30.9 |
| Relative humidity min / mean / max (%) | 13 / 77.0 / 100 | 27 / 80.4 / 100 |
| Precipitation max (mm/h) | 17.7 | 16.1 |
| Wind speed mean / max (km/h) | 9.7 / 44.0 | 15.4 / 57.5 |

Definitional ranges (`config.yaml → quality.valid_ranges`): temperature −50 to 60 °C, RH 0 to 100 %, precipitation ≥ 0, wind ≥ 0. A value that is missing or out of range is **flagged**, never deleted. Extreme but valid values (RH = 100 %, heavy rain) are kept, because they may be exactly the events that matter.

## Timestamps

The `time` values form a perfectly regular hourly sequence (26 304 = 3 × 365 × 24 + 24 for the 2024 leap day). The March gap and October duplicate that a daylight-saving local clock would produce are absent. The API therefore returned a **fixed-offset** clock (`utc_offset_seconds` in the file), and `src/data.py` also derives `timestamp_utc`. Hour-of-day analyses convert `timestamp_utc` to the true `Europe/Lisbon` clock (with summer time).

## How the data is used

All protocol steps are in `src/data.py → prepare_frames()`:

1. Both snapshots → chronological tables on the **common timeline** (identical timestamps for both cities).
2. **Warm-up** = first 12 months (2023, 8 760 h). **Evaluation** = 2024–2025 (17 544 h).
3. **τ_H** = median of the pooled warm-up humidity of both cities = **82 %** (17 520 observations), then frozen.
4. **Target T4:** y_t = 1 (humid) if RH_t ≥ τ_H, else 0 (dry).

Columns of each prepared stream (`prepare_frames(cfg)["frames"][city]`): `city, timestamp, timestamp_utc, temperature_2m, relative_humidity_2m, precipitation, wind_speed_10m, quality_flag, y, phase`.

## Limitations

* One grid cell per city: the cell value is not the climate inside a specific building or street.
* Outdoor conditions only. Indoor humidity in a warehouse also depends on ventilation and the building envelope.
* Three years: τ_H is estimated from a single year (2023).
