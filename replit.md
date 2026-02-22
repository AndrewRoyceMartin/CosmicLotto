# Powerball Planet Correlation Analyzer

## Overview
A Streamlit data analysis application that compares historical Powerball draw results with astronomical planetary positions at each draw date/time and location. It calculates planetary alignments, tests statistical correlations, and evaluates patterns.

## Project Architecture

### File Structure
```
├── app.py                 # Main Streamlit UI and workflow orchestration
├── db.py                  # SQLite database schema and connection management
├── ingest_powerball.py    # CSV import, validation, and draw ingestion
├── ephemeris.py           # Skyfield planetary position calculations
├── features.py            # Alignment feature engineering (bins, aspects)
├── analysis.py            # Statistical correlation testing with BH correction + plain-English summaries
├── forecast.py            # Future draw forecast with planetary position scoring
├── model.py               # V2 placeholder for backtesting/prediction
├── utils.py               # Shared helper functions (timezone, validation, JSON)
├── .streamlit/config.toml # Streamlit server config
└── data/
    └── app.db             # SQLite database
```

### Database (SQLite)
- **draws**: Historical draw results with UTC and local timestamps
- **planet_positions**: Computed planetary coordinates per draw
- **planet_features**: Derived alignment features per draw

### Key Libraries
- Streamlit (UI), Pandas/NumPy (data), Scipy (statistics)
- Skyfield with DE421 ephemeris (planetary positions)
- Plotly (interactive charts)

### Workflow
1. Upload CSV → Import draws into SQLite
2. Compute planetary positions via Skyfield
3. Generate alignment features (longitude bins, pairwise aspects)
4. Run two-proportion z-tests with Benjamini-Hochberg correction
5. View plain-English correlation summaries
6. Generate forecasts for future draws based on active planetary features
7. Filter and explore results

## Recent Changes
- 2026-02-22: Initial build of all modules (Phases 1-7)
- 2026-02-22: Added configurable draw time (default 20:30 Sydney) with DST-aware handling for date-only CSVs
- 2026-02-22: Added forecast module (forecast.py) with Thursday 20:30 draw scheduling, DST-aware timestamps, and UTC offset tracking
- 2026-02-22: Added plain-English correlation summaries (humanize_feature_name, summarize_correlations_plain_english in analysis.py)
- 2026-02-22: Added Forecast tab to app.py with draw cards, timestamp verification, and number scoring
- 2026-02-22: Upgraded forecast to generate 10 ranked game cards per draw with confidence indicators, combo generation from top-N scored numbers, and advanced settings (combo pool size, PB candidates)

## User Preferences
- Australian Powerball format supported (7 main balls + powerball)
- US Powerball format also supported (5 main balls + powerball)
- Default location: Sydney, Australia
- Default timezone: Australia/Sydney
- Default draw time: 20:30 (8:30 PM) — configurable in sidebar
- CSV date format: supports both full datetime and date-only (dd/mm/yyyy) with auto-applied draw time
