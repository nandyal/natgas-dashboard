# Natural Gas Dashboard

This project now includes a Python dashboard that recreates the main natural gas inventory and market analysis from the earlier R workflow.

## Files

- `refresh_eia_ng_inventory.py`: refreshes the EIA weekly Lower 48 storage series and rewrites the local CSV/JSON files
- `dashboard_data.py`: reusable inventory and market data helpers
- `app.py`: Streamlit dashboard
- `generate_static_report.py`: builds a static GitHub Pages report into `docs/index.html`
- `eia_ng_total_inventory_last_10_years.csv`: local EIA weekly inventory history

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Refresh inventory data

```powershell
$env:EIA_API_KEY="your_rotated_eia_api_key"
python refresh_eia_ng_inventory.py
```

## Run the dashboard

```powershell
streamlit run app.py
```

## Build the GitHub Pages report locally

```powershell
python generate_static_report.py
```

This writes the static report to `docs/index.html`.

## Build inventory shock sentiment analysis

```powershell
python -m pip install -r requirements-sentiment.txt
python inventory_sentiment_analysis.py
```

This creates precomputed FinBERT and VADER sentiment files for unusual inventory builds and drawdowns:

- `inventory_sentiment_events.csv`
- `inventory_sentiment_events.json`

## Docker

Build and run the dashboard:

```powershell
Copy-Item .env.example .env
# edit .env and set EIA_API_KEY
docker compose up --build natgas-dashboard
```

Then open `http://localhost:8501`.

Refresh the EIA inventory files inside Docker:

```powershell
Copy-Item .env.example .env
# edit .env and set EIA_API_KEY
docker compose run --rm natgas-refresh
```

The compose setup mounts this project folder into the container, so refreshed CSV and JSON files are written back into the local workspace.

## API key handling

- Rotate the old EIA API key because it was exposed in this chat session.
- Store the replacement key in a local `.env` file created from `.env.example`.
- `.env` is ignored by git, so the live key stays out of version control.
- GitHub Actions in this repo do not use the EIA API key. If you later automate refreshes in GitHub, store the key as a GitHub Actions secret and never commit it to the repository.
- The GitHub Pages workflow only builds the static report from files already in the repository. It does not use your EIA API key.
- The GitHub Pages workflow refreshes Yahoo Finance market data every 6 hours and republishes the static report automatically.

## Current dashboard coverage

- Weekly Lower 48 natural gas inventory history
- 10-year seasonal inventory range and current-year comparison
- Seasonal naive inventory forecast
- Inventory decomposition
- Static GitHub Pages report in `docs/index.html`
- Natural gas and related equities/ETF market tracking from Yahoo Finance
- Equal-weight natural gas equity portfolio view
- Correlation heatmap
- Monthly return table by ticker
