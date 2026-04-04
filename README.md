# Natural Gas Dashboard

This project now includes a Python dashboard that recreates the main natural gas inventory and market analysis from the earlier R workflow.

## Files

- `refresh_eia_ng_inventory.py`: refreshes the EIA weekly Lower 48 storage series and rewrites the local CSV/JSON files
- `dashboard_data.py`: reusable inventory and market data helpers
- `app.py`: Streamlit inventory dashboard
- `market_app.py`: separate Streamlit market dashboard
- `generate_static_report.py`: builds a static GitHub Pages report into `docs/index.html`
- `generate_market_report.py`: builds a separate static market report into `docs/market.html`
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

## Run the market dashboard

```powershell
streamlit run market_app.py
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

If you have a private Hugging Face or FinBERT access token, store it only in your local `.env` file as:

```env
FINBERT_API_KEY=your_rotated_finbert_api_key
```

The sentiment script reads `FINBERT_API_KEY` locally and does not require that key to be committed or pushed to GitHub.

## Build market sentiment analysis

```powershell
python -m pip install -r requirements-sentiment.txt
python market_sentiment_analysis.py
python generate_market_report.py
```

This creates:

- `market_sentiment_events.csv`
- `market_sentiment_events.json`
- `docs/market.html`

## Docker

Build and run the dashboard:

```powershell
Copy-Item .env.example .env
# edit .env and set EIA_API_KEY
docker compose up --build natgas-dashboard
```

Then open `http://localhost:8501`.

Run the separate market dashboard:

```powershell
docker compose up --build natgas-market-dashboard
```

Then open `http://localhost:8502`.

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
- The GitHub Pages workflow now also rebuilds the separate market report and market sentiment outputs.

## Current dashboard coverage

- Weekly Lower 48 natural gas inventory history
- 10-year seasonal inventory range and current-year comparison
- Seasonal naive inventory forecast
- Inventory decomposition
- Static GitHub Pages report in `docs/index.html`
- Separate market dashboard and GitHub Pages report for stocks, ETFs, and NG futures
- Optimized market portfolio view
- Stock, ETF, and futures sentiment with one-month forward price reaction
- Calendar monthly returns for all tracked stocks, ETFs, and futures
