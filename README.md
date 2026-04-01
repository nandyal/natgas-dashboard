# Natural Gas Dashboard

This project now includes a Python dashboard that recreates the main natural gas inventory and market analysis from the earlier R workflow.

## Files

- `refresh_eia_ng_inventory.py`: refreshes the EIA weekly Lower 48 storage series and rewrites the local CSV/JSON files
- `dashboard_data.py`: reusable inventory and market data helpers
- `app.py`: Streamlit dashboard
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

## Current dashboard coverage

- Weekly Lower 48 natural gas inventory history
- 10-year seasonal inventory range and current-year comparison
- Seasonal naive inventory forecast
- Inventory decomposition
- Natural gas and related equities/ETF market tracking from Yahoo Finance
- Equal-weight natural gas equity portfolio view
- Correlation heatmap
- Monthly return table by ticker
