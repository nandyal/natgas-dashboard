from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose


INVENTORY_CSV = "eia_ng_total_inventory_last_10_years.csv"
DEFAULT_TICKERS = ["NG=F", "RRC", "AR", "EQT", "CNX", "UNG", "USO", "FTI"]
PORTFOLIO_TICKERS = ["RRC", "AR", "EQT", "CNX"]


@dataclass(frozen=True)
class InventorySummary:
    latest_date: str
    latest_value: float
    weekly_change: float
    trailing_year_low: float
    trailing_year_high: float


def load_inventory_data(base_dir: Path) -> pd.DataFrame:
    path = base_dir / INVENTORY_CSV
    df = pd.read_csv(path, parse_dates=["period"])
    df = df.sort_values("period").reset_index(drop=True)
    df["week_of_year"] = df["period"].dt.isocalendar().week.astype(int)
    df["year"] = df["period"].dt.year
    df["inventory_52w_avg"] = df["value_bcf"].rolling(52).mean()
    df["inventory_52w_std"] = df["value_bcf"].rolling(52).std()
    return df


def summarize_inventory(df: pd.DataFrame) -> InventorySummary:
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    trailing_year = df.tail(52)
    return InventorySummary(
        latest_date=latest["period"].strftime("%Y-%m-%d"),
        latest_value=float(latest["value_bcf"]),
        weekly_change=float(latest["value_bcf"] - prev["value_bcf"]),
        trailing_year_low=float(trailing_year["value_bcf"].min()),
        trailing_year_high=float(trailing_year["value_bcf"].max()),
    )


def seasonal_inventory_profile(df: pd.DataFrame) -> pd.DataFrame:
    profile = (
        df.groupby("week_of_year")["value_bcf"]
        .agg(["mean", "min", "max"])
        .reset_index()
        .rename(columns={"mean": "avg_bcf", "min": "min_bcf", "max": "max_bcf"})
    )
    return profile


def latest_inventory_vs_history(df: pd.DataFrame) -> pd.DataFrame:
    latest_year = int(df["year"].max())
    prior_year = latest_year - 1
    view = df[df["year"].isin([prior_year, latest_year])][
        ["period", "year", "week_of_year", "value_bcf"]
    ].copy()
    return view


def seasonal_naive_inventory_forecast(df: pd.DataFrame, horizon: int = 13) -> pd.DataFrame:
    history = df[["period", "value_bcf"]].copy()
    if len(history) < 53:
        raise ValueError("Need at least 53 weeks of history for a seasonal naive forecast.")

    last_period = history["period"].max()
    future_dates = pd.date_range(last_period + pd.Timedelta(days=7), periods=horizon, freq="W-FRI")
    seasonal_source = history.tail(52)["value_bcf"].reset_index(drop=True)
    repeated_values = [float(seasonal_source.iloc[i % len(seasonal_source)]) for i in range(horizon)]

    forecast = pd.DataFrame({"period": future_dates, "forecast_bcf": repeated_values})
    return forecast


def inventory_decomposition(df: pd.DataFrame):
    series = df.set_index("period")["value_bcf"].asfreq("W-FRI")
    return seasonal_decompose(series, model="additive", period=52, extrapolate_trend="freq")


def fetch_market_prices(
    tickers: list[str] | None = None,
    start: str = "2016-01-01",
) -> pd.DataFrame:
    tickers = tickers or DEFAULT_TICKERS
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError("No market data returned from Yahoo Finance.")

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = data.to_frame(name=tickers[0])

    close = close.dropna(axis=1, how="all")
    close = close.dropna(how="all").ffill()
    if close.empty:
        raise ValueError("Yahoo Finance returned no usable closing-price data.")
    close.index = pd.to_datetime(close.index)
    return close


def normalized_prices(close: pd.DataFrame) -> pd.DataFrame:
    return close.divide(close.iloc[0]).multiply(100)


def daily_returns(close: pd.DataFrame) -> pd.DataFrame:
    return close.pct_change().dropna(how="all")


def monthly_returns(close: pd.DataFrame) -> pd.DataFrame:
    monthly_close = close.resample("ME").last()
    return monthly_close.pct_change().dropna(how="all")


def correlation_matrix(close: pd.DataFrame) -> pd.DataFrame:
    return daily_returns(close).corr()


def build_equal_weight_portfolio(close: pd.DataFrame, tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or PORTFOLIO_TICKERS
    available = [ticker for ticker in tickers if ticker in close.columns]
    if not available:
        raise ValueError("None of the requested portfolio tickers are available.")

    portfolio_close = close[available].dropna().copy()
    returns = portfolio_close.pct_change().dropna()
    weights = pd.Series(1 / len(available), index=available)
    portfolio_returns = returns.mul(weights, axis=1).sum(axis=1)
    cumulative = (1 + portfolio_returns).cumprod() * 100
    return pd.DataFrame(
        {
            "portfolio_return": portfolio_returns,
            "portfolio_index": cumulative,
        }
    )


def available_tickers(close: pd.DataFrame, requested: list[str]) -> tuple[list[str], list[str]]:
    available = [ticker for ticker in requested if ticker in close.columns]
    missing = [ticker for ticker in requested if ticker not in close.columns]
    return available, missing


def calendar_return_table(series: pd.Series) -> pd.DataFrame:
    values = series.dropna().copy()
    table = pd.DataFrame(
        {
            "year": values.index.year,
            "month": values.index.strftime("%b"),
            "return_pct": values.values * 100,
        }
    )
    pivot = table.pivot(index="year", columns="month", values="return_pct")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    existing = [month for month in month_order if month in pivot.columns]
    return pivot.reindex(columns=existing).sort_index(ascending=False)
