from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import genpareto
import yfinance as yf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf, adfuller, pacf


INVENTORY_CSV = "eia_ng_total_inventory_last_10_years.csv"
DEFAULT_TICKERS = ["NG=F", "RRC", "AR", "EQT", "CNX", "UNG", "USO", "FTI"]
PORTFOLIO_TICKERS = ["RRC", "AR", "EQT", "CNX", "UNG", "USO"]


@dataclass(frozen=True)
class InventorySummary:
    latest_date: str
    latest_value: float
    weekly_change: float
    trailing_year_low: float
    trailing_year_high: float


@dataclass(frozen=True)
class PortfolioSummary:
    method: str
    weights: pd.Series
    rebalance_weights: pd.DataFrame
    returns: pd.Series
    index: pd.Series
    kelly_growth_rate: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float


def load_inventory_data(base_dir: Path) -> pd.DataFrame:
    path = base_dir / INVENTORY_CSV
    df = pd.read_csv(path, parse_dates=["period"])
    df = df.sort_values("period").reset_index(drop=True)
    df["week_of_year"] = df["period"].dt.isocalendar().week.astype(int)
    df["year"] = df["period"].dt.year
    df["inventory_52w_avg"] = df["value_bcf"].rolling(52).mean()
    df["inventory_52w_std"] = df["value_bcf"].rolling(52).std()
    df.attrs["adf_inventory_level"] = adf_test_summary(df["value_bcf"], "Inventory level")
    df.attrs["adf_weekly_change"] = adf_test_summary(df["value_bcf"].diff(), "Weekly change")
    return df


def adf_test_summary(series: pd.Series, label: str) -> dict[str, float | str | int]:
    clean = pd.Series(series).dropna()
    if len(clean) < 20:
        return {
            "label": label,
            "adf_statistic": np.nan,
            "p_value": np.nan,
            "used_lags": 0,
            "nobs": int(len(clean)),
            "interpretation": "Insufficient observations for ADF test.",
        }
    statistic, p_value, used_lags, nobs, critical_values, _ = adfuller(clean, autolag="AIC")
    return {
        "label": label,
        "adf_statistic": float(statistic),
        "p_value": float(p_value),
        "used_lags": int(used_lags),
        "nobs": int(nobs),
        "critical_1pct": float(critical_values["1%"]),
        "critical_5pct": float(critical_values["5%"]),
        "critical_10pct": float(critical_values["10%"]),
        "interpretation": "Stationary at 5% significance." if p_value < 0.05 else "Cannot reject a unit root at 5% significance.",
    }


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
    clean = series.interpolate(limit_direction="both")
    return STL(clean, period=52, seasonal=53, trend=53, robust=True).fit()


def split_residual_components(
    residual: pd.Series,
    window: int = 4,
    ar_lags: int = 4,
) -> tuple[pd.Series, pd.Series]:
    clean = residual.dropna().copy()
    rolling_structured = clean.rolling(window=window, center=True, min_periods=1).mean()
    if len(clean) <= ar_lags + 20:
        structured = rolling_structured
    else:
        try:
            ar_model = AutoReg(clean, lags=ar_lags, old_names=False).fit()
            ar_structured = ar_model.fittedvalues.reindex(clean.index)
            structured = ar_structured.combine_first(rolling_structured)
        except Exception:
            structured = rolling_structured
    noise = clean - structured
    return structured, noise


def residual_acf_pacf_table(series: pd.Series, nlags: int = 26) -> pd.DataFrame:
    clean = pd.Series(series).dropna()
    max_lag = min(nlags, max(len(clean) // 2 - 1, 1))
    acf_values = acf(clean, nlags=max_lag, fft=False)
    pacf_values = pacf(clean, nlags=max_lag, method="ywm")
    confidence = 1.96 / np.sqrt(len(clean))
    return pd.DataFrame(
        {
            "lag": range(1, max_lag + 1),
            "acf": acf_values[1:],
            "pacf": pacf_values[1:],
            "confidence": confidence,
        }
    )


def rolling_residual_autocorrelation(
    series: pd.Series,
    window: int = 13,
    lag: int = 1,
) -> pd.Series:
    clean = pd.Series(series).dropna()
    return clean.rolling(window=window, min_periods=window).apply(
        lambda values: pd.Series(values).autocorr(lag=lag),
        raw=False,
    )


def residual_regime_alert(
    series: pd.Series,
    window: int = 13,
    lag: int = 1,
    fixed_threshold: float = 0.35,
    quantile: float = 0.90,
) -> dict[str, float | bool | str]:
    rolling = rolling_residual_autocorrelation(series, window=window, lag=lag).dropna()
    if rolling.empty:
        return {
            "latest_autocorrelation": np.nan,
            "threshold": fixed_threshold,
            "alert": False,
            "message": "Insufficient residual history for regime alert.",
        }
    quantile_threshold = float(rolling.abs().quantile(quantile))
    threshold = max(fixed_threshold, quantile_threshold)
    latest = float(rolling.iloc[-1])
    alert = abs(latest) >= threshold
    return {
        "latest_autocorrelation": latest,
        "threshold": threshold,
        "alert": bool(alert),
        "message": (
            "Residual autocorrelation is elevated; noise structure may be strengthening."
            if alert
            else "Residual autocorrelation is below the regime-shift alert threshold."
        ),
    }


def gpd_tail_var_thresholds(
    weekly_change: pd.Series,
    threshold_quantile: float = 0.90,
    var_quantile: float = 0.99,
) -> dict[str, float | int]:
    changes = pd.Series(weekly_change).dropna()
    if len(changes) < 100:
        raise ValueError("Need at least 100 weekly changes for EVT tail fitting.")

    def fit_one_tail(values: pd.Series) -> tuple[float, float, int]:
        threshold = float(values.quantile(threshold_quantile))
        exceedances = values[values > threshold] - threshold
        exceedances = exceedances[exceedances > 0]
        if len(exceedances) < 10:
            raise ValueError("Need at least 10 exceedances for GPD tail fitting.")
        shape, _, scale = genpareto.fit(exceedances, floc=0)
        exceedance_probability = len(exceedances) / len(values)
        tail_probability = 1 - var_quantile
        if abs(shape) < 1e-6:
            var_value = threshold + scale * np.log(exceedance_probability / tail_probability)
        else:
            var_value = threshold + (scale / shape) * ((exceedance_probability / tail_probability) ** shape - 1)
        return float(var_value), float(threshold), int(len(exceedances))

    upper_var, upper_threshold, upper_exceedances = fit_one_tail(changes[changes > 0])
    lower_var_abs, lower_threshold_abs, lower_exceedances = fit_one_tail((-changes[changes < 0]))
    return {
        "upper_var_bcf": upper_var,
        "lower_var_bcf": -lower_var_abs,
        "upper_tail_threshold_bcf": upper_threshold,
        "lower_tail_threshold_bcf": -lower_threshold_abs,
        "upper_exceedances": upper_exceedances,
        "lower_exceedances": lower_exceedances,
        "threshold_quantile": threshold_quantile,
        "var_quantile": var_quantile,
    }


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


def _optimize_portfolio_weights(
    returns: pd.DataFrame,
    method: str = "sharpe",
    risk_free_rate: float = 0.02,
    max_long_weight: float = 0.55,
) -> pd.Series:
    available = list(returns.columns)
    if len(returns) < 20:
        return pd.Series(1 / len(available), index=available)

    n_assets = len(available)
    signal = returns.mean()
    start = signal.clip(lower=0.0)
    if np.allclose(start.sum(), 0.0):
        start = pd.Series(1 / n_assets, index=available)
    else:
        start = start / start.sum()
        start = start.clip(lower=0.0, upper=max_long_weight)
        start = start / start.sum()
    start = start.to_numpy(dtype=float)
    bounds = [(0.0, min(max_long_weight, 1.0))] * n_assets
    constraints = [{"type": "eq", "fun": lambda w: sum(w) - 1}]

    annualized_returns = returns.mean() * 252
    annualized_cov = returns.cov() * 252

    def objective(weights: list[float]) -> float:
        weights_series = pd.Series(weights, index=available)
        portfolio_volatility = float(np.sqrt(weights_series.T @ annualized_cov @ weights_series))
        if portfolio_volatility <= 0:
            return 1e9
        if method == "min_volatility":
            return portfolio_volatility
        portfolio_return = float((annualized_returns * weights_series).sum())
        return -((portfolio_return - risk_free_rate) / portfolio_volatility)

    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if result.success:
        return pd.Series(result.x, index=available)
    return pd.Series(start, index=available)


def build_optimized_portfolio(
    close: pd.DataFrame,
    tickers: list[str] | None = None,
    method: str = "sharpe",
    risk_free_rate: float = 0.02,
    max_long_weight: float = 0.55,
    rebalance_years: int = 1,
) -> PortfolioSummary:
    tickers = tickers or PORTFOLIO_TICKERS
    available = [ticker for ticker in tickers if ticker in close.columns]
    if not available:
        raise ValueError("None of the requested portfolio tickers are available.")

    portfolio_close = close[available].dropna().copy()
    returns = portfolio_close.pct_change().dropna()
    if returns.empty:
        raise ValueError("Not enough price history to build portfolio returns.")

    rebalance_months = rebalance_years * 12
    rebalance_dates = [returns.index[0]]
    candidate_dates = returns.resample(f"{rebalance_months}ME").last().index
    for date in candidate_dates:
        actual_date = returns.index[returns.index >= date]
        if len(actual_date) > 0 and actual_date[0] not in rebalance_dates:
            rebalance_dates.append(actual_date[0])
    rebalance_dates = sorted(set(rebalance_dates))

    rebalance_history: list[pd.Series] = []
    period_returns: list[pd.Series] = []
    latest_weights = pd.Series(dtype=float)

    for idx, rebalance_date in enumerate(rebalance_dates):
        lookback = returns.loc[:rebalance_date].tail(252 * rebalance_years)
        if len(lookback) < 126:
            lookback = returns.loc[:rebalance_date]
        weights = _optimize_portfolio_weights(
            lookback,
            method=method,
            risk_free_rate=risk_free_rate,
            max_long_weight=max_long_weight,
        )
        latest_weights = weights
        rebalance_history.append(weights.rename(rebalance_date))

        if idx + 1 < len(rebalance_dates):
            next_date = rebalance_dates[idx + 1]
            mask = (returns.index >= rebalance_date) & (returns.index < next_date)
        else:
            mask = returns.index >= rebalance_date
        period_slice = returns.loc[mask]
        if not period_slice.empty:
            period_returns.append(period_slice.mul(weights, axis=1).sum(axis=1))

    portfolio_returns = pd.concat(period_returns).sort_index()
    cumulative = (1 + portfolio_returns).cumprod() * 100
    kelly_growth_rate = float(np.log1p(portfolio_returns).mean() * 252)
    annual_return = float(portfolio_returns.mean() * 252)
    annual_volatility = float(portfolio_returns.std() * (252 ** 0.5))
    sharpe_ratio = 0.0 if annual_volatility == 0 else float((annual_return - risk_free_rate) / annual_volatility)
    rebalance_weights = pd.DataFrame(rebalance_history).fillna(0.0)
    return PortfolioSummary(
        method=method,
        weights=latest_weights.sort_values(ascending=False),
        rebalance_weights=rebalance_weights,
        returns=portfolio_returns,
        index=cumulative,
        kelly_growth_rate=kelly_growth_rate,
        annual_return=annual_return,
        annual_volatility=annual_volatility,
        sharpe_ratio=sharpe_ratio,
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
