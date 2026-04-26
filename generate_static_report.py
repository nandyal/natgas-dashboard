from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency, genpareto, norm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import ccf

from dashboard_data import (
    DEFAULT_TICKERS,
    PORTFOLIO_TICKERS,
    build_optimized_portfolio,
    calendar_return_table,
    correlation_matrix,
    fetch_market_prices,
    inventory_decomposition,
    load_full_inventory_data,
    load_hdd_data,
    load_inventory_data,
    merge_inventory_hdd,
    monthly_returns,
    normalized_prices,
    residual_acf_pacf_table,
    residual_regime_alert,
    rolling_residual_autocorrelation,
    seasonal_inventory_profile,
    split_residual_components,
    summarize_inventory,
)


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
REPORT_PATH = DOCS_DIR / "index.html"
LEGACY_RELEASE_JSON = BASE_DIR / "weekly_natural_gas_inventory_2026-03-20.json"
MONTHLY_RETURN_TICKERS = ["NG=F", "UNG", "USO"]
SENTIMENT_CSV = BASE_DIR / "inventory_sentiment_events.csv"
FULL_HISTORY_INVENTORY_CSV = BASE_DIR / "eia_ng_total_inventory_full_history.csv"


def latest_release_payload(df: pd.DataFrame) -> dict:
    latest_period = pd.Timestamp(df["period"].max()).normalize()
    payload = {
        "release_name": "Weekly Natural Gas Storage Report",
        "current_week": latest_period.strftime("%Y-%m-%d"),
        "release_date": (latest_period + pd.Timedelta(days=6)).strftime("%Y-%m-%d"),
        "series": [],
    }
    if LEGACY_RELEASE_JSON.exists():
        legacy = json.loads(LEGACY_RELEASE_JSON.read_text(encoding="utf-8-sig"))
        if str(legacy.get("current_week")) == payload["current_week"]:
            payload = legacy
    return payload


def load_sentiment_events() -> pd.DataFrame:
    if not SENTIMENT_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(SENTIMENT_CSV, parse_dates=["period"])
    return df


def load_evt_inventory_history(default_df: pd.DataFrame) -> pd.DataFrame:
    if not FULL_HISTORY_INVENTORY_CSV.exists():
        return default_df
    df = pd.read_csv(FULL_HISTORY_INVENTORY_CSV, parse_dates=["period"])
    df = df.sort_values("period").reset_index(drop=True)
    df["week_of_year"] = df["period"].dt.isocalendar().week.astype(int)
    df["year"] = df["period"].dt.year
    df["inventory_52w_avg"] = df["value_bcf"].rolling(52).mean()
    df["inventory_52w_std"] = df["value_bcf"].rolling(52).std()
    return df


def summary_cards_html(summary, release: dict) -> str:
    return f"""
    <section class="summary-grid">
      <div class="card">
        <div class="eyebrow">Data week ending</div>
        <div class="metric">{pd.to_datetime(release['current_week']).strftime('%B %d, %Y')}</div>
      </div>
      <div class="card">
        <div class="eyebrow">Published by EIA</div>
        <div class="metric">{pd.to_datetime(release['release_date']).strftime('%B %d, %Y')}</div>
      </div>
      <div class="card">
        <div class="eyebrow">Lower 48 storage</div>
        <div class="metric">{summary.latest_value:,.0f} Bcf</div>
      </div>
      <div class="card">
        <div class="eyebrow">Weekly change</div>
        <div class="metric">{summary.weekly_change:+,.0f} Bcf</div>
      </div>
    </section>
    """


def nav_html() -> str:
    return """
    <nav class="nav">
      <a href="./index.html" class="active">Inventory Analysis</a>
      <a href="./market.html">Market Analysis</a>
    </nav>
    """


def inventory_history_chart(df: pd.DataFrame) -> str:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["period"],
            y=df["value_bcf"],
            name="Lower 48 storage",
            mode="lines",
            line=dict(color="#0f766e", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["period"],
            y=df["inventory_52w_avg"],
            name="52-week average",
            mode="lines",
            line=dict(color="#b45309", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title="Natural gas inventory history",
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        yaxis_title="Billion cubic feet",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def seasonality_chart(df: pd.DataFrame) -> str:
    profile = seasonal_inventory_profile(df)
    current = df[df["year"] == df["year"].max()][["week_of_year", "value_bcf"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=profile["week_of_year"],
            y=profile["max_bcf"],
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=profile["week_of_year"],
            y=profile["min_bcf"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(15,118,110,0.15)",
            line=dict(width=0),
            name="10-year range",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=profile["week_of_year"],
            y=profile["avg_bcf"],
            mode="lines",
            line=dict(color="#0f766e", width=3),
            name="10-year average",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=current["week_of_year"],
            y=current["value_bcf"],
            mode="lines",
            line=dict(color="#1d4ed8", width=2),
            name=str(df["year"].max()),
        )
    )
    fig.update_layout(
        title="Seasonal view: current year vs 10-year history",
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Week of year",
        yaxis_title="Billion cubic feet",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def decomposition_chart(df: pd.DataFrame) -> str:
    decomp = inventory_decomposition(df)
    structured_residual, noise_residual = split_residual_components(
        pd.Series(decomp.resid, index=decomp.observed.index)
    )
    clean = pd.DataFrame(
        {
            "period": decomp.observed.index,
            "observed": decomp.observed.values,
            "trend": decomp.trend.values,
            "seasonal": decomp.seasonal.values,
            "structured_residual": structured_residual.reindex(decomp.observed.index).values,
            "noise": noise_residual.reindex(decomp.observed.index).values,
        }
    ).dropna()

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Observed", "Trend", "Seasonal", "Structured Residual", "Noise"),
    )
    for idx, column in enumerate(
        ["observed", "trend", "seasonal", "structured_residual", "noise"],
        start=1,
    ):
        fig.add_trace(
            go.Scatter(
                x=clean["period"],
                y=clean[column],
                mode="lines",
                line=dict(color="#0f766e", width=2),
                showlegend=False,
            ),
            row=idx,
            col=1,
        )
    fig.update_layout(
        title="Decomposition of natural gas inventory",
        template="plotly_white",
        height=880,
        margin=dict(l=20, r=20, t=70, b=20),
    )
    fig.update_xaxes(title_text="Year", row=5, col=1)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def decomposition_analysis_html(df: pd.DataFrame) -> str:
    decomp = inventory_decomposition(df)
    resid = pd.Series(decomp.resid, index=decomp.observed.index).dropna()
    structured_residual, noise_residual = split_residual_components(resid)
    noise = noise_residual.dropna()
    lb = acorr_ljungbox(resid, lags=[13, 26], return_df=True)
    noise_lb = acorr_ljungbox(noise, lags=[13, 26], return_df=True)
    acf1 = resid.autocorr(1)
    noise_acf1 = noise.autocorr(1)
    winter_months = resid.groupby(resid.index.month).apply(lambda s: s.abs().mean())
    winter_peak_month = int(winter_months.sort_values(ascending=False).index[0])
    winter_peak_name = pd.Timestamp(year=2000, month=winter_peak_month, day=1).strftime("%B")
    top_weeks = (
        resid.groupby(resid.index.isocalendar().week.astype(int))
        .apply(lambda s: s.abs().mean())
        .sort_values(ascending=False)
        .head(4)
        .index.tolist()
    )
    noise_interpretation = (
        "The remaining Noise passes this white-noise check at the 5% level, so the STL plus AR(4) residual filter is absorbing most of the visible short-run structure."
        if noise_lb.loc[13, "lb_pvalue"] > 0.05 and noise_lb.loc[26, "lb_pvalue"] > 0.05
        else "The remaining Noise still shows serial dependence, so the decomposition should be treated as an incomplete split rather than a pure white-noise model."
    )
    return f"""
    <div class="analysis-box">
      <h3>Residual interpretation</h3>
      <p class="small">
        The residual component does not look like pure white noise. The residual series still shows strong short-run dependence, with lag-1 autocorrelation of
        <strong>{acf1:.2f}</strong>, and the Ljung-Box test rejects a pure-noise process at both 13 and 26 lags
        (<strong>p-values {lb.loc[13, 'lb_pvalue']:.4f}</strong> and <strong>{lb.loc[26, 'lb_pvalue']:.4f}</strong>).
      </p>
      <p class="small">
        The largest leftover disturbances cluster around winter transition weeks rather than appearing evenly spread through the year. Absolute residuals are largest in
        <strong>{winter_peak_name}</strong>, and the most volatile weeks are
        <strong>{", ".join(str(week) for week in top_weeks)}</strong>. In the figure, this leftover component is split using an AR(4) residual filter into a
        <strong>Structured Residual</strong> line, meant to capture winter weather shocks, storage-regime shifts, and short-memory residual momentum, and a faster-moving <strong>Noise</strong> line for the remaining short-run variation.
      </p>
      <p class="small">
        After the split, the Noise component has lag-1 autocorrelation of <strong>{noise_acf1:.2f}</strong>. The Ljung-Box p-values for Noise are
        <strong>{noise_lb.loc[13, 'lb_pvalue']:.4f}</strong> at 13 lags and <strong>{noise_lb.loc[26, 'lb_pvalue']:.4f}</strong> at 26 lags.
        Values above 0.05 support a white-noise interpretation; values below 0.05 indicate remaining serial dependence. {noise_interpretation}
      </p>
    </div>
    """


def adf_test_html(df: pd.DataFrame) -> str:
    tests = [df.attrs.get("adf_inventory_level", {}), df.attrs.get("adf_weekly_change", {})]
    rows = ""
    for test in tests:
        if not test:
            continue
        rows += (
            f"<tr><td>{test['label']}</td><td>{test['adf_statistic']:.2f}</td>"
            f"<td>{test['p_value']:.4f}</td><td>{test['used_lags']}</td>"
            f"<td>{test['nobs']}</td><td>{test['interpretation']}</td></tr>"
        )
    return f"""
    <div class="analysis-box">
      <h3>ADF stationarity check</h3>
      <p class="small">The Augmented Dickey-Fuller test checks whether the inventory series behaves like it has a unit root. The weekly-change test is included because storage levels are seasonal and persistent, while weekly changes are the closer input for shock analysis.</p>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Series</th><th>ADF statistic</th><th>p-value</th><th>Used lags</th><th>Observations</th><th>Interpretation</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </div>
    """


def noise_acf_pacf_chart(df: pd.DataFrame) -> str:
    decomp = inventory_decomposition(df)
    resid = pd.Series(decomp.resid, index=decomp.observed.index)
    _, noise = split_residual_components(resid)
    corr = residual_acf_pacf_table(noise, nlags=26)
    confidence = float(corr["confidence"].iloc[0]) if not corr.empty else 0.0
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Noise ACF", "Noise PACF"),
        shared_yaxes=True,
    )
    fig.add_trace(go.Bar(x=corr["lag"], y=corr["acf"], marker_color="#0f766e", name="ACF"), row=1, col=1)
    fig.add_trace(go.Bar(x=corr["lag"], y=corr["pacf"], marker_color="#1d4ed8", name="PACF"), row=1, col=2)
    for col in [1, 2]:
        fig.add_hline(y=confidence, line_dash="dash", line_color="#991b1b", row=1, col=col)
        fig.add_hline(y=-confidence, line_dash="dash", line_color="#991b1b", row=1, col=col)
    fig.update_layout(
        title="Noise residual ACF/PACF diagnostic",
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=70, b=20),
        showlegend=False,
    )
    fig.update_xaxes(title_text="Lag")
    fig.update_yaxes(title_text="Correlation")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def residual_regime_monitor_html(df: pd.DataFrame) -> str:
    decomp = inventory_decomposition(df)
    resid = pd.Series(decomp.resid, index=decomp.observed.index)
    _, noise = split_residual_components(resid)
    rolling = rolling_residual_autocorrelation(noise, window=13, lag=1)
    alert = residual_regime_alert(noise, window=13, lag=1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rolling.index,
            y=rolling.values,
            mode="lines",
            name="13-week rolling lag-1 autocorrelation",
            line=dict(color="#0f766e", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[rolling.index.min(), rolling.index.max()],
            y=[alert["threshold"], alert["threshold"]],
            mode="lines",
            name="Alert threshold",
            line=dict(color="#b45309", width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[rolling.index.min(), rolling.index.max()],
            y=[-alert["threshold"], -alert["threshold"]],
            mode="lines",
            name="Negative alert threshold",
            line=dict(color="#b45309", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title="Noise residual regime monitor",
        template="plotly_white",
        height=360,
        margin=dict(l=20, r=20, t=60, b=20),
        yaxis_title="13-week rolling autocorrelation",
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
    )
    latest = alert["latest_autocorrelation"]
    latest_text = "n/a" if pd.isna(latest) else f"{latest:.2f}"
    threshold_text = f"{alert['threshold']:.2f}"
    state = "Alert" if alert["alert"] else "Normal"
    return f"""
    <div class="analysis-box">
      <h3>Residual regime monitor</h3>
      <p class="small">This monitor tracks the 13-week rolling lag-1 autocorrelation of the final Noise residual. Rising absolute autocorrelation suggests the model is leaving more structure in the residual, which can indicate a storage-regime shift or a new short-memory pattern.</p>
      <p class="small"><strong>Status:</strong> {state}. <strong>Latest autocorrelation:</strong> {latest_text}. <strong>Alert threshold:</strong> {threshold_text}. {alert['message']}</p>
      {fig.to_html(full_html=False, include_plotlyjs=False)}
    </div>
    """


def normalized_prices_chart(close: pd.DataFrame) -> str:
    norm = normalized_prices(close)
    fig = go.Figure()
    palette = ["#0f766e", "#1d4ed8", "#b45309", "#7c3aed", "#dc2626", "#4d7c0f", "#0891b2", "#a16207"]
    for idx, column in enumerate(norm.columns):
        fig.add_trace(
            go.Scatter(
                x=norm.index,
                y=norm[column],
                mode="lines",
                name=column,
                line=dict(width=2, color=palette[idx % len(palette)]),
            )
        )
    fig.update_layout(
        title="Normalized prices",
        template="plotly_white",
        height=440,
        margin=dict(l=20, r=20, t=60, b=20),
        yaxis_title="Indexed to 100",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def correlation_chart(close: pd.DataFrame) -> str:
    corr = correlation_matrix(close).round(2)
    heat = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            text=corr.values,
            texttemplate="%{text:.2f}",
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
        )
    )
    heat.update_layout(
        title="Return correlation matrix",
        template="plotly_white",
        height=520,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return heat.to_html(full_html=False, include_plotlyjs=False)


def portfolio_chart(close: pd.DataFrame) -> str:
    portfolio = build_optimized_portfolio(close, PORTFOLIO_TICKERS)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=portfolio.index.index,
            y=portfolio.index.values,
            mode="lines",
            name="Optimized portfolio",
            line=dict(color="#0f766e", width=3),
        )
    )
    fig.update_layout(
        title="Optimized portfolio performance",
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        yaxis_title="Indexed to 100",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def portfolio_summary_html(close: pd.DataFrame) -> str:
    portfolio = build_optimized_portfolio(close, PORTFOLIO_TICKERS)
    weights = "".join(
        f"<tr><td>{ticker}</td><td>{weight * 100:.1f}%</td></tr>"
        for ticker, weight in portfolio.weights.items()
    )
    return f"""
    <div class="weights-box">
      <h3>Optimized allocation</h3>
      <p class="small">Long-only weights chosen to maximize risk-adjusted return using historical daily returns across the selected stocks and ETFs.</p>
      <table>
        <thead>
          <tr><th>Asset</th><th>Weight</th></tr>
        </thead>
        <tbody>{weights}</tbody>
      </table>
      <p class="small"><strong>Annualized return:</strong> {portfolio.annual_return * 100:.1f}%<br>
      <strong>Annualized volatility:</strong> {portfolio.annual_volatility * 100:.1f}%<br>
      <strong>Sharpe ratio:</strong> {portfolio.sharpe_ratio:.2f}</p>
    </div>
    """


def monthly_returns_table(close: pd.DataFrame) -> str:
    monthly = monthly_returns(close)
    tables = []
    for ticker in MONTHLY_RETURN_TICKERS:
        if ticker not in monthly.columns:
            continue
        frame = calendar_return_table(monthly[ticker])
        frame = frame[(frame.index >= 2020) & (frame.index <= 2026)]
        frame = frame.sort_index(ascending=True).fillna("")
        header = "".join(f"<th>{col}</th>" for col in ["Year", *frame.columns.tolist()])
        rows = ""
        for year, values in frame.iterrows():
            cells = "".join(
                f"<td>{value:.1f}%</td>" if value != "" else "<td></td>"
                for value in values.tolist()
            )
            rows += f"<tr><td>{year}</td>{cells}</tr>"
        tables.append(
            f"""
            <div class="mini-table">
              <h3>{ticker}</h3>
              <table>
                <thead><tr>{header}</tr></thead>
                <tbody>{rows}</tbody>
              </table>
            </div>
            """
        )
    return '<div class="mini-grid">' + "".join(tables) + "</div>"


def hdd_overlay_chart(merged_df: pd.DataFrame) -> str:
    plot_df = merged_df.dropna(subset=["us_hdd_weekly", "weekly_change_bcf"]).copy()
    plot_df["storage_draw_bcf"] = -plot_df["weekly_change_bcf"]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=plot_df["period"],
            y=plot_df["storage_draw_bcf"],
            name="Storage draw (Bcf)",
            marker_color=np.where(plot_df["storage_draw_bcf"] >= 0, "#b91c1c", "#2563eb"),
            opacity=0.55,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["period"],
            y=plot_df["us_hdd_weekly"],
            name="NOAA weekly HDD",
            mode="lines",
            line=dict(color="#1d4ed8", width=3),
        ),
        secondary_y=True,
    )
    if (plot_df["period"] == pd.Timestamp("2014-04-18")).any():
        polar = plot_df.loc[plot_df["period"] == pd.Timestamp("2014-04-18")].iloc[0]
        fig.add_annotation(
            x=polar["period"],
            y=polar["storage_draw_bcf"],
            text="2014 Polar Vortex hangover",
            showarrow=True,
            arrowhead=2,
            ay=-45,
            bgcolor="rgba(255,255,255,0.9)",
        )
    fig.update_layout(
        title="NOAA HDD versus weekly storage draws",
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="Storage draw (Bcf)", secondary_y=False)
    fig.update_yaxes(title_text="Heating degree days", secondary_y=True)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def hdd_scatter_chart(merged_df: pd.DataFrame) -> str:
    plot_df = merged_df.dropna(subset=["hdd_anomaly", "weekly_change_bcf"]).copy()
    plot_df["storage_draw_bcf"] = -plot_df["weekly_change_bcf"]
    x = plot_df["hdd_anomaly"].to_numpy(dtype=float)
    y = plot_df["storage_draw_bcf"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1) if len(plot_df) >= 2 else (0.0, float(np.nanmean(y)))
    line_x = np.linspace(np.nanmin(x), np.nanmax(x), 200) if len(plot_df) >= 2 else x
    line_y = slope * line_x + intercept if len(plot_df) >= 2 else np.repeat(intercept, len(line_x))
    colors = np.where(plot_df["extreme_hdd_week"], "#f59e0b", "#64748b")
    sizes = np.where(plot_df["extreme_storage_week"], 12, 8)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["hdd_anomaly"],
            y=plot_df["storage_draw_bcf"],
            mode="markers",
            marker=dict(color=colors, size=sizes, line=dict(color="#ffffff", width=0.5)),
            text=plot_df["period"].dt.strftime("%Y-%m-%d"),
            name="Weekly observations",
            hovertemplate="Week %{text}<br>HDD anomaly %{x:+.0f}<br>Storage draw %{y:+.0f} Bcf<extra></extra>",
        )
    )
    if len(plot_df) >= 2:
        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                line=dict(color="#0f766e", width=3, dash="dash"),
                name="Linear fit",
            )
        )
    fig.update_layout(
        title="HDD anomaly versus storage draw",
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="HDD anomaly",
        yaxis_title="Storage draw (Bcf)",
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def fit_gpd_tail_metrics(
    values: pd.Series,
    threshold_quantile: float = 0.90,
    return_years: int = 50,
    bootstrap_samples: int = 150,
) -> dict[str, float]:
    clean = pd.Series(values).dropna().astype(float)
    if len(clean) < 80:
        return {}
    threshold = float(clean.quantile(threshold_quantile))
    tail = clean[clean > threshold]
    exceedances = tail - threshold
    if len(exceedances) < 12:
        return {}

    shape, _, scale = genpareto.fit(exceedances, floc=0)
    exceedance_rate = len(exceedances) / len(clean)
    target_probability = 1 / (52 * return_years)
    if abs(shape) < 1e-6:
        return_level = threshold + scale * np.log(exceedance_rate / target_probability)
    else:
        return_level = threshold + (scale / shape) * ((exceedance_rate / target_probability) ** shape - 1)

    rng = np.random.default_rng(42)
    shape_samples: list[float] = []
    return_level_samples: list[float] = []
    values_array = clean.to_numpy(dtype=float)
    for _ in range(bootstrap_samples):
        sample = pd.Series(rng.choice(values_array, size=len(values_array), replace=True))
        sample_threshold = float(sample.quantile(threshold_quantile))
        sample_tail = sample[sample > sample_threshold]
        sample_exceedances = sample_tail - sample_threshold
        if len(sample_exceedances) < 12:
            continue
        try:
            sample_shape, _, sample_scale = genpareto.fit(sample_exceedances, floc=0)
        except Exception:
            continue
        sample_rate = len(sample_exceedances) / len(sample)
        if abs(sample_shape) < 1e-6:
            sample_return_level = sample_threshold + sample_scale * np.log(sample_rate / target_probability)
        else:
            sample_return_level = sample_threshold + (sample_scale / sample_shape) * ((sample_rate / target_probability) ** sample_shape - 1)
        shape_samples.append(float(sample_shape))
        return_level_samples.append(float(sample_return_level))

    shape_array = np.asarray(shape_samples, dtype=float)
    return_level_array = np.asarray(return_level_samples, dtype=float)
    return {
        "n_obs": float(len(clean)),
        "n_exceedances": float(len(exceedances)),
        "threshold": float(threshold),
        "shape": float(shape),
        "scale": float(scale),
        "return_level_50y": float(return_level),
        "shape_se": float(np.nanstd(shape_array, ddof=1)) if len(shape_array) > 1 else np.nan,
        "shape_ci_low": float(np.nanpercentile(shape_array, 5)) if len(shape_array) else np.nan,
        "shape_ci_high": float(np.nanpercentile(shape_array, 95)) if len(shape_array) else np.nan,
        "return_level_ci_low": float(np.nanpercentile(return_level_array, 5)) if len(return_level_array) else np.nan,
        "return_level_ci_high": float(np.nanpercentile(return_level_array, 95)) if len(return_level_array) else np.nan,
    }


def conditional_hdd_tail_frame(merged_df: pd.DataFrame) -> pd.DataFrame:
    plot_df = merged_df.dropna(subset=["weekly_change_bcf"]).copy()
    plot_df["storage_draw_bcf"] = -plot_df["weekly_change_bcf"]
    results: list[dict[str, float | str]] = []
    region_specs = [
        ("East HDD proxy", "east_hdd_anomaly"),
        ("Midwest HDD proxy", "midwest_hdd_anomaly"),
    ]
    tercile_labels = ["Low", "Medium", "High"]
    for region_label, anomaly_col in region_specs:
        if anomaly_col not in plot_df.columns or not plot_df[anomaly_col].notna().any():
            continue
        sample = plot_df.dropna(subset=[anomaly_col, "storage_draw_bcf"]).copy()
        sample["intensity"] = pd.qcut(sample[anomaly_col], q=3, labels=tercile_labels, duplicates="drop")
        for intensity in sample["intensity"].dropna().unique():
            subset = sample.loc[sample["intensity"] == intensity, "storage_draw_bcf"]
            metrics = fit_gpd_tail_metrics(subset)
            if not metrics:
                continue
            metrics["region"] = region_label
            metrics["intensity"] = str(intensity)
            results.append(metrics)
    return pd.DataFrame(results)


def conditional_return_level_chart(cond_df: pd.DataFrame) -> str:
    fig = go.Figure()
    tercile_order = ["Low", "Medium", "High"]
    palette = {"East HDD proxy": "#0f766e", "Midwest HDD proxy": "#1d4ed8"}
    for region in cond_df["region"].dropna().unique():
        region_df = cond_df[cond_df["region"] == region].copy()
        region_df["intensity"] = pd.Categorical(region_df["intensity"], categories=tercile_order, ordered=True)
        region_df = region_df.sort_values("intensity")
        errors_plus = region_df["return_level_ci_high"] - region_df["return_level_50y"]
        errors_minus = region_df["return_level_50y"] - region_df["return_level_ci_low"]
        fig.add_trace(
            go.Scatter(
                x=region_df["intensity"],
                y=region_df["return_level_50y"],
                mode="lines+markers",
                name=region,
                line=dict(color=palette.get(region, "#7c3aed"), width=3),
                marker=dict(size=10),
                error_y=dict(
                    type="data",
                    array=errors_plus.fillna(0).to_numpy(dtype=float),
                    arrayminus=errors_minus.fillna(0).to_numpy(dtype=float),
                    thickness=1.2,
                    width=4,
                ),
            )
        )
    fig.update_layout(
        title="Conditional return level plot: 1-in-50-year storage draw by HDD intensity",
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        yaxis_title="1-in-50-year storage draw (Bcf)",
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def shape_wald_test_table(cond_df: pd.DataFrame) -> tuple[str, str]:
    rows = []
    takeaways = []
    for region in cond_df["region"].dropna().unique():
        region_df = cond_df[cond_df["region"] == region].set_index("intensity")
        if "Low" not in region_df.index or "High" not in region_df.index:
            continue
        low = region_df.loc["Low"]
        high = region_df.loc["High"]
        denominator = np.sqrt((low["shape_se"] ** 2) + (high["shape_se"] ** 2))
        z_stat = (high["shape"] - low["shape"]) / denominator if denominator and np.isfinite(denominator) else np.nan
        p_value = float(2 * norm.sf(abs(z_stat))) if np.isfinite(z_stat) else np.nan
        reject = bool(np.isfinite(p_value) and p_value < 0.05)
        rows.append(
            f"<tr><td>{region}</td><td>{low['shape']:.2f}</td><td>{high['shape']:.2f}</td><td>{z_stat:.2f}</td><td>{p_value:.4f}</td><td>{'Yes' if reject else 'No'}</td></tr>"
        )
        takeaways.append(
            f"{region}: {'reject' if reject else 'do not reject'} H0 that xi_low equals xi_high (p={p_value:.4f})."
        )
    return "".join(rows), " ".join(takeaways)


def ccf_frame(merged_df: pd.DataFrame, column: str, max_lag: int = 6) -> pd.DataFrame:
    sample = merged_df.dropna(subset=[column, "weekly_change_bcf"]).copy()
    sample["storage_draw_bcf"] = -sample["weekly_change_bcf"]
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            corr = sample[column].shift(lag).corr(sample["storage_draw_bcf"])
        else:
            corr = sample[column].corr(sample["storage_draw_bcf"].shift(-lag))
        rows.append({"lag": lag, "correlation": float(corr) if pd.notna(corr) else np.nan})
    return pd.DataFrame(rows)


def ccf_chart(merged_df: pd.DataFrame) -> str:
    fig = go.Figure()
    specs = [
        ("US HDD anomaly", "hdd_anomaly", "#0f766e"),
        ("East HDD proxy anomaly", "east_hdd_anomaly", "#1d4ed8"),
        ("Midwest HDD proxy anomaly", "midwest_hdd_anomaly", "#b45309"),
    ]
    for label, column, color in specs:
        if column not in merged_df.columns or not merged_df[column].notna().any():
            continue
        frame = ccf_frame(merged_df, column, max_lag=6)
        fig.add_trace(
            go.Scatter(
                x=frame["lag"],
                y=frame["correlation"],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=3),
                marker=dict(size=7),
            )
        )
    fig.add_vline(x=1, line_color="#111827", line_dash="dot", line_width=1.5)
    fig.update_layout(
        title="Cross-correlation: HDD leads/lags storage draws",
        template="plotly_white",
        height=380,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Lag in weeks (positive means HDD leads storage draw)",
        yaxis_title="Correlation",
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def hdd_weather_section_html(merged_df: pd.DataFrame) -> str:
    if merged_df.empty or merged_df["us_hdd_weekly"].notna().sum() == 0:
        return """
        <section class="panel">
          <h2>Weather conditioning</h2>
          <p>No overlapping NOAA HDD history is available locally yet, so the weather merge section could not be rendered.</p>
        </section>
        """

    plot_df = merged_df.dropna(subset=["us_hdd_weekly", "weekly_change_bcf"]).copy()
    plot_df["storage_draw_bcf"] = -plot_df["weekly_change_bcf"]
    source_name = merged_df.attrs.get("hdd_source", "NOAA HDD data")
    matched_rows = int(merged_df.attrs.get("hdd_rows_merged", len(plot_df)))
    proxy_note = merged_df.attrs.get("regional_proxy_note", "")
    raw_correlation = float(plot_df["us_hdd_weekly"].corr(plot_df["storage_draw_bcf"]))
    anomaly_correlation = float(plot_df["hdd_anomaly"].corr(plot_df["storage_draw_bcf"]))
    cond_df = conditional_hdd_tail_frame(merged_df)
    shape_rows, shape_takeaway = shape_wald_test_table(cond_df)

    contingency = pd.crosstab(
        plot_df["extreme_hdd_week"].fillna(False),
        plot_df["extreme_storage_week"].fillna(False),
    ).reindex(index=[False, True], columns=[False, True], fill_value=0)
    chi2, p_value, _, _ = chi2_contingency(contingency.to_numpy())
    extreme_hdd_total = int(contingency.loc[True].sum())
    normal_hdd_total = int(contingency.loc[False].sum())
    p_draw_given_extreme = (
        contingency.loc[True, True] / extreme_hdd_total if extreme_hdd_total else np.nan
    )
    p_draw_given_normal = (
        contingency.loc[False, True] / normal_hdd_total if normal_hdd_total else np.nan
    )
    amplification = (
        p_draw_given_extreme / p_draw_given_normal
        if pd.notna(p_draw_given_extreme) and pd.notna(p_draw_given_normal) and p_draw_given_normal > 0
        else np.nan
    )

    tercile_labels = ["Low HDD anomaly", "Mid HDD anomaly", "High HDD anomaly"]
    plot_df["hdd_tercile"] = pd.qcut(plot_df["hdd_anomaly"], q=3, labels=tercile_labels, duplicates="drop")
    tercile_summary = (
        plot_df.dropna(subset=["hdd_tercile"])
        .groupby("hdd_tercile", observed=False)
        .agg(
            weeks=("period", "size"),
            avg_hdd=("us_hdd_weekly", "mean"),
            avg_anomaly=("hdd_anomaly", "mean"),
            avg_draw=("storage_draw_bcf", "mean"),
            extreme_draw_rate=("extreme_storage_week", "mean"),
        )
        .reset_index()
    )
    tercile_rows = "".join(
        f"<tr><td>{row.hdd_tercile}</td><td>{row.weeks:.0f}</td><td>{row.avg_hdd:.1f}</td><td>{row.avg_anomaly:+.1f}</td><td>{row.avg_draw:+.1f} Bcf</td><td>{row.extreme_draw_rate * 100:.1f}%</td></tr>"
        for row in tercile_summary.itertuples()
    )

    contingency_rows = f"""
      <tr><td>Extreme HDD week</td><td>{contingency.loc[True, True]:.0f}</td><td>{contingency.loc[True, False]:.0f}</td><td>{extreme_hdd_total:.0f}</td></tr>
      <tr><td>Normal HDD week</td><td>{contingency.loc[False, True]:.0f}</td><td>{contingency.loc[False, False]:.0f}</td><td>{normal_hdd_total:.0f}</td></tr>
    """

    return f"""
    <section class="panel">
      <h2>Weather conditioning</h2>
      <p>This section merges NOAA weekly heating degree day data into the EIA storage history using the instructed Friday alignment with a nearest-date tolerance of up to three days. It gives the physical weather context behind large inventory withdrawals and builds.</p>
      <p class="small"><strong>Local HDD source:</strong> {source_name}. <strong>Matched weekly rows:</strong> {matched_rows}. <strong>Raw HDD versus storage-draw correlation:</strong> {raw_correlation:.2f}. <strong>HDD anomaly versus storage-draw correlation:</strong> {anomaly_correlation:.2f}.</p>
      <p class="small">The raw HDD relationship stays strong because winter seasonality drives both weather demand and storage withdrawals together. The anomaly relationship is intentionally harder: it removes the normal seasonal pattern and focuses only on unusually cold or warm departures from that baseline. The anomaly now uses a rolling weekly normal: a 30-year average where enough history exists, otherwise a 10-year weekly average.</p>
      <p class="small">{proxy_note}</p>
      {hdd_overlay_chart(plot_df)}
      {hdd_scatter_chart(plot_df)}
      <div class="analysis-box">
        <h3>Extreme HDD contingency</h3>
        <p class="small">Extreme HDD weeks are defined as weeks with HDD anomaly above the 90th percentile. Extreme storage weeks are defined as weeks where the absolute storage change is above the 90th percentile. This is the direct weather-conditioning check from your merge note.</p>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Condition</th><th>Extreme storage week</th><th>Non-extreme storage week</th><th>Total</th></tr></thead>
            <tbody>{contingency_rows}</tbody>
          </table>
        </div>
        <p class="small"><strong>P(extreme storage | extreme HDD):</strong> {p_draw_given_extreme * 100:.1f}%.
        <strong>P(extreme storage | normal HDD):</strong> {p_draw_given_normal * 100:.1f}%.
        <strong>Amplification:</strong> {amplification:.1f}x.
        <strong>Chi-square p-value:</strong> {p_value:.4f}.</p>
      </div>
      <div class="analysis-box">
        <h3>HDD tercile conditioning</h3>
        <p class="small">Weeks are categorized into Low, Medium, and High HDD intensity terciles using the historical distribution of the rolling-normal anomaly. This is the conditioning layer for the weather-dependent tail model.</p>
        <div class="table-wrap">
          <table>
            <thead><tr><th>HDD tercile</th><th>Weeks</th><th>Avg HDD</th><th>Avg anomaly</th><th>Avg storage draw</th><th>Extreme storage rate</th></tr></thead>
            <tbody>{tercile_rows}</tbody>
          </table>
        </div>
      </div>
      <div class="analysis-box">
        <h3>Conditional return levels</h3>
        <p class="small">This plot shows how the 1-in-50-year storage draw changes as the weather regime moves from a mild tercile to a severe tercile. If the severe-cold line sits materially above the mild line, the storage tail is weather-dependent rather than static.</p>
        {conditional_return_level_chart(cond_df) if not cond_df.empty else ""}
      </div>
      <div class="analysis-box">
        <h3>Methodology: lead-lag check</h3>
        <p class="small">Positive lags mean HDD leads the storage report. The lag-1 point is the direct check for whether last week's NOAA weather shock maps into this week's EIA storage draw because of reporting timing.</p>
        {ccf_chart(merged_df)}
      </div>
      <div class="analysis-box">
        <h3>Shape-parameter test</h3>
        <p class="small">A Wald-style test compares the fitted GPD shape parameter in the Low and High HDD terciles. The null hypothesis is <strong>H0: xi_low = xi_high</strong>. Rejecting that null would support the claim that extreme weather changes not just the level of risk, but the shape of the storage-draw tail itself.</p>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Region</th><th>xi low</th><th>xi high</th><th>Wald z</th><th>p-value</th><th>Reject H0 at 5%</th></tr></thead>
            <tbody>{shape_rows}</tbody>
          </table>
        </div>
        <p class="small">{shape_takeaway}</p>
      </div>
    </section>
    """


def sentiment_section_html(
    sentiment_df: pd.DataFrame,
    market_close: pd.DataFrame,
    inventory_df: pd.DataFrame,
    evt_inventory_df: pd.DataFrame,
) -> str:
    if sentiment_df.empty:
        return """
        <section class="panel">
          <h2>Inventory shock sentiment Analysis</h2>
          <p>FinBERT and VADER sentiment has not been generated yet for unusual inventory shocks.</p>
        </section>
        """

    ngf_weekly = pd.Series(dtype=float)
    if "NG=F" in market_close.columns:
        ngf_weekly = market_close["NG=F"].pct_change() * 100

    sentiment_df = sentiment_df.copy()
    if not ngf_weekly.empty:
        sentiment_df["ngf_change_pct"] = sentiment_df["period"].map(ngf_weekly)
    else:
        sentiment_df["ngf_change_pct"] = pd.NA

    chart = sentiment_chart(sentiment_df)
    recent = sentiment_df.sort_values("period", ascending=False).head(8).copy()
    row_chunks = []
    for row in recent.itertuples():
        ngf_cell = f"{row.ngf_change_pct:+.1f}%" if pd.notna(row.ngf_change_pct) else ""
        evt_cell = "Yes" if bool(getattr(row, "is_extreme_tail_event", False)) else "No"
        row_chunks.append(
            "<tr>"
            f"<td>{row.period.strftime('%Y-%m-%d')}</td>"
            f"<td>{row.weekly_change_bcf:+.0f} Bcf</td>"
            f"<td>{ngf_cell}</td>"
            f"<td>{'<strong>' if row.abs_zscore > 3 else ''}{row.abs_zscore:.2f}{'</strong>' if row.abs_zscore > 3 else ''}</td>"
            f"<td>{evt_cell}</td>"
            f"<td>{row.inventory_signal}</td>"
            f"<td>{row.finbert_label}<br>({row.finbert_score:.2f})</td>"
            f"<td>{row.vader_compound:.2f}</td>"
            "</tr>"
        )
    rows = "".join(row_chunks)
    vader_mean = sentiment_df["vader_compound"].mean()
    finbert_counts = sentiment_df["finbert_label"].value_counts()
    top_finbert = finbert_counts.index[0] if not finbert_counts.empty else "n/a"
    return f"""
    <section class="panel">
      <h2>Inventory shock sentiment Analysis</h2>
      <p>This section scores unusually large weekly inventory builds and drawdowns using FinBERT and VADER on structured event summaries. It is intended as a sentiment layer over inventory shocks rather than a substitute for market fundamentals.</p>
      <p class="small">Important: the NLP labels reflect text tone, not necessarily gas-price direction. For example, FinBERT can classify a large inventory drawdown as textually negative even when the inventory shock is bullish for natural gas prices.</p>
      <p class="small">The chart overlays Peaks-over-Threshold tail cutoffs and Generalized Pareto Distribution Value-at-Risk lines. Amber bars are inventory changes that cross the fitted GPD VaR threshold and are classified as Extreme Tail Events.</p>
      {chart}
      <p class="small"><strong>Average VADER compound:</strong> {vader_mean:.2f}. <strong>Most common FinBERT label:</strong> {top_finbert}.</p>
      {finbert_draw_volatility_html(sentiment_df, market_close)}
      {inventory_signal_probability_html(sentiment_df, market_close)}
      <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Week ending</th>
            <th>Weekly change</th>
            <th>Weekly Natural Gas<br>Price Change</th>
            <th>Shock z-score</th>
            <th>EVT tail event</th>
            <th>Inventory signal</th>
            <th>FinBERT</th>
            <th>VADER</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
      </div>
      {extreme_event_analysis_html(evt_inventory_df)}
    </section>
    """


def inventory_signal_probability_html(sentiment_df: pd.DataFrame, market_close: pd.DataFrame) -> str:
    if "NG=F" not in market_close.columns:
        return ""
    ng_price = market_close["NG=F"].dropna()
    if ng_price.empty:
        return ""

    ng_returns = ng_price.pct_change().dropna()
    records = []
    for row in sentiment_df.dropna(subset=["period", "weekly_change_bcf", "finbert_label"]).itertuples():
        event_date = pd.Timestamp(row.period)
        forward_date = event_date + pd.Timedelta(days=7)
        if forward_date > ng_price.index.max():
            continue
        start_price = ng_price.asof(event_date)
        end_price = ng_price.asof(forward_date)
        trailing = ng_returns[ng_returns.index <= event_date].tail(63)
        if pd.isna(start_price) or pd.isna(end_price) or len(trailing) < 20:
            continue
        forward_return = (float(end_price) / float(start_price) - 1) * 100
        two_sigma_threshold = float(trailing.std() * (5 ** 0.5) * 2 * 100)
        records.append(
            {
                "finbert_label": row.finbert_label,
                "inventory_condition": "Inventory draw" if row.weekly_change_bcf < 0 else "Inventory build",
                "forward_1w_return_pct": forward_return,
                "two_sigma_threshold_pct": two_sigma_threshold,
                "two_sigma_price_spike": forward_return >= two_sigma_threshold,
            }
        )

    events = pd.DataFrame(records)
    if events.empty:
        return ""

    total_events = len(events)
    grouped = (
        events.groupby(["finbert_label", "inventory_condition"], dropna=False)
        .agg(
            events=("two_sigma_price_spike", "size"),
            two_sigma_spikes=("two_sigma_price_spike", "sum"),
            avg_forward_1w_return_pct=("forward_1w_return_pct", "mean"),
            avg_two_sigma_threshold_pct=("two_sigma_threshold_pct", "mean"),
        )
        .reset_index()
    )
    grouped["joint_probability_pct"] = grouped["two_sigma_spikes"] / total_events * 100
    grouped["conditional_probability_pct"] = grouped["two_sigma_spikes"] / grouped["events"] * 100
    grouped["conditional_probability_ci_pct"] = grouped.apply(
        lambda row: wilson_interval(
            int(row["two_sigma_spikes"]),
            int(row["events"]),
        ),
        axis=1,
    )
    grouped["conditional_probability_bootstrap_se_pct"] = grouped.apply(
        lambda row: bootstrap_binary_standard_error(
            successes=int(row["two_sigma_spikes"]),
            total=int(row["events"]),
        ),
        axis=1,
    )

    def sort_key(row) -> tuple[int, int]:
        label_order = {"negative": 0, "neutral": 1, "positive": 2}
        condition_order = {"Inventory draw": 0, "Inventory build": 1}
        return (
            label_order.get(str(row.finbert_label), 99),
            condition_order.get(str(row.inventory_condition), 99),
        )

    grouped = grouped.sort_values(by=["finbert_label", "inventory_condition"], key=lambda col: col)
    grouped = pd.DataFrame(sorted(grouped.itertuples(index=False), key=sort_key), columns=grouped.columns)
    rows = "".join(
        "<tr>"
        f"<td>{row.finbert_label}</td>"
        f"<td>{row.inventory_condition}</td>"
        f"<td>{row.events:.0f}</td>"
        f"<td>{row.two_sigma_spikes:.0f}</td>"
        f"<td>{row.joint_probability_pct:.1f}%</td>"
        f"<td>{row.conditional_probability_pct:.1f}%</td>"
        f"<td>{row.conditional_probability_ci_pct[0]:.1f}% to {row.conditional_probability_ci_pct[1]:.1f}%</td>"
        f"<td>{row.conditional_probability_bootstrap_se_pct:.1f}%</td>"
        f"<td>{row.avg_forward_1w_return_pct:+.1f}%</td>"
        f"<td>{row.avg_two_sigma_threshold_pct:.1f}%</td>"
        "</tr>"
        for row in grouped.itertuples(index=False)
    )
    focus = grouped[
        (grouped["finbert_label"] == "negative")
        & (grouped["inventory_condition"] == "Inventory draw")
    ]
    focus_text = ""
    if not focus.empty:
        focus_row = focus.iloc[0]
        focus_text = (
            f" For the key signal, negative FinBERT sentiment plus an inventory draw produced "
            f"{focus_row['two_sigma_spikes']:.0f} two-sigma upside NG=F spikes from {focus_row['events']:.0f} events, "
            f"a conditional probability of {focus_row['conditional_probability_pct']:.1f}% "
            f"with a Wilson interval of {focus_row['conditional_probability_ci_pct'][0]:.1f}% to {focus_row['conditional_probability_ci_pct'][1]:.1f}% "
            f"and bootstrap standard error of {focus_row['conditional_probability_bootstrap_se_pct']:.1f}%."
        )

    return f"""
      <div class="analysis-box">
        <h3>Signal table: FinBERT plus inventory shock</h3>
        <p class="small">This table estimates the probability of a forward one-week NG=F price spike after each sentiment/inventory condition. A price spike is defined as a one-week NG=F gain greater than two times trailing weekly volatility, estimated from the prior 63 trading days.{focus_text}</p>
        <div class="table-wrap">
          <table>
            <thead>
              <tr><th>FinBERT label</th><th>Inventory condition</th><th>Events</th><th>2-sigma spikes</th><th>Joint probability</th><th>P(spike | signal)</th><th>Wilson 90% CI</th><th>Bootstrap SE</th><th>Avg next-week NG=F return</th><th>Avg 2-sigma threshold</th></tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
      </div>
    """


def format_return_period(probability: float) -> str:
    if not np.isfinite(probability) or probability <= 0:
        return "not observed in sample"
    years = 1 / (probability * 52)
    if years < 1:
        weeks = 1 / probability
        return f"1-in-{weeks:.0f} weeks"
    return f"1-in-{years:.1f} years"


def wilson_interval(successes: int, total: int, confidence_level: float = 0.90) -> tuple[float, float]:
    if total <= 0:
        return (np.nan, np.nan)
    z = norm.ppf(0.5 + confidence_level / 2)
    phat = successes / total
    denominator = 1 + z**2 / total
    center = (phat + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total) / denominator
    return (max(0.0, (center - margin) * 100), min(100.0, (center + margin) * 100))


def bootstrap_binary_standard_error(successes: int, total: int, bootstrap_samples: int = 200) -> float:
    if total <= 1:
        return np.nan
    rng = np.random.default_rng(42)
    series = np.array([1] * successes + [0] * (total - successes), dtype=float)
    estimates = []
    for _ in range(bootstrap_samples):
        draw = rng.choice(series, size=total, replace=True)
        estimates.append(draw.mean() * 100)
    return float(np.std(estimates, ddof=1))


def bootstrap_evt_uncertainty(
    shocks: pd.Series,
    selected_shock: float,
    x_values: np.ndarray,
    threshold_quantile: float = 0.90,
    bootstrap_samples: int = 250,
) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(42)
    shock_values = np.asarray(shocks.dropna(), dtype=float)
    shape_samples: list[float] = []
    probability_samples: list[float] = []
    density_samples: list[np.ndarray] = []

    for _ in range(bootstrap_samples):
        sample = rng.choice(shock_values, size=len(shock_values), replace=True)
        sample_threshold = float(np.quantile(sample, threshold_quantile))
        sample_tail = sample[sample > sample_threshold]
        if len(sample_tail) < 10:
            continue
        sample_exceedances = sample_tail - sample_threshold
        try:
            sample_shape, _, sample_scale = genpareto.fit(sample_exceedances, floc=0)
        except Exception:
            continue
        sample_tail_probability = len(sample_tail) / len(sample)
        if selected_shock <= sample_threshold:
            sample_probability = float(np.mean(sample >= selected_shock))
        else:
            sample_probability = float(
                sample_tail_probability * genpareto.sf(selected_shock - sample_threshold, sample_shape, loc=0, scale=sample_scale)
            )
        density = genpareto.pdf(np.clip(x_values - sample_threshold, a_min=0, a_max=None), sample_shape, loc=0, scale=sample_scale)
        density = np.where(x_values >= sample_threshold, density, np.nan)
        shape_samples.append(float(sample_shape))
        probability_samples.append(sample_probability)
        density_samples.append(density)

    shape_array = np.asarray(shape_samples, dtype=float)
    probability_array = np.asarray(probability_samples, dtype=float)
    density_matrix = np.asarray(density_samples, dtype=float)
    return {
        "shape_lower": float(np.nanpercentile(shape_array, 5)) if len(shape_array) else np.nan,
        "shape_upper": float(np.nanpercentile(shape_array, 95)) if len(shape_array) else np.nan,
        "probability_lower": float(np.nanpercentile(probability_array, 5)) if len(probability_array) else np.nan,
        "probability_upper": float(np.nanpercentile(probability_array, 95)) if len(probability_array) else np.nan,
        "density_lower": np.nanpercentile(density_matrix, 5, axis=0) if density_matrix.size else np.full_like(x_values, np.nan),
        "density_upper": np.nanpercentile(density_matrix, 95, axis=0) if density_matrix.size else np.full_like(x_values, np.nan),
    }


def extreme_event_analysis_html(inventory_df: pd.DataFrame) -> str:
    decomp = inventory_decomposition(inventory_df)
    residual = pd.Series(decomp.resid, index=decomp.observed.index).dropna()
    shocks = residual.abs().dropna()
    if len(shocks) < 100:
        return ""

    threshold = float(shocks.quantile(0.90))
    tail = shocks[shocks > threshold]
    exceedances = tail - threshold
    if len(exceedances) < 10:
        return ""

    shape, _, scale = genpareto.fit(exceedances, floc=0)
    selected_date = shocks.idxmax()
    selected_shock = float(shocks.loc[selected_date])
    selected_signed = float(residual.loc[selected_date])
    residual_std = float(residual.std())
    selected_z = selected_shock / residual_std if residual_std > 0 else np.nan

    standard_probability = float(2 * norm.sf(selected_z)) if np.isfinite(selected_z) else np.nan
    tail_probability = len(tail) / len(shocks)
    if selected_shock <= threshold:
        gpd_probability = float((shocks >= selected_shock).mean())
    else:
        gpd_probability = float(tail_probability * genpareto.sf(selected_shock - threshold, shape, loc=0, scale=scale))

    x_values = np.linspace(threshold, max(float(tail.max()), selected_shock) * 1.05, 180)
    folded_normal_tail_probability = max(float(2 * norm.sf(threshold / residual_std)), 1e-12) if residual_std > 0 else np.nan
    gpd_density = genpareto.pdf(x_values - threshold, shape, loc=0, scale=scale)
    evt_bootstrap = bootstrap_evt_uncertainty(shocks, selected_shock, x_values)
    normal_density = (
        2 * norm.pdf(x_values, loc=0, scale=residual_std) / folded_normal_tail_probability
        if residual_std > 0 and np.isfinite(folded_normal_tail_probability)
        else np.full_like(x_values, np.nan)
    )

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=tail,
            histnorm="probability density",
            nbinsx=18,
            marker_color="#d97706",
            opacity=0.55,
            name="Observed tail shocks",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_values, x_values[::-1]]),
            y=np.concatenate([evt_bootstrap["density_upper"], evt_bootstrap["density_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(124,45,18,0.14)",
            line=dict(color="rgba(124,45,18,0)"),
            hoverinfo="skip",
            name="GPD 90% bootstrap band",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=gpd_density,
            mode="lines",
            line=dict(color="#7c2d12", width=3),
            name="Fitted GPD tail",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=normal_density,
            mode="lines",
            line=dict(color="#1d4ed8", width=2, dash="dash"),
            name="Normal tail expectation",
        )
    )
    fig.add_vline(x=selected_shock, line_color="#111827", line_width=2, line_dash="dot")
    fig.add_vline(x=threshold, line_color="#92400e", line_width=2, line_dash="dash")
    fig.update_layout(
        title="Tail fit plot: absolute STL residual shocks",
        template="plotly_white",
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Absolute residual shock (Bcf)",
        yaxis_title="Density",
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
        bargap=0.04,
    )

    standard_period = format_return_period(standard_probability)
    gpd_period = format_return_period(gpd_probability)
    gpd_period_lower = format_return_period(float(evt_bootstrap["probability_upper"]))
    gpd_period_upper = format_return_period(float(evt_bootstrap["probability_lower"]))
    direction = "positive" if selected_signed >= 0 else "negative"
    start_period = pd.Timestamp(inventory_df["period"].min()).strftime("%Y-%m-%d")
    end_period = pd.Timestamp(inventory_df["period"].max()).strftime("%Y-%m-%d")
    years = (pd.Timestamp(inventory_df["period"].max()) - pd.Timestamp(inventory_df["period"].min())).days / 365.25
    return f"""
      <details class="analysis-box" open>
        <summary><strong>Extreme Event Analysis</strong></summary>
        <p class="small">This panel is the EVT stress-test layer. It uses the full available weekly EIA inventory history from <strong>{start_period}</strong> to <strong>{end_period}</strong> ({years:.1f} years) only for tail-risk fitting, while the rest of the dashboard remains based on the 10-year working dataset. The orange bars show the empirical tail, the brown curve is the fitted GPD tail, and the dashed blue curve shows what a normal residual model would expect beyond the same tail threshold.</p>
        {fig.to_html(full_html=False, include_plotlyjs=False)}
        <p class="small">
          <strong>Selected full-history stress shock:</strong> {selected_date.strftime('%Y-%m-%d')}, {selected_signed:+.0f} Bcf residual ({direction}), or <strong>{selected_z:.2f} sigma</strong>.
          <strong>Tail threshold:</strong> {threshold:.0f} Bcf. <strong>GPD shape:</strong> {shape:.2f} with 90% bootstrap interval <strong>{evt_bootstrap["shape_lower"]:.2f} to {evt_bootstrap["shape_upper"]:.2f}</strong>. <strong>GPD scale:</strong> {scale:.1f}.
        </p>
        <p class="small">
          The week ending April 18, 2014, represents one of the most significant statistical dislocations in the history of the US natural gas market. The reason for this massive residual is that the market was experiencing the hangover of the 2013-2014 Polar Vortex, the most severe winter for natural gas demand in over a decade.
        </p>
        <p class="small">
          <strong>GPD-adjusted probability:</strong> the standard normal model implies this shock is roughly <strong>{standard_period}</strong>;
          the fitted GPD tail implies roughly <strong>{gpd_period}</strong>, with a 90% bootstrap return-period band of <strong>{gpd_period_lower}</strong> to <strong>{gpd_period_upper}</strong>. The gap is the fat-tail evidence: large storage shocks occur more often than a Gaussian model would suggest.
        </p>
      </details>
    """


def finbert_draw_volatility_html(sentiment_df: pd.DataFrame, market_close: pd.DataFrame) -> str:
    if "NG=F" not in market_close.columns:
        return ""
    draws = sentiment_df[
        (sentiment_df["weekly_change_bcf"] < 0)
        & sentiment_df["finbert_label"].notna()
    ].copy()
    if draws.empty:
        return ""

    ng_returns = market_close["NG=F"].pct_change().dropna()
    records = []
    for row in draws.itertuples():
        start = pd.Timestamp(row.period)
        end = start + pd.Timedelta(days=30)
        window = ng_returns[(ng_returns.index > start) & (ng_returns.index <= end)]
        if len(window) < 5:
            continue
        records.append(
            {
                "cohort": "Negative FinBERT bullish draw" if row.finbert_label == "negative" else "Other bullish draw",
                "realized_volatility": float(window.std() * (252 ** 0.5) * 100),
                "absolute_return": float(window.abs().mean() * 100),
            }
        )
    result = pd.DataFrame(records)
    if result.empty:
        return ""
    grouped = result.groupby("cohort").agg(
        events=("realized_volatility", "size"),
        avg_30d_realized_vol=("realized_volatility", "mean"),
        median_30d_realized_vol=("realized_volatility", "median"),
        avg_abs_daily_move=("absolute_return", "mean"),
    )
    desired_order = ["Negative FinBERT bullish draw", "Other bullish draw"]
    grouped = grouped.reindex([idx for idx in desired_order if idx in grouped.index])
    rows = "".join(
        f"<tr><td>{idx}</td><td>{row.events:.0f}</td><td>{row.avg_30d_realized_vol:.1f}%</td><td>{row.median_30d_realized_vol:.1f}%</td><td>{row.avg_abs_daily_move:.2f}%</td></tr>"
        for idx, row in grouped.iterrows()
    )
    takeaway = ""
    if "Negative FinBERT bullish draw" in grouped.index and "Other bullish draw" in grouped.index:
        neg_vol = grouped.loc["Negative FinBERT bullish draw", "avg_30d_realized_vol"]
        other_vol = grouped.loc["Other bullish draw", "avg_30d_realized_vol"]
        direction = "higher" if neg_vol > other_vol else "lower"
        takeaway = (
            f" In this sample, negative FinBERT labels during bullish physical draws were followed by "
            f"{direction} average 30-day realized volatility than other bullish draws "
            f"({neg_vol:.1f}% versus {other_vol:.1f}%)."
        )
    elif "Negative FinBERT bullish draw" in grouped.index:
        neg_vol = grouped.loc["Negative FinBERT bullish draw", "avg_30d_realized_vol"]
        events = int(grouped.loc["Negative FinBERT bullish draw", "events"])
        takeaway = (
            f" In the current sample, all bullish physical draws with valid forward NG=F data were labeled negative by FinBERT, "
            f"so there is no separate non-negative bullish-draw cohort for a direct comparison. These {events} events were followed by "
            f"average 30-day realized volatility of {neg_vol:.1f}%."
        )
    return f"""
      <div class="analysis-box">
        <h3>FinBERT tone versus post-draw volatility</h3>
        <p class="small">This test asks whether a textually <strong>negative</strong> FinBERT label during a physically <strong>bullish</strong> inventory draw is followed by higher NG=F volatility. Realized volatility is calculated from daily NG=F returns over the 30 calendar days after each event.{takeaway}</p>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Cohort</th><th>Events</th><th>Avg 30d realized vol</th><th>Median 30d realized vol</th><th>Avg absolute daily move</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
      </div>
    """


def sentiment_chart(sentiment_df: pd.DataFrame) -> str:
    df = sentiment_df.sort_values("period").copy()
    finbert_numeric = df["finbert_label"].map({"negative": -1, "neutral": 0, "positive": 1}).fillna(0)
    upper_var = df["evt_upper_var_bcf"].dropna().iloc[0] if "evt_upper_var_bcf" in df and df["evt_upper_var_bcf"].notna().any() else None
    lower_var = df["evt_lower_var_bcf"].dropna().iloc[0] if "evt_lower_var_bcf" in df and df["evt_lower_var_bcf"].notna().any() else None
    upper_tail = df["evt_upper_tail_threshold_bcf"].dropna().iloc[0] if "evt_upper_tail_threshold_bcf" in df and df["evt_upper_tail_threshold_bcf"].notna().any() else None
    lower_tail = df["evt_lower_tail_threshold_bcf"].dropna().iloc[0] if "evt_lower_tail_threshold_bcf" in df and df["evt_lower_tail_threshold_bcf"].notna().any() else None

    def bar_color(row) -> str:
        if bool(getattr(row, "is_extreme_tail_event", False)):
            return "#f59e0b"
        return "#b91c1c" if row.weekly_change_bcf < 0 else "#2563eb"

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=df["period"],
            y=df["weekly_change_bcf"],
            name="Weekly inventory change (Bcf)",
            marker_color=[bar_color(row) for row in df.itertuples()],
            opacity=0.55,
        ),
        secondary_y=False,
    )
    for value, name, color, dash in [
        (upper_tail, "Upper POT tail threshold", "#f97316", "dot"),
        (lower_tail, "Lower POT tail threshold", "#f97316", "dot"),
        (upper_var, "Upper GPD VaR", "#d97706", "dash"),
        (lower_var, "Lower GPD VaR", "#d97706", "dash"),
    ]:
        if value is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=[df["period"].min(), df["period"].max()],
                y=[value, value],
                name=name,
                mode="lines",
                line=dict(color=color, width=2, dash=dash),
            ),
            secondary_y=False,
        )
    fig.add_trace(
        go.Scatter(
            x=df["period"],
            y=df["vader_compound"],
            name="VADER compound",
            mode="lines+markers",
            line=dict(color="#0f766e", width=3),
            marker=dict(size=6),
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=df["period"],
            y=finbert_numeric,
            name="FinBERT sentiment",
            mode="lines+markers",
            line=dict(color="#7c3aed", width=2, dash="dot"),
            marker=dict(size=6),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Inventory shock size versus sentiment over time",
        template="plotly_white",
        height=460,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="Weekly change (Bcf)", secondary_y=False)
    fig.update_yaxes(
        title_text="Sentiment score",
        secondary_y=True,
        range=[-1.1, 1.1],
        tickvals=[-1, -0.5, 0, 0.5, 1],
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def lower48_table(df: pd.DataFrame) -> str:
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    year_ago = df[df["period"] <= latest["period"] - pd.Timedelta(days=364)].iloc[-1]
    five_year = latest["inventory_52w_avg"]
    rows = [
        ("Lower 48 Total", latest["value_bcf"], prev["value_bcf"], latest["value_bcf"] - prev["value_bcf"], year_ago["value_bcf"], latest["value_bcf"] - year_ago["value_bcf"], five_year),
    ]
    body = "".join(
        f"<tr><td>{name}</td><td>{current:,.0f}</td><td>{previous:,.0f}</td><td>{weekly:+,.0f}</td><td>{yrago:,.0f}</td><td>{delta:+,.0f}</td><td>{avg:,.0f}</td></tr>"
        for name, current, previous, weekly, yrago, delta, avg in rows
    )
    return f"""
    <table>
      <thead>
        <tr>
          <th>Region</th>
          <th>Current storage</th>
          <th>Previous week</th>
          <th>Weekly change</th>
          <th>Year ago</th>
          <th>Change vs year ago</th>
          <th>Rolling 52-week average</th>
        </tr>
      </thead>
      <tbody>{body}</tbody>
    </table>
    """


def regional_table(release: dict) -> str:
    if not release.get("series"):
        return """
        <div class="table-wrap">
          <p class="small">Regional storage detail is not available in the current cached payload yet. The page-level dates and Lower 48 totals above are updated through the latest EIA release.</p>
        </div>
        """
    rows = []
    for series in release["series"]:
        if series["name"] == "total lower 48 states":
            continue
        current, previous, year_ago = series["data"]
        calc = series["calculated"]
        rows.append(
            (
                series["name"].replace(" region", "").title(),
                current[1],
                previous[1],
                calc["net_change"],
                year_ago[1],
                calc["5yr-avg"],
                calc["pct-change_yrago"],
                calc["pct-chg_5yr-avg"],
            )
        )
    body = "".join(
        f"<tr><td>{region}</td><td>{current:,.0f}</td><td>{previous:,.0f}</td><td>{weekly:+,.0f}</td><td>{yrago:,.0f}</td><td>{avg:,.0f}</td><td>{pyr:+.1f}%</td><td>{pavg:+.1f}%</td></tr>"
        for region, current, previous, weekly, yrago, avg, pyr, pavg in rows
    )
    return f"""
    <table>
      <thead>
        <tr>
          <th>Region</th>
          <th>Current storage</th>
          <th>Previous week</th>
          <th>Weekly net change</th>
          <th>Same week year ago</th>
          <th>Five-year average</th>
          <th>Change vs year ago</th>
          <th>Change vs five-year average</th>
        </tr>
      </thead>
      <tbody>{body}</tbody>
    </table>
    """


def html_page(
    df: pd.DataFrame,
    release: dict,
    market_close: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    evt_inventory_df: pd.DataFrame,
    weather_hdd_df: pd.DataFrame,
) -> str:
    summary = summarize_inventory(df)
    refreshed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EIA Natural Gas Weekly Inventory Analysis</title>
  <style>
    :root {{
      --bg: #f7f7f2;
      --panel: #ffffff;
      --ink: #15231f;
      --muted: #5f6f69;
      --accent: #0f766e;
      --accent-2: #1d4ed8;
      --border: #d8ddd7;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background: linear-gradient(180deg, #eef4ef 0%, var(--bg) 240px);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 40px 20px 72px;
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    h1 {{ font-size: 2.6rem; letter-spacing: -0.03em; }}
    h2 {{ font-size: 1.6rem; margin-top: 34px; }}
    p {{ color: var(--muted); font-size: 1.02rem; line-height: 1.6; }}
    .lede {{
      max-width: 760px;
      font-size: 1.1rem;
    }}
    .stamp {{
      margin-top: 14px;
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .summary-grid {{
      margin: 24px 0 8px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 14px;
    }}
    .nav {{
      display: flex;
      gap: 14px;
      margin-bottom: 18px;
    }}
    .nav a {{
      color: #0f766e;
      text-decoration: none;
      font-weight: 600;
    }}
    .nav a.active {{
      color: #15231f;
    }}
    .card, .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(21, 35, 31, 0.05);
    }}
    .card {{
      padding: 18px 18px 20px;
    }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 0.72rem;
      color: var(--muted);
      margin-bottom: 10px;
    }}
    .metric {{
      font-size: 1.5rem;
      line-height: 1.2;
    }}
    .panel {{
      padding: 18px;
      margin-top: 18px;
    }}
    .two-up {{
      display: grid;
      gap: 18px;
      grid-template-columns: 1fr;
    }}
    .mini-grid {{
      display: grid;
      gap: 16px;
      grid-template-columns: 1fr;
    }}
    .mini-table h3 {{
      font-size: 1rem;
      margin-bottom: 10px;
    }}
    .mini-table {{
      overflow-x: auto;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    .weights-box {{
      margin-top: 14px;
      padding-top: 6px;
    }}
    .analysis-box {{
      margin-top: 14px;
      padding: 2px 0 0;
    }}
    .small {{
      font-size: 0.94rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.96rem;
    }}
    th, td {{
      padding: 12px 10px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
    }}
    @media (min-width: 900px) {{
      .two-up {{
        grid-template-columns: 1fr 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="wrap">
    {nav_html()}
    <h1>EIA Natural Gas Weekly Inventory Analysis</h1>
    <p class="lede">
      This page gives a quick, readable summary of the latest official U.S. Energy Information Administration weekly natural gas release. It highlights what changed this week, how current storage compares with history, and the cleaned tables behind the charts.
    </p>
    <div class="stamp">Report refreshed at {refreshed}</div>
    {summary_cards_html(summary, release)}

    <section class="panel">
      <h2>Lower 48 storage history</h2>
      <p>This chart tracks total Lower 48 working gas in storage and overlays a rolling 52-week average so the current cycle is easier to place in context.</p>
      {inventory_history_chart(df)}
    </section>

    <section class="panel">
      <h2>Seasonal position</h2>
      <p>This chart compares the current year with the historical weekly storage range and average.</p>
      {seasonality_chart(df)}
    </section>

    <section class="panel">
      <h2>Lower 48 summary table</h2>
      <p>This table summarizes the latest total storage level against last week, the same point last year, and the rolling one-year average.</p>
      {lower48_table(df)}
    </section>

    <section class="panel">
      <h2>Decomposition</h2>
      <p>The inventory observed series is split into longterm trend, sesonal change and random shifts.</p>
      {decomposition_chart(df)}
      {decomposition_analysis_html(df)}
      {adf_test_html(df)}
      {noise_acf_pacf_chart(df)}
      {residual_regime_monitor_html(df)}
    </section>

    {hdd_weather_section_html(weather_hdd_df)}

    <section class="panel">
      <h2>Regional storage table</h2>
      <p>This table summarizes the latest weekly storage release by reporting region. All storage values are shown in billion cubic feet.</p>
      {regional_table(release)}
    </section>

    {sentiment_section_html(sentiment_df, market_close, df, evt_inventory_df)}

    <section class="panel">
      <h2>Market analysis</h2>
      <p>The stock, ETF, natural gas futures, portfolio, and market-sentiment sections now live on a separate dedicated page.</p>
      <p><a href="./market.html">Open the Natural Gas Market and Portfolio Analysis page</a></p>
    </section>
  </main>
</body>
</html>
"""


def main() -> int:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_inventory_data(BASE_DIR)
    full_inventory_df = load_full_inventory_data(BASE_DIR)
    release = latest_release_payload(df)
    hdd_df = load_hdd_data(BASE_DIR, prefer_full_history=True)
    weather_hdd_df = merge_inventory_hdd(full_inventory_df, hdd_df)
    market_tickers = list(dict.fromkeys(DEFAULT_TICKERS + MONTHLY_RETURN_TICKERS))
    market_close = fetch_market_prices(market_tickers, start="2017-01-01")
    sentiment_df = load_sentiment_events()
    evt_inventory_df = load_evt_inventory_history(df)
    REPORT_PATH.write_text(
        html_page(df, release, market_close, sentiment_df, evt_inventory_df, weather_hdd_df),
        encoding="utf-8",
    )
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
