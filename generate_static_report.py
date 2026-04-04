from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.stats.diagnostic import acorr_ljungbox

from dashboard_data import (
    DEFAULT_TICKERS,
    PORTFOLIO_TICKERS,
    build_optimized_portfolio,
    calendar_return_table,
    correlation_matrix,
    fetch_market_prices,
    inventory_decomposition,
    load_inventory_data,
    monthly_returns,
    normalized_prices,
    seasonal_inventory_profile,
    split_residual_components,
    summarize_inventory,
)


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
REPORT_PATH = DOCS_DIR / "index.html"
LATEST_JSON = BASE_DIR / "weekly_natural_gas_inventory_2026-03-20.json"
MONTHLY_RETURN_TICKERS = ["NG=F", "UNG", "USO"]
SENTIMENT_CSV = BASE_DIR / "inventory_sentiment_events.csv"


def latest_release_payload() -> dict:
    return json.loads(LATEST_JSON.read_text(encoding="utf-8-sig"))


def load_sentiment_events() -> pd.DataFrame:
    if not SENTIMENT_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(SENTIMENT_CSV, parse_dates=["period"])
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
    lb = acorr_ljungbox(resid, lags=[13, 26], return_df=True)
    acf1 = resid.autocorr(1)
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
        <strong>{", ".join(str(week) for week in top_weeks)}</strong>. In the figure, this leftover component is split heuristically into a smoother
        <strong>Structured Residual</strong> line, meant to capture winter weather shocks and storage-regime shifts, and a faster-moving <strong>Noise</strong> line for the remaining short-run variation.
      </p>
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


def sentiment_section_html(sentiment_df: pd.DataFrame, market_close: pd.DataFrame) -> str:
    if sentiment_df.empty:
        return """
        <section class="panel">
          <h2>Inventory shock sentiment</h2>
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
        row_chunks.append(
            "<tr>"
            f"<td>{row.period.strftime('%Y-%m-%d')}</td>"
            f"<td>{row.weekly_change_bcf:+.0f} Bcf</td>"
            f"<td>{ngf_cell}</td>"
            f"<td>{row.abs_zscore:.2f}</td>"
            f"<td>{row.inventory_signal}</td>"
            f"<td>{row.finbert_label} ({row.finbert_score:.2f})</td>"
            f"<td>{row.vader_compound:.2f}</td>"
            "</tr>"
        )
    rows = "".join(row_chunks)
    vader_mean = sentiment_df["vader_compound"].mean()
    finbert_counts = sentiment_df["finbert_label"].value_counts()
    top_finbert = finbert_counts.index[0] if not finbert_counts.empty else "n/a"
    return f"""
    <section class="panel">
      <h2>Inventory shock sentiment</h2>
      <p>This section scores unusually large weekly inventory builds and drawdowns using FinBERT and VADER on structured event summaries. It is intended as a sentiment layer over inventory shocks rather than a substitute for market fundamentals.</p>
      <p class="small">Important: the NLP labels reflect text tone, not necessarily gas-price direction. For example, FinBERT can classify a large inventory drawdown as textually negative even when the inventory shock is bullish for natural gas prices.</p>
      {chart}
      <p class="small"><strong>Average VADER compound:</strong> {vader_mean:.2f}. <strong>Most common FinBERT label:</strong> {top_finbert}.</p>
      <table>
        <thead>
          <tr>
            <th>Week ending</th>
            <th>Weekly change</th>
            <th>NG=F change</th>
            <th>Shock z-score</th>
            <th>Inventory signal</th>
            <th>FinBERT</th>
            <th>VADER</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
    """


def sentiment_chart(sentiment_df: pd.DataFrame) -> str:
    df = sentiment_df.sort_values("period").copy()
    finbert_numeric = df["finbert_label"].map({"negative": -1, "neutral": 0, "positive": 1}).fillna(0)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=df["period"],
            y=df["weekly_change_bcf"],
            name="Weekly inventory change (Bcf)",
            marker_color=["#b91c1c" if x < 0 else "#2563eb" for x in df["weekly_change_bcf"]],
            opacity=0.55,
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
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


def html_page(df: pd.DataFrame, release: dict, market_close: pd.DataFrame, sentiment_df: pd.DataFrame) -> str:
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
    </section>

    <section class="panel">
      <h2>Regional storage table</h2>
      <p>This table summarizes the latest weekly storage release by reporting region. All storage values are shown in billion cubic feet.</p>
      {regional_table(release)}
    </section>

    {sentiment_section_html(sentiment_df, market_close)}

    <section class="panel">
      <h2>Market and portfolio</h2>
      <p>In this section, selected US Natural gas based company stocks, Natural Gas Fund ETF, US Oil Fund ETF normalized prices, correlation of their returns, returns of an optimized equity and ETF portfolio and monthly returns tables are analysed. The Source for stocks and ETF data is Yahoo Finance.</p>
      {normalized_prices_chart(market_close)}
    </section>

    <section class="two-up">
      <div class="panel">
        {correlation_chart(market_close)}
      </div>
      <div class="panel">
        {portfolio_chart(market_close)}
        {portfolio_summary_html(market_close)}
      </div>
    </section>

    <section class="panel">
      <h2>Monthly returns</h2>
      <p>Recent calendar-style monthly return tables for the tracked natural gas contract, equities, and ETFs.</p>
      {monthly_returns_table(market_close)}
    </section>
  </main>
</body>
</html>
"""


def main() -> int:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_inventory_data(BASE_DIR)
    release = latest_release_payload()
    market_tickers = list(dict.fromkeys(DEFAULT_TICKERS + MONTHLY_RETURN_TICKERS))
    market_close = fetch_market_prices(market_tickers, start="2019-01-01")
    sentiment_df = load_sentiment_events()
    REPORT_PATH.write_text(html_page(df, release, market_close, sentiment_df), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
