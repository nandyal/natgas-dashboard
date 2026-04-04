from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard_data import (
    DEFAULT_TICKERS,
    PORTFOLIO_TICKERS,
    build_optimized_portfolio,
    calendar_return_table,
    correlation_matrix,
    fetch_market_prices,
    monthly_returns,
    normalized_prices,
)


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
REPORT_PATH = DOCS_DIR / "market.html"
SENTIMENT_CSV = BASE_DIR / "market_sentiment_events.csv"
RECENT_SENTIMENT_CSV = BASE_DIR / "market_sentiment_recent.csv"


def load_market_sentiment() -> pd.DataFrame:
    if not SENTIMENT_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(SENTIMENT_CSV, parse_dates=["period"])


def load_recent_market_sentiment() -> pd.DataFrame:
    if not RECENT_SENTIMENT_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(RECENT_SENTIMENT_CSV, parse_dates=["as_of_date"])


def nav_html() -> str:
    return """
    <nav class="nav">
      <a href="./index.html">Inventory Analysis</a>
      <a href="./market.html" class="active">Market Analysis</a>
    </nav>
    """


def normalized_prices_chart(close: pd.DataFrame) -> str:
    norm = normalized_prices(close)
    fig = go.Figure()
    for column in norm.columns:
        fig.add_trace(go.Scatter(x=norm.index, y=norm[column], mode="lines", name=column))
    fig.update_layout(
        title="Normalized prices",
        template="plotly_white",
        height=460,
        margin=dict(l=20, r=20, t=60, b=20),
        yaxis_title="Indexed to 100",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def correlation_chart(close: pd.DataFrame) -> str:
    corr = correlation_matrix(close).round(2)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            text=corr.values,
            texttemplate="%{text:.2f}",
        )
    )
    fig.update_layout(title="Return correlation matrix", template="plotly_white", height=540, margin=dict(l=20, r=20, t=60, b=20))
    return fig.to_html(full_html=False, include_plotlyjs=False)


def portfolio_chart(close: pd.DataFrame) -> str:
    portfolio = build_optimized_portfolio(close, PORTFOLIO_TICKERS)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio.index.index, y=portfolio.index.values, mode="lines", line=dict(width=3, color="#0f766e"), name="Optimized portfolio"))
    fig.update_layout(title="Optimized portfolio performance", template="plotly_white", height=440, margin=dict(l=20, r=20, t=60, b=20), yaxis_title="Indexed to 100")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def portfolio_summary_html(close: pd.DataFrame) -> str:
    portfolio = build_optimized_portfolio(close, PORTFOLIO_TICKERS)
    rows = "".join(f"<tr><td>{ticker}</td><td>{weight * 100:.1f}%</td></tr>" for ticker, weight in portfolio.weights.items())
    return f"""
    <div class="panel">
      <h2>Optimized allocation</h2>
      <p>The portfolio is optimized on a long-only basis for risk-adjusted return using historical daily returns across selected stocks and ETFs.</p>
      <table>
        <thead><tr><th>Asset</th><th>Weight</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
      <p class="small"><strong>Annualized return:</strong> {portfolio.annual_return * 100:.1f}%<br>
      <strong>Annualized volatility:</strong> {portfolio.annual_volatility * 100:.1f}%<br>
      <strong>Sharpe ratio:</strong> {portfolio.sharpe_ratio:.2f}</p>
    </div>
    """


def monthly_tables_html(close: pd.DataFrame) -> str:
    monthly = monthly_returns(close)
    blocks = []
    for ticker in monthly.columns:
        table = calendar_return_table(monthly[ticker]).sort_index(ascending=True).fillna("")
        header = "".join(f"<th>{col}</th>" for col in ["Year", *table.columns.tolist()])
        rows = ""
        for year, values in table.iterrows():
            cells = "".join(f"<td>{v:.1f}%</td>" if v != "" else "<td></td>" for v in values.tolist())
            rows += f"<tr><td>{year}</td>{cells}</tr>"
        blocks.append(
            f"""
            <div class="mini-table">
              <h3>{ticker}</h3>
              <div class="table-wrap">
                <table>
                  <thead><tr>{header}</tr></thead>
                  <tbody>{rows}</tbody>
                </table>
              </div>
            </div>
            """
        )
    return '<div class="stack-grid">' + "".join(blocks) + "</div>"


def sentiment_chart(sentiment_df: pd.DataFrame) -> str:
    df = sentiment_df.copy()
    finbert_numeric = df["finbert_label"].map({"negative": -1, "neutral": 0, "positive": 1}).fillna(0)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df["period"], y=df["monthly_return_pct"], mode="markers", marker=dict(size=8), name="Monthly return (%)", text=df["ticker"]), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["period"], y=df["forward_1m_return_pct"], mode="markers", marker=dict(size=8, color="#1d4ed8"), name="Next complete month return (%)", text=df["ticker"]), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["period"], y=df["vader_compound"], mode="lines+markers", line=dict(color="#0f766e", width=2), name="VADER"), secondary_y=True)
    fig.add_trace(go.Scatter(x=df["period"], y=finbert_numeric, mode="lines+markers", line=dict(color="#7c3aed", width=2, dash="dot"), name="FinBERT"), secondary_y=True)
    fig.update_layout(title="Sentiment and one-month price reaction", template="plotly_white", height=500, margin=dict(l=20, r=20, t=60, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_yaxes(title_text="Price change (%)", secondary_y=False)
    fig.update_yaxes(title_text="Sentiment score", secondary_y=True, range=[-1.1, 1.1])
    return fig.to_html(full_html=False, include_plotlyjs=False)


def sentiment_table_html(recent_sentiment_df: pd.DataFrame) -> str:
    latest = recent_sentiment_df.sort_values("ticker").copy()
    rows = "".join(
        f"<tr><td>{row.ticker}</td><td>{row.as_of_date.strftime('%Y-%m-%d')}</td><td>{row.one_week_return_pct:+.1f}%</td><td>{row.finbert_label}<br>({row.finbert_score:.2f})</td><td>{row.vader_compound:.2f}</td><td>{row.one_month_return_pct:+.1f}%</td></tr>"
        for row in latest.itertuples()
    )
    return f"""
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>Ticker</th><th>As of date</th><th>1 week return</th><th>FinBERT</th><th>VADER</th><th>1 month return</th></tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """


def sentiment_section_html(sentiment_df: pd.DataFrame, recent_sentiment_df: pd.DataFrame) -> str:
    if sentiment_df.empty:
        return """
        <section class="panel">
          <h2>Stock, ETF, and futures sentiment</h2>
          <p>Market sentiment analysis has not been generated yet.</p>
        </section>
        """
    return f"""
    <section class="panel">
      <h2>Stock, ETF, and futures sentiment</h2>
      <p>This section applies FinBERT and VADER to monthly price-event summaries for each tracked stock, ETF, and natural gas futures series. The chart shows the longer historical pattern, while the table below summarizes the last 30 days as of today.</p>
      <p class="small">These NLP labels describe the tone of price-event summaries, so they should be read alongside realized price moves rather than as standalone trading signals.</p>
      {sentiment_chart(sentiment_df)}
      {sentiment_table_html(recent_sentiment_df)}
    </section>
    """


def page_html(close: pd.DataFrame, sentiment_df: pd.DataFrame, recent_sentiment_df: pd.DataFrame) -> str:
    refreshed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Natural Gas Market and Portfolio Analysis</title>
  <style>
    :root {{ --bg:#f7f7f2; --panel:#ffffff; --ink:#15231f; --muted:#5f6f69; --border:#d8ddd7; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Georgia, "Times New Roman", serif; background:linear-gradient(180deg,#eef4ef 0%, var(--bg) 240px); color:var(--ink); }}
    .wrap {{ max-width:1100px; margin:0 auto; padding:40px 20px 72px; }}
    .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:18px; box-shadow:0 10px 30px rgba(21,35,31,0.05); padding:18px; margin-top:18px; }}
    .nav {{ display:flex; gap:14px; margin-bottom:18px; }}
    .nav a {{ color:#0f766e; text-decoration:none; font-weight:600; }}
    .nav a.active {{ color:#15231f; }}
    .two-up {{ display:grid; gap:18px; grid-template-columns:1fr; }}
    .mini-grid {{ display:grid; gap:16px; grid-template-columns:1fr; }}
    .stack-grid {{ display:grid; gap:16px; grid-template-columns:1fr; }}
    .mini-table h3 {{ margin:0 0 10px; font-size:1rem; }}
    .table-wrap {{ overflow-x:auto; }}
    h1,h2,h3 {{ margin:0 0 12px; }}
    p {{ color:var(--muted); line-height:1.6; }}
    .small {{ font-size:0.94rem; }}
    table {{ width:100%; border-collapse:collapse; font-size:0.96rem; }}
    th,td {{ padding:12px 10px; border-bottom:1px solid var(--border); text-align:left; vertical-align:top; }}
    th {{ font-size:0.8rem; text-transform:uppercase; letter-spacing:0.04em; color:var(--muted); }}
    @media (min-width: 900px) {{ .two-up {{ grid-template-columns:1fr 1fr; }} .mini-grid {{ grid-template-columns:1fr 1fr; }} }}
  </style>
</head>
<body>
  <main class="wrap">
    {nav_html()}
    <h1>Natural Gas Market and Portfolio Analysis</h1>
    <p>This page separates the market layer from the storage analysis by focusing on stocks, ETFs, natural gas futures, optimized portfolio behavior, and stock-level sentiment.</p>
    <p class="small">Report refreshed at {refreshed}. Market data source: Yahoo Finance.</p>

    <section class="panel">
      <h2>Normalized prices</h2>
      <p>Selected U.S. natural gas-linked equities, ETFs, and front-month natural gas futures normalized to a common starting point.</p>
      {normalized_prices_chart(close)}
    </section>

    <section class="two-up">
      <div class="panel">{correlation_chart(close)}</div>
      <div class="panel">{portfolio_chart(close)}</div>
    </section>

    {portfolio_summary_html(close)}
    {sentiment_section_html(sentiment_df, recent_sentiment_df)}

    <section class="panel">
      <h2>Calendar monthly returns</h2>
      <p>This section includes all tracked stocks, ETFs, and natural gas futures.</p>
      {monthly_tables_html(close)}
    </section>
  </main>
</body>
</html>"""


def main() -> int:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    close = fetch_market_prices(DEFAULT_TICKERS, start="2019-01-01")
    sentiment_df = load_market_sentiment()
    recent_sentiment_df = load_recent_market_sentiment()
    REPORT_PATH.write_text(page_html(close, sentiment_df, recent_sentiment_df), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
