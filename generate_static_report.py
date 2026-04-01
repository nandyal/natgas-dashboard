from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard_data import (
    inventory_decomposition,
    load_inventory_data,
    seasonal_inventory_profile,
    summarize_inventory,
)


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
REPORT_PATH = DOCS_DIR / "index.html"
LATEST_JSON = BASE_DIR / "weekly_natural_gas_inventory_2026-03-20.json"


def latest_release_payload() -> dict:
    return json.loads(LATEST_JSON.read_text(encoding="utf-8-sig"))


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
    clean = pd.DataFrame(
        {
            "period": decomp.observed.index,
            "observed": decomp.observed.values,
            "trend": decomp.trend.values,
            "seasonal": decomp.seasonal.values,
            "random": decomp.resid.values,
        }
    ).dropna()

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Observed", "Trend", "Seasonal", "Random"),
    )
    for idx, column in enumerate(["observed", "trend", "seasonal", "random"], start=1):
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
        height=720,
        margin=dict(l=20, r=20, t=70, b=20),
    )
    fig.update_xaxes(title_text="Year", row=4, col=1)
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


def html_page(df: pd.DataFrame, release: dict) -> str:
    summary = summarize_inventory(df)
    refreshed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Natural Gas Weekly Energy Watcher</title>
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
    <h1>Natural Gas Weekly Energy Watcher</h1>
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
    </section>

    <section class="panel">
      <h2>Regional storage table</h2>
      <p>This table summarizes the latest weekly storage release by reporting region. All storage values are shown in billion cubic feet.</p>
      {regional_table(release)}
    </section>
  </main>
</body>
</html>
"""


def main() -> int:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_inventory_data(BASE_DIR)
    release = latest_release_payload()
    REPORT_PATH.write_text(html_page(df, release), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
