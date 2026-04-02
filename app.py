from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard_data import (
    PORTFOLIO_TICKERS,
    DEFAULT_TICKERS,
    available_tickers,
    build_optimized_portfolio,
    calendar_return_table,
    correlation_matrix,
    fetch_market_prices,
    inventory_decomposition,
    latest_inventory_vs_history,
    load_inventory_data,
    monthly_returns,
    normalized_prices,
    seasonal_inventory_profile,
    summarize_inventory,
)


BASE_DIR = Path(__file__).resolve().parent


st.set_page_config(page_title="Natural Gas Dashboard", layout="wide")


@st.cache_data(show_spinner=False)
def get_inventory_data() -> pd.DataFrame:
    return load_inventory_data(BASE_DIR)


@st.cache_data(ttl=3600, show_spinner=False)
def get_market_data(start_date: str) -> pd.DataFrame:
    return fetch_market_prices(start=start_date)


st.title("Natural Gas Market Dashboard")
st.caption("Inventory data: U.S. EIA weekly storage series. Market data: Yahoo Finance.")

inventory = get_inventory_data()
summary = summarize_inventory(inventory)
profile = seasonal_inventory_profile(inventory)
comparison = latest_inventory_vs_history(inventory)
decomposition = inventory_decomposition(inventory)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Inventory (BCF)", f"{summary.latest_value:,.0f}", summary.latest_date)
col2.metric("Weekly Change (BCF)", f"{summary.weekly_change:,.0f}")
col3.metric("52-Week Low (BCF)", f"{summary.trailing_year_low:,.0f}")
col4.metric("52-Week High (BCF)", f"{summary.trailing_year_high:,.0f}")

st.divider()
st.subheader("Inventory")

inventory_chart = go.Figure()
inventory_chart.add_trace(
    go.Scatter(
        x=inventory["period"],
        y=inventory["value_bcf"],
        name="Inventory",
        mode="lines",
        line=dict(color="#0d6e6e", width=3),
    )
)
inventory_chart.add_trace(
    go.Scatter(
        x=inventory["period"],
        y=inventory["inventory_52w_avg"],
        name="52-week average",
        mode="lines",
        line=dict(color="#c46b00", dash="dash"),
    )
)
inventory_chart.update_layout(
    height=420,
    margin=dict(l=20, r=20, t=40, b=20),
    yaxis_title="BCF",
)
st.plotly_chart(inventory_chart, width="stretch")

left, right = st.columns(2)

with left:
    season_chart = go.Figure()
    season_chart.add_trace(
        go.Scatter(
            x=profile["week_of_year"],
            y=profile["max_bcf"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    season_chart.add_trace(
        go.Scatter(
            x=profile["week_of_year"],
            y=profile["min_bcf"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(13,110,110,0.18)",
            line=dict(width=0),
            name="Historical range",
        )
    )
    season_chart.add_trace(
        go.Scatter(
            x=profile["week_of_year"],
            y=profile["avg_bcf"],
            mode="lines",
            line=dict(color="#0d6e6e", width=3),
            name="10-year average",
        )
    )
    season_chart.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Week of year", yaxis_title="BCF")
    st.plotly_chart(season_chart, width="stretch")

with right:
    compare_chart = px.line(
        comparison,
        x="week_of_year",
        y="value_bcf",
        color="year",
        markers=False,
        labels={"week_of_year": "Week of year", "value_bcf": "BCF", "year": "Year"},
        title="Current year vs prior year",
    )
    compare_chart.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(compare_chart, width="stretch")

left, right = st.columns(2)

with left:
    recent_chart = go.Figure()
    recent_chart.add_trace(
        go.Scatter(
            x=inventory["period"].tail(104),
            y=inventory["value_bcf"].tail(104),
            name="Inventory",
            mode="lines",
            line=dict(color="#0d6e6e", width=3),
        )
    )
    recent_chart.update_layout(
        title="Recent two-year inventory trend",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis_title="BCF",
    )
    st.plotly_chart(recent_chart, width="stretch")

with right:
    decomp_df = pd.DataFrame(
        {
            "period": decomposition.observed.index,
            "observed": decomposition.observed.values,
            "trend": decomposition.trend.values,
            "seasonal": decomposition.seasonal.values,
            "residual": decomposition.resid.values,
        }
    )
    decomp_clean = decomp_df.dropna().copy()
    decomp_chart = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("Observed", "Trend", "Seasonal", "Random"),
    )
    for idx, column in enumerate(["observed", "trend", "seasonal", "residual"], start=1):
        decomp_chart.add_trace(
            go.Scatter(
                x=decomp_clean["period"],
                y=decomp_clean[column],
                mode="lines",
                line=dict(color="#0d6e6e", width=2),
                showlegend=False,
            ),
            row=idx,
            col=1,
        )
    decomp_chart.update_yaxes(title_text="BCF", row=1, col=1)
    decomp_chart.update_yaxes(title_text="BCF", row=2, col=1)
    decomp_chart.update_yaxes(title_text="BCF", row=3, col=1)
    decomp_chart.update_yaxes(title_text="BCF", row=4, col=1)
    decomp_chart.update_xaxes(title_text="Year", row=4, col=1)
    decomp_chart.update_layout(
        title="Decomposition of Natural Gas Inventory",
        height=560,
        margin=dict(l=20, r=20, t=70, b=20),
    )
    st.plotly_chart(decomp_chart, width="stretch")

st.divider()
st.subheader("Market and Portfolio")

market_start = st.sidebar.date_input("Market data start", value=pd.Timestamp("2019-01-01"))
market = get_market_data(pd.Timestamp(market_start).strftime("%Y-%m-%d"))
available_market, missing_market = available_tickers(market, DEFAULT_TICKERS)
available_portfolio, missing_portfolio = available_tickers(market, PORTFOLIO_TICKERS)
norm = normalized_prices(market)
monthly = monthly_returns(market)
portfolio = build_optimized_portfolio(market, PORTFOLIO_TICKERS)
correlation = correlation_matrix(market[available_market].dropna())

if missing_market:
    st.info(f"Yahoo Finance did not return current data for: {', '.join(missing_market)}")

market_tabs = st.tabs(["Normalized Prices", "Correlation", "Portfolio", "Monthly Returns"])

with market_tabs[0]:
    normalized_chart = px.line(
        norm.reset_index(),
        x="Date",
        y=norm.columns,
        labels={"value": "Indexed to 100", "variable": "Ticker"},
        title="Normalized market prices",
    )
    normalized_chart.update_layout(height=480, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(normalized_chart, width="stretch")

with market_tabs[1]:
    heatmap = px.imshow(
        correlation.round(2),
        text_auto=True,
        color_continuous_scale="RdBu",
        origin="lower",
        aspect="auto",
        title="Daily return correlation matrix",
    )
    heatmap.update_layout(height=520, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(heatmap, width="stretch")

with market_tabs[2]:
    port_chart = go.Figure()
    port_chart.add_trace(
        go.Scatter(
            x=portfolio.index.index,
            y=portfolio.index.values,
            mode="lines",
            name="Optimized portfolio",
            line=dict(color="#0d6e6e", width=3),
        )
    )
    port_chart.update_layout(
        title="Optimized portfolio performance",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis_title="Indexed to 100",
    )
    st.plotly_chart(port_chart, width="stretch")

    rolling_vol = portfolio.returns.rolling(63).std() * (252 ** 0.5)
    st.line_chart(rolling_vol.rename("63-day rolling volatility"))
    st.dataframe(
        portfolio.weights.rename("weight").mul(100).round(1).to_frame(),
        width="stretch",
    )
    st.caption(
        f"Optimized portfolio metrics: annualized return {portfolio.annual_return * 100:.1f}%, "
        f"annualized volatility {portfolio.annual_volatility * 100:.1f}%, "
        f"Sharpe ratio {portfolio.sharpe_ratio:.2f}."
    )
    if missing_portfolio:
        st.caption(f"Portfolio currently excludes unavailable tickers: {', '.join(missing_portfolio)}")

with market_tabs[3]:
    ticker = st.selectbox("Monthly return table", market.columns.tolist(), index=0)
    table = calendar_return_table(monthly[ticker])
    st.dataframe(table.style.format("{:.1f}"), width="stretch")

st.sidebar.header("Run Notes")
st.sidebar.write("1. Refresh inventory data with `python refresh_eia_ng_inventory.py`.")
st.sidebar.write("2. Launch the dashboard with `streamlit run app.py`.")
