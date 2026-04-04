from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
MARKET_SENTIMENT_CSV = BASE_DIR / "market_sentiment_events.csv"
MARKET_SENTIMENT_RECENT_CSV = BASE_DIR / "market_sentiment_recent.csv"


st.set_page_config(page_title="Natural Gas Market Dashboard", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def get_market_data(start_date: str) -> pd.DataFrame:
    return fetch_market_prices(DEFAULT_TICKERS, start=start_date)


@st.cache_data(show_spinner=False)
def get_market_sentiment() -> pd.DataFrame:
    if not MARKET_SENTIMENT_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(MARKET_SENTIMENT_CSV, parse_dates=["period"])


@st.cache_data(show_spinner=False)
def get_recent_market_sentiment() -> pd.DataFrame:
    if not MARKET_SENTIMENT_RECENT_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(MARKET_SENTIMENT_RECENT_CSV, parse_dates=["as_of_date", "analysis_end_date"])


st.title("Natural Gas Market Dashboard")
st.caption("Separate market dashboard for stocks, ETFs, natural gas futures, portfolio construction, and sentiment.")

market_start = st.sidebar.date_input("Market data start", value=pd.Timestamp("2019-01-01"))
market = get_market_data(pd.Timestamp(market_start).strftime("%Y-%m-%d"))
sentiment = get_market_sentiment()
recent_sentiment = get_recent_market_sentiment()

norm = normalized_prices(market)
monthly = monthly_returns(market)
correlation = correlation_matrix(market)
portfolio = build_optimized_portfolio(market, PORTFOLIO_TICKERS)

tabs = st.tabs(["Normalized Prices", "Correlation", "Portfolio", "Sentiment", "Monthly Returns"])

with tabs[0]:
    fig = px.line(norm.reset_index(), x="Date", y=norm.columns, labels={"value": "Indexed to 100", "variable": "Ticker"})
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, width="stretch")

with tabs[1]:
    heatmap = px.imshow(correlation.round(2), text_auto=True, color_continuous_scale="RdBu", origin="lower", aspect="auto")
    heatmap.update_layout(height=540, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(heatmap, width="stretch")

with tabs[2]:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio.index.index, y=portfolio.index.values, mode="lines", line=dict(width=3, color="#0d6e6e"), name="Long-short Kelly portfolio"))
    fig.update_layout(title="Long-short Kelly portfolio performance", height=460, margin=dict(l=20, r=20, t=50, b=20), yaxis_title="Indexed to 100")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(portfolio.weights.rename("weight").mul(100).round(1).to_frame(), width="stretch")
    rebalance_table = portfolio.rebalance_weights.mul(100).round(1).copy()
    rebalance_table.index = rebalance_table.index.strftime("%Y-%m-%d")
    st.dataframe(rebalance_table, width="stretch")
    st.caption(
        f"Portfolio uses a bounded long-short Kelly Criterion allocation with an annual rebalance. "
        f"Short positions are exited after a 10% monthly loss, can flip long when RSI falls below 20, "
        f"and active long positions are sold when RSI rises above 80. "
        f"Annualized Kelly growth {portfolio.kelly_growth_rate * 100:.1f}%, "
        f"annualized return {portfolio.annual_return * 100:.1f}%, "
        f"annualized volatility {portfolio.annual_volatility * 100:.1f}%, "
        f"Sharpe ratio {portfolio.sharpe_ratio:.2f}."
    )

with tabs[3]:
    if sentiment.empty:
        st.write("Run `python market_sentiment_analysis.py` to generate stock, ETF, and futures sentiment results.")
    else:
        sentiment = sentiment.sort_values("period")
        finbert_numeric = sentiment["finbert_label"].map({"negative": -1, "neutral": 0, "positive": 1}).fillna(0)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=sentiment["period"],
                y=sentiment["monthly_return_pct"] / 4.0,
                mode="markers",
                name="1 week return (%)",
                text=sentiment["ticker"],
                marker=dict(color="#111111", size=8),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sentiment["period"],
                y=sentiment["forward_1m_return_pct"],
                mode="markers",
                name="Next complete month return (%)",
                text=sentiment["ticker"],
                marker=dict(color="#1d4ed8", size=8),
            )
        )
        fig.add_trace(go.Scatter(x=sentiment["period"], y=finbert_numeric, mode="lines+markers", name="FinBERT"))
        fig.update_layout(
            height=520,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(range=[-100, 100], title="Price change (%)"),
        )
        st.plotly_chart(fig, width="stretch")

        if not recent_sentiment.empty:
            st.dataframe(
                recent_sentiment[["ticker", "as_of_date", "analysis_end_date", "one_week_return_pct", "finbert_label", "finbert_score", "vader_compound", "one_month_return_pct"]].rename(
                    columns={
                        "as_of_date": "sentiment_anchor_date",
                        "analysis_end_date": "window_end_date",
                    }
                ),
                width="stretch",
            )

with tabs[4]:
    ticker = st.selectbox("Ticker", monthly.columns.tolist(), index=0)
    st.dataframe(calendar_return_table(monthly[ticker]).sort_index(ascending=True).style.format("{:.1f}"), width="stretch")
