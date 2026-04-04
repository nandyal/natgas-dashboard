from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard_data import (
    inventory_decomposition,
    latest_inventory_vs_history,
    load_inventory_data,
    seasonal_inventory_profile,
    split_residual_components,
    summarize_inventory,
)


BASE_DIR = Path(__file__).resolve().parent


st.set_page_config(page_title="Natural Gas Dashboard", layout="wide")


@st.cache_data(show_spinner=False)
def get_inventory_data() -> pd.DataFrame:
    return load_inventory_data(BASE_DIR)


st.title("Natural Gas Inventory Dashboard")
st.caption("Inventory data: U.S. EIA weekly storage series.")

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
    structured_residual, noise_residual = split_residual_components(
        pd.Series(decomposition.resid, index=decomposition.observed.index)
    )
    decomp_df = pd.DataFrame(
        {
            "period": decomposition.observed.index,
            "observed": decomposition.observed.values,
            "trend": decomposition.trend.values,
            "seasonal": decomposition.seasonal.values,
            "structured_residual": structured_residual.reindex(decomposition.observed.index).values,
            "noise": noise_residual.reindex(decomposition.observed.index).values,
        }
    )
    decomp_clean = decomp_df.dropna().copy()
    decomp_chart = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("Observed", "Trend", "Seasonal", "Structured Residual", "Noise"),
    )
    for idx, column in enumerate(
        ["observed", "trend", "seasonal", "structured_residual", "noise"],
        start=1,
    ):
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
    decomp_chart.update_yaxes(title_text="BCF", row=5, col=1)
    decomp_chart.update_xaxes(title_text="Year", row=5, col=1)
    decomp_chart.update_layout(
        title="Decomposition of Natural Gas Inventory",
        height=700,
        margin=dict(l=20, r=20, t=70, b=20),
    )
    st.plotly_chart(decomp_chart, width="stretch")

st.sidebar.header("Run Notes")
st.sidebar.write("1. Refresh inventory data with `python refresh_eia_ng_inventory.py`.")
st.sidebar.write("2. Launch the dashboard with `streamlit run app.py`.")
st.sidebar.write("3. Launch the separate market dashboard with `streamlit run market_app.py`.")
