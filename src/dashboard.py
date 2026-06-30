import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="AIgnition Forecaster",
    page_icon="📈",
    layout="wide"
)

def escape_dollar_signs(text):
    """Prevent Streamlit from interpreting $...$ as LaTeX math"""
    return text.replace("$", "\\$")

# ---------- Load Data ----------
@st.cache_data
def load_predictions():
    return pd.read_csv("output/predictions.csv")

@st.cache_data
def load_insights():
    with open("output/insights.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_features():
    return pd.read_parquet("features.parquet")


try:
    predictions = load_predictions()
    insights = load_insights()
    features = load_features()
except FileNotFoundError:
    st.error("Run `./run.sh` first to generate predictions before launching the dashboard.")
    st.stop()


# ---------- Header ----------
st.title("📈 Probabilistic Revenue Forecasting")
st.markdown("**AI-Assisted Forecasting Utility for Digital Marketing Agencies**")
st.caption(f"Based on {len(features):,} historical campaign-day records from {features['date'].min().date()} to {features['date'].max().date()}")
st.markdown("---")


# ---------- Sidebar ----------
st.sidebar.header("⚙️ Controls")
period = st.sidebar.selectbox("Forecast Window", [30, 60, 90], index=0)
channels_available = predictions[predictions["forecast_level"] == "channel"]["channel"].unique()
selected_channels = st.sidebar.multiselect("Channels", channels_available, default=list(channels_available))


# ---------- Executive Summary ----------
st.header("🎯 Executive Summary")
st.info(escape_dollar_signs(insights.get("overall", "No summary available.")))


# ---------- Top Level Metrics ----------
channel_data = predictions[
    (predictions["forecast_level"] == "channel") &
    (predictions["period_days"] == period) &
    (predictions["channel"].isin(selected_channels))
]

if channel_data.empty:
    st.warning("Please select at least one channel from the sidebar.")
    st.stop()

total_p50 = channel_data["revenue_p50"].sum()
total_p10 = channel_data["revenue_p10"].sum()
total_p90 = channel_data["revenue_p90"].sum()
avg_roas = channel_data["roas_p50"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric(f"Forecasted Revenue ({period}d)", f"${total_p50:,.0f}")
col2.metric("Pessimistic (P10)", f"${total_p10:,.0f}")
col3.metric("Optimistic (P90)", f"${total_p90:,.0f}")
col4.metric("Avg Forecasted ROAS", f"{avg_roas:.2f}x")

st.markdown("---")


# ---------- Revenue Forecast Chart ----------
# ---------- Consistent Color Map (used across all charts) ----------
CHANNEL_COLORS = {
    "google": "#4285F4",   # Google Blue (official Google brand color)
    "meta": "#FF6B9D",     # Meta-inspired pink/purple gradient tone (distinct from Google blue)
    "bing": "#00897B"      # Bing-inspired teal/green (distinct from both)
}
# ---------- Revenue Forecast Chart ----------
# ---------- Revenue Forecast Chart ----------
st.header("💰 Revenue Forecast by Channel")

fig = go.Figure()
for channel in selected_channels:
    row = channel_data[channel_data["channel"] == channel]
    if row.empty:
        continue
    fig.add_trace(go.Bar(
        name=channel.capitalize(),
        x=[channel.capitalize()],
        y=[row["revenue_p50"].values[0]],
        marker=dict(
            color=CHANNEL_COLORS.get(channel, "#888888"),
            line=dict(color="rgba(255,255,255,0.15)", width=1),
            cornerradius=8
        ),
        error_y=dict(
            type="data",
            symmetric=False,
            array=[row["revenue_p90"].values[0] - row["revenue_p50"].values[0]],
            arrayminus=[row["revenue_p50"].values[0] - row["revenue_p10"].values[0]],
            color="rgba(255,255,255,0.4)",
            thickness=1.5,
            width=4
        ),
        width=0.5
    ))

fig.update_layout(
    yaxis_title="Forecasted Revenue ($)",
    showlegend=True,
    height=420,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Arial, sans-serif", size=13, color="#E0E0E0"),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1
    ),
    xaxis=dict(
        showgrid=False,
        zeroline=False
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False
    ),
    bargap=0.4,
    margin=dict(t=20, b=20, l=20, r=20)
)
st.plotly_chart(fig, use_container_width=True)


# ---------- ROAS Comparison ----------
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📊 ROAS by Channel")
    roas_fig = px.bar(
        channel_data,
        x="channel",
        y="roas_p50",
        color="channel",
        color_discrete_map=CHANNEL_COLORS,
        labels={"roas_p50": "Forecasted ROAS", "channel": "Channel"}
    )
    roas_fig.update_traces(
        marker=dict(
            line=dict(color="rgba(255,255,255,0.15)", width=1),
            cornerradius=8
        ),
        width=0.5
    )
    roas_fig.update_layout(
        showlegend=False,
        height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", size=13, color="#E0E0E0"),
        xaxis=dict(title="Channel", showgrid=False, zeroline=False),
        yaxis=dict(
            title="Forecasted ROAS",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False
        ),
        bargap=0.4,
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(roas_fig, use_container_width=True)

with col_b:
    st.subheader("🥧 Revenue Share by Channel")
    pie_fig = px.pie(
        channel_data,
        names="channel",
        values="revenue_p50",
        hole=0.55,
        color="channel",
        color_discrete_map=CHANNEL_COLORS
    )
    pie_fig.update_traces(
        marker=dict(line=dict(color="#0E1117", width=2)),
        textfont=dict(size=13, color="white"),
        textposition="outside",
        pull=[0.02, 0.02, 0.02]
    )
    pie_fig.update_layout(
        height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", size=13, color="#E0E0E0"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1
        ),
        margin=dict(t=20, b=20, l=20, r=80)
    )
    st.plotly_chart(pie_fig, use_container_width=True)
# ---------- Campaign Type Breakdown ----------
st.header("🔍 Campaign Type Breakdown")

camptype_data = predictions[
    (predictions["forecast_level"] == "campaign_type") &
    (predictions["period_days"] == period) &
    (predictions["channel"].isin(selected_channels))
]

camptype_fig = px.bar(
    camptype_data.sort_values("revenue_p50", ascending=True),
    x="revenue_p50",
    y="campaign_type",
    color="channel",
    color_discrete_map=CHANNEL_COLORS,
    orientation="h",
    labels={"revenue_p50": "Forecasted Revenue ($)", "campaign_type": "Campaign Type"}
)
camptype_fig.update_traces(
    marker=dict(
        line=dict(color="rgba(255,255,255,0.15)", width=1),
        cornerradius=6
    )
)
camptype_fig.update_layout(
    height=480,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Arial, sans-serif", size=13, color="#E0E0E0"),
    xaxis=dict(
        title="Forecasted Revenue ($)",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False
    ),
    yaxis=dict(
        title="Campaign Type",
        showgrid=False,
        zeroline=False
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1
    ),
    bargap=0.3,
    margin=dict(t=20, b=20, l=20, r=20)
)
st.plotly_chart(camptype_fig, use_container_width=True)


# ---------- Campaign-Level Breakdown ----------
st.header("📋 Campaign-Level Forecast")

campaign_data_all = predictions[
    (predictions["forecast_level"] == "campaign") &
    (predictions["period_days"] == period) &
    (predictions["channel"].isin(selected_channels))
].copy()

if not campaign_data_all.empty:
    camp_col1, camp_col2 = st.columns([1, 3])
    with camp_col1:
        sort_by = st.selectbox("Sort by", ["revenue_p50", "roas_p50", "revenue_p90"], key="camp_sort")
    with camp_col2:
        show_top = st.slider("Show top N campaigns", 5, len(campaign_data_all), min(20, len(campaign_data_all)), key="camp_top")

    campaign_display = (
        campaign_data_all
        .sort_values(sort_by, ascending=False)
        .head(show_top)
        .reset_index(drop=True)
    )

    camp_fig = px.bar(
        campaign_display.sort_values("revenue_p50", ascending=True),
        x="revenue_p50",
        y="campaign_name",
        color="channel",
        color_discrete_map=CHANNEL_COLORS,
        orientation="h",
        labels={"revenue_p50": "Forecasted Revenue ($)", "campaign_name": "Campaign"},
        hover_data={"roas_p50": ":.2f", "revenue_p10": True, "revenue_p90": True}
    )
    camp_fig.update_traces(
        marker=dict(line=dict(color="rgba(255,255,255,0.15)", width=1), cornerradius=5)
    )
    camp_fig.update_layout(
        height=max(400, show_top * 28),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", size=12, color="#E0E0E0"),
        xaxis=dict(title="Forecasted Revenue ($)", showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False),
        yaxis=dict(title="", showgrid=False, zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        bargap=0.25,
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(camp_fig, use_container_width=True)

    with st.expander("View campaign forecast table"):
        table_cols = ["channel", "campaign_type", "campaign_name", "revenue_p10", "revenue_p50", "revenue_p90", "roas_p50"]
        st.dataframe(
            campaign_data_all[table_cols].sort_values("revenue_p50", ascending=False).reset_index(drop=True),
            use_container_width=True
        )
else:
    st.info("No campaign-level forecasts available for selected channels.")

st.markdown("---")


# ---------- Budget Simulator ----------
st.header("💡 Budget Simulation")
sim_channel = st.selectbox("Select channel to simulate", channels_available)
st.caption("💡 Tip: check the matching tab in 'AI-Generated Channel Insights' below for context on this channel.")
budget_change = st.slider("Budget change (%)", -50, 100, 0, step=10)

multiplier = 1 + (budget_change / 100)
base_row = channel_data[channel_data["channel"] == sim_channel]

if not base_row.empty:
    base_revenue = base_row["revenue_p50"].values[0]
    scaled_revenue = base_revenue * (multiplier ** 0.5)  # diminishing returns
    delta_val = scaled_revenue - base_revenue

    sim_col1, sim_col2 = st.columns(2)
    sim_col1.metric(f"Current Forecasted Revenue ({period}d)", f"${base_revenue:,.0f}")

    if budget_change == 0:
        sim_col2.metric(
        f"Simulated Revenue ({period}d, {budget_change:+d}% budget)",
        f"${scaled_revenue:,.0f}",
        delta=None if budget_change == 0 else f"{delta_val:,.0f}"
    )
    else:
        sim_col2.metric(
            f"Simulated Revenue ({budget_change:+d}% budget)",
            f"${scaled_revenue:,.0f}",
            delta=f"{delta_val:,.0f}"
        )

st.markdown("---")  # Visual separator before AI insights


# ---------- Channel Deep Dive (AI Insights) ----------
st.header("🤖 AI-Generated Channel Insights")
def escape_dollar_signs(text):
    """Prevent Streamlit from interpreting $...$ as LaTeX math"""
    return text.replace("$", "\\$")

tabs = st.tabs([ch.capitalize() for ch in channels_available])
for tab, channel in zip(tabs, channels_available):
    with tab:
        raw_text = insights.get(channel, "No insight available for this channel.")
        st.markdown(escape_dollar_signs(raw_text))


# ---------- Raw Data ----------
with st.expander("📋 View Raw Predictions Data"):
    st.dataframe(predictions, use_container_width=True)

st.markdown("---")
st.caption("Built for NetElixir AIgnition 3.0 Hackathon | Probabilistic Revenue Forecasting")