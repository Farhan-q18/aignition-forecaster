"""
AIgnition Forecaster — demo dashboard (Streamlit).

Sits on top of the pipeline outputs (output/*.csv|json + pickle/model.pkl).
Run the pipeline first:  bash run.sh
Optional AI layers:      python src/anomalies.py && python src/llm_insights.py
Launch:                  streamlit run src/dashboard.py
"""

import json
import os
import pickle
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from response_curves import curve_points, scale_factors

st.set_page_config(page_title="AIgnition Forecaster", page_icon="◆", layout="wide")

# ---------------- Theme ----------------

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

THEMES = {
    "dark": {
        "bg": "#0C0E12", "panel": "#14171E", "text": "#E6E8EC",
        "muted": "#8A8F98", "accent": "#E8A33D", "grid": "rgba(255,255,255,0.07)",
        "band": "rgba(232,163,61,0.18)", "median": "#E8A33D",
        "history": "#5B8DEF", "good": "#3FB68B", "bad": "#E05D5D",
    },
    "light": {
        "bg": "#FAFAF7", "panel": "#FFFFFF", "text": "#1A1D23",
        "muted": "#6B7280", "accent": "#B4690E", "grid": "rgba(0,0,0,0.08)",
        "band": "rgba(180,105,14,0.15)", "median": "#B4690E",
        "history": "#2456C4", "good": "#12805C", "bad": "#C03434",
    },
}
T = THEMES[st.session_state.theme]

CHANNEL_COLORS = {"google": "#5B8DEF", "meta": "#C77DDB", "bing": "#3FB68B"}

st.markdown(f"""<style>
.stApp {{ background-color: {T['bg']}; color: {T['text']}; }}
[data-testid="stSidebar"] {{ background-color: {T['panel']}; }}
[data-testid="stMetric"] {{ background-color: {T['panel']}; padding: 14px 16px;
    border-radius: 10px; border: 1px solid {T['grid']}; }}
[data-testid="stMetricLabel"] {{ color: {T['muted']}; }}
h1, h2, h3, h4, p, li, span, label {{ color: {T['text']}; }}
.small-muted {{ color: {T['muted']}; font-size: 0.85rem; }}
</style>""", unsafe_allow_html=True)


def plotly_layout(fig, height=380, **kwargs):
    fig.update_layout(
        height=height,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif", size=13, color=T["text"]),
        xaxis=dict(showgrid=False, zeroline=False, color=T["muted"]),
        yaxis=dict(showgrid=True, gridcolor=T["grid"], zeroline=False, color=T["muted"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=30, b=20, l=10, r=10),
        **kwargs,
    )
    return fig


def esc(text):
    return str(text).replace("$", "\\$")


# ---------------- Data loading ----------------

@st.cache_data
def load_csv(path):
    return pd.read_csv(path) if os.path.isfile(path) else None

@st.cache_data
def load_json(path):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data
def load_features():
    return pd.read_parquet("features.parquet") if os.path.isfile("features.parquet") else None

@st.cache_resource
def load_model():
    if os.path.isfile("pickle/model.pkl"):
        with open("pickle/model.pkl", "rb") as f:
            return pickle.load(f)
    return None


predictions = load_csv("output/predictions.csv")
paths = load_csv("output/forecast_paths.csv")
scorecard = load_csv("output/backtest_scorecard.csv")
insights = load_json("output/insights.json") or {}
anomalies = load_json("output/anomalies.json") or {}
health = load_json("output/data_health.json") or {}
features = load_features()
model = load_model()

if predictions is None or features is None:
    st.error("Run `bash run.sh` first to generate predictions.")
    st.stop()

CHANNELS = sorted(predictions[predictions["forecast_level"] == "channel"]["channel"].unique())


def insight_for_channel(ch):
    if "channels" in insights:
        return insights["channels"].get(ch)
    return insights.get(ch)  # legacy schema


# ---------------- Sidebar ----------------

with st.sidebar:
    st.markdown("## ◆ AIgnition Forecaster")
    st.markdown('<p class="small-muted">Probabilistic revenue forecasting for paid media</p>',
                unsafe_allow_html=True)
    page = st.radio("Navigate", [
        "Overview",
        "Drill-Down",
        "Budget Simulator",
        "Accuracy Scorecard",
        "AI Insights & Risks",
        "Data Health & Methodology",
    ], label_visibility="collapsed")
    st.divider()
    theme_choice = st.toggle("Light mode", value=(st.session_state.theme == "light"))
    if theme_choice != (st.session_state.theme == "light"):
        st.session_state.theme = "light" if theme_choice else "dark"
        st.rerun()
    st.divider()
    period = st.select_slider("Forecast window (days)", [30, 60, 90], value=30)
    st.markdown(f'<p class="small-muted">History through {features["date"].max().date()}</p>',
                unsafe_allow_html=True)


def band_chart(level, channel=None, campaign_type=None, height=420, title=None):
    """Uncertainty-band chart: history line + shaded P10-P90 + median."""
    if paths is None:
        st.info("forecast_paths.csv not found — re-run the pipeline.")
        return
    sel = paths[paths["forecast_level"] == level]
    sel = sel[sel["channel"] == channel] if channel else sel[sel["channel"].isna()]
    if campaign_type:
        sel = sel[sel["campaign_type"] == campaign_type]
    hist = sel[sel["kind"] == "history"]
    fc = sel[sel["kind"] == "forecast"]
    if fc.empty:
        st.info("No forecast path available for this selection.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["week_end"], y=hist["p50"], mode="lines",
                             name="Actual (weekly)", line=dict(color=T["history"], width=2)))
    # bridge: connect history to forecast
    bridge_x = [hist["week_end"].iloc[-1]] + list(fc["week_end"])
    fig.add_trace(go.Scatter(x=bridge_x, y=[hist["p50"].iloc[-1]] + list(fc["p90"]),
                             mode="lines", line=dict(width=0), showlegend=False,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=bridge_x, y=[hist["p50"].iloc[-1]] + list(fc["p10"]),
                             mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=T["band"], name="P10–P90 range"))
    fig.add_trace(go.Scatter(x=bridge_x, y=[hist["p50"].iloc[-1]] + list(fc["p50"]),
                             mode="lines", name="Median forecast (P50)",
                             line=dict(color=T["median"], width=2.5, dash="dash")))
    plotly_layout(fig, height=height,
                  title=dict(text=title or "", font=dict(size=15, color=T["muted"])))
    st.plotly_chart(fig, use_container_width=True)


def get_rows(level, period_days, **filters):
    sel = predictions[(predictions["forecast_level"] == level) &
                      (predictions["period_days"] == period_days)]
    for col, val in filters.items():
        sel = sel[sel[col] == val]
    return sel


# ================= PAGE: Overview =================

if page == "Overview":
    st.title("Where is my revenue heading?")
    st.markdown(f'<p class="small-muted">Blended ecommerce revenue across Google, Meta and Bing '
                f'— forecast as a probability range, not a single guess.</p>',
                unsafe_allow_html=True)

    bl = get_rows("blended", period)
    if not bl.empty:
        r = bl.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Expected revenue · next {period}d", f"${r['revenue_p50']:,.0f}")
        c2.metric("If things go badly (P10)", f"${r['revenue_p10']:,.0f}")
        c3.metric("If things go well (P90)", f"${r['revenue_p90']:,.0f}")
        c4.metric("Blended ROAS (P50)", f"{r['roas_p50']:.2f}x")

    # What changed since last period
    end = features["date"].max()
    last30 = features[features["date"] > end - pd.Timedelta(days=30)]
    prev30 = features[(features["date"] <= end - pd.Timedelta(days=30)) &
                      (features["date"] > end - pd.Timedelta(days=60))]
    rev_delta = (last30["revenue"].sum() - prev30["revenue"].sum()) / max(prev30["revenue"].sum(), 1)
    spend_delta = (last30["spend"].sum() - prev30["spend"].sum()) / max(prev30["spend"].sum(), 1)
    st.markdown(f"**What changed:** last 30 days brought **${last30['revenue'].sum():,.0f}** revenue "
                f"({rev_delta:+.0%} vs the prior 30) on **${last30['spend'].sum():,.0f}** spend "
                f"({spend_delta:+.0%}).")

    if insights.get("overall"):
        st.info(esc(insights["overall"]))

    st.subheader("Blended forecast with uncertainty")
    band_chart("blended", title="Weekly revenue — actuals, then the P10–P90 forecast band")

    st.subheader("Which channel is carrying the number?")
    ch_rows = get_rows("channel", period)
    col_a, col_b = st.columns([3, 2])
    with col_a:
        fig = go.Figure()
        for _, row in ch_rows.iterrows():
            fig.add_trace(go.Bar(
                name=row["channel"].capitalize(), x=[row["channel"].capitalize()],
                y=[row["revenue_p50"]],
                marker=dict(color=CHANNEL_COLORS.get(row["channel"], "#888"), cornerradius=8),
                error_y=dict(type="data", symmetric=False,
                             array=[row["revenue_p90"] - row["revenue_p50"]],
                             arrayminus=[row["revenue_p50"] - row["revenue_p10"]],
                             color=T["muted"], thickness=1.5, width=5),
                width=0.55))
        plotly_layout(fig, height=360, showlegend=False,
                      yaxis_title=f"{period}d revenue forecast ($)")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = go.Figure(go.Pie(
            labels=[c.capitalize() for c in ch_rows["channel"]],
            values=ch_rows["revenue_p50"], hole=0.6,
            marker=dict(colors=[CHANNEL_COLORS.get(c, "#888") for c in ch_rows["channel"]],
                        line=dict(color=T["bg"], width=2))))
        plotly_layout(fig, height=360)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(ch_rows[["channel", "revenue_p10", "revenue_p50", "revenue_p90",
                          "roas_p10", "roas_p50", "roas_p90"]]
                 .rename(columns={"revenue_p10": "rev P10", "revenue_p50": "rev P50",
                                  "revenue_p90": "rev P90"}),
                 use_container_width=True, hide_index=True)

# ================= PAGE: Drill-Down =================

elif page == "Drill-Down":
    st.title("Which campaigns are worth scaling — and which should I pause?")

    ch = st.selectbox("Channel", CHANNELS, format_func=str.capitalize)
    band_chart("channel", channel=ch, height=340,
               title=f"{ch.capitalize()} — weekly revenue and forecast band")

    st.subheader("Campaign types")
    ct_rows = get_rows("campaign_type", period, channel=ch).sort_values("revenue_p50")
    if not ct_rows.empty:
        fig = go.Figure(go.Bar(
            x=ct_rows["revenue_p50"], y=ct_rows["campaign_type"], orientation="h",
            marker=dict(color=CHANNEL_COLORS.get(ch, "#888"), cornerradius=6),
            error_x=dict(type="data", symmetric=False,
                         array=ct_rows["revenue_p90"] - ct_rows["revenue_p50"],
                         arrayminus=ct_rows["revenue_p50"] - ct_rows["revenue_p10"],
                         color=T["muted"], thickness=1.5)))
        plotly_layout(fig, height=max(260, 60 * len(ct_rows)),
                      xaxis_title=f"{period}d revenue forecast ($)")
        st.plotly_chart(fig, use_container_width=True)

        pick_ct = st.selectbox("Inspect a campaign type's forecast band",
                               ["(none)"] + list(ct_rows["campaign_type"]))
        if pick_ct != "(none)":
            band_chart("campaign_type", channel=ch, campaign_type=pick_ct, height=320)

    st.subheader("Campaigns")
    camp = get_rows("campaign", period, channel=ch).copy()
    if camp.empty:
        st.info("No campaign-level forecasts for this channel (insufficient history).")
    else:
        camp["uncertainty"] = (camp["revenue_p90"] - camp["revenue_p10"]).clip(lower=0)
        sort_by = st.radio("Rank by", ["revenue_p50", "roas_p50"], horizontal=True,
                           format_func=lambda c: {"revenue_p50": "Forecast revenue",
                                                  "roas_p50": "Forecast ROAS"}[c])
        show = camp.sort_values(sort_by, ascending=False)
        st.dataframe(
            show[["campaign_name", "campaign_type", "revenue_p10", "revenue_p50",
                  "revenue_p90", "roas_p50"]].reset_index(drop=True),
            use_container_width=True, hide_index=True, height=420)
        scale = show[show["roas_p50"] >= show["roas_p50"].quantile(0.75)].head(5)
        pause = show[(show["roas_p50"] <= show["roas_p50"].quantile(0.25)) &
                     (show["revenue_p50"] < show["revenue_p50"].median())].tail(5)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Scale candidates** (top-quartile forecast ROAS)")
            for _, r in scale.iterrows():
                st.markdown(f"- `{r['campaign_name']}` — {r['roas_p50']:.2f}x, "
                            f"\\${r['revenue_p50']:,.0f} forecast")
        with col2:
            st.markdown("**Review/pause candidates** (bottom-quartile ROAS, low revenue)")
            for _, r in pause.iterrows():
                st.markdown(f"- `{r['campaign_name']}` — {r['roas_p50']:.2f}x, "
                            f"\\${r['revenue_p50']:,.0f} forecast")

# ================= PAGE: Budget Simulator =================

elif page == "Budget Simulator":
    st.title("I have budget left this quarter — where should it go?")
    st.markdown('<p class="small-muted">Moves each channel\'s planned spend through its fitted '
                'saturation curve (not a naive linear scale). Bands show curve uncertainty '
                'compounded with forecast uncertainty.</p>', unsafe_allow_html=True)

    if model is None:
        st.error("pickle/model.pkl not found — run `python src/train.py` first.")
        st.stop()
    curves = model.get("response_curves", {})

    ch_rows = get_rows("channel", period).set_index("channel")
    st.subheader("Planned budget change per channel")
    cols = st.columns(len(CHANNELS))
    multipliers = {}
    for col, ch in zip(cols, CHANNELS):
        with col:
            pct = st.slider(f"{ch.capitalize()}", -50, 100, 0, 5, format="%+d%%")
            multipliers[ch] = 1 + pct / 100

    base_total, sim_total = {"p10": 0, "p50": 0, "p90": 0}, {"p10": 0, "p50": 0, "p90": 0}
    rows_out = []
    for ch in CHANNELS:
        if ch not in ch_rows.index:
            continue
        base = ch_rows.loc[ch]
        s10, s50, s90 = scale_factors(curves.get(f"channel::{ch}"), multipliers[ch])
        sim = {"p10": base["revenue_p10"] * s10, "p50": base["revenue_p50"] * s50,
               "p90": base["revenue_p90"] * s90}
        for q in base_total:
            base_total[q] += base[f"revenue_{q}"]
            sim_total[q] += sim[q]
        marginal_roas = None
        if multipliers[ch] != 1.0:
            spend0 = base["revenue_p50"] / base["roas_p50"] if base["roas_p50"] else 0
            extra_spend = spend0 * (multipliers[ch] - 1)
            extra_rev = sim["p50"] - base["revenue_p50"]
            marginal_roas = extra_rev / extra_spend if extra_spend else None
        rows_out.append({"channel": ch, "budget": f"{multipliers[ch]:+.0%}"[1:],
                         "base P50": base["revenue_p50"], "simulated P50": sim["p50"],
                         "delta": sim["p50"] - base["revenue_p50"],
                         "marginal ROAS": marginal_roas})

    c1, c2, c3 = st.columns(3)
    delta = sim_total["p50"] - base_total["p50"]
    c1.metric(f"Baseline {period}d revenue (P50)", f"${base_total['p50']:,.0f}")
    c2.metric("Simulated (P50)", f"${sim_total['p50']:,.0f}", delta=f"{delta:+,.0f}")
    c3.metric("Simulated range (P10–P90)",
              f"${sim_total['p10']:,.0f} – ${sim_total['p90']:,.0f}")
    st.markdown('<p class="small-muted">Channel bands are summed for display; the pipeline\'s '
                'blended forecast simulates channels jointly.</p>', unsafe_allow_html=True)

    df_out = pd.DataFrame(rows_out)
    df_out["marginal ROAS"] = df_out["marginal ROAS"].map(
        lambda v: f"{v:.2f}x" if v is not None and pd.notna(v) else "—")
    for col in ["base P50", "simulated P50", "delta"]:
        df_out[col] = df_out[col].map(lambda v: f"${v:,.0f}")
    st.dataframe(df_out, use_container_width=True, hide_index=True)

    st.subheader("Why more budget ≠ proportionally more revenue")
    pick = st.selectbox("Channel response curve", CHANNELS, format_func=str.capitalize)
    cp = curve_points(curves.get(f"channel::{pick}"))
    if cp is None:
        st.info("No fitted curve for this channel.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cp["multiplier"], y=cp["scale_p90"], mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=cp["multiplier"], y=cp["scale_p10"], mode="lines",
                                 line=dict(width=0), fill="tonexty", fillcolor=T["band"],
                                 name="curve uncertainty"))
        fig.add_trace(go.Scatter(x=cp["multiplier"], y=cp["scale_p50"], mode="lines",
                                 name="revenue scale", line=dict(color=T["median"], width=2.5)))
        fig.add_trace(go.Scatter(x=cp["multiplier"], y=cp["multiplier"], mode="lines",
                                 name="linear (no saturation)",
                                 line=dict(color=T["muted"], width=1.5, dash="dot")))
        plotly_layout(fig, height=380, xaxis_title="budget multiplier",
                      yaxis_title="revenue multiplier")
        st.plotly_chart(fig, use_container_width=True)
        gap = curves.get(f"channel::{pick}")
        if gap:
            s = scale_factors(gap, 2.0)[1]
            verdict = ("still has headroom — extra dollars keep buying revenue"
                       if s > 1.6 else
                       "is close to saturation — extra dollars buy very little")
            st.markdown(f"**Reading it:** doubling {pick.capitalize()}'s budget is predicted to "
                        f"multiply revenue by **{s:.2f}x**, i.e. this channel {verdict}.")

# ================= PAGE: Accuracy Scorecard =================

elif page == "Accuracy Scorecard":
    st.title("Can I actually trust these numbers?")
    st.markdown('<p class="small-muted">We back-tested the model by rolling back to five '
                'historical dates, forecasting forward with only the data available then, and '
                'scoring against what actually happened.</p>', unsafe_allow_html=True)

    if scorecard is None or scorecard.empty:
        st.error("No backtest scorecard found — run `python src/train.py`.")
        st.stop()

    bl = scorecard[scorecard["series"] == "blended"]
    if not bl.empty:
        cols = st.columns(len(bl))
        for col, (_, r) in zip(cols, bl.iterrows()):
            col.metric(f"Blended · {int(r['period_days'])}d horizon",
                       f"±{r['mape']:.0%} typical error",
                       f"{r['coverage_p10_p90']:.0%} band coverage", delta_color="off")
        row60 = bl[bl["period_days"] == 60]
        if not row60.empty:
            r = row60.iloc[0]
            st.markdown(
                f"**Plain English:** on 60-day blended revenue, this model's median forecast has "
                f"historically been off by **{r['mape']:.0%}** on average, and the P10–P90 range "
                f"contained the actual outcome **{r['coverage_p10_p90']:.0%}** of the time "
                f"(target: 80%).")

    st.subheader("Per-series accuracy")
    disp = scorecard.copy()
    disp["mape"] = disp["mape"].map(lambda v: f"{v:.0%}")
    disp["smape"] = disp["smape"].map(lambda v: f"{v:.0%}")
    disp["coverage_p10_p90"] = disp["coverage_p10_p90"].map(lambda v: f"{v:.0%}")
    st.dataframe(disp.rename(columns={
        "series": "series", "period_days": "horizon (days)", "n": "backtests",
        "mape": "MAPE", "smape": "sMAPE", "coverage_p10_p90": "P10–P90 coverage"}),
        use_container_width=True, hide_index=True, height=480)

    st.markdown("""
**How to read this honestly:**
- The blended forecast is the most reliable (larger aggregates are easier to predict).
- The hardest windows straddle the holiday ramp — with only two observed holiday
  seasons, the exact spike *timing* shifts by a week or two between years.
- Uncertainty bands were **calibrated** on these backtests (per-channel residual
  scaling), which is why the blended band hits its 80% coverage target.
- Campaign-type series with heavy zero-inflation (e.g. small Bing types) have high
  percentage errors on tiny denominators — read absolute numbers there.""")

    detail = load_csv("output/backtest_detail.csv")
    if detail is not None:
        with st.expander("Every backtest, individually"):
            st.dataframe(detail, use_container_width=True, hide_index=True)

# ================= PAGE: AI Insights & Risks =================

elif page == "AI Insights & Risks":
    st.title("Why did performance move — and should I be worried?")

    if not insights:
        st.warning("No AI insights found. Generate them with "
                   "`python src/anomalies.py && python src/llm_insights.py` "
                   "(requires ANTHROPIC_API_KEY or OPENAI_API_KEY). "
                   "Anomaly detection below is fully offline.")
    else:
        prov = insights.get("provider"), insights.get("model")
        if prov[0]:
            st.markdown(f'<p class="small-muted">Generated via {prov[0]} · {prov[1]} — '
                        'grounded in statistically detected anomalies, not free association.</p>',
                        unsafe_allow_html=True)
        if insights.get("overall"):
            st.info(esc(insights["overall"]))

        tabs = st.tabs([c.capitalize() for c in CHANNELS])
        for tab, ch in zip(tabs, CHANNELS):
            with tab:
                text = insight_for_channel(ch)
                st.markdown(esc(text) if text else "_No insight generated._")

        anomaly_insights = insights.get("anomaly_insights", [])
        if anomaly_insights:
            st.subheader("Flagged anomalies, interpreted")
            for a in anomaly_insights:
                conf = a.get("confidence", "low")
                icon = {"high": "🟠", "medium": "🟡", "low": "⚪"}.get(conf, "⚪")
                where = a.get("campaign_type") or a.get("channel", "")
                title = f"{icon} {a.get('type','anomaly').replace('_',' ')} — {where}"
                if a.get("week_end"):
                    title += f" · week of {a['week_end']}"
                with st.expander(title):
                    st.markdown(f"**What happened:** {esc(a.get('summary',''))}")
                    st.markdown(f"**Likely cause:** {esc(a.get('likely_cause',''))}")
                    st.markdown(f"**Recommended action:** {esc(a.get('recommended_action',''))}")
                    stats = {k: a[k] for k in
                             ("actual", "expected", "z_score", "recent_roas",
                              "historical_roas") if k in a}
                    st.markdown(f'<p class="small-muted">Statistical basis: {stats}</p>',
                                unsafe_allow_html=True)

    st.subheader("Operational risk flags (statistical, offline)")
    caps = anomalies.get("budget_caps", [])
    drift = anomalies.get("roas_drift", [])
    outliers = anomalies.get("revenue_outliers", [])
    c1, c2, c3 = st.columns(3)
    c1.metric("Campaigns at/above budget cap", len(caps))
    c2.metric("ROAS drift alerts", len(drift))
    c3.metric("Revenue outlier weeks (26w)", len(outliers))
    if caps:
        st.markdown("**Budget caps limiting scale** — these campaigns spend at or above their "
                    "stated daily budget; raising forecasts there requires raising caps first:")
        st.dataframe(pd.DataFrame(caps).head(10), use_container_width=True, hide_index=True)
    if drift:
        st.markdown("**ROAS drifting outside its historical range:**")
        st.dataframe(pd.DataFrame(drift), use_container_width=True, hide_index=True)

# ================= PAGE: Data Health & Methodology =================

elif page == "Data Health & Methodology":
    st.title("What is this built on — and what are we assuming?")

    st.subheader("Data health by source")
    if health:
        cols = st.columns(len(health))
        for col, (ch, h) in zip(cols, sorted(health.items())):
            with col:
                st.markdown(f"**{ch.capitalize()}**")
                st.markdown(f"- {h['rows']:,} rows · {h['campaigns']} campaigns")
                st.markdown(f"- {h['date_min']} → {h['date_max']}")
                null_b = h["null_daily_budget_rows"]
                st.markdown(f"- {'⚠️ ' if null_b else ''}{null_b} null budget rows")
                st.markdown(f"- {h['zero_revenue_day_share']:.0%} zero-revenue days")
                uncls = h.get("unclassified_campaign_types", 0)
                st.markdown(f"- {'⚠️ ' if uncls else '✅ '}{uncls} unclassified campaign types")
    else:
        st.info("Run the pipeline to generate output/data_health.json.")

    st.subheader("The taxonomy problem we solved")
    st.markdown("""
Google and Bing ship clean campaign-type columns; **Meta buries the type in the campaign
name** (`Prospecting_DPA_Campaign_04`, `Remarketing_Brand_Campaign_02`, ...). Every agency
hits this: three platforms, three vocabularies, no single view. We parse Meta names with a
rule-based classifier into a normalized taxonomy (Prospecting / Remarketing / Generic /
Advantage+, with Brand/DPA sub-tags) and normalize Google/Bing casing to match. Names that
match no rule are flagged `Unclassified` for review rather than silently guessed.""")
    if features is not None:
        meta_view = (features[features["channel"] == "meta"]
                     .groupby("campaign_type")["campaign_name"].nunique()
                     .rename("campaigns").reset_index())
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Before** — raw Meta names")
            st.dataframe(features[features["channel"] == "meta"]["campaign_name"]
                         .drop_duplicates().head(8), use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**After** — normalized types")
            st.dataframe(meta_view, use_container_width=True, hide_index=True)

    st.subheader("Explicit assumptions")
    st.markdown("""
1. **Meta's `conversion` column is treated as conversion *value* (revenue), not a count.**
   Its ratio to spend (median ≈4.1x) matches Google's revenue/cost (≈2.6x) and Bing's
   Revenue/Spend — a count would imply implausible per-conversion economics. Toggle:
   `--meta-revenue-mode count --meta-aov <value>` in `generate_features.py`.
2. **Zero-inflation is handled by weekly aggregation** (85% of Bing campaign-days have zero
   revenue). Forecasting happens on weekly sums where zeros wash out; bands come from block
   bootstrap of weekly residuals, not a Gaussian on daily data.
3. **ROAS bands treat near-term spend as planned** (recent 28-day daily average × window);
   the uncertainty shown is revenue uncertainty over that planned spend.
4. **Budget simulation uses fitted saturation curves, not an MMM** — deliberately in scope
   per the brief. It answers marginal-budget questions, not cross-channel attribution.
5. **The AI layer never runs in the scored pipeline** — `run.sh` is fully offline;
   anomaly detection is statistical and offline; only insight *interpretation* calls an LLM,
   from the demo backend.""")

    st.subheader("Method, in one paragraph")
    st.markdown("""
Each series is aggregated to complete weeks, decomposed into a robust rolling-median trend ×
multiplicative week-of-year seasonal factors (estimated against an annual-window trend so the
holiday cycle isn't absorbed), extrapolated with a damped Theil-Sen slope, and Monte-Carlo
simulated 10,000× by block-bootstrapping residuals (preserving autocorrelation). 30/60/90-day
totals take P10/P50/P90 **per simulated path**, and the blended total resamples all channels
with the same time blocks to preserve cross-channel correlation. Band widths are calibrated on
walk-forward backtests to hit 80% coverage. Full details: `docs/TECHNICAL_DOCUMENTATION.md`.""")

    if model:
        with st.expander("Model artifact (pickle/model.pkl)"):
            st.json({k: v for k, v in model.items() if k != "response_curves"} |
                    {"response_curves": f"{len(model.get('response_curves', {}))} fitted curves"})

st.divider()
st.markdown('<p class="small-muted">AIgnition 3.0 · Probabilistic Revenue Forecasting · '
            'forecasts are P10–P90 Monte-Carlo bands, calibrated by walk-forward backtesting</p>',
            unsafe_allow_html=True)
