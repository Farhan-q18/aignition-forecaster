import os
import json
import pandas as pd
import numpy as np
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
MODEL = "gpt-4o-mini"


def get_causal_drivers(df, channel):
    ch = df[df["channel"] == channel].copy().sort_values("date")
    last_30 = ch.tail(30)
    prev_30 = ch.iloc[-60:-30] if len(ch) >= 60 else ch.head(max(1, len(ch) // 2))

    def pct(last, prev):
        if prev == 0:
            return None
        return round(((last - prev) / prev) * 100, 1)

    rev_last = last_30["revenue"].sum()
    rev_prev = prev_30["revenue"].sum()
    spend_last = last_30["spend"].sum()
    spend_prev = prev_30["spend"].sum()
    clicks_last = last_30["clicks"].sum()
    clicks_prev = prev_30["clicks"].sum()
    imp_last = last_30["impressions"].sum()
    imp_prev = prev_30["impressions"].sum()
    conv_last = last_30["conversions"].sum()
    conv_prev = prev_30["conversions"].sum()

    roas_last = rev_last / spend_last if spend_last > 0 else 0
    roas_prev = rev_prev / spend_prev if spend_prev > 0 else 0
    cpc_last = spend_last / clicks_last if clicks_last > 0 else 0
    cpc_prev = spend_prev / clicks_prev if clicks_prev > 0 else 0
    ctr_last = clicks_last / imp_last if imp_last > 0 else 0
    ctr_prev = clicks_prev / imp_prev if imp_prev > 0 else 0
    cvr_last = conv_last / clicks_last if clicks_last > 0 else 0
    cvr_prev = conv_prev / clicks_prev if clicks_prev > 0 else 0

    return {
        "revenue_trend_pct": pct(rev_last, rev_prev),
        "spend_trend_pct": pct(spend_last, spend_prev),
        "roas_last": round(roas_last, 2),
        "roas_prev": round(roas_prev, 2),
        "roas_trend_pct": pct(roas_last, roas_prev),
        "cpc_last": round(cpc_last, 2),
        "cpc_prev": round(cpc_prev, 2),
        "cpc_trend_pct": pct(cpc_last, cpc_prev),
        "ctr_last_pct": round(ctr_last * 100, 3),
        "ctr_trend_pct": pct(ctr_last, ctr_prev),
        "cvr_last_pct": round(cvr_last * 100, 3),
        "cvr_trend_pct": pct(cvr_last, cvr_prev),
        "last_30d_revenue": round(rev_last, 2),
        "prev_30d_revenue": round(rev_prev, 2),
        "last_30d_spend": round(spend_last, 2),
    }


def get_top_campaigns(predictions_df, channel, n=3):
    rows = predictions_df[
        (predictions_df["channel"] == channel) &
        (predictions_df["forecast_level"] == "campaign") &
        (predictions_df["period_days"] == 30)
    ].sort_values("revenue_p50", ascending=False).head(n)

    result = []
    for _, row in rows.iterrows():
        result.append(
            f"  - {row['campaign_name']} ({row.get('campaign_type', '')}): "
            f"${row['revenue_p50']:,.0f} rev, {row['roas_p50']:.2f}x ROAS"
        )
    return "\n".join(result) if result else "  No campaign data available"


def get_forecast_summary(predictions_df, channel):
    rows = predictions_df[
        (predictions_df["channel"] == channel) &
        (predictions_df["forecast_level"] == "channel")
    ]
    summary = {}
    for _, row in rows.iterrows():
        p = int(row["period_days"])
        summary[f"{p}d"] = {
            "revenue_p10": row["revenue_p10"],
            "revenue_p50": row["revenue_p50"],
            "revenue_p90": row["revenue_p90"],
            "roas_p10": row["roas_p10"],
            "roas_p50": row["roas_p50"],
            "roas_p90": row["roas_p90"],
        }
    return summary


def _fmt_pct(v):
    if v is None:
        return "N/A"
    sign = "+" if v > 0 else ""
    return f"{sign}{v}%"


def generate_channel_insight(df, predictions_df, channel):
    d = get_causal_drivers(df, channel)
    fc = get_forecast_summary(predictions_df, channel)
    top_camps = get_top_campaigns(predictions_df, channel)

    prompt = f"""You are a senior digital marketing analyst at a top ecommerce agency.
Analyze the data below and write a causally-grounded channel insight.

CHANNEL: {channel.upper()}

PERFORMANCE TREND (last 30d vs prior 30d):
- Revenue: {_fmt_pct(d['revenue_trend_pct'])}  (${d['last_30d_revenue']:,} vs ${d['prev_30d_revenue']:,})
- Spend:   {_fmt_pct(d['spend_trend_pct'])}  (last 30d: ${d['last_30d_spend']:,})
- ROAS:    {d['roas_prev']}x → {d['roas_last']}x  ({_fmt_pct(d['roas_trend_pct'])})
- CPC:     ${d['cpc_prev']} → ${d['cpc_last']}  ({_fmt_pct(d['cpc_trend_pct'])})
- CTR:     {d['ctr_last_pct']}%  ({_fmt_pct(d['ctr_trend_pct'])})
- Conv Rate: {d['cvr_last_pct']}%  ({_fmt_pct(d['cvr_trend_pct'])})

TOP CAMPAIGNS — 30d revenue forecast:
{top_camps}

REVENUE FORECAST:
- 30d: ${fc.get('30d', {}).get('revenue_p50', 0):,.0f}  (P10 ${fc.get('30d', {}).get('revenue_p10', 0):,.0f} – P90 ${fc.get('30d', {}).get('revenue_p90', 0):,.0f})
- 60d: ${fc.get('60d', {}).get('revenue_p50', 0):,.0f}
- 90d: ${fc.get('90d', {}).get('revenue_p50', 0):,.0f}
- ROAS forecast range (30d): {fc.get('30d', {}).get('roas_p10', 0):.2f}x – {fc.get('30d', {}).get('roas_p90', 0):.2f}x

Write exactly 5 labeled sections. Cite specific numbers and causal chains (e.g. "CPC rose X% which compressed ROAS by Y%"). Be direct.

1. PERFORMANCE SUMMARY: What happened and the primary causal driver (2 sentences)
2. KEY DRIVERS: The 1-2 metrics most responsible for the current trajectory (2 sentences)
3. FORECAST OUTLOOK: What the P10–P90 spread signals about confidence and risk (1-2 sentences)
4. BUDGET RECOMMENDATION: Increase / hold / decrease and why (1-2 sentences)
5. RISK FLAGS: One specific metric to watch that could invalidate the forecast (1 sentence)

Max 220 words total."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=380,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Insight generation failed: {e}"


def generate_overall_insight(df, predictions_df):
    total_30d = predictions_df[
        (predictions_df["forecast_level"] == "channel") &
        (predictions_df["period_days"] == 30)
    ]["revenue_p50"].sum()

    total_60d = predictions_df[
        (predictions_df["forecast_level"] == "channel") &
        (predictions_df["period_days"] == 60)
    ]["revenue_p50"].sum()

    total_90d = predictions_df[
        (predictions_df["forecast_level"] == "channel") &
        (predictions_df["period_days"] == 90)
    ]["revenue_p50"].sum()

    total_spend = df["spend"].sum()
    total_revenue = df["revenue"].sum()
    blended_roas = round(total_revenue / total_spend, 2) if total_spend > 0 else 0

    channel_lines = []
    for ch in ["google", "meta", "bing"]:
        row = predictions_df[
            (predictions_df["channel"] == ch) &
            (predictions_df["forecast_level"] == "channel") &
            (predictions_df["period_days"] == 30)
        ]
        if not row.empty:
            rev = row["revenue_p50"].values[0]
            roas = row["roas_p50"].values[0]
            share = rev / total_30d * 100 if total_30d > 0 else 0
            channel_lines.append(
                f"  - {ch.capitalize()}: ${rev:,.0f} ({share:.0f}% share, {roas:.2f}x ROAS)"
            )

    prompt = f"""You are a senior ecommerce marketing strategist.
Write an executive summary of the blended paid-media forecast.

HISTORICAL BLENDED PERFORMANCE:
- ROAS: {blended_roas}x
- Total Revenue: ${total_revenue:,.0f}
- Total Spend: ${total_spend:,.0f}

BLENDED REVENUE FORECAST:
- Next 30 days: ${total_30d:,.0f}
- Next 60 days: ${total_60d:,.0f}
- Next 90 days: ${total_90d:,.0f}

CHANNEL BREAKDOWN (30d forecast):
{chr(10).join(channel_lines)}

Write exactly 4 sentences:
1. Overall revenue outlook — state the 30/60/90d numbers and the implied run-rate
2. Biggest opportunity channel — cite its ROAS and revenue share
3. Biggest risk — name a specific metric or channel
4. One concrete budget reallocation recommendation

Max 120 words. Be direct and data-driven."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=220,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Overall insight generation failed: {e}"


def run_insights(df, predictions_df):
    print("\nGenerating AI insights...")
    insights = {}

    print("  >> Overall executive summary...")
    insights["overall"] = generate_overall_insight(df, predictions_df)

    for channel in ["google", "meta", "bing"]:
        print(f"  >> {channel}...")
        insights[channel] = generate_channel_insight(df, predictions_df, channel)

    return insights


def save_insights(insights, output_path="output/insights.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(insights, f, indent=2)
    print(f"Insights saved to {output_path}")


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    if not OPENAI_API_KEY:
        print("No OPENAI_API_KEY found in environment. Skipping AI insights.")
        sys.exit(0)

    print("Loading data...")
    df = pd.read_parquet("features.parquet")
    predictions_df = pd.read_csv("output/predictions.csv")

    insights = run_insights(df, predictions_df)
    save_insights(insights)
