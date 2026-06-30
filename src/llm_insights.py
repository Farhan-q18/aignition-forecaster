import os 
import json
import pandas as pd
import numpy as np
from openai import OpenAI

# Configure OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
client = OpenAI(api_key=OPENAI_API_KEY)


def get_channel_stats(df, channel):
    """Get historical stats for a channel"""
    channel_df = df[df["channel"] == channel].copy()

    total_revenue = channel_df["revenue"].sum()
    total_spend = channel_df["spend"].sum()
    avg_roas = round(total_revenue / total_spend, 2) if total_spend > 0 else 0
    avg_ctr = round(channel_df["ctr"].mean() * 100, 2)
    avg_cpc = round(channel_df["cpc"].mean(), 2)

    channel_df = channel_df.sort_values("date")
    last_30 = channel_df.tail(30)["revenue"].sum()
    prev_30 = channel_df.iloc[-60:-30]["revenue"].sum()
    trend = round(((last_30 - prev_30) / prev_30) * 100, 1) if prev_30 > 0 else 0

    return {
        "channel": channel,
        "total_revenue": round(total_revenue, 2),
        "total_spend": round(total_spend, 2),
        "avg_roas": avg_roas,
        "avg_ctr_pct": avg_ctr,
        "avg_cpc": avg_cpc,
        "revenue_trend_30d_pct": trend,
        "last_30d_revenue": round(last_30, 2),
        "prev_30d_revenue": round(prev_30, 2)
    }


def get_forecast_summary(predictions_df, channel):
    """Extract forecast numbers for a channel"""
    channel_preds = predictions_df[
        (predictions_df["channel"] == channel) &
        (predictions_df["forecast_level"] == "channel")
    ]

    summary = {}
    for _, row in channel_preds.iterrows():
        period = int(row["period_days"])
        summary[f"{period}d"] = {
            "revenue_p10": row["revenue_p10"],
            "revenue_p50": row["revenue_p50"],
            "revenue_p90": row["revenue_p90"],
            "roas_p50": row["roas_p50"]
        }
    return summary


def generate_channel_insight(df, predictions_df, channel):
    """Generate AI insight for a single channel using OpenAI"""
    stats = get_channel_stats(df, channel)
    forecast = get_forecast_summary(predictions_df, channel)

    prompt = f"""
You are a senior digital marketing analyst at a top ecommerce agency.
Analyze this channel performance data and provide actionable insights.

CHANNEL: {channel.upper()}

HISTORICAL PERFORMANCE:
- Total Revenue: ${stats['total_revenue']:,}
- Total Spend: ${stats['total_spend']:,}
- Average ROAS: {stats['avg_roas']}x
- Average CTR: {stats['avg_ctr_pct']}%
- Average CPC: ${stats['avg_cpc']}
- Revenue Trend (last 30d vs prev 30d): {stats['revenue_trend_30d_pct']}%
- Last 30d Revenue: ${stats['last_30d_revenue']:,}
- Previous 30d Revenue: ${stats['prev_30d_revenue']:,}

FORECASTED REVENUE:
- Next 30 days: ${forecast.get('30d', {}).get('revenue_p50', 0):,.0f} (P10: ${forecast.get('30d', {}).get('revenue_p10', 0):,.0f} - P90: ${forecast.get('30d', {}).get('revenue_p90', 0):,.0f})
- Next 60 days: ${forecast.get('60d', {}).get('revenue_p50', 0):,.0f} (P10: ${forecast.get('60d', {}).get('revenue_p10', 0):,.0f} - P90: ${forecast.get('60d', {}).get('revenue_p90', 0):,.0f})
- Next 90 days: ${forecast.get('90d', {}).get('revenue_p50', 0):,.0f} (P10: ${forecast.get('90d', {}).get('revenue_p10', 0):,.0f} - P90: ${forecast.get('90d', {}).get('revenue_p90', 0):,.0f})
- Forecasted ROAS: {forecast.get('30d', {}).get('roas_p50', 0)}x

Please provide:
1. PERFORMANCE SUMMARY: 2-3 sentences on current channel health
2. KEY DRIVERS: What is driving performance up or down
3. FORECAST OUTLOOK: What the forecast range means for the business
4. BUDGET RECOMMENDATION: Should spend increase, decrease or stay same
5. RISK FLAGS: Any concerns the agency should watch

Keep it concise, data-driven and actionable. Max 200 words total.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Insight generation failed: {e}"


def generate_overall_insight(df, predictions_df):
    """Generate overall blended forecast insight"""

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

    prompt = f"""
You are a senior ecommerce marketing strategist.
Provide an executive summary of the blended forecast across all channels.

BLENDED HISTORICAL ROAS: {blended_roas}x
TOTAL HISTORICAL REVENUE: ${total_revenue:,.0f}
TOTAL HISTORICAL SPEND: ${total_spend:,.0f}

BLENDED REVENUE FORECAST:
- Next 30 days: ${total_30d:,.0f}
- Next 60 days: ${total_60d:,.0f}
- Next 90 days: ${total_90d:,.0f}

CHANNELS: Google Ads, Meta Ads, Microsoft (Bing) Ads

Write a 3-4 sentence executive summary that:
1. States the overall revenue outlook
2. Highlights the biggest opportunity
3. Flags the biggest risk
4. Gives one strategic recommendation

Be direct, confident and data-driven.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Overall insight generation failed: {e}"


def run_insights(df, predictions_df):
    """Generate all insights and return as dictionary"""
    print("\nGenerating AI insights...")

    insights = {}

    print("  >> Generating overall executive summary...")
    insights["overall"] = generate_overall_insight(df, predictions_df)

    for channel in ["google", "meta", "bing"]:
        print(f"  >> Generating insight for {channel}...")
        insights[channel] = generate_channel_insight(df, predictions_df, channel)

    return insights


def save_insights(insights, output_path="output/insights.json"):
    """Save insights to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(insights, f, indent=2)
    print(f"Insights saved to {output_path}")


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from forecast import run_all_forecasts

    print("Loading data and forecasts...")
    df, channel_fcs, camptype_fcs = run_all_forecasts()

    predictions_df = pd.read_csv("output/predictions.csv")

    insights = run_insights(df, predictions_df)

    save_insights(insights)

    print("\n=== AI INSIGHTS ===")
    for key, value in insights.items():
        print(f"\n--- {key.upper()} ---")
        print(value)