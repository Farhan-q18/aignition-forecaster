"""
AI-assisted causal inference layer — demo service, NOT part of run.sh.

Grounding contract: anomalies are detected statistically first
(src/anomalies.py); this module hands the LLM only the *detected* anomalies
plus compact structured context, and asks for causal hypotheses grounded
strictly in the numbers given — never invented external events.

Provider: Anthropic API first (ANTHROPIC_API_KEY), OpenAI fallback
(OPENAI_API_KEY). Model overridable via ANTHROPIC_MODEL / OPENAI_MODEL.

Outputs output/insights.json:
{
  "overall": "...executive summary...",
  "channels": {"google": "...", ...},
  "anomaly_insights": [{...anomaly, "summary", "likely_cause",
                        "confidence", "recommended_action"}, ...],
  "provider": "anthropic|openai", "model": "..."
}
"""

import json
import os
import re
import sys

import pandas as pd

ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

MAX_ANOMALIES_TO_EXPLAIN = 8


def _get_client():
    if ANTHROPIC_KEY:
        import anthropic
        return "anthropic", anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    if OPENAI_KEY:
        from openai import OpenAI
        return "openai", OpenAI(api_key=OPENAI_KEY)
    return None, None


def _call_llm(provider, client, prompt, max_tokens=400):
    if provider == "anthropic":
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


def _parse_json(text):
    """Extract the first JSON object from an LLM reply."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


# ---------- structured context builders (plain pandas, no LLM) ----------

def get_causal_drivers(df, channel):
    ch = df[df["channel"] == channel].copy().sort_values("date")
    last_30 = ch[ch["date"] > ch["date"].max() - pd.Timedelta(days=30)]
    prev_30 = ch[(ch["date"] <= ch["date"].max() - pd.Timedelta(days=30)) &
                 (ch["date"] > ch["date"].max() - pd.Timedelta(days=60))]

    def agg(frame):
        rev, spend = frame["revenue"].sum(), frame["spend"].sum()
        clicks, imps = frame["clicks"].sum(), frame["impressions"].sum()
        return {
            "revenue": rev, "spend": spend,
            "roas": rev / spend if spend > 0 else 0,
            "cpc": spend / clicks if clicks > 0 else 0,
            "ctr": clicks / imps if imps > 0 else 0,
        }

    last, prev = agg(last_30), agg(prev_30)

    def pct(a, b):
        return round((a - b) / b * 100, 1) if b else None

    return {
        "last_30d": {k: round(v, 4) for k, v in last.items()},
        "prev_30d": {k: round(v, 4) for k, v in prev.items()},
        "trend_pct": {k: pct(last[k], prev[k]) for k in last},
    }


def get_forecast_context(predictions_df, channel=None):
    level = "channel" if channel else "blended"
    rows = predictions_df[predictions_df["forecast_level"] == level]
    if channel:
        rows = rows[rows["channel"] == channel]
    out = {}
    for _, row in rows.iterrows():
        out[f"{int(row['period_days'])}d"] = {
            "revenue_p10": row["revenue_p10"],
            "revenue_p50": row["revenue_p50"],
            "revenue_p90": row["revenue_p90"],
            "roas_p50": row["roas_p50"],
        }
    return out


def _select_anomalies(anomalies):
    """Material, high-|z| anomalies first; cap the count (cost + focus)."""
    outliers = [a for a in anomalies.get("revenue_outliers", [])
                if max(a["actual"], a["expected"]) >= 1000]
    outliers.sort(key=lambda a: -abs(a["z_score"]))
    picked = outliers[:MAX_ANOMALIES_TO_EXPLAIN - len(anomalies.get("roas_drift", []))]
    return picked + anomalies.get("roas_drift", [])


# ---------- LLM-facing generation ----------

def explain_anomaly(provider, client, anomaly, drivers, forecast_ctx):
    prompt = f"""You are a digital marketing analyst. A statistical monitor flagged this anomaly.
Interpret it using ONLY the numbers provided — do not invent external events (no holidays,
outages, or competitor actions unless the dates themselves imply seasonality).

ANOMALY (statistically detected):
{json.dumps(anomaly, indent=1)}

CHANNEL CONTEXT (last 30d vs prior 30d):
{json.dumps(drivers, indent=1)}

FORECAST CONTEXT (probabilistic, P10-P90):
{json.dumps(forecast_ctx, indent=1)}

Reply with ONLY a JSON object:
{{"summary": "<=25 words, plain English, for a business owner",
 "likely_cause": "<=35 words, causal hypothesis grounded in the numbers above",
 "confidence": "high|medium|low",
 "recommended_action": "<=25 words, one concrete next step"}}"""
    reply = _call_llm(provider, client, prompt, max_tokens=300)
    parsed = _parse_json(reply)
    if parsed is None:
        parsed = {"summary": reply[:140], "likely_cause": "",
                  "confidence": "low", "recommended_action": ""}
    return {**anomaly, **parsed}


def generate_channel_insight(provider, client, df, predictions_df, channel,
                             channel_anomalies):
    drivers = get_causal_drivers(df, channel)
    fc = get_forecast_context(predictions_df, channel)

    top = predictions_df[
        (predictions_df["channel"] == channel) &
        (predictions_df["forecast_level"] == "campaign") &
        (predictions_df["period_days"] == 30)
    ].nlargest(3, "revenue_p50")
    top_lines = [f"- {r['campaign_name']} ({r['campaign_type']}): "
                 f"${r['revenue_p50']:,.0f} 30d revenue, {r['roas_p50']:.2f}x ROAS"
                 for _, r in top.iterrows()]

    prompt = f"""You are a senior digital marketing analyst. Write a causally-grounded insight for
the {channel.upper()} channel using ONLY the data below.

PERFORMANCE (last 30d vs prior 30d): {json.dumps(drivers)}

PROBABILISTIC FORECAST (P10/P50/P90): {json.dumps(fc)}

TOP CAMPAIGNS (30d forecast):
{chr(10).join(top_lines) if top_lines else '- none'}

STATISTICALLY DETECTED ANOMALIES (recent):
{json.dumps(channel_anomalies[:4]) if channel_anomalies else 'none'}

Write exactly 5 labeled sections, citing specific numbers and causal chains:
1. PERFORMANCE SUMMARY: what happened + primary driver (2 sentences)
2. KEY DRIVERS: the 1-2 metrics most responsible (2 sentences)
3. FORECAST OUTLOOK: what the P10-P90 spread signals about risk (1-2 sentences)
4. BUDGET RECOMMENDATION: increase/hold/decrease and why (1-2 sentences)
5. RISK FLAGS: one specific metric to watch (1 sentence)
Max 220 words."""
    return _call_llm(provider, client, prompt, max_tokens=450)


def generate_overall_insight(provider, client, df, predictions_df, anomalies):
    blended = get_forecast_context(predictions_df)
    total_spend, total_rev = df["spend"].sum(), df["revenue"].sum()

    channel_lines = []
    ch_rows = predictions_df[(predictions_df["forecast_level"] == "channel") &
                             (predictions_df["period_days"] == 30)]
    for _, r in ch_rows.iterrows():
        channel_lines.append(f"- {r['channel']}: ${r['revenue_p50']:,.0f} 30d P50, "
                             f"{r['roas_p50']:.2f}x ROAS")

    n_caps = len(anomalies.get("budget_caps", []))
    prompt = f"""You are a senior ecommerce marketing strategist. Executive summary of the blended
paid-media forecast, using ONLY the data below.

HISTORICAL: blended ROAS {total_rev / total_spend:.2f}x, total revenue ${total_rev:,.0f}, total spend ${total_spend:,.0f}
BLENDED FORECAST (P10/P50/P90): {json.dumps(blended)}
CHANNELS (30d): {chr(10).join(channel_lines)}
OPERATIONAL: {n_caps} campaigns are at/above their stated daily budget caps.

Write exactly 4 sentences:
1. Revenue outlook — 30/60/90d P50 numbers and what the P10-P90 spread implies
2. Biggest opportunity channel — cite ROAS and share
3. Biggest risk — a specific metric or channel
4. One concrete budget reallocation recommendation
Max 130 words."""
    return _call_llm(provider, client, prompt, max_tokens=280)


def run_insights(df, predictions_df, anomalies):
    provider, client = _get_client()
    if provider is None:
        print("No ANTHROPIC_API_KEY or OPENAI_API_KEY found — skipping AI insights.")
        return None
    model = ANTHROPIC_MODEL if provider == "anthropic" else OPENAI_MODEL
    print(f"Generating AI insights via {provider} ({model})...")

    insights = {"provider": provider, "model": model, "channels": {},
                "anomaly_insights": []}

    print("  >> executive summary...")
    insights["overall"] = generate_overall_insight(
        provider, client, df, predictions_df, anomalies)

    outliers = anomalies.get("revenue_outliers", [])
    for ch in sorted(df["channel"].unique()):
        print(f"  >> {ch} channel insight...")
        ch_anoms = [a for a in outliers if a.get("channel") == ch]
        insights["channels"][ch] = generate_channel_insight(
            provider, client, df, predictions_df, ch, ch_anoms)

    for anomaly in _select_anomalies(anomalies):
        label = anomaly.get("campaign_type") or anomaly.get("channel")
        print(f"  >> explaining anomaly: {anomaly['type']} @ {label}...")
        try:
            insights["anomaly_insights"].append(explain_anomaly(
                provider, client, anomaly,
                get_causal_drivers(df, anomaly["channel"]),
                get_forecast_context(predictions_df, anomaly["channel"])))
        except Exception as e:
            print(f"     failed: {e}")

    return insights


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI insights (demo layer, needs API key)")
    parser.add_argument("--features", default="features.parquet")
    parser.add_argument("--predictions", default="output/predictions.csv")
    parser.add_argument("--anomalies", default="output/anomalies.json")
    parser.add_argument("--out", default="output/insights.json")
    args = parser.parse_args()

    df = pd.read_parquet(args.features)
    predictions_df = pd.read_csv(args.predictions)
    if os.path.isfile(args.anomalies):
        with open(args.anomalies) as f:
            anomalies = json.load(f)
    else:
        print(f"NOTE: {args.anomalies} not found — run src/anomalies.py first "
              "for anomaly-grounded insights.")
        anomalies = {}

    insights = run_insights(df, predictions_df, anomalies)
    if insights is None:
        sys.exit(0)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(insights, f, indent=2)
    print(f"Insights saved to {args.out}")


if __name__ == "__main__":
    main()
