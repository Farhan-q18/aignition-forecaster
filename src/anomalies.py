"""
Statistical anomaly detection — the grounding layer for AI insights.

Anomalies are detected STATISTICALLY first (robust z-scores on decomposition
residuals, budget-cap proximity, ROAS drift); the LLM only interprets what the
statistics flagged, with the surrounding structured context. It is never asked
to "find anomalies" in raw numbers — that's neither reliable nor auditable.

Runs fully offline. Output: output/anomalies.json.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from forecast import daily_series, decompose, weekly_series, MIN_WEEKS_FORECAST

Z_THRESHOLD = 2.5
ROAS_DRIFT_WEEKS = 4


def _robust_z(resid):
    """z-scores using median/MAD — outliers shouldn't inflate their own bar."""
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    scale = 1.4826 * mad if mad > 0 else (np.std(resid) or 1.0)
    return (resid - med) / scale


def residual_anomalies(df, channel, campaign_type=None, last_n_weeks=26):
    """Weeks whose decomposition residual is a robust-z outlier."""
    mask = df["channel"] == channel
    label = {"level": "channel", "channel": channel}
    if campaign_type is not None:
        mask &= df["campaign_type"] == campaign_type
        label = {"level": "campaign_type", "channel": channel,
                 "campaign_type": campaign_type}

    weekly = weekly_series(daily_series(df[mask]))
    if len(weekly) < MIN_WEEKS_FORECAST:
        return []

    trend_fwd, seasonal_fwd, resid = decompose(weekly)
    z = _robust_z(resid)
    expected = weekly.values - resid

    out = []
    for i in range(max(0, len(weekly) - last_n_weeks), len(weekly)):
        if abs(z[i]) >= Z_THRESHOLD:
            out.append({
                **label,
                "type": "revenue_outlier",
                "week_end": str(weekly.index[i].date()),
                "actual": round(float(weekly.values[i]), 2),
                "expected": round(float(expected[i]), 2),
                "z_score": round(float(z[i]), 2),
                "direction": "spike" if z[i] > 0 else "drop",
            })
    return out


def budget_cap_flags(df, lookback_days=28, threshold=0.90):
    """Campaigns whose recent daily spend sits at/near their budget cap —
    an operational constraint on scaling that the forecast can't see."""
    end = df["date"].max()
    recent = df[df["date"] > end - pd.Timedelta(days=lookback_days)]
    flags = []
    for (ch, name), grp in recent.groupby(["channel", "campaign_name"]):
        budget = grp["daily_budget"].dropna()
        if budget.empty:
            continue
        budget = float(budget.median())
        if budget <= 0:
            continue
        util = float(grp["spend"].mean()) / budget
        if util >= threshold:
            flags.append({
                "level": "campaign",
                "type": "budget_cap",
                "channel": ch,
                "campaign_name": name,
                "avg_daily_spend": round(float(grp["spend"].mean()), 2),
                "daily_budget": round(budget, 2),
                "utilization": round(util, 3),
            })
    return sorted(flags, key=lambda f: -f["utilization"])


def roas_drift_flags(df, window_weeks=ROAS_DRIFT_WEEKS, z_threshold=2.0):
    """Channels/campaign-types whose recent ROAS drifted outside its own
    historical weekly distribution."""
    flags = []
    groups = [("channel", ch, df["channel"] == ch)
              for ch in df["channel"].unique()]
    for ch in df["channel"].unique():
        for ct in df[df["channel"] == ch]["campaign_type"].unique():
            groups.append(("campaign_type", (ch, ct),
                           (df["channel"] == ch) & (df["campaign_type"] == ct)))

    for level, key, mask in groups:
        sub = df[mask]
        wk = sub.groupby(pd.Grouper(key="date", freq="W"))[["spend", "revenue"]].sum()
        wk = wk[wk["spend"] > 0]
        if len(wk) < 20:
            continue
        roas = wk["revenue"] / wk["spend"]
        hist, recent = roas.iloc[:-window_weeks], roas.iloc[-window_weeks:]
        if hist.std() == 0 or len(recent) < window_weeks:
            continue
        z = (recent.mean() - hist.mean()) / (hist.std() / np.sqrt(window_weeks))
        if abs(z) >= z_threshold:
            flag = {
                "level": level,
                "type": "roas_drift",
                "recent_roas": round(float(recent.mean()), 2),
                "historical_roas": round(float(hist.mean()), 2),
                "z_score": round(float(z), 2),
                "direction": "up" if z > 0 else "down",
            }
            if level == "channel":
                flag["channel"] = key
            else:
                flag["channel"], flag["campaign_type"] = key
            flags.append(flag)
    return flags


def detect_all(df):
    anomalies = []
    for ch in sorted(df["channel"].unique()):
        anomalies += residual_anomalies(df, ch)
        for ct in sorted(df[df["channel"] == ch]["campaign_type"].unique()):
            anomalies += residual_anomalies(df, ch, ct)
    return {
        "revenue_outliers": anomalies,
        "budget_caps": budget_cap_flags(df),
        "roas_drift": roas_drift_flags(df),
    }


def main():
    parser = argparse.ArgumentParser(description="Statistical anomaly detection")
    parser.add_argument("--features", default="features.parquet")
    parser.add_argument("--out", default="output/anomalies.json")
    args = parser.parse_args()

    df = pd.read_parquet(args.features)
    result = detect_all(df)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Detected {len(result['revenue_outliers'])} revenue outliers, "
          f"{len(result['budget_caps'])} budget-cap flags, "
          f"{len(result['roas_drift'])} ROAS drift flags -> {args.out}")


if __name__ == "__main__":
    main()
