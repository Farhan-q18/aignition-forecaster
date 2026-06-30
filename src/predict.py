import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from forecast import (
    aggregate_campaign_forecasts,
    aggregate_forecasts,
    forecast_campaign,
    forecast_campaign_type,
    forecast_channel,
)

PERIODS = [30, 60, 90]
COL_ORDER = [
    "forecast_level", "channel", "campaign_type", "campaign_name",
    "period_days", "revenue_p10", "revenue_p50", "revenue_p90",
    "roas_p10", "roas_p50", "roas_p90",
]


def validate_campaigns(df):
    issues = []
    for channel in df["channel"].unique():
        ch_df = df[df["channel"] == channel]
        for campaign, grp in ch_df.groupby("campaign_name"):
            if grp["revenue"].sum() == 0 and grp["spend"].sum() == 0:
                issues.append(f"[{channel}] '{campaign}': zero revenue and zero spend — likely inactive")
            if (grp["spend"] < 0).any():
                issues.append(f"[{channel}] '{campaign}': negative spend detected")
    return issues


def estimate_daily_spend(df, channel, campaign_type=None, campaign_name=None):
    mask = df["channel"] == channel
    if campaign_type is not None:
        mask &= df["campaign_type"] == campaign_type
    if campaign_name is not None:
        mask &= df["campaign_name"] == campaign_name
    daily = df[mask].groupby("date")["spend"].sum()
    return daily.mean() if len(daily) > 0 else 0.0


def add_roas_columns(summary_df, df):
    rows = []
    for _, row in summary_df.iterrows():
        channel = row["channel"]
        campaign_type = row.get("campaign_type") if pd.notna(row.get("campaign_type")) else None
        campaign_name = row.get("campaign_name") if pd.notna(row.get("campaign_name")) else None
        period = row["period_days"]

        daily_spend = estimate_daily_spend(df, channel, campaign_type, campaign_name)
        est_spend = daily_spend * period

        def safe_roas(rev):
            return round(rev / est_spend, 4) if est_spend > 0 else 0.0

        row = row.copy()
        row["roas_p10"] = safe_roas(row["revenue_p10"])
        row["roas_p50"] = safe_roas(row["revenue_p50"])
        row["roas_p90"] = safe_roas(row["revenue_p90"])
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="AIgnition Forecaster — prediction pipeline")
    parser.add_argument("--features", default="features.parquet")
    parser.add_argument("--model", default="./pickle/model.pkl")
    parser.add_argument("--output", default="./output/predictions.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.model)), exist_ok=True)

    print("Loading features...")
    df = pd.read_parquet(args.features)
    np.random.seed(42)

    print("\nValidating campaign consistency...")
    issues = validate_campaigns(df)
    if issues:
        print(f"  {len(issues)} issue(s) found:")
        for issue in issues[:10]:
            print(f"    {issue}")
        if len(issues) > 10:
            print(f"    ... and {len(issues) - 10} more")
    else:
        print("  No consistency issues found.")

    channels = df["channel"].unique()

    # --- Channel-level ---
    print("\nForecasting by channel...")
    channel_fcs = []
    for channel in channels:
        fc = forecast_channel(df, channel)
        if fc is not None:
            channel_fcs.append(fc)

    channel_summary = aggregate_forecasts(channel_fcs)
    channel_summary["forecast_level"] = "channel"
    channel_summary = add_roas_columns(channel_summary, df)

    # --- Campaign-type level ---
    print("\nForecasting by campaign type...")
    camptype_fcs = []
    for channel in channels:
        for ct in df[df["channel"] == channel]["campaign_type"].unique():
            print(f"  >> {channel} / {ct}")
            fc = forecast_campaign_type(df, channel, ct)
            if fc is not None:
                camptype_fcs.append(fc)

    camptype_summary = aggregate_forecasts(camptype_fcs)
    camptype_summary["forecast_level"] = "campaign_type"
    camptype_summary = add_roas_columns(camptype_summary, df)

    # --- Campaign level ---
    print("\nForecasting by campaign...")
    campaign_fcs = []
    skipped = 0
    for channel in channels:
        for campaign in df[df["channel"] == channel]["campaign_name"].unique():
            fc = forecast_campaign(df, channel, campaign)
            if fc is not None:
                print(f"  >> {channel} / {campaign}")
                campaign_fcs.append(fc)
            else:
                skipped += 1

    campaign_summary = pd.DataFrame()
    if campaign_fcs:
        campaign_summary = aggregate_campaign_forecasts(campaign_fcs)
        campaign_summary["forecast_level"] = "campaign"
        campaign_summary = add_roas_columns(campaign_summary, df)

    print(f"  Campaigns forecast: {len(campaign_fcs)}, skipped (insufficient data): {skipped}")

    # --- Combine ---
    parts = [channel_summary, camptype_summary]
    if not campaign_summary.empty:
        parts.append(campaign_summary)

    predictions = pd.concat(parts, ignore_index=True)
    for col in COL_ORDER:
        if col not in predictions.columns:
            predictions[col] = None
    predictions = predictions[COL_ORDER]

    predictions.to_csv(args.output, index=False)
    print(f"\nPredictions written to {args.output} — {len(predictions)} rows")

    model_artifact = {"channels": list(channels), "periods": PERIODS}
    with open(args.model, "wb") as f:
        pickle.dump(model_artifact, f)
    print(f"Model artifact saved to {args.model}")


if __name__ == "__main__":
    main()
