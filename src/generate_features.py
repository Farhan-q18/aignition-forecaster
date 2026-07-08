import argparse
import json
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from taxonomy import (
    classify_meta_campaigns,
    normalize_bing_type,
    normalize_google_type,
)


def load_bing(data_dir):
    path = os.path.join(data_dir, "bing_campaign_stats.csv")
    df = pd.read_csv(path, index_col=0)
    df = df.rename(columns={
        "TimePeriod": "date",
        "Revenue": "revenue",
        "Spend": "spend",
        "Clicks": "clicks",
        "Impressions": "impressions",
        "Conversions": "conversions",
        "CampaignType": "campaign_type",
        "CampaignName": "campaign_name",
        "CampaignId": "campaign_id",
        "DailyBudget": "daily_budget"
    })
    df["campaign_type"] = df["campaign_type"].map(normalize_bing_type)
    df["channel"] = "bing"
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "channel", "campaign_id", "campaign_name",
               "campaign_type", "revenue", "spend", "clicks",
               "impressions", "conversions", "daily_budget"]]


def load_google(data_dir):
    path = os.path.join(data_dir, "google_ads_campaign_stats.csv")
    df = pd.read_csv(path, index_col=0)
    df = df.rename(columns={
        "segments_date": "date",
        "metrics_conversions_value": "revenue",
        "metrics_clicks": "clicks",
        "metrics_conversions": "conversions",
        "metrics_impressions": "impressions",
        "campaign_advertising_channel_type": "campaign_type",
        "campaign_name": "campaign_name",
        "campaign_id": "campaign_id",
        "campaign_budget_amount": "daily_budget"
    })
    # Convert micros to actual dollars
    df["spend"] = df["metrics_cost_micros"] / 1_000_000
    df["campaign_type"] = df["campaign_type"].map(normalize_google_type)
    df["channel"] = "google"
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "channel", "campaign_id", "campaign_name",
               "campaign_type", "revenue", "spend", "clicks",
               "impressions", "conversions", "daily_budget"]]


def load_meta(data_dir, revenue_mode="value", assumed_aov=75.0):
    """Load Meta data.

    ASSUMPTION (documented in docs/TECHNICAL_DOCUMENTATION.md): Meta's
    `conversion` column is treated as conversion VALUE (revenue), not a count —
    its ratio to spend (median ~4.1x) matches Google's revenue/cost and Bing's
    Revenue/Spend. `revenue_mode="count"` flips the interpretation and applies
    an assumed AOV instead.
    """
    path = os.path.join(data_dir, "meta_ads_campaign_stats.csv")
    df = pd.read_csv(path, index_col=0)
    df = df.rename(columns={
        "date_start": "date",
        "clicks": "clicks",
        "impressions": "impressions",
        "campaign_name": "campaign_name",
        "campaign_id": "campaign_id",
        "daily_budget": "daily_budget"
    })

    if revenue_mode == "count":
        df["revenue"] = df["conversion"] * assumed_aov
        df["conversions"] = df["conversion"]
    else:
        df["revenue"] = df["conversion"]
        df["conversions"] = 0.0  # no separate conversion count under this interpretation

    # Campaign type is embedded in the campaign name on Meta — parse it into
    # the normalized taxonomy (Prospecting/Remarketing/Generic/Advantage+,
    # with Brand/DPA sub-tags).
    type_map = classify_meta_campaigns(df["campaign_name"].unique())
    df["campaign_type"] = df["campaign_name"].map(
        lambda n: type_map[n]["campaign_type"]
    )
    unmatched = sorted(n for n, info in type_map.items() if not info["matched"])
    if unmatched:
        print(f"  WARNING: {len(unmatched)} Meta campaign name(s) did not match "
              f"any taxonomy rule (typed as 'Unclassified'): {unmatched[:5]}")

    df["channel"] = "meta"
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "channel", "campaign_id", "campaign_name",
               "campaign_type", "revenue", "spend", "clicks",
               "impressions", "conversions", "daily_budget"]]


def engineer_features(df):
    df = df.sort_values(["channel", "campaign_name", "date"])

    # Basic ROAS
    df["roas"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], 0)

    # CTR and CPC
    df["ctr"] = np.where(df["impressions"] > 0, df["clicks"] / df["impressions"], 0)
    df["cpc"] = np.where(df["clicks"] > 0, df["spend"] / df["clicks"], 0)

    # Date features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Rolling features (7-day) per campaign
    for col in ["revenue", "spend", "roas", "clicks"]:
        df[f"{col}_7d_avg"] = (
            df.groupby("campaign_name")[col]
            .transform(lambda x: x.rolling(7, min_periods=1).mean())
        )

    # Lag features
    for col in ["revenue", "spend", "roas"]:
        df[f"{col}_lag1"] = df.groupby("campaign_name")[col].shift(1)
        df[f"{col}_lag7"] = df.groupby("campaign_name")[col].shift(7)

    df = df.fillna(0)
    return df


def build_data_health(df):
    """Per-channel data-quality report: null budgets, zero-revenue days, ranges."""
    health = {}
    for channel, grp in df.groupby("channel"):
        zero_rev_days = float((grp["revenue"] == 0).mean())
        health[channel] = {
            "rows": int(len(grp)),
            "campaigns": int(grp["campaign_name"].nunique()),
            "date_min": str(grp["date"].min().date()),
            "date_max": str(grp["date"].max().date()),
            "null_daily_budget_rows": int(grp["daily_budget"].isna().sum()),
            "zero_revenue_day_share": round(zero_rev_days, 4),
            "unclassified_campaign_types": int(
                (grp["campaign_type"] == "Unclassified").sum()
            ),
        }
    return health


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out", default="features.parquet")
    parser.add_argument(
        "--meta-revenue-mode", choices=["value", "count"], default="value",
        help="Interpret Meta's `conversion` column as revenue ('value', default) "
             "or as a conversion count multiplied by --meta-aov ('count')."
    )
    parser.add_argument("--meta-aov", type=float, default=75.0,
                        help="Assumed average order value when --meta-revenue-mode=count")
    parser.add_argument("--health-out", default="output/data_health.json")
    args = parser.parse_args()

    print("Loading Bing data...")
    bing = load_bing(args.data_dir)

    print("Loading Google data...")
    google = load_google(args.data_dir)

    print("Loading Meta data...")
    meta = load_meta(args.data_dir, revenue_mode=args.meta_revenue_mode,
                     assumed_aov=args.meta_aov)

    print("Combining all channels...")
    df = pd.concat([bing, google, meta], ignore_index=True)

    health = build_data_health(df)
    os.makedirs(os.path.dirname(os.path.abspath(args.health_out)), exist_ok=True)
    with open(args.health_out, "w") as f:
        json.dump(health, f, indent=2)
    for ch, h in health.items():
        print(f"  [{ch}] {h['rows']} rows, {h['campaigns']} campaigns, "
              f"{h['null_daily_budget_rows']} null budgets, "
              f"{h['zero_revenue_day_share']:.0%} zero-revenue days")

    print("Engineering features...")
    df = engineer_features(df)

    df.to_parquet(args.out, index=False)
    print(f"Features saved to {args.out} — shape: {df.shape}")


if __name__ == "__main__":
    main()