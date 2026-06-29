import argparse
import pandas as pd
import numpy as np
import os

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
    df["channel"] = "google"
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "channel", "campaign_id", "campaign_name",
               "campaign_type", "revenue", "spend", "clicks",
               "impressions", "conversions", "daily_budget"]]


def load_meta(data_dir):
    path = os.path.join(data_dir, "meta_ads_campaign_stats.csv")
    df = pd.read_csv(path, index_col=0)
    df = df.rename(columns={
        "date_start": "date",
        "conversion": "revenue",
        "spend": "spend",
        "clicks": "clicks",
        "impressions": "impressions",
        "campaign_name": "campaign_name",
        "campaign_id": "campaign_id",
        "daily_budget": "daily_budget"
    })
    df["campaign_type"] = "Paid_Social"
    df["conversions"] = 0.0  # Meta doesn't have direct conversions count
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out", default="features.parquet")
    args = parser.parse_args()

    print("Loading Bing data...")
    bing = load_bing(args.data_dir)

    print("Loading Google data...")
    google = load_google(args.data_dir)

    print("Loading Meta data...")
    meta = load_meta(args.data_dir)

    print("Combining all channels...")
    df = pd.concat([bing, google, meta], ignore_index=True)

    print("Engineering features...")
    df = engineer_features(df)

    df.to_parquet(args.out, index=False)
    print(f"Features saved to {args.out} — shape: {df.shape}")


if __name__ == "__main__":
    main()