import argparse
import pandas as pd
import numpy as np
import pickle
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from forecast import run_all_forecasts, aggregate_forecasts


def compute_roas_forecast(df, channel_forecasts):
    """Compute ROAS ranges based on historical spend patterns"""
    results = []

    for fc in channel_forecasts:
        if fc is None:
            continue

        channel = fc["channel"].iloc[0]
        channel_df = df[df["channel"] == channel]

        # Historical average spend per day
        avg_daily_spend = channel_df[channel_df["spend"] > 0]["spend"].mean()
        avg_daily_spend = avg_daily_spend if not np.isnan(avg_daily_spend) else 1.0

        for period in [30, 60, 90]:
            projected_spend = avg_daily_spend * period

            fc_copy = fc.copy()
            fc_copy["ds"] = pd.to_datetime(fc_copy["ds"])
            start_date = fc_copy["ds"].min()
            end_date = start_date + pd.Timedelta(days=period)
            window = fc_copy[fc_copy["ds"] <= end_date]

            revenue_p10 = window["yhat_lower"].sum()
            revenue_p50 = window["yhat"].sum()
            revenue_p90 = window["yhat_upper"].sum()

            results.append({
                "channel": channel,
                "period_days": period,
                "projected_spend": round(projected_spend, 2),
                "roas_p10": round(revenue_p10 / projected_spend, 4) if projected_spend > 0 else 0,
                "roas_p50": round(revenue_p50 / projected_spend, 4) if projected_spend > 0 else 0,
                "roas_p90": round(revenue_p90 / projected_spend, 4) if projected_spend > 0 else 0,
            })

    return pd.DataFrame(results)


def build_output(channel_forecasts, camptype_forecasts, roas_df):
    """Combine all forecasts into final predictions.csv format"""

    # Channel level forecasts
    channel_summary = aggregate_forecasts(channel_forecasts)
    channel_summary["campaign_type"] = "ALL"
    channel_summary["forecast_level"] = "channel"

    # Campaign type level forecasts
    camptype_summary = aggregate_forecasts(camptype_forecasts)
    camptype_summary["forecast_level"] = "campaign_type"
    if "campaign_type" not in camptype_summary.columns:
        camptype_summary["campaign_type"] = "UNKNOWN"

    # Combine
    combined = pd.concat([channel_summary, camptype_summary], ignore_index=True)

    # Add ROAS
    combined = combined.merge(
        roas_df[["channel", "period_days", "roas_p10", "roas_p50", "roas_p90"]],
        on=["channel", "period_days"],
        how="left"
    )

    # Fill missing ROAS for campaign type rows
    combined["roas_p10"] = combined["roas_p10"].fillna(0)
    combined["roas_p50"] = combined["roas_p50"].fillna(0)
    combined["roas_p90"] = combined["roas_p90"].fillna(0)

    # Round all numbers
    for col in ["revenue_p10", "revenue_p50", "revenue_p90",
                "roas_p10", "roas_p50", "roas_p90"]:
        combined[col] = combined[col].round(2)

    # Reorder columns
    combined = combined[[
        "forecast_level", "channel", "campaign_type",
        "period_days",
        "revenue_p10", "revenue_p50", "revenue_p90",
        "roas_p10", "roas_p50", "roas_p90"
    ]]

    combined = combined.sort_values(
        ["forecast_level", "channel", "period_days"]
    ).reset_index(drop=True)

    return combined


def save_model(df, channel_forecasts, model_path):
    """Save forecast state as pickle"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model_data = {
        "channel_forecasts": channel_forecasts,
        "feature_columns": list(df.columns),
        "channels": list(df["channel"].unique()),
        "date_range": {
            "start": str(df["date"].min()),
            "end": str(df["date"].max())
        }
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"Model saved to {model_path}")


def load_model(model_path):
    """Load pickle model"""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="features.parquet")
    parser.add_argument("--model", default="./pickle/model.pkl")
    parser.add_argument("--output", default="./output/predictions.csv")
    args = parser.parse_args()

    print("Running forecasts...")
    df, channel_fcs, camptype_fcs = run_all_forecasts(
        features_path=args.features,
        periods=90
    )

    print("\nComputing ROAS forecasts...")
    roas_df = compute_roas_forecast(df, channel_fcs)

    print("Building output...")
    predictions = build_output(channel_fcs, camptype_fcs, roas_df)

    print("Saving model pickle...")
    save_model(df, channel_fcs, args.model)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    predictions.to_csv(args.output, index=False)

    print(f"\nDone! Predictions saved to {args.output}")
    print(f"Shape: {predictions.shape}")
    print("\nPreview:")
    print(predictions.head(10).to_string())


if __name__ == "__main__":
    main()