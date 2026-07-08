"""
Prediction entry point — the scored pipeline (called by run.sh).

Loads the committed model artifact (pickle/model.pkl: calibrated sigma,
budget-response curves, config), reads the unified feature table, and writes
output/predictions.csv with probabilistic 30/60/90-day revenue + ROAS
forecasts at four levels: blended, channel, campaign_type, campaign.

No network access, no interactivity, no retraining of the artifact —
the decomposition is estimated on the provided history (that's forecasting),
but curves and calibration come straight from the pickle.
"""

import argparse
import json
import os
import pickle
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from forecast import (
    forecast_blended,
    forecast_group,
    recent_daily_spend,
    window_percentiles,
)
from response_curves import scale_factors

COL_ORDER = [
    "forecast_level", "channel", "campaign_type", "campaign_name",
    "period_days", "revenue_p10", "revenue_p50", "revenue_p90",
    "roas_p10", "roas_p50", "roas_p90", "budget_multiplier",
]


def validate_campaigns(df):
    """Campaign-consistency checks surfaced before forecasting."""
    issues = []
    for channel in df["channel"].unique():
        ch_df = df[df["channel"] == channel]
        for campaign, grp in ch_df.groupby("campaign_name"):
            if grp["revenue"].sum() == 0 and grp["spend"].sum() == 0:
                issues.append(f"[{channel}] '{campaign}': zero revenue and zero spend — likely inactive")
            if (grp["spend"] < 0).any():
                issues.append(f"[{channel}] '{campaign}': negative spend detected")
        n_unclassified = (ch_df["campaign_type"] == "Unclassified").sum()
        if n_unclassified:
            issues.append(f"[{channel}] {n_unclassified} rows with unclassifiable campaign type")
    return issues


def summarize(result, df, mask, periods, level, channel=None,
              campaign_type=None, campaign_name=None):
    """Turn a forecast result into 30/60/90-day rows with revenue + ROAS bands.

    ROAS treats near-term spend as planned/deterministic (recent 28-day daily
    average x window length), so the ROAS band inherits the revenue band's
    shape. Documented as a modeling decision.
    """
    rows = []
    daily_spend = recent_daily_spend(df, mask)
    for period in periods:
        p10, p50, p90 = window_percentiles(result["sims"], period)
        est_spend = daily_spend * period

        def roas(rev):
            return round(rev / est_spend, 4) if est_spend > 0 else 0.0

        rows.append({
            "forecast_level": level,
            "channel": channel,
            "campaign_type": campaign_type,
            "campaign_name": campaign_name,
            "period_days": period,
            "revenue_p10": round(p10, 2),
            "revenue_p50": round(p50, 2),
            "revenue_p90": round(p90, 2),
            "roas_p10": roas(p10),
            "roas_p50": roas(p50),
            "roas_p90": roas(p90),
            "budget_multiplier": None,
        })
    return rows


def path_rows(result, level, channel=None, campaign_type=None):
    """Weekly forecast path (for the dashboard's uncertainty-band charts)."""
    rows = []
    hist = result["history"].tail(52)
    for ds, val in hist.items():
        rows.append({"forecast_level": level, "channel": channel,
                     "campaign_type": campaign_type, "week_end": str(ds.date()),
                     "kind": "history", "p10": None, "p50": round(float(val), 2),
                     "p90": None})
    for i, ds in enumerate(result["future_week_ends"]):
        rows.append({"forecast_level": level, "channel": channel,
                     "campaign_type": campaign_type, "week_end": str(ds.date()),
                     "kind": "forecast",
                     "p10": round(float(result["p10"][i]), 2),
                     "p50": round(float(result["p50"][i]), 2),
                     "p90": round(float(result["p90"][i]), 2)})
    return rows


def load_budgets(budgets_arg):
    """Budget multipliers from a JSON string or file path,
    e.g. {"google": 1.2, "meta": 0.9}."""
    if not budgets_arg:
        return {}
    if os.path.isfile(budgets_arg):
        with open(budgets_arg) as f:
            return json.load(f)
    return json.loads(budgets_arg)


def budget_scenario_rows(prediction_rows, budgets, model):
    """Re-scale channel forecasts through the fitted response curves.

    Revenue scales by f(m*s0)/f(s0) with a bootstrap band on the curve itself;
    ROAS re-derives from scaled revenue over scaled spend.
    """
    curves = model.get("response_curves", {})
    out = []
    for row in prediction_rows:
        if row["forecast_level"] != "channel":
            continue
        multiplier = budgets.get(row["channel"])
        if multiplier is None or multiplier == 1.0:
            continue
        curve = curves.get(f"channel::{row['channel']}")
        s10, s50, s90 = scale_factors(curve, float(multiplier))
        new = dict(row)
        new["forecast_level"] = "channel_budget"
        # Curve uncertainty compounds with forecast uncertainty: apply the
        # low scale to P10 and the high scale to P90.
        new["revenue_p10"] = round(row["revenue_p10"] * s10, 2)
        new["revenue_p50"] = round(row["revenue_p50"] * s50, 2)
        new["revenue_p90"] = round(row["revenue_p90"] * s90, 2)
        for q, s in (("p10", s10), ("p50", s50), ("p90", s90)):
            new[f"roas_{q}"] = round(row[f"roas_{q}"] * s / multiplier, 4)
        new["budget_multiplier"] = multiplier
        out.append(new)
    return out


def main():
    parser = argparse.ArgumentParser(description="AIgnition Forecaster — prediction pipeline")
    parser.add_argument("--features", default="features.parquet")
    parser.add_argument("--model", default="./pickle/model.pkl")
    parser.add_argument("--output", default="./output/predictions.csv")
    parser.add_argument("--paths-output", default=None,
                        help="Weekly forecast paths CSV (defaults to "
                             "<output dir>/forecast_paths.csv)")
    parser.add_argument("--budgets", default=None,
                        help='Per-channel budget multipliers as JSON string or '
                             'file path, e.g. \'{"google": 1.2, "meta": 0.9}\'')
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        sys.exit(f"ERROR: model artifact not found at {args.model} — "
                 "the committed pickle/model.pkl is required.")
    with open(args.model, "rb") as f:
        model = pickle.load(f)
    print(f"Loaded model artifact v{model.get('model_version')} "
          f"(trained on {model.get('trained_on', {}).get('date_min')} -> "
          f"{model.get('trained_on', {}).get('date_max')})")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    paths_output = args.paths_output or os.path.join(out_dir, "forecast_paths.csv")

    df = pd.read_parquet(args.features)
    periods = model.get("periods", [30, 60, 90])
    n_sims = model.get("n_sims", 10_000)
    seed = model.get("seed", 42)
    sigma = model.get("sigma_by_channel", {})
    end = df["date"].max()

    print("\nValidating campaign consistency...")
    issues = validate_campaigns(df)
    for issue in issues[:10]:
        print(f"  {issue}")
    if len(issues) > 10:
        print(f"  ... and {len(issues) - 10} more")
    if not issues:
        print("  No consistency issues found.")

    rows, paths = [], []
    all_true = pd.Series(True, index=df.index)

    # --- Channel level (also feeds the blended joint simulation) ---
    print("\nForecasting by channel...")
    channel_results = {}
    for ch in sorted(df["channel"].unique()):
        mask = df["channel"] == ch
        result = forecast_group(df, mask, sigma_scale=sigma.get(ch, 1.0),
                                seed=seed, n_sims=n_sims, end=end)
        if result is None:
            print(f"  >> {ch}: insufficient history, skipped")
            continue
        channel_results[ch] = result
        rows += summarize(result, df, mask, periods, "channel", channel=ch)
        paths += path_rows(result, "channel", channel=ch)
        print(f"  >> {ch}: ok")

    # --- Blended (jointly simulated — preserves cross-channel correlation) ---
    print("Forecasting blended total (joint simulation)...")
    blended = forecast_blended(channel_results, n_sims=n_sims,
                               sigma_scale=sigma.get("blended", 1.0), seed=seed)
    if blended is not None:
        rows = summarize(blended, df, all_true, periods, "blended") + rows
        paths = path_rows(blended, "blended") + paths

    # --- Campaign-type level ---
    print("Forecasting by campaign type...")
    for ch in sorted(df["channel"].unique()):
        for ct in sorted(df[df["channel"] == ch]["campaign_type"].unique()):
            mask = (df["channel"] == ch) & (df["campaign_type"] == ct)
            result = forecast_group(df, mask, sigma_scale=sigma.get(ch, 1.0),
                                    seed=seed, n_sims=n_sims, end=end)
            if result is None:
                continue
            rows += summarize(result, df, mask, periods, "campaign_type",
                              channel=ch, campaign_type=ct)
            paths += path_rows(result, "campaign_type", channel=ch, campaign_type=ct)

    # --- Campaign level ---
    print("Forecasting by campaign...")
    n_ok, n_skipped = 0, 0
    for ch in sorted(df["channel"].unique()):
        ch_df = df[df["channel"] == ch]
        for campaign in sorted(ch_df["campaign_name"].unique()):
            mask = (df["channel"] == ch) & (df["campaign_name"] == campaign)
            result = forecast_group(df, mask, sigma_scale=sigma.get(ch, 1.0),
                                    seed=seed, n_sims=n_sims, end=end)
            if result is None:
                n_skipped += 1
                continue
            ct = df[mask]["campaign_type"].mode().iloc[0]
            rows += summarize(result, df, mask, periods, "campaign", channel=ch,
                              campaign_type=ct, campaign_name=campaign)
            n_ok += 1
    print(f"  Campaigns forecast: {n_ok}, skipped (insufficient history): {n_skipped}")

    # --- Budget scenarios (through the fitted response curves) ---
    budgets = load_budgets(args.budgets)
    if budgets:
        print(f"Applying budget scenarios: {budgets}")
        rows += budget_scenario_rows(rows, budgets, model)

    predictions = pd.DataFrame(rows)[COL_ORDER]
    predictions.to_csv(args.output, index=False)
    print(f"\nPredictions written to {args.output} — {len(predictions)} rows")

    pd.DataFrame(paths).to_csv(paths_output, index=False)
    print(f"Forecast paths written to {paths_output}")


if __name__ == "__main__":
    main()
