"""
Training entry point — produces the committed pickle/model.pkl artifact.

This runs at DEVELOPMENT time only (never inside run.sh). It fits everything
that should NOT be re-estimated on held-out test data:
- budget response curves per channel and per (channel, campaign_type),
  with bootstrap parameter uncertainty;
- per-channel residual scale factors (sigma) calibrated by walk-forward
  backtesting so the P10-P90 band historically covers ~80% of actuals;
- the forecast configuration (horizon, simulation count, seed, taxonomy
  version, the Meta revenue-column interpretation).

At prediction time, predict.py LOADS this artifact and applies the stored
curves/calibration to whatever data is in data/ — the decomposition itself is
(re)estimated on the provided history, which is inherent to forecasting, but
no curve fitting or calibration happens in the scored path.

Usage:
    python src/generate_features.py --data-dir ./data --out features.parquet
    python src/train.py --features features.parquet --model ./pickle/model.pkl
"""

import argparse
import os
import pickle
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backtest import calibrate_sigma, run_backtest, summarize_backtest
from forecast import HORIZON_WEEKS, N_SIMS, SEED
from response_curves import fit_curve

MODEL_VERSION = 2


def fit_all_curves(df):
    curves = {}
    print("Fitting budget response curves...")
    for ch in sorted(df["channel"].unique()):
        curve = fit_curve(df, df["channel"] == ch)
        curves[f"channel::{ch}"] = curve
        print(f"  channel::{ch}: {'ok (' + curve['form'] + ')' if curve else 'insufficient data'}")
        for ct in sorted(df[df["channel"] == ch]["campaign_type"].unique()):
            mask = (df["channel"] == ch) & (df["campaign_type"] == ct)
            curve = fit_curve(df, mask)
            if curve is not None:
                curves[f"camptype::{ch}::{ct}"] = curve
                print(f"  camptype::{ch}::{ct}: ok ({curve['form']})")
    return curves


def main():
    parser = argparse.ArgumentParser(description="Train the forecaster artifact")
    parser.add_argument("--features", default="features.parquet")
    parser.add_argument("--model", default="./pickle/model.pkl")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Reuse sigma=1.0 everywhere (faster dev loop)")
    parser.add_argument("--backtest-out", default="scorecard/backtest_scorecard.csv")
    parser.add_argument("--backtest-detail-out", default="scorecard/backtest_detail.csv")
    args = parser.parse_args()

    df = pd.read_parquet(args.features)

    curves = fit_all_curves(df)

    if args.skip_calibration:
        sigma = {}
    else:
        print("Calibrating uncertainty via walk-forward backtest...")
        sigma = calibrate_sigma(df)

    print("Running final walk-forward backtest for the accuracy scorecard...")
    results = run_backtest(df, sigma_by_channel=sigma)
    scorecard = summarize_backtest(results)
    os.makedirs(os.path.dirname(os.path.abspath(args.backtest_out)), exist_ok=True)
    results.to_csv(args.backtest_detail_out, index=False)
    scorecard.to_csv(args.backtest_out, index=False)

    artifact = {
        "model_version": MODEL_VERSION,
        "seed": SEED,
        "n_sims": N_SIMS,
        "horizon_weeks": HORIZON_WEEKS,
        "periods": [30, 60, 90],
        "sigma_by_channel": sigma,
        "response_curves": curves,
        "meta_revenue_mode": "value",  # `conversion` treated as revenue — see docs
        "trained_on": {
            "rows": int(len(df)),
            "date_min": str(df["date"].min().date()),
            "date_max": str(df["date"].max().date()),
            "channels": sorted(df["channel"].unique().tolist()),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.model)), exist_ok=True)
    with open(args.model, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\nModel artifact saved to {args.model}")
    print(f"  curves: {sum(1 for c in curves.values() if c)} fitted")
    print(f"  sigma calibration: {sigma or 'default (1.0)'}")


if __name__ == "__main__":
    main()
