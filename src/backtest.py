"""
Walk-forward backtesting + uncertainty calibration.

For several historical cutoff dates, forecast 30/60/90 days forward using ONLY
data available up to the cutoff, then compare against realized actuals:
- MAPE / sMAPE per grain and horizon (point accuracy)
- P10-P90 interval coverage (should be ~80% if uncertainty is calibrated)

`calibrate_sigma` grid-searches a residual scale factor so that empirical
coverage hits the 80% target — the chosen factor is stored in model.pkl and
applied at prediction time. This is the honest answer to "can I trust the
band?": we widen/narrow it until it historically contains ~80% of actuals.
"""

import numpy as np
import pandas as pd

from forecast import (
    N_SIMS,
    daily_series,
    forecast_blended,
    forecast_group,
    window_percentiles,
)

CUTOFF_OFFSETS_DAYS = [270, 225, 180, 135, 90]  # counted back from data end
PERIODS = [30, 60, 90]
BT_SIMS = 4000  # fewer sims per backtest fit — enough for stable percentiles
COVERAGE_TARGET = 0.80
SIGMA_GRID = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]


def actual_window_sum(df, mask, cutoff, period_days):
    sub = df[mask & (df["date"] > cutoff) &
             (df["date"] <= cutoff + pd.Timedelta(days=period_days))]
    return float(sub["revenue"].sum())


def _series_specs(df, include_campaign_types=True):
    """(label, channel, mask) for every backtested series."""
    specs = [("blended", None, pd.Series(True, index=df.index))]
    for ch in sorted(df["channel"].unique()):
        specs.append((f"channel:{ch}", ch, df["channel"] == ch))
    if include_campaign_types:
        top = (df.groupby(["channel", "campaign_type"])["revenue"].sum()
               .sort_values(ascending=False).head(6).index)
        for ch, ct in top:
            specs.append((f"camptype:{ch}:{ct}", ch,
                          (df["channel"] == ch) & (df["campaign_type"] == ct)))
    return specs


def run_backtest(df, sigma_scale=1.0, sigma_by_channel=None,
                 include_campaign_types=True, verbose=False):
    """Walk-forward backtest. Returns a tidy DataFrame of per-cutoff results."""
    end = df["date"].max()
    specs = _series_specs(df, include_campaign_types)
    rows = []

    for offset in CUTOFF_OFFSETS_DAYS:
        cutoff = end - pd.Timedelta(days=offset)
        hist = df[df["date"] <= cutoff]
        if hist.empty:
            continue

        channel_results = {}
        for ch in sorted(hist["channel"].unique()):
            scale = (sigma_by_channel or {}).get(ch, sigma_scale)
            channel_results[ch] = forecast_group(
                hist, hist["channel"] == ch, sigma_scale=scale,
                n_sims=BT_SIMS, end=cutoff)

        for label, ch, mask in specs:
            if label == "blended":
                scale = (sigma_by_channel or {}).get("blended", sigma_scale)
                result = forecast_blended(channel_results, n_sims=BT_SIMS,
                                          sigma_scale=scale)
            elif label.startswith("channel:"):
                result = channel_results.get(ch)
            else:
                scale = (sigma_by_channel or {}).get(ch, sigma_scale)
                hist_mask = (hist["channel"] == ch) & \
                            (hist["campaign_type"] == label.split(":", 2)[2])
                result = forecast_group(hist, hist_mask, sigma_scale=scale,
                                        n_sims=BT_SIMS, end=cutoff)
            if result is None:
                continue

            for period in PERIODS:
                if offset < period:
                    continue  # not enough realized future to score
                p10, p50, p90 = window_percentiles(result["sims"], period)
                actual = actual_window_sum(df, mask, cutoff, period)
                if actual <= 0:
                    continue
                ape = abs(p50 - actual) / actual
                sape = abs(p50 - actual) / ((abs(p50) + abs(actual)) / 2)
                rows.append({
                    "series": label,
                    "cutoff": str(cutoff.date()),
                    "period_days": period,
                    "actual": round(actual, 2),
                    "p10": round(p10, 2),
                    "p50": round(p50, 2),
                    "p90": round(p90, 2),
                    "ape": round(ape, 4),
                    "sape": round(sape, 4),
                    "covered": bool(p10 <= actual <= p90),
                })
        if verbose:
            print(f"  cutoff {cutoff.date()} done")

    return pd.DataFrame(rows)


def summarize_backtest(results):
    """Aggregate to the scorecard shown in the dashboard."""
    if results.empty:
        return pd.DataFrame()
    g = results.groupby(["series", "period_days"])
    out = g.agg(
        n=("ape", "size"),
        mape=("ape", "mean"),
        smape=("sape", "mean"),
        coverage_p10_p90=("covered", "mean"),
    ).reset_index()
    for col in ["mape", "smape", "coverage_p10_p90"]:
        out[col] = out[col].round(4)
    return out


def calibrate_sigma(df, verbose=True):
    """Grid-search a per-channel residual scale so P10-P90 coverage ~= 80%.

    Channel-level only (campaign types inherit their channel's factor;
    blended gets its own). Returns dict channel -> sigma_scale.
    """
    per_scale = {}
    for scale in SIGMA_GRID:
        if verbose:
            print(f"  backtesting sigma_scale={scale}...")
        per_scale[scale] = run_backtest(df, sigma_scale=scale,
                                        include_campaign_types=False)

    chosen = {}
    for label in ["blended"] + sorted(df["channel"].unique()):
        series = "blended" if label == "blended" else f"channel:{label}"
        best_scale, best_gap = 1.0, np.inf
        for scale, res in per_scale.items():
            sub = res[res["series"] == series]
            if sub.empty:
                continue
            gap = abs(sub["covered"].mean() - COVERAGE_TARGET)
            if gap < best_gap:
                best_gap, best_scale = gap, scale
        chosen[label] = best_scale
        if verbose:
            print(f"  calibrated sigma[{label}] = {best_scale}")
    return chosen


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Walk-forward backtest")
    parser.add_argument("--features", default="features.parquet")
    parser.add_argument("--out", default="output/backtest_scorecard.csv")
    parser.add_argument("--detail-out", default="output/backtest_detail.csv")
    args = parser.parse_args()

    df = pd.read_parquet(args.features)
    print("Running walk-forward backtest "
          f"({len(CUTOFF_OFFSETS_DAYS)} cutoffs x {PERIODS} day windows)...")
    results = run_backtest(df, verbose=True)
    scorecard = summarize_backtest(results)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    results.to_csv(args.detail_out, index=False)
    scorecard.to_csv(args.out, index=False)
    print(scorecard.to_string(index=False))
    print(f"\nScorecard -> {args.out}\nDetail -> {args.detail_out}")
