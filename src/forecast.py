"""
Probabilistic forecast engine.

Methodology (see docs/TECHNICAL_DOCUMENTATION.md):
1. Aggregate each revenue series to WEEKLY grain — this tames the zero-inflated
   daily noise (Bing has ~85% zero-revenue days) instead of forcing a Gaussian
   onto a spike-and-zeros daily series.
2. STL decomposition (annual period on the weekly grain) into
   trend + seasonal + residual, when the series is long enough; robust linear
   (Theil-Sen) trend otherwise.
3. Trend extrapolated with a Theil-Sen fit; seasonal component repeated
   cyclically over the horizon.
4. Monte Carlo: residuals resampled via BLOCK bootstrap (preserves
   autocorrelation, unlike iid) N times and added to trend+seasonal to produce
   a full distribution of future paths.
5. Aggregate-window (30/60/90-day) sums computed PER SIMULATION, then
   P10/P50/P90 taken from the distribution of window sums — not from summing
   per-day percentiles.
6. Blended total simulated JOINTLY: the same bootstrap time-blocks are applied
   across all channels' residuals so historical cross-channel correlation is
   preserved (independent sums would give unrealistically narrow bands).

All randomness flows through a seeded numpy Generator — runs are reproducible.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

warnings.filterwarnings("ignore")

N_SIMS = 10_000
BLOCK_SIZE = 4           # weeks per bootstrap block
HORIZON_WEEKS = 14       # 14 weeks = 98 days, covers the 90-day window
TREND_WINDOW = 13        # weeks for the responsive rolling-median trend
LONG_TREND_WINDOW = 53   # weeks — smooths through the annual cycle
SLOPE_WINDOW = 26        # weeks of trend used to estimate the forward slope
SLOPE_DAMP = 0.5         # damp trend extrapolation (walk-forward validated)
MIN_WEEKS_SEASONAL = 80  # ~1.5 annual cycles before trusting a seasonal index
MIN_WEEKS_FORECAST = 16
# Mild shrink toward 1: the holiday cycle repeats at near-identical magnitude
# year over year (peak weeks ~10x baseline in BOTH 2024 and 2025), so heavy
# shrinkage systematically underpredicts Q4 AND contaminates the post-holiday
# trend (under-deseasonalized holiday weeks inflate the January trend level).
SEASONAL_SHRINK = 0.9
SEED = 42


def daily_series(df, value_col="revenue", end=None):
    """Sum a value to daily grain over the observed range (0-filled).

    `end` extends/anchors the series end so every series in a run shares the
    same weekly grid (needed to align residuals for the joint blended sim).
    """
    daily = df.groupby("date")[value_col].sum()
    end = end or daily.index.max()
    full = pd.date_range(daily.index.min(), end, freq="D")
    return daily.reindex(full, fill_value=0.0)


def weekly_series(daily):
    """Collapse a daily series into complete 7-day weeks counted back from the
    LAST date, so the final week is always complete and the forecast starts
    exactly at data-end + 1 day."""
    n_weeks = len(daily) // 7
    if n_weeks == 0:
        return pd.Series(dtype=float)
    trimmed = daily.iloc[len(daily) - n_weeks * 7:]
    values = trimmed.values.reshape(n_weeks, 7).sum(axis=1)
    week_ends = trimmed.index[6::7]
    return pd.Series(values, index=week_ends)


def _seasonal_factors(y, weeks_of_year):
    """Multiplicative week-of-year factors estimated against a long-window
    rolling-median trend (the annual window smooths THROUGH the holiday cycle
    instead of absorbing it).

    Week-of-year grain matters here: the Q4 spike concentrates in the Black
    Friday->Christmas weeks, and a month-grain index dilutes it (verified in
    walk-forward backtests). Each factor pools ratios from the target week
    +-1 week (circularly), then is shrunk toward 1 because each week is only
    observed 2-3 times.
    """
    trend_long = (pd.Series(y)
                  .rolling(LONG_TREND_WINDOW, center=True, min_periods=13)
                  .median()
                  .values)
    floor = np.percentile(y[y > 0], 10) if (y > 0).any() else 1.0
    trend_long = np.clip(trend_long, floor, None)
    ratio = np.clip(y / trend_long, 0.1, 8.0)

    factors = {}
    for w in range(1, 53):
        neighbors = [(w - 2) % 52 + 1, w, w % 52 + 1]
        vals = ratio[np.isin(weeks_of_year, neighbors)]
        if len(vals) >= 2:
            f = float(np.median(vals))
            factors[w] = 1.0 + (f - 1.0) * SEASONAL_SHRINK
        else:
            factors[w] = 1.0
    return factors


def _week_of_year(index):
    return np.minimum(index.isocalendar().week.values.astype(int), 52)


def decompose(weekly):
    """Decomposition -> (trend_forward, seasonal_factor_forward, residuals).

    The forecast is trend * seasonal_factor + bootstrapped residuals.

    Design notes:
    - With only ~2.4 annual cycles, STL's annual seasonal component overfits
      badly (it absorbs most of the noise, leaving residuals ~10x too small
      and bands miscalibrated). And this data has a large MULTIPLICATIVE
      holiday cycle (Q4 revenue is several times baseline, growing with the
      account) — an additive index estimated on 2024 undershoots 2025.
    - So: (1) estimate multiplicative month-of-year factors against a
      53-week rolling-median trend; (2) deseasonalize; (3) fit the responsive
      trend (13-week rolling median, Theil-Sen slope over the last 26 weeks,
      anchored at the final trend value) on the deseasonalized series;
      (4) residuals = y - trend * factor, on the original scale, feed the
      block bootstrap so band width reflects real week-to-week noise.
    - Day-of-week seasonality is handled implicitly by weekly aggregation.
    - Short series (< ~1.5 annual cycles): factors default to 1 (trend-only).
    """
    y = weekly.values.astype(float)
    n = len(y)
    t = np.arange(n)
    steps = np.arange(1, HORIZON_WEEKS + 1)
    woy = _week_of_year(weekly.index)

    if n >= MIN_WEEKS_SEASONAL and y.sum() > 0:
        factors = _seasonal_factors(y, woy)
    else:
        factors = {w: 1.0 for w in range(1, 53)}

    seasonal = np.array([factors[w] for w in woy])
    y_deseasonalized = y / seasonal

    trend = (pd.Series(y_deseasonalized)
             .rolling(TREND_WINDOW, center=True, min_periods=1)
             .median()
             .values)
    tail = min(n, SLOPE_WINDOW)
    slope, _, _, _ = theilslopes(trend[-tail:], t[-tail:])
    trend_fwd = np.clip(trend[-1] + SLOPE_DAMP * slope * steps, 0.0, None)

    future_ends = weekly.index[-1] + pd.to_timedelta(steps * 7, unit="D")
    seasonal_fwd = np.array([factors[w] for w in _week_of_year(future_ends)])

    resid = y - trend * seasonal
    return trend_fwd, seasonal_fwd, resid


def block_bootstrap_paths(resid, n_sims, horizon, rng, sigma_scale=1.0):
    """Sample (n_sims, horizon) residual paths via moving-block bootstrap."""
    resid = np.asarray(resid, dtype=float)
    block = min(BLOCK_SIZE, len(resid))
    n_blocks = int(np.ceil(horizon / block))
    starts = rng.integers(0, len(resid) - block + 1, size=(n_sims, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_sims, -1)
    return resid[idx[:, :horizon]] * sigma_scale


def forecast_series(weekly, n_sims=N_SIMS, sigma_scale=1.0, seed=SEED):
    """Forecast one weekly series HORIZON_WEEKS ahead.

    Returns dict with future week-end dates, the (n_sims, horizon) simulation
    matrix (clipped at 0 — revenue can't be negative), P10/50/90 paths, and the
    raw components (used by the joint blended simulation).
    """
    if len(weekly) < MIN_WEEKS_FORECAST:
        return None

    rng = np.random.default_rng(seed)
    trend_fwd, seasonal_fwd, resid = decompose(weekly)
    point = trend_fwd * seasonal_fwd

    resid_paths = block_bootstrap_paths(resid, n_sims, HORIZON_WEEKS, rng, sigma_scale)
    sims = np.clip(point[None, :] + resid_paths, 0.0, None)

    last_end = weekly.index[-1]
    future_ends = pd.date_range(last_end + pd.Timedelta(days=7),
                                periods=HORIZON_WEEKS, freq="7D")
    return {
        "future_week_ends": future_ends,
        "sims": sims,
        "p10": np.percentile(sims, 10, axis=0),
        "p50": np.percentile(sims, 50, axis=0),
        "p90": np.percentile(sims, 90, axis=0),
        "point": point,
        "resid": pd.Series(resid, index=weekly.index),
        "history": weekly,
    }


def window_weights(period_days, horizon=HORIZON_WEEKS):
    """Fraction of each forecast week falling inside the first `period_days`."""
    days_into = np.arange(horizon) * 7
    return np.clip(period_days - days_into, 0, 7) / 7.0


def window_percentiles(sims, period_days):
    """P10/P50/P90 of the `period_days`-window revenue sum across simulations."""
    w = window_weights(period_days, sims.shape[1])
    totals = sims @ w
    return (float(np.percentile(totals, 10)),
            float(np.percentile(totals, 50)),
            float(np.percentile(totals, 90)))


def forecast_blended(channel_results, n_sims=N_SIMS, sigma_scale=1.0, seed=SEED):
    """Jointly simulate the blended (all-channel) total.

    Residuals from all channels are aligned on common weeks and resampled with
    the SAME bootstrap blocks, preserving the cross-channel correlation
    structure. The blended point forecast is the sum of per-channel
    trend+seasonal components.
    """
    results = {ch: r for ch, r in channel_results.items() if r is not None}
    if not results:
        return None

    resid_df = pd.DataFrame({ch: r["resid"] for ch, r in results.items()}).dropna()
    if len(resid_df) < MIN_WEEKS_FORECAST:
        return None

    rng = np.random.default_rng(seed)
    joint_resid = resid_df.values.sum(axis=1)  # same weeks summed => correlation kept
    resid_paths = block_bootstrap_paths(joint_resid, n_sims, HORIZON_WEEKS,
                                        rng, sigma_scale)

    point = np.sum([r["point"] for r in results.values()], axis=0)
    sims = np.clip(point[None, :] + resid_paths, 0.0, None)

    # All channels end on the same date in practice; take the latest to be safe.
    last_end = max(r["history"].index[-1] for r in results.values())
    future_ends = pd.date_range(last_end + pd.Timedelta(days=7),
                                periods=HORIZON_WEEKS, freq="7D")
    history = pd.DataFrame({ch: r["history"] for ch, r in results.items()}).sum(axis=1)
    return {
        "future_week_ends": future_ends,
        "sims": sims,
        "p10": np.percentile(sims, 10, axis=0),
        "p50": np.percentile(sims, 50, axis=0),
        "p90": np.percentile(sims, 90, axis=0),
        "point": point,
        "history": history,
    }


def forecast_group(df, mask, sigma_scale=1.0, seed=SEED, n_sims=N_SIMS, end=None):
    """Convenience: weekly-aggregate and forecast the rows selected by mask.

    Pass `end` (usually the global max date of the dataset) so every series
    shares one weekly grid — required for the joint blended simulation.
    """
    sub = df[mask]
    if sub.empty:
        return None
    return forecast_series(weekly_series(daily_series(sub, end=end)),
                           n_sims=n_sims, sigma_scale=sigma_scale, seed=seed)


def recent_daily_spend(df, mask, lookback_days=28):
    """Recent average daily spend for the selected rows — used to turn revenue
    forecasts into ROAS forecasts (spend is treated as planned/deterministic)."""
    sub = df[mask]
    if sub.empty:
        return 0.0
    daily = sub.groupby("date")["spend"].sum()
    cutoff = daily.index.max() - pd.Timedelta(days=lookback_days - 1)
    recent = daily[daily.index >= cutoff]
    # Include zero-spend days within the window
    return float(recent.sum() / lookback_days)
