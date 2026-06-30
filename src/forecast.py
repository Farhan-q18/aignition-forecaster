import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")


def prepare_series(df, value_col="revenue"):
    series = df.groupby("date")[value_col].sum().reset_index()
    series = series.sort_values("date")
    series = series.set_index("date")
    
    # Fill missing dates with 0
    full_index = pd.date_range(start=series.index.min(), 
                               end=series.index.max(), 
                               freq="D")
    series = series.reindex(full_index, fill_value=0)
    series.index.freq = "D"
    
    return series[value_col]

def run_forecast(series, periods=90):
    """Holt-Winters Exponential Smoothing with confidence intervals"""
    # Need at least 2 full seasons (14 days) of data
    if len(series) < 14:
        return None

    try:
        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add",
            seasonal_periods=7,  # weekly seasonality
            initialization_method="estimated"
        ).fit(optimized=True)

        forecast_mean = model.forecast(periods)

        # Generate prediction intervals via Monte Carlo simulation
        residuals = model.resid
        std = residuals.std() * 0.3
        

        simulations = []
        for _ in range(500):
            noise = np.random.normal(0, std, periods)
            simulations.append(forecast_mean.values + noise)

        sims = np.array(simulations)
        p10 = np.percentile(sims, 10, axis=0)
        p50 = np.percentile(sims, 50, axis=0)
        p90 = np.percentile(sims, 90, axis=0)

        forecast_df = pd.DataFrame({
            "ds": forecast_mean.index,
            "yhat": p50,
            "yhat_lower": p10,
            "yhat_upper": p90
        })

        # Clip negatives to 0
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            forecast_df[col] = forecast_df[col].clip(lower=0)

        return forecast_df

    except Exception as e:
        print(f"  Forecast error: {e}")
        return None


def forecast_channel(df, channel, periods=90):
    channel_df = df[df["channel"] == channel].copy()
    series = prepare_series(channel_df, "revenue")

    print(f"  >> {channel} ({len(series)} days of data)")
    forecast = run_forecast(series, periods)

    if forecast is not None:
        forecast["channel"] = channel
    return forecast


def forecast_campaign_type(df, channel, campaign_type, periods=90):
    filtered = df[
        (df["channel"] == channel) &
        (df["campaign_type"] == campaign_type)
    ].copy()

    series = prepare_series(filtered, "revenue")

    if len(series) < 14:
        return None

    forecast = run_forecast(series, periods)
    if forecast is not None:
        forecast["channel"] = channel
        forecast["campaign_type"] = campaign_type
    return forecast


def simulate_budget_impact(df, channel, budget_multiplier=1.2, periods=90):
    channel_df = df[df["channel"] == channel].copy()
    avg_roas = channel_df[channel_df["spend"] > 0]["roas"].mean()
    avg_roas = avg_roas if not np.isnan(avg_roas) else 1.0

    series = prepare_series(channel_df, "revenue")
    forecast = run_forecast(series, periods)

    if forecast is None:
        return None

    # Diminishing returns scaling
    scale = np.sqrt(budget_multiplier)
    forecast["yhat"] *= scale
    forecast["yhat_lower"] *= scale
    forecast["yhat_upper"] *= scale
    forecast["channel"] = channel
    forecast["budget_multiplier"] = budget_multiplier
    forecast["assumed_roas"] = round(avg_roas, 4)
    return forecast


def aggregate_forecasts(forecasts_list, periods=[30, 60, 90]):
    results = []
    for fc in forecasts_list:
        if fc is None:
            continue

        fc = fc.copy()
        fc["ds"] = pd.to_datetime(fc["ds"])
        start_date = fc["ds"].min()

        channel = fc["channel"].iloc[0] if "channel" in fc.columns else "all"
        camp_type = fc["campaign_type"].iloc[0] if "campaign_type" in fc.columns else None

        for period in periods:
            end_date = start_date + pd.Timedelta(days=period)
            window = fc[fc["ds"] <= end_date]

            row = {
                "period_days": period,
                "channel": channel,
                "revenue_p10": round(window["yhat_lower"].sum(), 2),
                "revenue_p50": round(window["yhat"].sum(), 2),
                "revenue_p90": round(window["yhat_upper"].sum(), 2),
            }
            if camp_type:
                row["campaign_type"] = camp_type

            results.append(row)

    return pd.DataFrame(results)


def forecast_campaign(df, channel, campaign_name, periods=90):
    filtered = df[
        (df["channel"] == channel) &
        (df["campaign_name"] == campaign_name)
    ].copy()

    series = prepare_series(filtered, "revenue")
    if len(series) < 14:
        return None

    forecast = run_forecast(series, periods)
    if forecast is not None:
        forecast["channel"] = channel
        forecast["campaign_name"] = campaign_name
        forecast["campaign_type"] = filtered["campaign_type"].mode().iloc[0]
    return forecast


def aggregate_campaign_forecasts(forecasts_list, periods=[30, 60, 90]):
    results = []
    for fc in forecasts_list:
        if fc is None:
            continue

        fc = fc.copy()
        fc["ds"] = pd.to_datetime(fc["ds"])
        start_date = fc["ds"].min()

        channel = fc["channel"].iloc[0]
        campaign_name = fc["campaign_name"].iloc[0]
        campaign_type = fc["campaign_type"].iloc[0] if "campaign_type" in fc.columns else None

        for period in periods:
            end_date = start_date + pd.Timedelta(days=period)
            window = fc[fc["ds"] <= end_date]
            results.append({
                "period_days": period,
                "channel": channel,
                "campaign_type": campaign_type,
                "campaign_name": campaign_name,
                "revenue_p10": round(window["yhat_lower"].sum(), 2),
                "revenue_p50": round(window["yhat"].sum(), 2),
                "revenue_p90": round(window["yhat_upper"].sum(), 2),
            })

    return pd.DataFrame(results)


def run_all_forecasts(features_path="features.parquet", periods=90):
    print("Loading features...")
    df = pd.read_parquet(features_path)

    channels = df["channel"].unique()
    all_channel_forecasts = []
    all_camptype_forecasts = []

    print("\nForecasting by channel...")
    for channel in channels:
        fc = forecast_channel(df, channel, periods=periods)
        if fc is not None:
            all_channel_forecasts.append(fc)

    print("\nForecasting by campaign type...")
    for channel in channels:
        camp_types = df[df["channel"] == channel]["campaign_type"].unique()
        for ct in camp_types:
            print(f"  >> {channel} / {ct}")
            fc = forecast_campaign_type(df, channel, ct, periods=periods)
            if fc is not None:
                all_camptype_forecasts.append(fc)

    return df, all_channel_forecasts, all_camptype_forecasts


if __name__ == "__main__":
    df, channel_fcs, camptype_fcs = run_all_forecasts()

    print("\n=== Channel Forecast Summary (30/60/90 days) ===")
    summary = aggregate_forecasts(channel_fcs)
    print(summary.to_string())

    print("\n=== Campaign Type Forecast Summary ===")
    ct_summary = aggregate_forecasts(camptype_fcs)
    print(ct_summary.to_string())