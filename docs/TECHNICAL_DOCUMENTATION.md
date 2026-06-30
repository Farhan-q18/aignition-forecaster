# Technical Documentation
## AIgnition Forecaster — Probabilistic Revenue Forecasting

---

## 1. Forecasting Methodology

The system produces **probabilistic, aggregate-period revenue forecasts** for 30, 60, and 90-day windows at three granularity levels: channel, campaign type, and individual campaign.

### Model: Holt-Winters Exponential Smoothing

We use **additive Holt-Winters Exponential Smoothing** (also called Triple Exponential Smoothing) from the `statsmodels` library. The model captures:

- **Level** — the baseline revenue signal
- **Trend** — the additive upward or downward trajectory
- **Seasonality** — weekly patterns (period = 7 days), since paid media performance follows consistent weekday/weekend cycles

Model parameters are optimized automatically via maximum likelihood estimation (`optimized=True`). No manual tuning is required.

**Why Holt-Winters over other options:**

| Option | Reason for / against |
|---|---|
| Holt-Winters (chosen) | Interpretable, handles trend + weekly seasonality, no external dependencies, fast |
| Facebook Prophet | More robust to holidays and structural breaks, but heavier dependency and slower |
| ML models (XGBoost, LSTM) | Higher capacity but require far more data per campaign; most campaigns have <365 days |
| ARIMA/SARIMA | Similar capability but manual order selection; Holt-Winters is simpler with equivalent performance on short horizons |

### Probabilistic Confidence Intervals via Monte Carlo Simulation

Rather than using analytical prediction intervals (which assume normality), we simulate uncertainty:

1. Fit the Holt-Winters model and extract model residuals
2. Compute residual standard deviation, scaled by a conservative factor (0.3) to avoid overconfident intervals
3. Run 500 Monte Carlo simulations: each simulation adds Gaussian noise drawn from `N(0, σ)` to the point forecast
4. Report P10, P50, and P90 percentiles across the 500 simulations as the forecast range

Forecasted values are clipped to zero to prevent negative revenue predictions.

### Budget Impact Simulation

Budget scenarios use a **square-root (diminishing returns) scaling**:

```
scaled_revenue = base_revenue × sqrt(budget_multiplier)
```

This models the empirical observation that doubling spend rarely doubles revenue — efficiency typically declines as budgets scale up.

---

## 2. Data Preprocessing

### Source Datasets

| Channel | File | Rows | Date range | Campaigns |
|---|---|---|---|---|
| Google Ads | `google_ads_campaign_stats.csv` | 19,272 | Jan 2024 – Jun 2026 | 92 |
| Meta Ads | `meta_ads_campaign_stats.csv` | 3,417 | May 2024 – Jun 2026 | 16 |
| Microsoft Bing Ads | `bing_campaign_stats.csv` | 2,873 | May 2024 – Jun 2026 | 28 |

### Schema Normalization

Each source has different column names and conventions. The `generate_features.py` module normalizes all three into a unified schema:

| Unified column | Google source | Meta source | Bing source |
|---|---|---|---|
| `date` | `segments_date` | `date_start` | `TimePeriod` |
| `revenue` | `metrics_conversions_value` | `conversion` | `Revenue` |
| `spend` | `metrics_cost_micros / 1,000,000` | `spend` | `Spend` |
| `clicks` | `metrics_clicks` | `clicks` | `Clicks` |
| `impressions` | `metrics_impressions` | `impressions` | `Impressions` |
| `conversions` | `metrics_conversions` | `0` (not reported) | `Conversions` |
| `campaign_type` | `campaign_advertising_channel_type` | `Paid_Social` (fixed) | `CampaignType` |

**Google spend conversion:** Google reports spend in micros (millionths of a dollar). This is converted: `spend = metrics_cost_micros / 1,000,000`.

**Meta conversions:** Meta does not report a conversions count column; `conversions` is set to 0 for Meta rows. Revenue is taken directly from the `conversion` column (which represents conversion value, not count).

### Feature Engineering

After normalization, the following features are derived:

| Feature | Formula |
|---|---|
| `roas` | `revenue / spend` (0 if spend = 0) |
| `ctr` | `clicks / impressions` (0 if impressions = 0) |
| `cpc` | `spend / clicks` (0 if clicks = 0) |
| `day_of_week` | 0 = Monday, 6 = Sunday |
| `is_weekend` | 1 if day_of_week >= 5 |
| `{col}_7d_avg` | 7-day rolling mean per campaign (revenue, spend, roas, clicks) |
| `{col}_lag1` | 1-day lag per campaign (revenue, spend, roas) |
| `{col}_lag7` | 7-day lag per campaign (revenue, spend, roas) |

Missing values from lag/rolling features at the start of each campaign's history are filled with 0.

### Time Series Preparation for Forecasting

Before fitting the model, each channel/campaign subset is:
1. Grouped by date and summed (aggregating across campaigns within the level)
2. Sorted chronologically
3. Reindexed to a continuous daily index with missing dates filled as 0
4. Required to have at least 14 days of data (two full weekly cycles) — series shorter than this are skipped

### Campaign Consistency Validation

On each pipeline run, campaigns are checked for:
- **Zero activity**: campaigns where both total revenue and total spend are 0 (likely inactive or misconfigured)
- **Negative spend**: data quality flag for anomalous records

Validation issues are printed to console but do not halt the pipeline.

---

## 3. ROAS Forecasting

The model does not directly forecast spend — only revenue. ROAS forecasts are derived as:

```
estimated_spend = historical_daily_avg_spend × forecast_period_days
roas_p{n} = revenue_p{n} / estimated_spend
```

This is computed at the appropriate granularity for each row: channel-level rows use channel-level historical spend, campaign-type rows use campaign-type spend, and campaign rows use campaign-level spend.

**Assumption:** future spend closely follows historical average daily spend. This is a simplifying assumption; actual spend may vary with budget changes.

---

## 4. Assumptions and Limitations

### Assumptions

- **Attribution is correct**: channel-level revenue attribution from the platform reporting APIs is treated as the source of truth. No cross-channel attribution correction is applied.
- **Stationarity in trend and seasonality**: Holt-Winters assumes the trend and seasonal patterns observed historically will continue. Structural breaks (new campaigns, major budget changes, market events) are not modeled.
- **Spend continuity**: future spend is assumed to match historical daily averages unless a budget multiplier is explicitly provided.
- **Weekly seasonality is dominant**: the model uses `seasonal_periods=7`. Annual or monthly seasonality is not modeled due to limited data history on some channels.
- **No cross-channel interaction effects**: channels are forecast independently. Cannibalization or complementary effects between channels are not captured.

### Limitations

- Campaigns with fewer than 14 days of data are excluded from forecasting (24 of 136 campaigns).
- The Monte Carlo confidence intervals assume Gaussian residuals, which may not hold for channels with heavy-tailed revenue distributions.
- The ROAS forecast is derived from revenue forecasts and historical spend averages — it does not independently model spend dynamics.
- Media Mix Modeling (MMM) and custom attribution are explicitly out of scope per the challenge constraints.
- The `seasonal_periods=7` parameter may be suboptimal for channels with strong monthly or promotional cycles.

---

## 5. AI Integration Strategy

### LLM Role

OpenAI GPT-4o-mini is used as a **causal reasoning and interpretation layer** — not as a forecasting component. The statistical model produces the numbers; the LLM explains what they mean and why.

### What the LLM receives

For each channel, the prompt includes:
- Revenue, spend, ROAS, CPC, CTR, and conversion rate for the last 30 days versus the prior 30 days, with computed percentage changes
- Top 3 campaigns by 30-day revenue forecast (name, type, revenue, ROAS)
- P10/P50/P90 revenue forecasts for 30, 60, and 90 days
- ROAS forecast range (P10–P90)

### What the LLM is asked to produce

The prompt enforces a structured 5-section output:
1. **Performance Summary** — what happened and the primary causal driver
2. **Key Drivers** — the 1-2 metrics most responsible (with explicit causal language)
3. **Forecast Outlook** — what the P10–P90 spread signals about confidence
4. **Budget Recommendation** — increase / hold / decrease with reasoning
5. **Risk Flags** — one specific metric to watch

Temperature is set to 0.4 (lower than default) to produce consistent, data-grounded outputs rather than creative interpretations.

### Causal Reasoning Design

The prompts are structured to elicit causal chains rather than narrative summaries. For example, the prompt explicitly instructs: *"Cite specific numbers and causal chains (e.g. CPC rose X% which compressed ROAS by Y%)"*. The LLM is given pre-computed trend deltas for CPC, CTR, and conversion rate — the specific metrics that explain ROAS movement — so it can reason about cause and effect rather than just reporting numbers.

### Graceful Degradation

If no `OPENAI_API_KEY` is set, the LLM insights step is skipped without failing the pipeline. The dashboard handles a missing `insights.json` by showing a fallback message.
