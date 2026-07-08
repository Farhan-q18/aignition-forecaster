# Technical Documentation

## 1. Data unification

Three per-platform CSVs (one row per campaign per day) are unified into one canonical
table before anything else; every downstream module reads only this table.

Canonical schema: `date, channel (google/meta/bing), campaign_id, campaign_name,
campaign_type, spend, revenue, clicks, impressions, conversions, daily_budget`
plus engineered features (ROAS, CTR, CPC, calendar features, rolling means, lags).

Per-source handling:

| source | key transformations |
|---|---|
| Google | `metrics_cost_micros / 1,000,000` → dollars (verified against implied CPC); `metrics_conversions_value` → revenue; channel type normalized (`PERFORMANCE_MAX` → `Performance Max`); 14 null `campaign_budget_amount` rows kept and reported in `data_health.json` |
| Meta | **no revenue or campaign-type columns** — see §1.1 and §1.2; 7 null `daily_budget` rows reported |
| Bing | clean; types normalized; 85% zero-revenue campaign-days → drives the zero-inflation design (§2.2) |

### 1.1 Meta revenue assumption (explicit, flagged)

Meta's `conversion` column is treated as **conversion value (revenue), not a count**:
its ratio to spend (median ≈ 4.1, mean ≈ 11.2) sits in the same range as Google's
`conversions_value/cost` (median ≈ 2.6) and Bing's `Revenue/Spend`. Interpreted as a
count, it would imply implausible per-conversion economics at typical AOVs.

This is a config toggle, not a buried constant:
`python src/generate_features.py --meta-revenue-mode count --meta-aov 75` switches to
"count × assumed AOV" if the value interpretation is challenged.

### 1.2 Campaign taxonomy unification (campaign-consistency deliverable)

Meta's campaign type is embedded in the campaign name (`Prospecting_DPA_Campaign_04`).
`src/taxonomy.py` parses names with ordered regex rules into a normalized taxonomy —
primary type (Prospecting / Remarketing / Generic / Advantage+) plus a Brand/DPA
sub-tag. Names matching no rule are typed `Unclassified` and surfaced in
`data_health.json` and the dashboard (for LLM/manual review in the demo layer — the
scored pipeline never calls a network). Google/Bing types are normalized to the same
vocabulary. `predict.py` additionally validates campaign consistency (inactive
campaigns, negative spend, unclassifiable types) before forecasting.

## 2. Forecasting methodology

### 2.1 Core: decomposition + block-bootstrap Monte Carlo

Each series (blended, per-channel, per-campaign-type, per-campaign) is:

1. **Aggregated to complete 7-day weeks** counted back from the last observed date
   (so the forecast starts exactly at data-end + 1 day, and all series share one
   weekly grid anchored at the global max date — required for joint simulation).
2. **Decomposed** into `trend × week-of-year seasonal factor + residual`:
   - *Seasonal factors*: multiplicative week-of-year indices estimated as the median
     ratio of the series to a 53-week rolling-median trend (the annual window smooths
     **through** the holiday cycle instead of absorbing it). Each factor pools the
     target week ±1 (robust to the ~1-week year-over-year shift in holiday timing)
     and is mildly shrunk toward 1 (factor 0.9). Applied only when the series spans
     ≥ 80 weeks. Day-of-week seasonality is handled implicitly by weekly aggregation.
   - *Trend*: 13-week centered rolling median of the **deseasonalized** series
     (robust to outlier weeks), extrapolated with a Theil-Sen slope over the last
     26 weeks, **damped ×0.5** (both choices validated by walk-forward backtest).
3. **Monte-Carlo simulated 10,000×**: residuals are resampled via **moving-block
   bootstrap** (block = 4 weeks — preserves autocorrelation, unlike iid resampling)
   and added to `trend × seasonal`; paths are clipped at 0.
4. **Aggregate-window percentiles**: 30/60/90-day revenue totals are computed **per
   simulated path** (fractional final week weighted by day count), then P10/P50/P90
   are taken from the distribution of window totals — *not* by summing per-day
   percentiles, which misstates aggregate uncertainty.

Why not STL with an annual period? With only ~2.4 observed annual cycles, STL's
seasonal component absorbs most of the noise (measured residual σ ~10× too small),
producing badly biased points and unrealistically narrow bands. The custom
decomposition was adopted after that failure mode showed up in backtests.

Why not Prophet/ARIMA out of the box? The dominant structure here is a sharp
multiplicative holiday spike (peak weeks ≈ 10× baseline, near-identical magnitude in
both observed years) over a noisy, regime-shifting baseline — the pooled-ratio
seasonal estimator plus robust trend handles this directly and transparently, and
every design choice above was accepted/rejected on walk-forward error, not aesthetics.

### 2.2 Zero-inflation

Bing has 85% zero-revenue campaign-days (median Revenue/Spend = 0); Meta 31%. Daily
Gaussian models are mis-specified for such series. Handling: forecast on **weekly
sums** (zeros mostly wash out), block-bootstrap the *empirical* residuals (no
distributional assumption), clip simulated paths at 0. Sparse series at campaign
grain are skipped below 16 weeks of history rather than fabricated.

### 2.3 Blended total: joint simulation

Channels co-move (shared seasonality and demand). Summing independently simulated
channel distributions would understate blended uncertainty. Instead, per-channel
residuals are aligned on the common weekly grid and resampled with the **same
bootstrap blocks** across channels, preserving the empirical cross-channel
correlation. Measured effect on the 30-day blended band: P10–P90 of ≈ [292k, 568k]
jointly vs ≈ [303k, 497k] under independence — the joint band is honestly wider.

### 2.4 ROAS

Near-term spend is treated as **planned** (recent 28-day average daily spend ×
window length): `roas_pXX = revenue_pXX / planned_spend`. The ROAS band therefore
reflects revenue uncertainty over a known budget — the operationally relevant
question ("what return will my planned spend earn?").

### 2.5 Budget-response curves (simulator; deliberately not an MMM)

Per channel and per (channel, campaign-type), weekly `(spend, revenue)` pairs are fit
with two candidate saturating forms via `scipy.optimize.curve_fit` — Hill
`v·s/(s+k)` and logarithmic `a·log1p(s/c)` — keeping the lower-SSE form. Curve
uncertainty comes from resampling the weekly pairs (200 bootstrap refits). A budget
multiplier `m` rescales the revenue forecast by `f(m·s₀)/f(s₀)` with a P10/P50/P90
band from the bootstrap fits; curve uncertainty compounds with forecast uncertainty
(low scale applied to P10, high to P90). Fitted curves live in `model.pkl` — they are
trained artifacts, never refit in the scored run.

Findings on this data: Google is far from saturation (2× budget → ≈1.96× revenue),
Meta mildly saturated (→ ≈1.87×), **Bing fully saturated (→ ≈1.02×)** — incremental
Bing spend buys essentially nothing.

### 2.6 Uncertainty calibration (walk-forward)

Band width is **calibrated, not assumed**: per-channel residual scale factors are
grid-searched so that empirical P10–P90 coverage on walk-forward backtests hits the
80% target (chosen: blended 2.5, google 2.5, meta 1.5, bing 1.0). Stored in
`model.pkl`, applied at prediction time.

## 3. Validation — walk-forward backtest

Five historical cutoffs (90–270 days before data end); at each, the model forecasts
30/60/90 days using only prior data; scored on MAPE, sMAPE and P10–P90 coverage.
Results ship in `output/backtest_scorecard.csv` and render as the dashboard's
**Accuracy Scorecard**.

Headline (calibrated model): blended MAPE ≈ 32% with **80% band coverage (on
target)**. Honest failure analysis: the worst windows straddle the holiday ramp —
with two observed holiday seasons, spike *timing* shifts ±1 week between years and a
cutoff placed mid-ramp misses high. Channel-level coverage is below target for Meta
(40%) — reported as-is in the scorecard rather than hidden; the blended number is the
primary decision quantity and is calibrated.

## 4. AI integration strategy

Strict grounding contract, in three layers:

1. **Statistical detection first** (`src/anomalies.py`, fully offline): robust
   z-scores (median/MAD) on decomposition residuals flag revenue outlier weeks;
   budget-cap proximity flags campaigns spending ≥90% of (or above) their stated
   daily budget; ROAS drift tests recent 4-week ROAS against the series' own history.
2. **LLM interpretation** (`src/llm_insights.py`, demo service only): each *detected*
   anomaly is sent with compact structured context (drivers, forecast bands) and the
   instruction to ground hypotheses strictly in the numbers given; output is strict
   JSON `{summary, likely_cause, confidence, recommended_action}` rendered in the UI.
   Anthropic API first (`ANTHROPIC_MODEL`, default `claude-sonnet-4-6`), OpenAI
   fallback. Channel narratives and the executive summary use the same grounding.
3. **Never in the scored path**: `run.sh` makes no network calls of any kind.

## 5. Limitations

- Response curves are correlational (observed spend↔revenue co-variation), not causal
  incrementality; a full MMM is explicitly out of scope per the brief.
- Two observed holiday cycles bound how well spike timing can be predicted; forecasts
  straddling the Q4 ramp carry the widest honest uncertainty.
- Meta revenue interpretation rests on the §1.1 assumption (toggleable).
- ROAS bands do not model spend uncertainty (documented as planned-spend by design).
- Campaign-level series shorter than 16 weeks are not forecast.
