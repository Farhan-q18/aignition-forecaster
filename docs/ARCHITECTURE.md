# Architecture Overview

## Two artifacts, deliberately separated

1. **Scored pipeline** (`run.sh`) — deterministic, offline, zero-interaction. This is
   what the automated grader runs against held-out data.
2. **Demo product** (dashboard + AI layer) — reads the pipeline's outputs; this is
   where network calls (LLM API) live.

```
                     SCORED PIPELINE (run.sh — no network, no prompts)
 ┌────────────────────────────────────────────────────────────────────────┐
 │  data/*.csv                                                            │
 │     │  generate_features.py  (unify schemas, taxonomy, data health)    │
 │     ▼                                                                  │
 │  features.parquet ──► predict.py ◄── pickle/model.pkl (committed,      │
 │                          │            loaded — curves, calibration,    │
 │                          │            config; trained by train.py at   │
 │                          │            DEV time only)                   │
 │                          ▼                                             │
 │   output/predictions.csv · forecast_paths.csv · data_health.json       │
 └────────────────────────────────────────────────────────────────────────┘

                     DEMO LAYER (separate processes; network allowed)
 ┌────────────────────────────────────────────────────────────────────────┐
 │  anomalies.py (offline stats) ──► output/anomalies.json                │
 │  llm_insights.py (Anthropic/OpenAI API) ──► output/insights.json       │
 │  dashboard.py (Streamlit + Plotly) reads ALL outputs + model.pkl       │
 └────────────────────────────────────────────────────────────────────────┘
```

## Components

| component | role |
|---|---|
| `src/taxonomy.py` | cross-platform campaign-type unifier (rule-based parser for Meta names, normalizers for Google/Bing) |
| `src/generate_features.py` | CSV → canonical table → engineered features → `features.parquet` + `data_health.json` |
| `src/forecast.py` | forecast engine: weekly aggregation, robust decomposition, block-bootstrap Monte Carlo, joint blended simulation, window percentiles |
| `src/response_curves.py` | saturating budget-response curves (Hill / log, `curve_fit` + bootstrap CI) |
| `src/backtest.py` | walk-forward backtesting (MAPE/sMAPE/coverage) + sigma calibration grid search |
| `src/train.py` | **dev-time** trainer: fits curves, calibrates sigma, writes `pickle/model.pkl` + backtest scorecard |
| `src/predict.py` | **scored** entry: loads `model.pkl`, validates campaign consistency, forecasts all levels, applies budget scenarios, writes outputs |
| `src/anomalies.py` | offline statistical anomaly detection (residual z-scores, budget caps, ROAS drift) |
| `src/llm_insights.py` | LLM causal-interpretation layer (Anthropic first, OpenAI fallback), structured JSON in/out |
| `src/dashboard.py` | Streamlit demo: overview, drill-down, budget simulator, accuracy scorecard, AI insights, data health/methodology; dark/light theme |

## The network-access tension, resolved explicitly

The brief requires an LLM-based causal layer **and** a scored pipeline with no network
access. Resolution: `run.sh` runs only feature generation + prediction (steps [1/2]
and [2/2]); anomaly *detection* is statistical and offline; anomaly *interpretation*
(`llm_insights.py`) is invoked separately for the demo and gracefully no-ops without
an API key. This separation is by design, not an oversight.

## Model artifact contract

`pickle/model.pkl` is committed and **loaded** (never written) by the scored run. It
contains plain dicts only (no fitted library objects → robust to library-version
drift): response-curve parameters + bootstrap parameter sets, per-channel calibrated
sigma factors, simulation config (seed 42, 10,000 sims, horizon), and training-data
metadata. `predict.py` fails loudly if it is missing.

## Reproducibility

- All randomness flows through seeded `numpy.random.default_rng` generators.
- `run.sh` uses `set -euo pipefail`; relative paths only; reads whatever files are in
  `data/` with the expected schemas (no row-count or date-range assumptions).
- Exact dependency pins in `requirements.txt`; Python 3.11.

## Frontend stack

Streamlit + Plotly (`graph_objects`), single file, six pages via sidebar navigation.
Dark and light themes are hand-defined palettes (near-black + amber accent for dark;
warm paper white for light) applied through injected CSS variables and matching
Plotly layout colors — toggled at runtime, defaulting to dark
(`.streamlit/config.toml`). Uncertainty is always drawn as a shaded P10–P90 band
around a dashed median line, joined continuously to the actuals line.
