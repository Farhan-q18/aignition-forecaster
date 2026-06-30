# Architecture Overview
## AIgnition Forecaster

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        run.sh (orchestrator)                    │
│                                                                 │
│  [1] generate_features.py → [2] predict.py → [3] llm_insights  │
└─────────────────────────────────────────────────────────────────┘
         ↓                          ↓                   ↓
   features.parquet          predictions.csv       insights.json
                                    ↓                   ↓
                          ┌─────────────────────────────────┐
                          │       dashboard.py (Streamlit)  │
                          └─────────────────────────────────┘
```

---

## Frontend Stack

**Framework:** Streamlit  
**Charting:** Plotly (graph_objects + express)  
**Entry point:** `src/dashboard.py`  
**Launch:** `streamlit run src/dashboard.py`

The dashboard is a single-page application with the following sections:

| Section | What it shows |
|---|---|
| Executive Summary | LLM-generated blended forecast narrative |
| Top-level Metrics | Forecasted revenue P10/P50/P90 and avg ROAS for selected period |
| Revenue Forecast by Channel | Bar chart with error bars (P10–P90 range) per channel |
| ROAS by Channel | Bar chart of forecasted ROAS per channel |
| Revenue Share | Donut chart of revenue mix across channels |
| Campaign Type Breakdown | Horizontal bar chart of forecast by campaign type, colored by channel |
| Campaign-Level Forecast | Sortable bar chart of individual campaign forecasts (112 campaigns) |
| Budget Simulator | Interactive slider that scales revenue using diminishing returns model |
| AI Channel Insights | Tabbed LLM-generated causal analysis per channel |
| Raw Data | Expandable table of the full predictions.csv |

**Controls:** Sidebar with forecast window selector (30 / 60 / 90 days) and channel multi-select filter. All charts update reactively.

---

## Backend Stack

**Language:** Python 3.x  
**Key libraries:**

| Library | Role |
|---|---|
| `pandas` | Data loading, transformation, feature engineering |
| `numpy` | Numerical operations, Monte Carlo simulation |
| `statsmodels` | Holt-Winters Exponential Smoothing model |
| `pyarrow` / `fastparquet` | Parquet file I/O for features |
| `openai` | LLM API client (GPT-4o-mini) |
| `pickle` | Model artifact serialization |

**No web server or database** — the backend is a batch pipeline that writes flat files consumed by the dashboard.

---

## Forecasting Pipeline

```
data/
├── google_ads_campaign_stats.csv
├── meta_ads_campaign_stats.csv
└── bing_campaign_stats.csv
        │
        ▼
┌─────────────────────────┐
│   generate_features.py  │  Schema normalization, spend unit conversion,
│                         │  feature engineering (ROAS, CTR, CPC, rolling
│                         │  averages, lag features), validation
└─────────────────────────┘
        │
        ▼
  features.parquet  (unified campaign-day rows for all channels)
        │
        ▼
┌─────────────────────────┐
│      predict.py         │  Orchestrates 3-level forecasting:
│                         │
│  ┌──────────────────┐   │  Level 1 — Channel (3 series)
│  │   forecast.py    │   │  Level 2 — Campaign type (11 series)
│  │                  │   │  Level 3 — Campaign (112/136 series)
│  │ Holt-Winters     │   │
│  │ Monte Carlo P10/ │   │  Each series: Holt-Winters fit →
│  │ P50/P90          │   │  500-simulation Monte Carlo →
│  └──────────────────┘   │  P10/P50/P90 for 30/60/90d windows
│                         │
│  + Budget scenarios     │  Optional: budget multiplier scaling
│  + ROAS derivation      │  with sqrt(multiplier) diminishing returns
│  + Validation           │
└─────────────────────────┘
        │
        ▼
  output/predictions.csv  (378 base rows + budget scenario rows)
  pickle/model.pkl         (model artifact / metadata)
```

### Pipeline arguments

```bash
./run.sh [DATA_DIR] [MODEL_PATH] [OUTPUT_PATH] [BUDGETS_JSON]

# Examples:
./run.sh                                          # all defaults
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv '{"google": 1.2, "meta": 0.9}'
```

`predict.py` can also be invoked directly:

```bash
python src/predict.py \
  --features features.parquet \
  --model ./pickle/model.pkl \
  --output ./output/predictions.csv \
  --budgets '{"google": 1.2, "meta": 0.8}'
```

---

## LLM Integration Workflow

```
features.parquet + predictions.csv
        │
        ▼
┌──────────────────────────────────────────────────┐
│              llm_insights.py                     │
│                                                  │
│  For each channel (google, meta, bing):          │
│  1. Compute causal drivers from features:        │
│     - Revenue, spend, ROAS trend (last 30d       │
│       vs prior 30d)                              │
│     - CPC trend, CTR trend, conversion rate      │
│  2. Extract top 3 campaigns from predictions.csv │
│  3. Extract P10/P50/P90 forecast from            │
│     predictions.csv                              │
│  4. Build structured prompt with all context     │
│  5. Call GPT-4o-mini (temp=0.4)                  │
│  6. Parse 5-section structured response          │
│                                                  │
│  + Overall blended executive summary             │
└──────────────────────────────────────────────────┘
        │
        ▼
  output/insights.json
        │
        ▼
  dashboard.py reads and displays in tabs
```

### Prompt design principles

- **Causal framing**: prompts explicitly provide trend deltas for CPC, CTR, and conversion rate so the LLM can reason about *why* ROAS moved, not just *that* it moved
- **Structured output**: each prompt specifies exactly 5 labeled sections with word/sentence constraints
- **Low temperature (0.4)**: favors consistent, data-grounded outputs over creative interpretation
- **Graceful degradation**: if `OPENAI_API_KEY` is absent, the step exits cleanly without failing the pipeline

---

## File Structure

```
aignition-forecaster/
├── run.sh                        # Pipeline orchestrator
├── features.parquet              # Generated: unified feature store
├── requirements.txt              # Python dependencies
├── data/
│   ├── google_ads_campaign_stats.csv
│   ├── meta_ads_campaign_stats.csv
│   └── bing_campaign_stats.csv
├── src/
│   ├── generate_features.py      # Step 1: ingest + feature engineering
│   ├── forecast.py               # Core forecasting library (Holt-Winters)
│   ├── predict.py                # Step 2: pipeline entry point
│   ├── llm_insights.py           # Step 3: AI causal analysis
│   ├── dashboard.py              # Streamlit frontend
│   └── utils.py                  # Shared constants and helpers
├── output/
│   └── predictions.csv           # Generated: all forecast rows
│   └── insights.json             # Generated: LLM narrative insights
├── pickle/
│   └── model.pkl                 # Generated: model artifact
└── docs/
    ├── TECHNICAL_DOCUMENTATION.md
    └── ARCHITECTURE.md
```

---

## Data Flow Summary

```
CSV files (3 channels)
    → generate_features.py
    → features.parquet  (single unified store)
    → predict.py + forecast.py
    → predictions.csv   (378 rows: channel + campaign_type + campaign levels)
    → llm_insights.py   (reads features + predictions)
    → insights.json     (4 AI narratives: overall + 3 channels)
    → dashboard.py      (reads predictions + insights, serves UI)
```
