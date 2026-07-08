# AIgnition Forecaster — Probabilistic Revenue Forecasting

A forecasting utility for a digital marketing agency: ingests Google / Meta / Bing
campaign data, produces **probabilistic** (P10–P90 Monte-Carlo, not point-estimate)
revenue and ROAS forecasts at blended / channel / campaign-type / campaign level for
30/60/90-day horizons, simulates alternative media budgets through fitted saturation
curves, and explains anomalies via a statistically-grounded LLM layer.

## Requirements

- **Python 3.11** (developed and verified on 3.11.9)
- Dependencies pinned exactly in `requirements.txt`

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Run the scored pipeline

```bash
bash run.sh
# equivalent to:
bash run.sh ./data ./pickle/model.pkl ./output/predictions.csv
```

- Reads all three CSVs from `data/` dynamically (same schema, any row counts).
- Loads the committed trained artifact `pickle/model.pkl` (no retraining at run time).
- Writes `output/predictions.csv` fresh on every run, plus `output/forecast_paths.csv`
  (weekly forecast bands) and `output/data_health.json`.
- Fully offline: **no network calls anywhere in `run.sh`**. The LLM insight layer is a
  separate demo service (see below).

Optional 4th argument — per-channel budget multipliers:

```bash
bash run.sh ./data ./pickle/model.pkl ./output/predictions.csv '{"google": 1.2, "meta": 0.9}'
```

### Output format (`output/predictions.csv`)

One row per `(forecast_level, entity, period_days)`:

| column | meaning |
|---|---|
| `forecast_level` | `blended` \| `channel` \| `campaign_type` \| `campaign` \| `channel_budget` |
| `channel` / `campaign_type` / `campaign_name` | entity identifiers (blank where not applicable) |
| `period_days` | 30, 60 or 90 |
| `revenue_p10/p50/p90` | aggregate-window revenue percentiles from 10,000 Monte-Carlo paths |
| `roas_p10/p50/p90` | ROAS percentiles over planned spend |
| `budget_multiplier` | set on `channel_budget` scenario rows only |

## Retrain the model artifact (development time only)

```bash
python src/generate_features.py --data-dir ./data --out features.parquet
python src/train.py            # fits response curves + calibrates uncertainty via backtest
```

## Demo dashboard + AI insights (separate from the scored pipeline)

```bash
python src/anomalies.py                       # statistical anomaly detection (offline)
export ANTHROPIC_API_KEY=...                  # or OPENAI_API_KEY
python src/llm_insights.py                    # LLM interpretation of detected anomalies
streamlit run src/dashboard.py
```

## Documentation

- [docs/TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md) — methodology,
  assumptions, validation, limitations
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — components and the network-access separation
- [docs/DEMO_WALKTHROUGH.md](docs/DEMO_WALKTHROUGH.md) — presenting the product end-to-end
