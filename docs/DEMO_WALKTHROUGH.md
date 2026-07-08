# Demo Walkthrough

The order a judge would want to see it — each step answers a real question a business
owner has, and each maps to a graded deliverable.

## 0. Setup (before the demo)

```bash
bash run.sh                                        # scored pipeline: features + forecasts
python src/anomalies.py                            # offline anomaly detection
ANTHROPIC_API_KEY=... python src/llm_insights.py   # LLM interpretation (or OPENAI_API_KEY)
streamlit run src/dashboard.py
```

## 1. Data ingestion — "three platforms, three vocabularies, one clean view"

Open **Data Health & Methodology**.

- Show the per-source health cards: 25.5k rows, 136 campaigns, null budgets counted
  (not silently dropped), Bing's 85% zero-revenue days called out.
- Show the **before/after taxonomy table**: raw Meta names like
  `Prospecting_DPA_Campaign_04` on the left, the normalized Prospecting/Remarketing/
  Generic/Advantage+ taxonomy on the right. Say: *"Google and Bing ship a type column;
  Meta buries it in the name. Every agency has this mess — we parse it, and anything
  unparseable is flagged for review, never guessed."*
- Point at the **explicit assumptions** list — especially Meta's `conversion` column
  treated as revenue, with the documented toggle. This preempts the judges' likeliest
  data question.

## 2. Forecast generation — "where is my revenue heading?"

Open **Overview**.

- Headline metrics: expected revenue, P10 ("if things go badly"), P90 ("if things go
  well"). Say: *"Every number here is a probability band from 10,000 Monte-Carlo
  simulations, not a single guess."*
- The band chart: two years of weekly actuals — point at the two near-identical
  holiday spikes — then the shaded P10–P90 band. Toggle 30/60/90 days in the sidebar.
- Channel contribution: who carries the number, with error bars.

Open **Drill-Down**: pick Google → campaign types → campaigns table ranked by
forecast ROAS, with scale/pause candidate lists. *"Which of my 92 Google campaigns
deserve more money? This is the answer, with uncertainty attached."*

## 3. Budget simulation — the decision tool

Open **Budget Simulator**.

- Frame it: *"I have \$50k left this quarter — where should it go?"*
- Drag Google to +50%: revenue responds nearly linearly (curve has headroom).
- Drag Bing to +100%: revenue barely moves. Show the response-curve chart — Bing's
  curve is flat against the dotted "linear" line. Say: *"Doubling Bing's budget buys
  ~2% more revenue — this channel is saturated. That insight alone pays for the tool."*
- Point at the marginal-ROAS column: the per-channel answer to "where does the next
  dollar go?"

## 4. Trust — the accuracy scorecard

Open **Accuracy Scorecard**.

- *"Can I actually trust this?"* — the model was rolled back to five historical dates
  and re-forecast blind. Blended 30/60/90-day error and the 80% band coverage are
  measured, not asserted.
- Be candid about the hard cases (holiday-ramp timing with only two observed
  seasons) — honesty here scores better than hiding it.

## 5. AI insights — grounded, not generated

Open **AI Insights & Risks**.

- Executive summary + per-channel causal narratives.
- Expand an anomaly card: *statistics* flagged the week (robust z-score on
  decomposition residuals), the LLM only interpreted it — summary, likely cause,
  confidence, recommended action, with the statistical basis printed underneath.
- Operational flags: campaigns spending at/above their stated daily caps (a real
  scaling constraint the forecast can't fix), ROAS drift alerts.
- If asked "does the graded pipeline call an LLM?" — no: `run.sh` is fully offline;
  this layer is a separate demo service (see ARCHITECTURE.md).

## 6. Close

Toggle light mode to show the theme is deliberate. Recap: one clean canonical
dataset → calibrated probabilistic forecasts → a budget decision tool → measured
trust → grounded AI narrative. Product, not a chart.
