"""
Budget-response (saturation) curves for the budget simulator.

Scope note (per the hackathon brief): this is deliberately a simple
response-curve layer fitted on historical (spend, revenue) pairs — NOT a media
mix model. It answers "if I move next period's budget on channel X by +-Y%,
what happens to revenue?" with diminishing returns and a confidence band.

Approach:
- Weekly (spend, revenue) pairs per channel and per (channel, campaign_type).
- Two candidate saturating forms fitted with scipy.optimize.curve_fit:
    hill: revenue = v * spend / (spend + k)        (plateaus at v)
    log:  revenue = a * log1p(spend / c)           (keeps growing, slowly)
  The one with lower SSE wins.
- Uncertainty: bootstrap the weekly points (resample with replacement, refit
  B times) so the *curve itself* has a confidence band, which propagates into
  the simulated revenue-scale factor.
- Applying a scenario: at budget multiplier m, the revenue scale is
  f(m * s0) / f(s0) evaluated per bootstrap fit -> P10/P50/P90 scale factors.
  A multiplier of 1.0 is always scale 1.0 (the curve reshapes the forecast,
  it doesn't replace it).
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

B_BOOTSTRAP = 200
SEED = 42


def hill(s, v, k):
    return v * s / (s + k)


def log_response(s, a, c):
    return a * np.log1p(s / c)


FORMS = {
    "hill": (hill, lambda s, y: ([max(y.max() * 2, 1.0), max(np.median(s), 1.0)],
                                 ([0, 1e-6], [np.inf, np.inf]))),
    "log": (log_response, lambda s, y: ([max(y.max() / 5, 1.0), max(np.median(s), 1.0)],
                                        ([0, 1e-6], [np.inf, np.inf]))),
}


def weekly_pairs(df, mask):
    """Weekly (spend, revenue) pairs for the selected rows."""
    sub = df[mask]
    if sub.empty:
        return None
    wk = sub.groupby(pd.Grouper(key="date", freq="W"))[["spend", "revenue"]].sum()
    wk = wk[(wk["spend"] > 0)]
    return wk if len(wk) >= 12 else None


def _fit_one(s, y):
    """Fit both candidate forms, return (form_name, params, sse) of the best."""
    best = None
    for name, (fn, p0_fn) in FORMS.items():
        p0, bounds = p0_fn(s, y)
        try:
            params, _ = curve_fit(fn, s, y, p0=p0, bounds=bounds, maxfev=5000)
            sse = float(np.sum((fn(s, *params) - y) ** 2))
            if best is None or sse < best[2]:
                best = (name, list(map(float, params)), sse)
        except Exception:
            continue
    return best


def fit_curve(df, mask, seed=SEED):
    """Fit a saturating response curve with bootstrap parameter uncertainty.

    Returns a plain-dict artifact (pickle-friendly, no scipy objects):
    {form, params, boot_params, base_weekly_spend, n_points} or None.
    """
    wk = weekly_pairs(df, mask)
    if wk is None:
        return None

    s, y = wk["spend"].values, wk["revenue"].values
    best = _fit_one(s, y)
    if best is None:
        return None
    form, params, _ = best

    rng = np.random.default_rng(seed)
    boot_params = []
    n = len(s)
    for _ in range(B_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        fit = _fit_one(s[idx], y[idx])
        if fit is not None and fit[0] == form:
            boot_params.append(fit[1])

    return {
        "form": form,
        "params": params,
        "boot_params": boot_params,
        "base_weekly_spend": float(np.median(s)),
        "n_points": int(n),
    }


def _evaluate(form, params, s):
    fn = FORMS[form][0]
    return fn(np.asarray(s, dtype=float), *params)


def scale_factors(curve, multiplier):
    """Revenue scale factor for a spend multiplier, with a bootstrap band.

    Returns (p10, p50, p90) scale factors. Falls back to sqrt(multiplier)
    (generic diminishing returns) when no curve is available.
    """
    if multiplier == 1.0:
        return 1.0, 1.0, 1.0
    if curve is None:
        s = float(np.sqrt(max(multiplier, 0.0)))
        return s, s, s

    s0 = curve["base_weekly_spend"]
    param_sets = curve["boot_params"] if curve["boot_params"] else [curve["params"]]
    ratios = []
    for p in param_sets:
        base = _evaluate(curve["form"], p, [s0])[0]
        new = _evaluate(curve["form"], p, [s0 * multiplier])[0]
        if base > 0:
            ratios.append(new / base)
    if not ratios:
        s = float(np.sqrt(max(multiplier, 0.0)))
        return s, s, s
    ratios = np.array(ratios)
    return (float(np.percentile(ratios, 10)),
            float(np.percentile(ratios, 50)),
            float(np.percentile(ratios, 90)))


def curve_points(curve, multipliers=None):
    """Evaluate the median curve over a range of budget multipliers — used by
    the dashboard to draw the response curve with its band."""
    if curve is None:
        return None
    if multipliers is None:
        multipliers = np.linspace(0.25, 2.5, 46)
    rows = []
    for m in multipliers:
        p10, p50, p90 = scale_factors(curve, float(m))
        rows.append({"multiplier": float(m), "scale_p10": p10,
                     "scale_p50": p50, "scale_p90": p90})
    return pd.DataFrame(rows)
