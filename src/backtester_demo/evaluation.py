"""Forecast evaluation, including tests that expose a "lag-1 copy" forecaster.

All inputs are log-vol series aligned on origin date t (forecast of t+1..t+h).

Key diagnostics beyond RMSE:
- skill_vs_naive : 1 - MSE(model) / MSE(naive). <= 0 means no value added.
- change_corr    : corr(forecast - naive, target - naive). Does the model
                   predict the *change* in vol? A lagged copy scores ~0 here
                   even when its level RMSE looks fine.
- Diebold-Mariano: is the loss difference vs a benchmark significant
                   (HAC variance, Harvey et al. small-sample correction)?
- lag_profile    : corr(forecast_t, realised_{t+k}) for k in [-K, K]. A
                   persistence-like forecast peaks at k <= 0 (it tracks the
                   past); real predictive content shifts mass to k > 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _align(*series: pd.Series) -> list[pd.Series]:
    df = pd.concat(series, axis=1).dropna()
    return [df.iloc[:, i] for i in range(df.shape[1])]


def qlike(forecast_lv: pd.Series, target_lv: pd.Series) -> pd.Series:
    """Patton (2011) QLIKE on variances, normalised so a perfect forecast scores 0."""
    ratio = np.exp(2 * (target_lv - forecast_lv))
    return ratio - np.log(ratio) - 1


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, h: int) -> tuple[float, float]:
    """DM statistic for mean(loss_a - loss_b); negative => a is better. Two-sided p."""
    d = (loss_a - loss_b).dropna().to_numpy()
    n = len(d)
    d_c = d - d.mean()
    lag = max(h - 1, 0)
    gamma = [d_c @ d_c / n] + [d_c[k:] @ d_c[:-k] / n for k in range(1, lag + 1)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    dm = d.mean() / np.sqrt(max(var, 1e-18))
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(dm), df=n - 1)
    return float(dm), float(p)


def mincer_zarnowitz(forecast: pd.Series, target: pd.Series) -> dict:
    """target = a + b * forecast; unbiased & efficient forecast has a=0, b=1."""
    f, y = _align(forecast, target)
    b, a = np.polyfit(f, y, 1)
    r2 = np.corrcoef(f, y)[0, 1] ** 2
    return {"mz_a": a, "mz_b": b, "mz_r2": r2}


def forecast_metrics(forecast: pd.Series, target: pd.Series, naive: pd.Series,
                     benchmark: pd.Series, h: int) -> dict:
    f, y, n, bm = _align(forecast, target, naive, benchmark)
    se = (f - y) ** 2
    se_naive = (n - y) ** 2
    ql, ql_bm = qlike(f, y), qlike(bm, y)
    dm, p = diebold_mariano(ql, ql_bm, h)
    return {
        "rmse": float(np.sqrt(se.mean())),
        "qlike": float(ql.mean()),
        "skill_vs_naive": float(1 - se.mean() / se_naive.mean()),
        "change_corr": float(np.corrcoef(f - n, y - n)[0, 1]),
        "dm_vs_har": dm,
        "dm_p": p,
        **mincer_zarnowitz(f, y),
    }


def lag_profile(forecast: pd.Series, realised_daily: pd.Series, max_lag: int = 5) -> pd.Series:
    """corr(forecast_t, realised_{t+k}). realised_daily is the 1-day log vol series."""
    out = {}
    for k in range(-max_lag, max_lag + 1):
        f, r = _align(forecast, realised_daily.shift(-k))
        out[k] = np.corrcoef(f, r)[0, 1]
    return pd.Series(out, name=forecast.name)
