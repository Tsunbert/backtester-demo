"""Naive persistence and HAR-RV baselines (log space).

All forecasts are indexed by origin date t: the value at t is the forecast of
forward_log_vol(var, h) made with information up to the close of t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester_demo import config
from backtester_demo.splits import first_position, label_inside_split, split_labels
from backtester_demo.volatility import forward_log_vol, trailing_log_vol


def naive(var: pd.Series) -> pd.DataFrame:
    """Tomorrow (and every later day) looks like today: log sqrt(v_t)."""
    lv = 0.5 * np.log(var)
    return pd.DataFrame({h: lv for h in config.HORIZONS})


def har_regressors(var: pd.Series) -> pd.DataFrame:
    """Corsi (2009) HAR components in log space: daily, weekly, monthly."""
    return pd.DataFrame({
        "const": 1.0,
        "d": trailing_log_vol(var, 1),
        "w": trailing_log_vol(var, 5),
        "m": trailing_log_vol(var, 22),
    })


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def har_static(var: pd.Series) -> pd.DataFrame:
    """HAR fitted once on embargoed train rows and applied to every date.

    Used as the LSTM anchor: the LSTM learns the residual on top of it.
    """
    X = har_regressors(var)
    out = {}
    for h in config.HORIZONS:
        y = forward_log_vol(var, h)
        fit_rows = (split_labels(var.index) == "train") & label_inside_split(var.index, h)
        fit_rows &= X.notna().all(axis=1) & y.notna()
        beta = _ols(X[fit_rows].to_numpy(), y[fit_rows].to_numpy())
        out[h] = X.to_numpy() @ beta
    return pd.DataFrame(out, index=var.index)


def har_walk_forward(var: pd.Series, refit_every: int = config.GARCH_REFIT_EVERY) -> pd.DataFrame:
    """Expanding-window HAR, refit every `refit_every` days from the start of val.

    At a refit at position i only rows j with j + h <= i are used, i.e. labels
    fully observed by the close of day i.
    """
    X = har_regressors(var)
    Xv = X.to_numpy()
    valid_x = X.notna().all(axis=1).to_numpy()
    start = first_position(var.index, "val")
    out = pd.DataFrame(np.nan, index=var.index, columns=list(config.HORIZONS))
    for h in config.HORIZONS:
        y = forward_log_vol(var, h).to_numpy()
        for i in range(start, len(var), refit_every):
            rows = np.arange(i - h + 1)
            rows = rows[valid_x[rows] & ~np.isnan(y[rows])]
            beta = _ols(Xv[rows], y[rows])
            seg = slice(i, min(i + refit_every, len(var)))
            out.iloc[seg, out.columns.get_loc(h)] = Xv[seg] @ beta
    return out
