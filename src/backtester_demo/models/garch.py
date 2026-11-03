"""Walk-forward GARCH-family forecasts with no look-ahead.

Every `refit_every` days (position i) the model is estimated on returns
[0, i) only. The fitted parameters are then frozen and the variance recursion
is run forward over the next block, so the forecast made at origin t uses
returns up to and including r_t and nothing later. To guarantee that, the
model object is built on returns truncated at the end of the block.

GARCH models close-to-close returns, so its raw forecast lives on the
close-to-close scale. `calibrate_to` maps it onto any other estimator with a
Mincer-Zarnowitz style regression fitted on the validation period only.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model

from backtester_demo import config
from backtester_demo.splits import first_position, label_inside_split, split_labels


def log_returns_pct(df: pd.DataFrame) -> pd.Series:
    return (100 * np.log(df["Close"] / df["Close"].shift(1))).dropna()


def garch_walk_forward(df: pd.DataFrame, spec: dict,
                       refit_every: int = config.GARCH_REFIT_EVERY) -> pd.DataFrame:
    """Forecast log sqrt(mean daily variance over t+1..t+h) for every origin t from val on."""
    r = log_returns_pct(df)
    max_h = max(config.HORIZONS)
    start = first_position(r.index, "val")
    blocks = []
    for i in range(start, len(r), refit_every):
        end = min(i + refit_every, len(r))
        model = arch_model(r.iloc[:end], mean="Constant", vol="GARCH", rescale=False, **spec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit(last_obs=i, disp="off")
            # Origins i-1 .. end-2 forecast days i .. end-1 onward; origin end-1 is
            # covered by the next block's first origin (i_next - 1).
            fc = res.forecast(start=i - 1, horizon=max_h, reindex=False)
        blocks.append(fc.variance if end == len(r) else fc.variance.iloc[: end - i])
    var_pct = pd.concat(blocks)
    var_pct = var_pct[~var_pct.index.duplicated(keep="first")]
    out = {}
    for h in config.HORIZONS:
        mean_var = var_pct.iloc[:, :h].mean(axis=1) / 100**2
        out[h] = 0.5 * np.log(mean_var)
    return pd.DataFrame(out).reindex(df.index)


def calibrate_to(raw: pd.DataFrame, var: pd.Series) -> pd.DataFrame:
    """Map a close-to-close vol forecast onto estimator `var` using val-period OLS.

    Fit rows: val dates whose label is fully inside val. Applied to all dates;
    only test-period values are ever evaluated or traded.
    """
    from backtester_demo.volatility import forward_log_vol

    out = {}
    val = split_labels(var.index) == "val"
    for h in config.HORIZONS:
        y = forward_log_vol(var, h)
        rows = val & label_inside_split(var.index, h) & raw[h].notna() & y.notna()
        b, a = np.polyfit(raw.loc[rows, h], y[rows], 1)
        out[h] = a + b * raw[h]
    return pd.DataFrame(out)
