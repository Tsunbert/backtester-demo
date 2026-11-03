"""Daily realised-variance estimators and forward volatility targets.

Every estimator returns a *daily variance* contribution v_t that uses only the
prices of day t (and C_{t-1} for those that include the overnight gap), so
averaging v over a window gives that window's realised variance.

    close_to_close   r_t^2,  r_t = ln(C_t / C_{t-1})          includes overnight
    parkinson        ln(H/L)^2 / (4 ln 2)                       intraday only
    garman_klass     0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2      intraday only
    rogers_satchell  ln(H/O) ln(H/C) + ln(L/O) ln(L/C)          intraday, drift-robust
    yang_zhang       ln(O_t/C_{t-1})^2 + k ln(C/O)^2 + (1-k) RS  includes overnight

Intraday-only estimators miss the overnight gap and therefore sit at a lower
level than close-to-close. That level gap matters when a forecast of one
estimator is used to size positions whose risk is close-to-close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester_demo import config

_LN2 = np.log(2.0)
# Yang-Zhang weight for a ~20-day window: k = 0.34 / (1.34 + (n+1)/(n-1)).
_YZ_K = 0.34 / (1.34 + 21 / 19)


def close_to_close(df: pd.DataFrame) -> pd.Series:
    return np.log(df["Close"] / df["Close"].shift(1)) ** 2


def parkinson(df: pd.DataFrame) -> pd.Series:
    return np.log(df["High"] / df["Low"]) ** 2 / (4 * _LN2)


def garman_klass(df: pd.DataFrame) -> pd.Series:
    hl = np.log(df["High"] / df["Low"])
    co = np.log(df["Close"] / df["Open"])
    return 0.5 * hl**2 - (2 * _LN2 - 1) * co**2


def rogers_satchell(df: pd.DataFrame) -> pd.Series:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    return np.log(h / o) * np.log(h / c) + np.log(l / o) * np.log(l / c)


def yang_zhang(df: pd.DataFrame) -> pd.Series:
    overnight = np.log(df["Open"] / df["Close"].shift(1)) ** 2
    open_close = np.log(df["Close"] / df["Open"]) ** 2
    return overnight + _YZ_K * open_close + (1 - _YZ_K) * rogers_satchell(df)


ESTIMATOR_FUNCS = {
    "close_to_close": close_to_close,
    "parkinson": parkinson,
    "garman_klass": garman_klass,
    "rogers_satchell": rogers_satchell,
    "yang_zhang": yang_zhang,
}


def daily_variance(df: pd.DataFrame, estimator: str) -> pd.Series:
    v = ESTIMATOR_FUNCS[estimator](df)
    return v.clip(lower=config.VAR_FLOOR).rename(estimator)


def all_daily_variances(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([daily_variance(df, e) for e in config.ESTIMATORS], axis=1)


def forward_log_vol(var: pd.Series, h: int) -> pd.Series:
    """Target at t: log sqrt(mean(v_{t+1}, ..., v_{t+h})). Uses no data from <= t."""
    fwd_mean = var[::-1].rolling(h).mean()[::-1].shift(-1)
    return 0.5 * np.log(fwd_mean)


def trailing_log_vol(var: pd.Series, window: int) -> pd.Series:
    """log sqrt(mean(v_{t-window+1}, ..., v_t)). Known at the close of t."""
    return 0.5 * np.log(var.rolling(window).mean())


def annualise(log_vol: pd.Series | pd.DataFrame):
    """Daily log-vol -> annualised vol in %."""
    return np.exp(log_vol) * np.sqrt(config.TRADING_DAYS) * 100
