"""Turn forecasts into target weights. Weights at t are decided at the close of t.

The backtest executes them at the next open, so nothing here needs shifting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester_demo import config
from backtester_demo.splits import split_labels


def level_ratio(variances: pd.DataFrame, estimator: str) -> float:
    """sqrt(mean v_est / mean v_cc) on train: how much lower an estimator reads
    than close-to-close (intraday-only estimators miss the overnight gap)."""
    train = split_labels(variances.index) == "train"
    v = variances[train]
    return float(np.sqrt(v[estimator].mean() / v["close_to_close"].mean()))


def vol_target_weights(forecast_lv: pd.Series, ratio: float,
                       target_ann: float = config.TARGET_VOL_ANN,
                       max_lev: float = config.MAX_LEVERAGE) -> pd.Series:
    """w_t = target / forecast, with the target expressed in the estimator's own units."""
    target_daily = target_ann / np.sqrt(config.TRADING_DAYS) * ratio
    return (target_daily / np.exp(forecast_lv)).clip(upper=max_lev)


def sma_long_short(close: pd.Series, fast: int = config.SMA_FAST,
                   slow: int = config.SMA_SLOW) -> pd.Series:
    """+1 when SMA_fast > SMA_slow, else -1. Parameters fixed a priori, never tuned."""
    sig = np.sign(close.rolling(fast).mean() - close.rolling(slow).mean())
    return sig.where(close.rolling(slow).count() == slow)


def buy_and_hold(close: pd.Series) -> pd.Series:
    return pd.Series(1.0, index=close.index)
