"""Orchestration: models x estimators forecast grid, metrics and backtest weights."""
from __future__ import annotations

import pickle
import time

import pandas as pd

from backtester_demo import config, evaluation, strategies
from backtester_demo.models import baselines, garch, lstm
from backtester_demo.splits import split_labels
from backtester_demo.volatility import all_daily_variances, forward_log_vol

MODELS = ("naive", "har", "garch", "gjr_garch", "lstm_direct", "lstm_residual")


def targets(variances: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {e: pd.DataFrame({h: forward_log_vol(variances[e], h) for h in config.HORIZONS})
            for e in config.ESTIMATORS}


def forecast_grid(df: pd.DataFrame, use_cache: bool = True, verbose: bool = True) -> dict:
    """{(model, estimator): DataFrame[horizon]} of log-vol forecasts by origin date."""
    path = config.ARTIFACTS_DIR / f"forecasts_{config.TICKER.lower()}_{df.index[-1]:%Y%m%d}.pkl"
    if use_cache and path.exists():
        return pickle.loads(path.read_bytes())

    def log(msg):
        if verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    variances = all_daily_variances(df)
    raw_garch = {}
    for name, spec in config.GARCH_SPECS.items():
        log(f"walk-forward {name}")
        raw_garch[name] = garch.garch_walk_forward(df, spec)

    grid = {}
    for e in config.ESTIMATORS:
        v = variances[e]
        grid[("naive", e)] = baselines.naive(v)
        grid[("har", e)] = baselines.har_walk_forward(v)
        for name, raw in raw_garch.items():
            grid[(name, e)] = garch.calibrate_to(raw, v)
        log(f"LSTM direct   -> {e}")
        grid[("lstm_direct", e)] = lstm.lstm_forecast(df, variances, e, residual=False)
        log(f"LSTM residual -> {e}")
        grid[("lstm_residual", e)] = lstm.lstm_forecast(df, variances, e, residual=True)

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(grid))
    return grid


def test_mask(index: pd.DatetimeIndex) -> pd.Series:
    return split_labels(index) == "test"


def metrics_table(grid: dict, tgts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Test-period forecast metrics for every (model, estimator, horizon)."""
    rows = []
    for (model, e), fc in grid.items():
        mask = test_mask(fc.index)
        for h in config.HORIZONS:
            m = evaluation.forecast_metrics(
                fc.loc[mask, h], tgts[e].loc[mask, h],
                grid[("naive", e)].loc[mask, h], grid[("har", e)].loc[mask, h], h)
            rows.append({"model": model, "estimator": e, "horizon": h, **m})
    return pd.DataFrame(rows)


def strategy_weights(df: pd.DataFrame, grid: dict, variances: pd.DataFrame,
                     tgts: dict[str, pd.DataFrame], h: int = config.TRADE_HORIZON) -> pd.DataFrame:
    """Columns: 'vt|model|estimator' vol-target weights plus benchmarks."""
    w = {"buy_hold": strategies.buy_and_hold(df["Close"]),
         "sma_long_short": strategies.sma_long_short(df["Close"])}
    for (model, e), fc in grid.items():
        ratio = strategies.level_ratio(variances, e)
        w[f"vt|{model}|{e}"] = strategies.vol_target_weights(fc[h], ratio)
        if e == "garman_klass":
            w[f"sma_ls_vt|{model}|{e}"] = w["sma_long_short"] * w[f"vt|{model}|{e}"]
    for e in config.ESTIMATORS:
        # Perfect-foresight ceiling: sizes with the realised future vol. Look-ahead on purpose.
        ratio = strategies.level_ratio(variances, e)
        w[f"vt|ORACLE|{e}"] = strategies.vol_target_weights(tgts[e][h], ratio)
    return pd.DataFrame(w)
