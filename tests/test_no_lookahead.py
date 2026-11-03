"""Look-ahead tests: a forecast made at t must not change when the future changes.

The core trick: run a model on the full history and on history truncated (or
scrambled) after a cut-off date; every forecast with origin <= cut-off must be
identical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester_demo import backtest, config, volatility as vol
from backtester_demo.models import baselines, garch, lstm
from backtester_demo.splits import label_inside_split


def synthetic_ohlc(n: int = 6900, seed: int = 0) -> pd.DataFrame:
    """GARCH-like random walk with plausible OHLC, business-day index from 2000."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    sig = np.empty(n)
    r = np.empty(n)
    s2 = 1e-4
    for t in range(n):
        sig[t] = np.sqrt(s2)
        r[t] = sig[t] * rng.standard_normal()
        s2 = 2e-6 + 0.08 * r[t] ** 2 + 0.9 * s2
    close = 100 * np.exp(np.cumsum(r))
    open_ = close * np.exp(sig * 0.3 * rng.standard_normal(n))
    hi = np.maximum(open_, close) * np.exp(np.abs(sig * rng.standard_normal(n)) * 0.6)
    lo = np.minimum(open_, close) * np.exp(-np.abs(sig * rng.standard_normal(n)) * 0.6)
    return pd.DataFrame({"Open": open_, "High": hi, "Low": lo, "Close": close,
                         "Volume": 1e6}, index=idx)


def scramble_after(df: pd.DataFrame, cut: pd.Timestamp, seed: int = 1) -> pd.DataFrame:
    """Replace everything after `cut` with different (but valid) prices."""
    out = df.copy()
    future = out.index > cut
    alt = synthetic_ohlc(len(out), seed=seed)
    scale = df.loc[~future, "Close"].iloc[-1] / alt.loc[future, "Close"].iloc[0]
    for c in ("Open", "High", "Low", "Close"):
        out.loc[future, c] = alt.loc[future, c].to_numpy() * scale
    return out


@pytest.fixture(scope="module")
def df():
    return synthetic_ohlc()


@pytest.fixture(scope="module")
def cut(df):
    return df.index[df.index > pd.Timestamp(config.VAL_END)][40]


def test_forward_target_uses_only_future(df):
    var = vol.daily_variance(df, "garman_klass")
    t = 100
    bumped = var.copy()
    bumped.iloc[: t + 1] *= 10           # change today and the past
    for h in config.HORIZONS:
        a = vol.forward_log_vol(var, h).iloc[t]
        b = vol.forward_log_vol(bumped, h).iloc[t]
        assert a == pytest.approx(b)
        expected = 0.5 * np.log(var.iloc[t + 1: t + 1 + h].mean())
        assert a == pytest.approx(expected)


def test_embargo_drops_boundary_rows(df):
    for h in config.HORIZONS:
        ok = label_inside_split(df.index, h)
        last_train = df.index[df.index <= pd.Timestamp(config.TRAIN_END)][-h:]
        assert not ok.loc[last_train].any()
        assert not ok.iloc[-h:].any()


@pytest.mark.parametrize("est", config.ESTIMATORS)
def test_estimators_are_causal(df, cut, est):
    a = vol.daily_variance(df, est)
    b = vol.daily_variance(scramble_after(df, cut), est)
    pd.testing.assert_series_equal(a.loc[:cut], b.loc[:cut])


def test_har_walk_forward_is_causal(df, cut):
    v = vol.daily_variance(df, "garman_klass")
    v2 = vol.daily_variance(scramble_after(df, cut), "garman_klass")
    a, b = baselines.har_walk_forward(v), baselines.har_walk_forward(v2)
    pd.testing.assert_frame_equal(a.loc[:cut], b.loc[:cut])


def test_garch_walk_forward_is_causal(df, cut):
    spec = config.GARCH_SPECS["garch"]
    a = garch.garch_walk_forward(df, spec)
    b = garch.garch_walk_forward(scramble_after(df, cut), spec)
    common = a.loc[:cut].dropna().index
    assert len(common) > 500
    np.testing.assert_allclose(a.loc[common].to_numpy(), b.loc[common].to_numpy(), rtol=1e-6)


def test_lstm_features_are_causal(df, cut):
    V = vol.all_daily_variances(df)
    df2 = scramble_after(df, cut)
    V2 = vol.all_daily_variances(df2)
    a = lstm.build_features(df, V, "garman_klass")
    b = lstm.build_features(df2, V2, "garman_klass")
    pd.testing.assert_frame_equal(a.loc[:cut], b.loc[:cut])


def test_lstm_forecast_is_causal(df, cut, monkeypatch):
    # Test-period prices must not reach training, scaling or earlier forecasts.
    monkeypatch.setattr(config, "LSTM_EPOCHS", 2)
    monkeypatch.setattr(config, "LSTM_SEEDS", (0,))
    df2 = scramble_after(df, cut)
    a = lstm.lstm_forecast(df, vol.all_daily_variances(df), "garman_klass")
    b = lstm.lstm_forecast(df2, vol.all_daily_variances(df2), "garman_klass")
    pd.testing.assert_frame_equal(a.loc[:cut], b.loc[:cut], atol=1e-5, rtol=1e-5)


def test_backtest_executes_next_open(df):
    # Go long only on the last val close; P&L must start at the next open.
    w = pd.Series(0.0, index=df.index)
    first_test = df.index[df.index > pd.Timestamp(config.VAL_END)][0]
    decision_day = df.index[df.index.get_loc(first_test) + 5]
    w.loc[decision_day:] = 1.0
    pf = backtest.run(df, w.to_frame("x"))
    orders = pf.orders.records_readable
    assert orders["Timestamp"].iloc[0] == df.index[df.index.get_loc(decision_day) + 1]
    assert orders["Price"].iloc[0] == pytest.approx(
        df["Open"].iloc[df.index.get_loc(decision_day) + 1] * (1 + config.SLIPPAGE))
