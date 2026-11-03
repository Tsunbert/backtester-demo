"""Central configuration: dates, horizons, estimators and backtest settings.

The chronological split is the backbone of the no-look-ahead guarantee:

    train  : START .. TRAIN_END      fit LSTM weights, HAR anchor, feature scalers
    val    : TRAIN_END .. VAL_END    LSTM early stopping, GARCH calibration,
                                     vol-target normalisation
    test   : VAL_END .. today        every reported metric and every backtest

Samples whose forward target window crosses a split boundary are dropped
(EMBARGO = longest horizon), so no training label is built from val/test prices.
"""
from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

TICKER = "SPY"
START = "2000-01-01"
TRAIN_END = "2018-12-31"
VAL_END = "2022-12-31"

TRADING_DAYS = 252

# Forecast horizons in trading days. Target(t, h) = realised vol over t+1..t+h.
HORIZONS = (1, 5, 20)
EMBARGO = max(HORIZONS)

# Realised-volatility estimators that make up the "target grid".
ESTIMATORS = ("close_to_close", "parkinson", "garman_klass", "rogers_satchell", "yang_zhang")

# Daily variance floor: protects log() from zero-range / zero-return days.
VAR_FLOOR = 1e-8

# LSTM
WINDOW = 60
LSTM_HIDDEN = 32
LSTM_LAYERS = 1
LSTM_DROPOUT = 0.1
LSTM_LR = 1e-3
LSTM_WEIGHT_DECAY = 1e-4
LSTM_EPOCHS = 60
LSTM_PATIENCE = 8
LSTM_BATCH = 128
LSTM_SEEDS = (0, 1, 2)

# GARCH walk-forward
GARCH_REFIT_EVERY = 21
GARCH_SPECS = {
    "garch": dict(p=1, o=0, q=1, dist="t"),
    "gjr_garch": dict(p=1, o=1, q=1, dist="t"),
}

# Backtest
TRADE_HORIZON = 5            # which forecast horizon drives position sizing
TARGET_VOL_ANN = 0.12        # annualised vol target, close-to-close units
MAX_LEVERAGE = 1.0           # vectorbt OSS cannot borrow cash for longs; >1 is silently capped
FEES = 0.0002                # 2 bps per trade value
SLIPPAGE = 0.0003            # 3 bps
INIT_CASH = 100_000.0
SMA_FAST, SMA_SLOW = 50, 200  # long/short baseline, fixed a priori (not tuned)
