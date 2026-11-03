"""Download and cache daily OHLCV for the traded instrument."""
from __future__ import annotations

import warnings

import pandas as pd

from backtester_demo import config


def _download(ticker: str, start: str) -> pd.DataFrame:
    import yfinance as yf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(ticker, start=start, auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    raw.index.name = "Date"
    return raw


def adjust_ohlc(raw: pd.DataFrame) -> pd.DataFrame:
    """Scale O/H/L/C by Adj Close / Close so dividends are in the price series.

    Intraday ratios (H/L, C/O, ...) are unchanged, so range-based estimators are
    unaffected; close-to-close returns and the backtest become total-return.
    """
    factor = raw["Adj Close"] / raw["Close"]
    out = pd.DataFrame({c: raw[c] * factor for c in ("Open", "High", "Low", "Close")})
    out["Volume"] = raw["Volume"].astype(float)
    return out


def load_ohlcv(ticker: str = config.TICKER, start: str = config.START,
               refresh: bool = False) -> pd.DataFrame:
    """Adjusted daily OHLCV, cached to data/<ticker>.parquet."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = config.DATA_DIR / f"{ticker.lower()}.parquet"
    if path.exists() and not refresh:
        raw = pd.read_parquet(path)
    else:
        raw = _download(ticker, start)
        raw.to_parquet(path)
    df = adjust_ohlc(raw)
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)].dropna()
    return df.sort_index()
