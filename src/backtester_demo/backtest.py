"""vectorbt backtest of target-weight strategies.

Timing convention (the one place look-ahead could sneak into the P&L):
weights decided at the close of t are executed at the OPEN of t+1, i.e. the
weight frame is shifted by one bar and filled at the open price. Positions are
then marked to market at each close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt

from backtester_demo import config


def run(ohlc: pd.DataFrame, weights: pd.DataFrame, start: str | None = None) -> vbt.Portfolio:
    """Backtest every column of `weights` (decided at close t) from `start`."""
    start = start or config.VAL_END
    exec_w = weights.shift(1)
    mask = ohlc.index > pd.Timestamp(start)
    close = ohlc.loc[mask, "Close"]
    open_ = ohlc.loc[mask, "Open"]
    size = exec_w.loc[mask].reindex(columns=weights.columns)
    return vbt.Portfolio.from_orders(
        close=pd.DataFrame({c: close for c in size.columns}),
        size=size,
        size_type="targetpercent",
        direction="both",
        price=pd.DataFrame({c: open_ for c in size.columns}),
        fees=config.FEES,
        slippage=config.SLIPPAGE,
        init_cash=config.INIT_CASH,
        freq="1D",
    )


def summary(pf: vbt.Portfolio) -> pd.DataFrame:
    """One row per strategy: the numbers that matter for vol targeting."""
    rets = pf.returns()
    ann = np.sqrt(config.TRADING_DAYS)
    rolling_vol = rets.rolling(63).std() * ann
    out = pd.DataFrame({
        "total_return_%": pf.total_return() * 100,
        "cagr_%": ((1 + pf.total_return()) ** (config.TRADING_DAYS / len(rets)) - 1) * 100,
        "ann_vol_%": rets.std() * ann * 100,
        "sharpe": rets.mean() / rets.std() * ann,
        "sortino": rets.mean() / rets[rets < 0].std() * ann,
        "max_dd_%": pf.max_drawdown() * 100,
        "vol_of_vol_%": rolling_vol.std() * 100,
        "trades": pf.orders.count(),
    })
    out["calmar"] = out["cagr_%"] / out["max_dd_%"].abs()
    return out
