# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Demo: do better volatility forecasts make a better strategy? SPY daily OHLC (2000→today, yfinance, dividend-adjusted) → five realised-vol estimators → a **models × estimators forecast grid** (naive, HAR-RV, GARCH-t, GJR-GARCH-t, LSTM direct, LSTM HAR-residual; horizons 1/5/20) → forecast evaluation → **vectorbt** vol-targeting backtest plus a 50/200 SMA long/short baseline. It extends a colleague's forecasting-only study (github.com/BogdanT54/Neural-networks-volatility-prediction) by attaching a backtester and fixing its leakage issues (no embargo at split boundaries, GARCH on close-to-close scale scored against Garman–Klass).

The demo narrative lives in `notebooks/demo.ipynb`; all logic is in `src/backtester_demo/`.

## Commands

```bash
uv sync                                    # env from uv.lock (Python 3.12)
uv run pytest -q -W ignore                 # look-ahead test suite (~1 min)
uv run pytest -q -k garch                  # single test
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/demo.ipynb
```

Forecast grid is cached as `artifacts/forecasts_spy_<lastdate>.pkl` (~6 min on CPU to rebuild; delete it or pass `use_cache=False` after changing a model). SPY data is cached in `data/spy.parquet` (`load_ohlcv(refresh=True)` re-downloads). Both dirs are gitignored.

## Architecture

- `config.py` — single source for split dates (train ≤2018, val 2019–2022, test 2023→), horizons, estimators, LSTM/GARCH/backtest params. Modules read `config.X` at call time, so tests monkeypatch it.
- `volatility.py` — each estimator returns a **daily variance** v_t; `forward_log_vol(v, h)` = log√mean(v_{t+1..t+h}) is the target, indexed by origin date t.
- Every forecast in the grid is a DataFrame indexed by origin date t, columns = horizons, values in **daily log-vol**; it must use only data ≤ close of t.
- `splits.py` — `label_inside_split` implements the embargo (drop rows whose label window crosses a split boundary).
- `models/` — HAR and GARCH are walk-forward (refit every 21d from val start); GARCH raw output is close-to-close scale and is mapped to each estimator via `calibrate_to` (OLS on val only). LSTM trains once on train, early-stops on val, 3-seed ensemble; `residual=True` predicts target − train-fitted HAR.
- `pipeline.py` — builds grid, test-only `metrics_table`, and `strategy_weights` (column names `vt|<model>|<estimator>`, `sma_ls_vt|…`, `buy_hold`, `sma_long_short`, `vt|ORACLE|…` = deliberate look-ahead ceiling).
- `backtest.py` — weights decided at close t are `shift(1)`ed and filled at the **next open** in `Portfolio.from_orders(size_type="targetpercent")`. vectorbt OSS can't borrow cash for longs, so `MAX_LEVERAGE` >1 is silently capped.
- `strategies.level_ratio` converts intraday-only estimators (Parkinson/GK/RS read ~17% below close-to-close) to close-to-close units before sizing.

## Non-negotiable modelling rules (look-ahead bias)

- Anything that fits parameters (scalers, HAR betas, GARCH params, calibrations, vol-target constants) may only see train, or val for selection/calibration — never test.
- New models must be added to `tests/test_no_lookahead.py` with the "scramble prices after a cut date, forecasts before it must not change" pattern.
- Always benchmark against naive and HAR, and report `skill_vs_naive`, `change_corr` and Diebold-Mariano vs HAR — an LSTM that reproduces lag-1 has learned nothing.
