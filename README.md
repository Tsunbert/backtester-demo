# backtester-demo

**Do better volatility forecasts make a better strategy?** This repo forecasts SPY volatility with naive persistence, HAR-RV, GARCH, GJR-GARCH and two LSTM variants, across five realised-volatility estimators. It then backtests the forecasts with [vectorbt](https://vectorbt.dev) as a vol-targeting strategy, alongside a simple 50/200 SMA long/short.

It builds on [BogdanT54/Neural-networks-volatility-prediction](https://github.com/BogdanT54/Neural-networks-volatility-prediction), which compared LSTM / Seq2Seq / GARCH forecasts on NYSE stocks, and adds the step from forecast to P&L.

## Quick start

```bash
uv sync
uv run pytest -q -W ignore          # look-ahead tests
uv run backtester-demo              # builds the forecast grid (~6 min first time) and prints results
uv run jupyter lab notebooks/demo.ipynb
```

## What's in the demo notebook

1. **OHLC visualisations**: candlesticks with SMA and volume, train/val/test shading, return distribution, volatility clustering.
2. **Volatility spectrum**: close-to-close, Parkinson, Garman–Klass, Rogers–Satchell, Yang–Zhang. Shows each estimator's level (the intraday-only ones miss the overnight gap) and its noise.
3. **Forecast grid**: 6 models × 5 estimators × horizons 1/5/20 days.
4. **Evaluation on the sealed test set** (2023→today): skill vs naive, QLIKE, Mincer–Zarnowitz, Diebold–Mariano vs HAR, and lag-1 copy diagnostics (change-correlation, lag profile).
5. **vectorbt backtest**: vol targeting at 12% annualised, executed at the next open with fees and slippage. Compared against buy & hold, the SMA long/short (raw and vol-scaled), and a perfect-foresight ORACLE as the ceiling.

## How look-ahead is prevented

| Component | Guard |
|---|---|
| Split | train ≤ 2018, val 2019–2022, test 2023→. All reported numbers are test-only. |
| Labels | target(t, h) = realised vol over t+1…t+h. Rows whose label crosses a split boundary are embargoed. |
| HAR | expanding refit every 21 days, using only labels observed by the refit date. |
| GARCH | refit every 21 days on data up to that date, then frozen parameters and a one-step variance recursion. Mapped to non-close-to-close estimators by a calibration fit on val only. |
| LSTM | scaler, target normalisation and weights fit on train; val only for early stopping. |
| Backtest | weights decided at close *t* are executed at the open of *t+1*. |

`tests/test_no_lookahead.py` checks this empirically. For each model it scrambles all prices after a cut-off date and asserts that every forecast made before the cut-off is unchanged.

### Avoiding the "LSTM learns lag-1" trap

Daily realised vol is very persistent and very noisy. An LSTM trained with MSE on tomorrow's vol, with today's vol as an input, can settle on a smoothed copy of today. That copy looks convincing on a chart but adds nothing. The repo handles this in three ways:

- `lstm_residual` predicts the residual over a train-fitted HAR anchor. If it learns nothing, it collapses to HAR, not to lag-1.
- `lstm_direct` (the naive formulation) is kept as a control.
- Every model is scored on **skill vs naive**, **change-correlation** (does it predict the *move* in vol?) and a **Diebold–Mariano** test against HAR.

## Results (test period 2023-01-03 → 2026-11-02, run on 2026-11-03)

**Forecasting:** mean skill vs naive at h=5, averaged over estimators (1 − MSE/MSE_naive):

| lstm_direct | lstm_residual | gjr_garch | har | garch |
|---|---|---|---|---|
| 0.669 | 0.665 | 0.650 | 0.602 | 0.586 |

Both LSTMs beat HAR significantly at h=1 and h=5 (Diebold–Mariano). They also predict the *change* in vol better than HAR, with change-correlation of about 0.62 at h=1 on Garman–Klass. So neither is a lag-1 copy. Range-based estimators are far less noisy than close-to-close: the lag-1 autocorrelation of daily log-vol is about 0.66 for range-based estimators against 0.14 for close-to-close.

**Backtest (12% vol target, long-only, next-open execution, 5 bps costs):**

| strategy | Sharpe | ann. vol | max DD | vol-of-vol |
|---|---|---|---|---|
| buy & hold | 1.34 | 14.9% | −18.8% | 5.5% |
| vol-target, HAR / Garman–Klass | 1.36 | 11.9% | −13.2% | 1.7% |
| vol-target, LSTM-residual / Garman–Klass | 1.32 | 12.0% | −14.3% | 1.8% |
| vol-target, naive / close-to-close | 1.01 | 12.6% | −19.4% | 3.1% |
| vol-target, ORACLE (perfect foresight) | 2.11 | 11.2% | −9.5% | 1.2% |
| SMA 50/200 long/short | 0.58 | 14.9% | −27.0% | 5.4% |

What it shows: vol targeting keeps buy-and-hold's Sharpe while cutting drawdown by about 30% and stabilising realised vol about 3×. Choosing a **good estimator** (range-based rather than close-to-close) matters more for P&L than choosing a fancier **model**. The LSTM's forecast edge over HAR is statistically real but does not survive into Sharpe over a single 4-year path. The ORACLE row shows how much would be available to a much better forecaster.
