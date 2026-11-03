def main() -> None:
    """Build (or load) the forecast grid and print test-period results."""
    import warnings

    warnings.filterwarnings("ignore")
    import pandas as pd

    from backtester_demo import backtest, config, data, pipeline
    from backtester_demo.volatility import all_daily_variances

    pd.set_option("display.width", 160)
    df = data.load_ohlcv()
    variances = all_daily_variances(df)
    grid = pipeline.forecast_grid(df)
    tgts = pipeline.targets(variances)
    metrics = pipeline.metrics_table(grid, tgts)
    h = config.TRADE_HORIZON
    print(f"\nForecast skill vs naive on test, h={h} (rows: model, cols: estimator)")
    print(metrics[metrics.horizon == h].pivot(index="model", columns="estimator",
                                               values="skill_vs_naive").round(3))
    summary = backtest.summary(backtest.run(df, pipeline.strategy_weights(df, grid, variances, tgts)))
    print("\nTop strategies by Sharpe (test period)")
    print(summary.sort_values("sharpe", ascending=False).round(2).head(12))
