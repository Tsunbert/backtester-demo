"""Plotly figures for the demo notebook."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtester_demo import config
from backtester_demo.volatility import annualise

TEMPLATE = "plotly_white"
SPLIT_COLORS = {"train": "rgba(99,110,250,0.06)", "val": "rgba(255,161,90,0.10)",
                "test": "rgba(0,204,150,0.10)"}


def _shade_splits(fig: go.Figure, index: pd.DatetimeIndex, row=None, col=None):
    bounds = [("train", index[0], pd.Timestamp(config.TRAIN_END)),
              ("val", pd.Timestamp(config.TRAIN_END), pd.Timestamp(config.VAL_END)),
              ("test", pd.Timestamp(config.VAL_END), index[-1])]
    for name, a, b in bounds:
        if b <= index[0] or a >= index[-1]:
            continue
        fig.add_vrect(x0=max(a, index[0]), x1=min(b, index[-1]), fillcolor=SPLIT_COLORS[name],
                      line_width=0, layer="below", row=row, col=col,
                      annotation_text=name, annotation_position="top left")


def ohlc_chart(df: pd.DataFrame, start: str | None = None) -> go.Figure:
    d = df.loc[start:] if start else df
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
                        vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"],
                                 close=d["Close"], name=config.TICKER), row=1, col=1)
    for n, color in ((config.SMA_FAST, "#636efa"), (config.SMA_SLOW, "#ef553b")):
        sma = df["Close"].rolling(n).mean().loc[d.index]
        fig.add_trace(go.Scatter(x=d.index, y=sma, name=f"SMA {n}", line=dict(width=1.2, color=color)),
                      row=1, col=1)
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], name="Volume", marker_color="#9aa0a6"), row=2, col=1)
    fig.update_layout(template=TEMPLATE, height=600, xaxis_rangeslider_visible=False,
                      title=f"{config.TICKER} daily OHLC (dividend-adjusted)")
    _shade_splits(fig, d.index, row=1, col=1)
    return fig


def returns_overview(df: pd.DataFrame) -> go.Figure:
    r = np.log(df["Close"]).diff().dropna() * 100
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Daily log returns (%) vs normal",
                                                        "Rolling 21d close-to-close vol (ann. %)"))
    fig.add_trace(go.Histogram(x=r, nbinsx=200, histnorm="probability density", name="returns",
                               marker_color="#636efa"), row=1, col=1)
    x = np.linspace(r.min(), r.max(), 400)
    pdf = np.exp(-0.5 * ((x - r.mean()) / r.std()) ** 2) / (r.std() * np.sqrt(2 * np.pi))
    fig.add_trace(go.Scatter(x=x, y=pdf, name="normal fit", line=dict(color="#ef553b")), row=1, col=1)
    rv = r.rolling(21).std() * np.sqrt(config.TRADING_DAYS)
    fig.add_trace(go.Scatter(x=rv.index, y=rv, name="21d vol", line=dict(width=1, color="#00a087")),
                  row=1, col=2)
    fig.update_layout(template=TEMPLATE, height=380, showlegend=False,
                      title=f"Fat tails & volatility clustering (kurtosis {r.kurt():.1f})")
    return fig


def estimators_chart(variances: pd.DataFrame, window: int = 21, start: str | None = None) -> go.Figure:
    lv = 0.5 * np.log(variances.rolling(window).mean())
    d = annualise(lv).loc[start:] if start else annualise(lv)
    fig = px.line(d, labels={"value": "ann. vol %", "variable": "estimator"},
                  title=f"Realised vol by estimator ({window}d window, annualised)")
    fig.update_traces(line=dict(width=1.2))
    fig.update_layout(template=TEMPLATE, height=420)
    return fig


def estimator_table(variances: pd.DataFrame) -> pd.DataFrame:
    """Level vs close-to-close and how noisy each 1-day estimate is."""
    lv = 0.5 * np.log(variances)
    return pd.DataFrame({
        "mean_ann_vol_%": (np.sqrt(variances.mean() * config.TRADING_DAYS) * 100),
        "level_vs_cc": np.sqrt(variances.mean() / variances["close_to_close"].mean()),
        "autocorr_lag1_logvol": lv.apply(lambda s: s.autocorr(1)),
        "std_logvol": lv.std(),
    }).round(3)


def estimator_corr(variances: pd.DataFrame) -> go.Figure:
    c = (0.5 * np.log(variances)).corr()
    fig = px.imshow(c, text_auto=".2f", color_continuous_scale="Blues", zmin=0, zmax=1,
                    title="Correlation of daily log-vol across estimators")
    fig.update_layout(template=TEMPLATE, height=420)
    return fig


def forecast_vs_realised(grid: dict, tgts: dict, estimator: str, h: int, models,
                         start: str | None = None) -> go.Figure:
    start = start or config.VAL_END
    y = annualise(tgts[estimator][h]).loc[start:]
    fig = go.Figure(go.Scatter(x=y.index, y=y, name="realised", line=dict(color="black", width=1)))
    for m in models:
        f = annualise(grid[(m, estimator)][h]).loc[start:]
        fig.add_trace(go.Scatter(x=f.index, y=f, name=m, line=dict(width=1.3)))
    fig.update_layout(template=TEMPLATE, height=430, yaxis_title="ann. vol %",
                      title=f"Test-period forecasts vs realised — {estimator}, h={h}")
    return fig


def lag_profile_chart(profiles: pd.DataFrame) -> go.Figure:
    fig = px.line(profiles, markers=True, labels={"index": "k (days)", "value": "corr"},
                  title="corr(forecast_t, realised log-vol_{t+k}) — k>0 is genuine foresight")
    fig.add_vline(x=0, line_dash="dot")
    fig.update_layout(template=TEMPLATE, height=400, legend_title="model")
    return fig


def metric_heatmap(metrics: pd.DataFrame, metric: str, h: int, fmt: str = ".3f",
                   reverse: bool = False) -> go.Figure:
    m = metrics[metrics["horizon"] == h].pivot(index="model", columns="estimator", values=metric)
    scale = "RdYlGn_r" if reverse else "RdYlGn"
    fig = px.imshow(m, text_auto=fmt, color_continuous_scale=scale, aspect="auto",
                    title=f"{metric} on test, h={h}")
    fig.update_layout(template=TEMPLATE, height=380)
    return fig


def equity_curves(pf, columns, title="Equity curves (test period)") -> go.Figure:
    v = pf.value()[list(columns)] / config.INIT_CASH
    fig = px.line(v, labels={"value": "growth of $1", "variable": "strategy"}, title=title)
    fig.update_traces(line=dict(width=1.4))
    fig.update_layout(template=TEMPLATE, height=450)
    return fig


def drawdowns(pf, columns) -> go.Figure:
    v = pf.value()[list(columns)]
    dd = (v / v.cummax() - 1) * 100
    fig = px.area(dd, labels={"value": "drawdown %", "variable": "strategy"}, title="Drawdowns")
    fig.update_traces(line=dict(width=0.8), stackgroup=None, fill="tozeroy")
    fig.update_layout(template=TEMPLATE, height=380)
    return fig


def rolling_vol_vs_target(pf, columns, window: int = 63) -> go.Figure:
    rv = pf.returns()[list(columns)].rolling(window).std() * np.sqrt(config.TRADING_DAYS) * 100
    fig = px.line(rv, labels={"value": "ann. vol %", "variable": "strategy"},
                  title=f"Realised strategy vol ({window}d) vs {config.TARGET_VOL_ANN:.0%} target")
    fig.add_hline(y=config.TARGET_VOL_ANN * 100, line_dash="dash", line_color="black")
    fig.update_layout(template=TEMPLATE, height=400)
    return fig


def weights_chart(weights: pd.DataFrame, columns, start: str | None = None) -> go.Figure:
    w = weights[list(columns)].loc[start or config.VAL_END:]
    fig = px.line(w, labels={"value": "weight", "variable": "strategy"},
                  title="Target exposure decided at each close")
    fig.update_traces(line=dict(width=1))
    fig.update_layout(template=TEMPLATE, height=380)
    return fig


def summary_heatmap(summary: pd.DataFrame, metric: str, fmt: str = ".2f",
                    reverse: bool = False) -> go.Figure:
    vt = summary[summary.index.str.startswith("vt|")].copy()
    parts = vt.index.str.split("|")
    vt["model"], vt["estimator"] = parts.str[1], parts.str[2]
    m = vt.pivot(index="model", columns="estimator", values=metric)
    fig = px.imshow(m, text_auto=fmt, aspect="auto",
                    color_continuous_scale="RdYlGn_r" if reverse else "RdYlGn",
                    title=f"Vol-target backtest: {metric} (test period)")
    fig.update_layout(template=TEMPLATE, height=400)
    return fig
