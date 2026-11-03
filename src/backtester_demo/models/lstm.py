"""LSTM volatility forecaster, trained once on train and early-stopped on val.

Two variants, both multi-horizon (one output per config.HORIZONS):

- residual=True  (the honest one): the network predicts
      target - HAR_anchor
  where HAR_anchor is a HAR fitted on train only. Persistence is already in the
  anchor, so the network only gets credit for information HAR cannot explain.
  If it learns nothing, it collapses to HAR, not to a lagged copy of the input.

- residual=False (the trap, kept for the demo): the network predicts the target
  directly. Because today's vol is an input and daily vol is ~70% noise, MSE is
  minimised by a smoothed copy of yesterday, which looks great on a chart and
  adds nothing over the naive baseline.

Leakage guards: features are known at the close of t; scaler and target
normalisation are fit on train rows only; train/val samples whose label window
crosses a split boundary are dropped; val is used only for early stopping.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from backtester_demo import config
from backtester_demo.models.baselines import har_static
from backtester_demo.splits import label_inside_split, split_labels
from backtester_demo.volatility import forward_log_vol, trailing_log_vol


def build_features(df: pd.DataFrame, variances: pd.DataFrame, target: str) -> pd.DataFrame:
    """Per-day inputs, each computable at the close of t."""
    feats = {f"lv_{e}": 0.5 * np.log(variances[e]) for e in config.ESTIMATORS}
    feats["lv_w"] = trailing_log_vol(variances[target], 5)
    feats["lv_m"] = trailing_log_vol(variances[target], 22)
    feats["ret"] = np.log(df["Close"] / df["Close"].shift(1))
    feats["overnight"] = np.log(df["Open"] / df["Close"].shift(1))
    return pd.DataFrame(feats)


class VolLSTM(nn.Module):
    def __init__(self, n_features: int, n_outputs: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, config.LSTM_HIDDEN, num_layers=config.LSTM_LAYERS,
                            batch_first=True)
        self.drop = nn.Dropout(config.LSTM_DROPOUT)
        self.head = nn.Linear(config.LSTM_HIDDEN, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1]))


def _windows(arr: np.ndarray, ends: np.ndarray, window: int) -> np.ndarray:
    return np.stack([arr[e - window + 1: e + 1] for e in ends]).astype(np.float32)


def _train_one(Xtr, ytr, Xva, yva, seed: int) -> VolLSTM:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = VolLSTM(Xtr.shape[-1], ytr.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=config.LSTM_LR,
                            weight_decay=config.LSTM_WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xva_t, yva_t = torch.from_numpy(Xva), torch.from_numpy(yva)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad = np.inf, None, 0
    for _ in range(config.LSTM_EPOCHS):
        model.train()
        for idx in torch.randperm(len(Xtr_t), generator=gen).split(config.LSTM_BATCH):
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva_t), yva_t).item()
        if val_loss < best - 1e-5:
            best, bad = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= config.LSTM_PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def lstm_forecast(df: pd.DataFrame, variances: pd.DataFrame, target: str,
                  residual: bool = True) -> pd.DataFrame:
    """Train on train, early-stop on val, forecast every date with a full window."""
    var = variances[target]
    feats = build_features(df, variances, target)
    y = pd.DataFrame({h: forward_log_vol(var, h) for h in config.HORIZONS})
    anchor = har_static(var) if residual else pd.DataFrame(0.0, index=y.index, columns=y.columns)
    resid = y - anchor

    labels = split_labels(df.index).to_numpy()
    usable = (feats.notna().all(axis=1) & anchor.notna().all(axis=1)).to_numpy()
    # A window ending at t needs WINDOW clean rows behind it.
    full_window = pd.Series(usable).rolling(config.WINDOW).sum().to_numpy() == config.WINDOW
    label_ok = label_inside_split(df.index, max(config.HORIZONS)).to_numpy() & resid.notna().all(axis=1).to_numpy()

    tr = np.where(full_window & label_ok & (labels == "train"))[0]
    va = np.where(full_window & label_ok & (labels == "val"))[0]
    pred_ends = np.where(full_window)[0]

    f = feats.to_numpy()
    mu, sd = f[tr].mean(axis=0), f[tr].std(axis=0) + 1e-8
    f = np.nan_to_num((f - mu) / sd)
    r = resid.to_numpy()
    r_mu, r_sd = r[tr].mean(axis=0), r[tr].std(axis=0) + 1e-8

    Xtr, Xva = _windows(f, tr, config.WINDOW), _windows(f, va, config.WINDOW)
    ytr = ((r[tr] - r_mu) / r_sd).astype(np.float32)
    yva = ((r[va] - r_mu) / r_sd).astype(np.float32)
    Xall = torch.from_numpy(_windows(f, pred_ends, config.WINDOW))

    preds = []
    for seed in config.LSTM_SEEDS:
        model = _train_one(Xtr, ytr, Xva, yva, seed)
        with torch.no_grad():
            preds.append(model(Xall).numpy())
    pred = np.mean(preds, axis=0) * r_sd + r_mu

    out = pd.DataFrame(np.nan, index=df.index, columns=list(config.HORIZONS))
    out.iloc[pred_ends] = pred + anchor.to_numpy()[pred_ends]
    return out
