"""Chronological split labels with an embargo on forward-looking targets."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester_demo import config


def split_labels(index: pd.DatetimeIndex) -> pd.Series:
    labels = np.where(index <= pd.Timestamp(config.TRAIN_END), "train",
                      np.where(index <= pd.Timestamp(config.VAL_END), "val", "test"))
    return pd.Series(labels, index=index, name="split")


def label_inside_split(index: pd.DatetimeIndex, h: int) -> pd.Series:
    """True where the target window t+1..t+h lies in the same split as t.

    Rows near a boundary (whose label would be built from the next split's
    prices) and the last h rows (label not yet observed) are False.
    """
    labels = split_labels(index)
    ahead = labels.shift(-h)
    return (labels == ahead).rename("label_ok")


def first_position(index: pd.DatetimeIndex, split: str) -> int:
    return int(np.argmax(split_labels(index).to_numpy() == split))
