
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DatasetBundle:
    X: pd.DataFrame
    y: pd.Series
    protected_df: Optional[pd.DataFrame] = None
    metadata: Optional[dict] = None


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def binary_target(logit, rng):
    probability = sigmoid(logit)
    return pd.Series(
        rng.binomial(1, probability),
        name="target"
    )


def train_test_validation_split(
    bundle: DatasetBundle,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
):
    """
    Simple deterministic split preserving row alignment between:
    X, y, protected_df.
    """

    rng = np.random.default_rng(seed)

    n = len(bundle.X)
    indices = np.arange(n)
    rng.shuffle(indices)

    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    def take(df, idx):
        if df is None:
            return None
        return df.iloc[idx].reset_index(drop=True)

    return {
        "train": DatasetBundle(
            bundle.X.iloc[train_idx].reset_index(drop=True),
            bundle.y.iloc[train_idx].reset_index(drop=True),
            take(bundle.protected_df, train_idx),
            bundle.metadata,
        ),
        "validation": DatasetBundle(
            bundle.X.iloc[val_idx].reset_index(drop=True),
            bundle.y.iloc[val_idx].reset_index(drop=True),
            take(bundle.protected_df, val_idx),
            bundle.metadata,
        ),
        "test": DatasetBundle(
            bundle.X.iloc[test_idx].reset_index(drop=True),
            bundle.y.iloc[test_idx].reset_index(drop=True),
            take(bundle.protected_df, test_idx),
            bundle.metadata,
        ),
    }


def perturb_features(
    X: pd.DataFrame,
    epsilon=0.03,
    seed=42,
):

    rng = np.random.default_rng(seed)

    noise = rng.uniform(
        -epsilon,
        epsilon,
        size=X.shape,
    )

    return X + noise