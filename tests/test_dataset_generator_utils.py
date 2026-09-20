import numpy as np
import pandas as pd

from dataset_generator_utils import (
    DatasetBundle,
    binary_target,
    perturb_features,
    sigmoid,
    train_test_validation_split,
)


def test_train_test_validation_split_preserves_alignment_and_sizes():
    X = pd.DataFrame({"feature": list(range(10))})
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], name="target")
    protected = pd.DataFrame({"group": ["A", "B"] * 5})

    bundle = DatasetBundle(X=X, y=y, protected_df=protected)
    splits = train_test_validation_split(bundle, train_frac=0.70, val_frac=0.15, seed=42)

    assert set(splits) == {"train", "validation", "test"}
    assert len(splits["train"].X) == 7
    assert len(splits["validation"].X) == 1
    assert len(splits["test"].X) == 2

    assert splits["train"].X.index.equals(pd.RangeIndex(0, 7))
    assert splits["validation"].X.index.equals(pd.RangeIndex(0, 1))
    assert splits["test"].X.index.equals(pd.RangeIndex(0, 2))

    assert splits["train"].y.name == "target"
    assert splits["validation"].protected_df is not None
    assert splits["test"].protected_df is not None
    assert len(splits["train"].protected_df) == len(splits["train"].X)
    assert len(splits["validation"].protected_df) == len(splits["validation"].X)
    assert len(splits["test"].protected_df) == len(splits["test"].X)


def test_perturb_features_returns_same_shape_and_changes_values():
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})

    perturbed = perturb_features(X, epsilon=0.03, seed=42)

    assert perturbed.shape == X.shape
    assert list(perturbed.columns) == list(X.columns)
    assert not perturbed.equals(X)


def test_sigmoid_and_binary_target_are_consistent():
    x = 0.0
    assert sigmoid(x) == 0.5

    rng = np.random.default_rng(42)
    result = binary_target(logit=0.0, rng=rng)

    assert result.name == "target"
    assert len(result) == 1
    assert set(result.unique()).issubset({0, 1})
