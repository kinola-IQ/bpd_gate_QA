import pandas as pd
import numpy as np
from dataset_generator_utils import DatasetBundle

def corrupt_X_to_numpy(bundle: DatasetBundle):
    """This corresponds to the DataFrame-type validation scenario"""
    return DatasetBundle(
        X=bundle.X.to_numpy(),
        y=bundle.y,
        protected_df=bundle.protected_df,
        metadata=bundle.metadata,
    )

def corrupt_target_length(bundle: DatasetBundle):
    """for Row mismatch scenerios"""
    return DatasetBundle(
        X=bundle.X,
        y=bundle.y.iloc[:-10].reset_index(drop=True),
        protected_df=bundle.protected_df,
        metadata=bundle.metadata,
    )

def make_single_class_target(
    bundle: DatasetBundle,
    positive=True,
):

    value = 1 if positive else 0

    return DatasetBundle(
        X=bundle.X,
        y=pd.Series(
            np.full(len(bundle.X), value),
            name="target",
        ),
        protected_df=bundle.protected_df,
        metadata=bundle.metadata,
    )

def corrupt_protected_alignment(bundle: DatasetBundle):
    """Protected dataframe mismatch"""
    bad_protected = bundle.protected_df.iloc[:10].copy()

    return DatasetBundle(
        X=bundle.X,
        y=bundle.y,
        protected_df=pd.DataFrame(bad_protected),
        metadata=bundle.metadata,
    )

def invalid_model_card():
    """Invalid model-card dataset metadata"""
    return {
        "model_name": None,
        "version": "",
        "use_case": 123,
    }

def make_invalid_latencies(n=100):
    values = np.random.default_rng(42).gamma(
        9.0,
        8.5,
        n,
    )

    values[0] = -3.0

    return values