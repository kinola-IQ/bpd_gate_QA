import numpy as np
import pandas as pd
from dataset_generator_utils import (
    DatasetBundle, ordinal_target, multiclass_target,
)


def generate_underwriting_dataset(
    n=4000,
    seed=42,
) -> DatasetBundle:
    """
    Ordinal 3-class underwriting outcome: decline < refer < accept.
    Deliberate proxy (distance_to_branch_km -> region) and a
    deliberate gender effect, same shape as the binary credit-scoring
    generator, so proxy_correlation / disparate_impact / equalised_odds
    have a known-answer target on the ordinal path too.
    """
    rng = np.random.default_rng(seed)
    class_order = ["decline", "refer", "accept"]

    region = rng.choice(
        ["Lagos", "Abuja", "Kano", "Port Harcourt"],
        n, p=[0.40, 0.25, 0.20, 0.15],
    )
    gender = rng.choice(["F", "M"], n, p=[0.45, 0.55])

    income = (
        rng.lognormal(11.6, 0.45, n)
        * pd.Series(region).map({
            "Lagos": 1.35, "Abuja": 1.20,
            "Port Harcourt": 1.00, "Kano": 0.70,
        }).to_numpy()
    )
    debt_to_income = np.clip(rng.beta(2, 5, n) * 1.4, 0.01, 0.95)
    months_employed = np.clip(rng.normal(48, 30, n), 0, None)
    existing_loans = rng.poisson(1.1, n)

    distance_to_branch = (
        pd.Series(region).map({
            "Lagos": 2.0, "Abuja": 3.5,
            "Port Harcourt": 6.0, "Kano": 14.0,
        }).to_numpy()
        + rng.normal(0, 1.1, n)
    )

    # Continuous risk index -> cut into 3 ordered outcomes.
    # Deliberate gender effect, same as the binary version.
    risk_index = (
        3.0 * np.log(np.maximum(income, 1) / 50_000)
        - 6.0 * debt_to_income
        + 0.03 * months_employed
        - 0.5 * existing_loans
        + 0.9 * (gender == "M")
    )

    y = ordinal_target(
        risk_index,
        thresholds=[-1.0, 1.0],   # 2 cuts -> 3 classes
        class_labels=class_order,
        rng=rng,
        noise_scale=0.4,
    )

    X = pd.DataFrame({
        "monthly_income_ngn": income.round(2),
        "months_employed": months_employed.round(),
        "existing_loans": existing_loans.astype(float),
        "debt_to_income": debt_to_income.round(4),
        "distance_to_branch_km": distance_to_branch.round(2),
    })

    protected_df = pd.DataFrame({"gender": gender, "region": region})

    return DatasetBundle(
        X=X, y=y, protected_df=protected_df,
        metadata={
            "problem": "underwriting_tiering",
            "task": "multiclass",
            "class_order": class_order,
            "proxy_feature": "distance_to_branch_km",
            "protected_attributes": ["gender", "region"],
        },
    )


def generate_ordinal_kappa_dataset(
    n=3000,
    seed=42,
) -> DatasetBundle:
    """
    Known-answer test for quadratic_kappa vs plain accuracy.
    Every prediction is exactly one step off in the SAME direction --
    accuracy is uniformly bad, but errors are all "near misses", so
    quadratic_kappa should score noticeably better than accuracy would
    suggest. Ships y_true and a fixed y_pred column together (metadata
    carries the intended prediction) rather than a model to fit.
    """
    rng = np.random.default_rng(seed)
    class_order = ["low", "medium", "high"]

    x = rng.normal(0, 1, n)
    y_true = ordinal_target(
        x, thresholds=[-0.4, 0.4],
        class_labels=class_order, rng=rng,
    )

    # Deliberately shift every true class up by one step (near misses only).
    shift = {"low": "medium", "medium": "high", "high": "high"}
    y_pred_fixed = y_true.map(shift)

    X = pd.DataFrame({"signal": x.round(4)})
    X["y_pred_fixed"] = y_pred_fixed.to_numpy()  # carried alongside, not a feature to train on

    return DatasetBundle(
        X=X, y=y_true,
        metadata={
            "purpose": "quadratic_kappa_vs_accuracy",
            "task": "multiclass",
            "class_order": class_order,
            "note": "X['y_pred_fixed'] is the intended y_pred -- drop it "
                    "before fitting anything, use it directly as y_pred.",
        },
    )


def generate_nominal_product_category_dataset(
    n=5000,
    seed=42,
) -> DatasetBundle:
    """
    Clean nominal (no order) multiclass baseline -- loan_intent-style
    categories with no natural ranking. Used to confirm that requesting
    an ordinal-only metric (quadratic_kappa, ordinal_mae) on a nominal
    task raises GateConfigurationError -> blocking CHECK_ERROR, since
    there is no class_order to compute "how wrong" against. Use
    metric="accuracy"/"f1"/etc for this dataset's actual gate runs.
    """
    rng = np.random.default_rng(seed)
    categories = ["education", "medical", "business", "personal", "home_improvement"]

    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)

    logits = np.column_stack([
        0.8 * x1 - 0.3 * x2,
        -0.5 * x1 + 0.9 * x3,
        0.4 * x2 + 0.4 * x3,
        -0.2 * x1 - 0.2 * x2,
        0.3 * x1 + 0.3 * x2 - 0.3 * x3,
    ])
    y = multiclass_target(logits, categories, rng)

    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

    return DatasetBundle(
        X=X, y=y,
        metadata={
            "problem": "loan_intent_classification",
            "task": "multiclass",
            "class_order": None,  # nominal -- no order to give the gate
        },
    )


def generate_multiclass_disparate_impact_dataset(
    n=4500,
    seed=42,
) -> DatasetBundle:
    """
    3-class ordinal outcome with a deliberate group effect on tier
    assignment -- known-answer trigger for disparate_impact /
    equalised_odds on a multiclass task (checked per-class /
    favourable-class, not just a single binary rate).
    """
    rng = np.random.default_rng(seed)
    class_order = ["decline", "refer", "accept"]

    group = rng.choice(["A", "B"], n, p=[0.5, 0.5])
    qualification = np.clip(rng.normal(70, 15, n), 0, 100)

    group_effect = np.where(group == "A", 8.0, -8.0)
    risk_index = 0.5 * qualification + group_effect

    y = ordinal_target(
        risk_index, thresholds=[29, 41],
        class_labels=class_order, rng=rng, noise_scale=3.0,
    )

    X = pd.DataFrame({"qualification_score": qualification.round(2)})
    protected_df = pd.DataFrame({"group": group})

    return DatasetBundle(
        X=X, y=y, protected_df=protected_df,
        metadata={
            "purpose": "multiclass_disparate_impact",
            "task": "multiclass",
            "class_order": class_order,
            "favourable_classes": ["accept"],
        },
    )
