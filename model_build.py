"""
Loan-approval model build (used by the BLOCK scenario: block.yml).

Importing this module is cheap: it only exposes the shared helpers, the file
paths and the loan model card. Training and writing the artifacts only happens
when it is run as a script (`python model_build.py`), which is what block.yml
does before running the gate.

FIX (2026-10-21): previously the data load, preprocessing and XGBoost training
ran at import time, and every runner (pass / review / block) imported the same
`model_card` from here. That single card (credit_scoring + stratified_split)
made every scenario BLOCK on `validation_strategy`. Each runner now owns a
model card that describes its own model; this file only holds the loan one.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

# ---------------------------------------------------------------------------
# Shared helpers (imported by the run_model_gate_*.py scripts)
# ---------------------------------------------------------------------------
DECISION_MAP = {
    "PASS": "PASS",
    "NEEDS_REVIEW": "REVIEW",
    "BLOCKED": "BLOCK",
}

MAX_DETAIL_LEN = 200


def clean_detail(text) -> str:
    """Flatten and truncate long governance findings."""
    flat = " ".join(str(text).split())
    if len(flat) > MAX_DETAIL_LEN:
        flat = flat[: MAX_DETAIL_LEN - 3] + "..."
    return flat


def build_reason(report, max_items: int = 5) -> str:
    """Build a compact governance reason string.

    On BLOCK only the blocking findings are listed (they are what must be
    fixed); on REVIEW every non-OK finding is listed.
    """
    if report.gate_status == "PASS":
        return "All checks passed."

    flags = list(report.flags)
    if report.gate_status == "BLOCKED":
        flags = [flag for flag in flags if flag.blocking]

    shown = " | ".join(
        f"{flag.check_name}: {clean_detail(flag.detail)}" for flag in flags[:max_items]
    )
    extra = len(flags) - max_items
    return shown + (f" | (+{extra} more)" if extra > 0 else "")


# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------
MODEL_PATH = "model_params.pkl"
X_train_path = "X_train.csv"
X_val_path = "X_val.csv"
y_val_path = "y_val.csv"
prot_val_path = "prot_val.csv"

# ---------------------------------------------------------------------------
# Model card for THIS model (loan approval / XGBoost)
# ---------------------------------------------------------------------------
# FIX: the old card claimed "Internal loan dataset", "Contractual necessity"
# and dpia_completed=True ("leave at True, else compliance will fail at push").
# loan_data_new.csv is a public loan-approval dataset with no customer data,
# and no DPIA exists for it, so the card now says so. A model card that is
# edited to make the gate pass defeats the gate.
#
# validation_strategy stays "stratified_split" because that is what this file
# actually does: the dataset has no date column, so an out-of-time split is not
# possible. credit_scoring is high-risk, so the gate is RIGHT to block this
# model until it is validated out-of-time and a DPIA is done. That is the
# expected outcome of the BLOCK scenario.
loan_model_card = {
    "use_case": "credit_scoring",
    "legal_basis": "Not applicable: public loan-approval dataset, no company-X customer data",
    "data_minimization_justification": "Affordability signals only; age and gender excluded from the model.",
    "training_data_source": "Public loan-approval dataset (loan_data_new.csv); not internal customer data",
    "dpia_completed": False,
    "influences_decision_about_person": True,
    "explainability_method": "SHAP",
    "validation_strategy": "stratified_split",
}

# Backwards-compatible alias for anything that still imports `model_card`.
model_card = loan_model_card

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
target_column = "loan_status"
log_numeric_cols = ["person_income", "loan_amount"]
plain_numeric_cols = [
    "employee_experience",
    "loan_interest_rate",
    "loan_percentage",
    "credit_history",
    "credit_score",
]
ordinal_cols = ["education"]
education_order = ["high school", "associate", "bachelor", "master", "doctorate"]
binary_cols = ["previous_loan"]
nominal_cols = ["loan_intent", "home_ownership"]


def load_loan_data(path: str = "./loan_data_new.csv") -> pd.DataFrame:
    loan_df = pd.read_csv(path)
    loan_df.columns = loan_df.columns.str.strip().str.replace(" ", "_").str.lower()
    loan_df = loan_df.rename(columns={"home_onwership": "home_ownership"})
    for column in loan_df.select_dtypes(include="object").columns:
        loan_df[column] = loan_df[column].str.lower()
    return loan_df


def band_age(age: pd.Series) -> pd.Series:
    """FIX: raw age gave the fairness checks one group per year of age
    (e.g. "43 is least likely to be approved"). Band it, like the fixture does."""
    return pd.cut(
        age, [0, 29, 44, 59, 200], labels=["under_30", "30_to_44", "45_to_59", "60_plus"]
    ).astype(str)


def build():
    # Imported here so that importing this module does not need xgboost.
    from xgboost import XGBClassifier

    loan_df = load_loan_data()

    X = loan_df[log_numeric_cols + plain_numeric_cols + ordinal_cols + binary_cols + nominal_cols]
    y = loan_df[target_column]
    protected_df = pd.DataFrame(
        {"age_band": band_age(loan_df["age"]), "gender": loan_df["gender"]}
    )

    X_train, X_val, y_train, y_val, _, prot_val = train_test_split(
        X, y, protected_df, test_size=0.25, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "log_num",
                FunctionTransformer(np.log1p, validate=False, feature_names_out="one-to-one"),
                log_numeric_cols,
            ),
            ("plain_num", StandardScaler(), plain_numeric_cols),
            ("ordinal", OrdinalEncoder(categories=[education_order]), ordinal_cols),
            ("binary", OrdinalEncoder(), binary_cols),
            ("nominal", OneHotEncoder(handle_unknown="ignore"), nominal_cols),
        ]
    )
    preprocessor.fit(X_train)
    feature_names = preprocessor.get_feature_names_out()

    X_train_enc = pd.DataFrame(preprocessor.transform(X_train), columns=feature_names, index=X_train.index)
    X_val_enc = pd.DataFrame(preprocessor.transform(X_val), columns=feature_names, index=X_val.index)

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="logloss", random_state=42
    )
    model.fit(X_train_enc, y_train)

    joblib.dump(model, MODEL_PATH)
    X_train_enc.to_csv(X_train_path, index=False)
    X_val_enc.to_csv(X_val_path, index=False)
    y_val.to_csv(y_val_path, index=False)
    prot_val.to_csv(prot_val_path, index=False)


if __name__ == "__main__":
    build()