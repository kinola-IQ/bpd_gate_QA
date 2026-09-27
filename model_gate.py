import json
import sys

import numpy as np
import pandas as pd
from bdp_model_gate import ModelGate, StructuredGateContext
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from xgboost import XGBClassifier

DECISION_MAP = {"PASS": "PASS", "NEEDS_REVIEW": "REVIEW", "BLOCKED": "BLOCK"}
MAX_DETAIL_LEN = 200  # keep each flag's detail short enough that the joined reason stays a sane size for a PR/issue body


def clean_detail(text: str) -> str:
    flat = " ".join(
        text.split()) 
    if len(flat) > MAX_DETAIL_LEN:
        flat = flat[:MAX_DETAIL_LEN - 3] + "..."
    return flat


def build_reason(report) -> str:
    if report.gate_status == "PASS":
        return "All checks passed."
    if report.gate_status == "BLOCKED":
        relevant = [f for f in report.flags if f.blocking]
    else:  # NEEDS_REVIEW
        relevant = report.flags
    return " | ".join(
        f"{f.check_name}: {clean_detail(f.detail)}" for f in relevant
    )


def main():
    loan_df = pd.read_csv("./loan_data_new.csv")

    loan_df.columns = (
        loan_df.columns.str.strip().str.replace(" ", "_").str.lower()
    )
    loan_df = loan_df.rename(columns={"home_onwership": "home_ownership"})

    textual_cols = loan_df.select_dtypes(include="object").columns
    for col in textual_cols:
        loan_df[col] = loan_df[col].str.lower()

    target_column = "loan_status"
    log_numeric_cols = ["person_income", "loan_amount"]
    plain_numeric_cols = ["employee_experience", "loan_interest_rate",
                          "loan_percentage", "credit_history", "credit_score"]
    ordinal_cols = ["education"]
    education_order = ["high school", "associate",
                       "bachelor", "master", "doctorate"]
    binary_cols = ["previous_loan"]
    nominal_cols = ["loan_intent", "home_ownership"]

    X = loan_df[log_numeric_cols + plain_numeric_cols +
                ordinal_cols + binary_cols + nominal_cols]
    y = loan_df[target_column]
    protected_df = loan_df[["age", "gender"]]

    X_train, X_val, y_train, y_val, prot_train, prot_val = train_test_split(
        X, y, protected_df, test_size=0.25, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(transformers=[
        ("log_num", FunctionTransformer(np.log1p, validate=False), log_numeric_cols),
        ("plain_num", StandardScaler(), plain_numeric_cols),
        ("ordinal", OrdinalEncoder(
            categories=[education_order]), ordinal_cols),
        ("binary", OrdinalEncoder(), binary_cols),
        ("nominal", OneHotEncoder(handle_unknown="ignore"), nominal_cols),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            n_estimators=200, 
            max_depth=4, 
            learning_rate=0.1,
            eval_metric="logloss", 
            random_state=42)
         ),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict_proba(X_val)[:, 1]

    model_card = {
        "use_case": "credit_scoring",
        "legal_basis": "Contractual necessity (NDPA 2023)",
        "data_minimization_justification": "Affordability signals only.",
        "training_data_source": "Internal loan dataset",
        "dpia_completed": False,          # <-- Yo! Dumininu, review, TRUE or FALSE?, switching supress the decision reasons 
        "influences_decision_about_person": True,
        "explainability_method": "SHAP",
        "validation_strategy": "stratified_split",
    }

    context = StructuredGateContext(
        model=pipeline, X=X_val, y_true=y_val, y_pred=y_pred,
        protected_df=prot_val, X_train=X_train,
        model_card=model_card, task="binary",
    )

    report = ModelGate().run(context)

    decision = DECISION_MAP[report.gate_status]
    reason = build_reason(report)

    print(json.dumps({"decision": decision, "reason": reason}))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({"decision": "BLOCK",
              "reason": f"run_model_gate.py crashed: {e}"}))
        sys.exit(1)
