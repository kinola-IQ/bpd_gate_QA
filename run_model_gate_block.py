"""
BLOCK scenario (block.yml): the loan-approval XGBoost model built by model_build.py.

block.yml runs `python model_build.py` first, so the CSVs and model_params.pkl
read here are always freshly built from the same code.

Expected decision: BLOCK, for real reasons:
  - compliance_mapping:     credit_scoring is high-risk and no DPIA was done
  - validation_strategy:    high-risk model validated with a random (stratified) split
  - adversarial_robustness: prediction flip rate under perturbation above 0.05
"""
import json

import joblib
import pandas as pd
from bdp_model_gate import ModelGate, StructuredGateContext

from model_build import (
    DECISION_MAP,
    MODEL_PATH,
    X_train_path,
    X_val_path,
    build_reason,
    clean_detail,
    loan_model_card,  # FIX: the card that describes THIS model, not a shared one
    prot_val_path,
    y_val_path,
)


def main():
    # FIX: loading moved inside main() so a load error is caught below and
    # reported as BLOCK instead of crashing the step with an empty JSON file.
    model = joblib.load(MODEL_PATH)
    X_train = pd.read_csv(X_train_path)
    X_val = pd.read_csv(X_val_path)
    y_val = pd.read_csv(y_val_path).iloc[:, 0]
    prot_val = pd.read_csv(prot_val_path)

    context = StructuredGateContext(
        model=model,
        X_train=X_train,
        X=X_val,
        y_true=y_val,
        y_pred=model.predict_proba(X_val)[:, 1],
        protected_df=prot_val,
        model_card=loan_model_card,
        task="binary",
    )
    # Full default check set. Do not drop checks (e.g. shap_subgroup_gap) in CI.
    report = ModelGate().run(context)

    print(json.dumps({"decision": DECISION_MAP[report.gate_status], "reason": build_reason(report)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # a crash must never look like a pass
        print(json.dumps({"decision": "BLOCK", "reason": f"gate crashed: {clean_detail(e)}"}))
