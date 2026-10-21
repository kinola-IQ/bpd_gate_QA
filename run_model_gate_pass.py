"""
PASS scenario (pass.yml): the synthetic adversarial LogisticRegression model
(adversarial_model.pkl, features x1..x3) on synthetic data.

Expected decision: PASS.

FIXES (2026-10-21):
  1. X_train and X_val were the SAME file (validation_data_adversarial.csv),
     so split_overlap saw 100% of validation rows in training and blocked.
     The model's original training set was not saved, so the file is now cut
     into two disjoint halves: the first half is the reference ("training")
     distribution for the drift / overlap checks, the second half is scored.
  2. It used the loan model's card (credit_scoring + stratified_split), which
     blocked on validation_strategy. It now has its own card describing this
     synthetic model honestly.
  3. protected_df was commented out; the synthetic data has no protected
     attributes, so it is simply not passed (fairness checks report N/A).
"""
import json

import joblib
import pandas as pd
from bdp_model_gate import ModelGate, StructuredGateContext

from model_build import DECISION_MAP, build_reason, clean_detail

MODEL_PATH = "adversarial_model.pkl"
DATA_PATH = "validation_data_adversarial.csv"
LABELS_PATH = "validation_labels_adversarial.csv"

# Card for THIS model. It is synthetic and not used on people, so it is not a
# high-risk use case and a random split is an acceptable validation strategy.
pass_model_card = {
    "use_case": "ci_pipeline_test",
    "legal_basis": "Not applicable: synthetic data, no personal data",
    "data_minimization_justification": "Three synthetic numeric features (x1..x3); no personal attributes.",
    "training_data_source": "Synthetic adversarial fixture (validation_data_adversarial.csv); not real customer data",
    "dpia_completed": False,  # not required: not a high-risk use case
    "influences_decision_about_person": False,
    "explainability_method": "SHAP",
    "validation_strategy": "random_split",
}


def main():
    model = joblib.load(MODEL_PATH)
    X = pd.read_csv(DATA_PATH)
    y = pd.read_csv(LABELS_PATH).iloc[:, 0]

    # Disjoint halves -> no train/validation overlap.
    half = len(X) // 2
    X_train = X.iloc[:half].reset_index(drop=True)
    X_val = X.iloc[half:].reset_index(drop=True)
    y_val = y.iloc[half:].reset_index(drop=True)

    context = StructuredGateContext(
        model=model,
        X_train=X_train,
        X=X_val,
        y_true=y_val,
        y_pred=model.predict_proba(X_val)[:, 1],
        model_card=pass_model_card,
        task="binary",
    )
    report = ModelGate().run(context)

    print(json.dumps({"decision": DECISION_MAP[report.gate_status], "reason": build_reason(report)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # a crash must never look like a pass
        print(json.dumps({"decision": "BLOCK", "reason": f"gate crashed: {clean_detail(e)}"}))
