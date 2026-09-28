import json

import joblib
from bdp_model_gate import ModelGate, StructuredGateContext

from model_build import (
    DECISION_MAP,
    X_train_enc,
    X_val_enc,
    build_reason,
    model_card,
    prot_val,
    y_val,
)

model = joblib.load("model_params.pkl")
def main():
    context = StructuredGateContext(
        model=model,
        X_train=X_train_enc,
        X=X_val_enc,
        y_true=y_val,
        y_pred=model.predict_proba(X_val_enc)[:, 1],
        protected_df=prot_val,
        model_card=model_card,
    )
    # context = StructuredGateContext(
    # model=model, X=X_val_enc, y_true=y_val, y_pred=y_pred,
    # protected_df=prot_val, X_train=X_train_enc,
    # model_card=model_card, task="binary",)

    report = ModelGate().run(context)

    decision = DECISION_MAP[report.gate_status]
    reason = build_reason(report)

    result = {
        "decision": DECISION_MAP[report.gate_status],
        "reason": build_reason(report),
        }
    
    print(json.dumps(result))

if __name__ == "__main__":
    main()
