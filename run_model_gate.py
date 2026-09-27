from bdp_model_gate import ModelGate, StructuredGateContext
import joblib
import json

from model_build import (
    DECISION_MAP, build_reason, X_train, X_val, y_val, prot_val,
    model_card
)

model = joblib.load("model_params.pkl")

def main():
    context = StructuredGateContext(
        model=model,
        X_train=X_train,
        X=X_val,
        y_true=y_val,
        y_pred=model.predict_proba(X_val)[:, 1],
        protected_df=prot_val,
        model_card=model_card,
    )

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
