import json
import pandas as pd
import joblib
from bdp_model_gate import ModelGate, StructuredGateContext
# from bdp_model_gate.structured import default_structured_checks (GateConfig)

from model_build import (
    DECISION_MAP,
    X_train_path,
    X_val_path,
    build_reason,
    model_card,
    prot_val_path,
    y_val_path,
)


model = joblib.load("model_params.pkl")
X_train = pd.read_csv(X_train_path)
X_val = pd.read_csv(X_val_path)
y_val = pd.read_csv(y_val_path).iloc[:, 0]
prot_val = pd.read_csv(prot_val_path)

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
    """
    This block ignores the shap_subgroup_gap to enable fast local execution
    PS: Do not uncomment before push!
    config = GateConfig()
    checks = [c for c in default_structured_checks(config) if c.name != "shap_subgroup_gap"]
    report = ModelGate(checks=checks).run(context)"""
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
