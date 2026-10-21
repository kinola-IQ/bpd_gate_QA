"""
REVIEW scenario (review.yml): a synthetic loan-style model with a planted
proxy-for-gender feature (`sector`) and historical label bias.

Expected decision: REVIEW (non-blocking proxy_correlation / equalised_odds
findings, nothing blocking).

FIXES (2026-10-21):
  1. It used the loan model's shared card (credit_scoring + stratified_split),
     which always blocked on validation_strategy.
  2. review.yml never ran model_build.py, so it read stale committed CSVs
     (raw German-credit data with text columns) and XGBoost crashed.
  3. The loan model cannot produce REVIEW honestly: it is credit_scoring
     (high-risk), the data has no date column (so no out-of-time split), and
     it fails adversarial_robustness. Those are blocking, which is the BLOCK
     scenario. This scenario therefore uses the synthetic generator from
     ci_fixture_gate.py, which is time-ordered (real out-of-time split) and is
     built so that only non-blocking fairness/proxy findings fire.
"""
import json
import warnings

from bdp_model_gate import GateConfig, ModelGate, StructuredGateContext
from bdp_model_gate.structured import default_structured_checks
from sklearn.ensemble import GradientBoostingClassifier

from ci_fixture_gate import SEED, make_data
from model_build import DECISION_MAP, build_reason, clean_detail

warnings.filterwarnings("ignore")

# Card for THIS model: synthetic, so not a high-risk use case; validated
# out-of-time because the rows are ordered by month and cut by position.
review_model_card = {
    "use_case": "ci_pipeline_test",
    "legal_basis": "Not applicable: synthetic data, no personal data",
    "data_minimization_justification": "Synthetic features only; protected attributes excluded from the model.",
    "training_data_source": "Synthetic CI fixture (ci_fixture_gate.make_data('review'), seed 7); not real customer data",
    "dpia_completed": False,  # not required: not a high-risk use case
    "influences_decision_about_person": True,
    "explainability_method": "SHAP",
    "validation_strategy": "out_of_time",
}


def main():
    X, y, protected = make_data("review")

    # Out-of-time split: rows are time-ordered, so cut by position (no shuffle).
    cut = int(len(X) * 0.75)
    X_train, X_val = X.iloc[:cut], X.iloc[cut:]
    y_train, y_val = y.iloc[:cut], y.iloc[cut:]
    prot_val = protected.iloc[cut:]

    model = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.08, subsample=0.8, random_state=SEED
    ).fit(X_train, y_train)

    # proxy_correlation needs enough permutations to reach significance across
    # its comparisons; 1000 (the default) is too few here, 1500 is enough.
    config = GateConfig()
    config.uncertainty.bootstrap_samples = 1500

    context = StructuredGateContext(
        model=model,
        X_train=X_train,
        X=X_val,
        y_true=y_val,
        y_pred=model.predict_proba(X_val)[:, 1],
        protected_df=prot_val,
        model_card=review_model_card,
        task="binary",
    )
    report = ModelGate(checks=default_structured_checks(config)).run(context)

    print(json.dumps({"decision": DECISION_MAP[report.gate_status], "reason": build_reason(report)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # a crash must never look like a pass
        print(json.dumps({"decision": "BLOCK", "reason": f"gate crashed: {clean_detail(e)}"}))
