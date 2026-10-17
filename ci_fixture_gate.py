"""
CI fixture for the BDP Model Gate workflow.

SYNTHETIC DATA. Purpose: prove that gate.yml routes PASS / REVIEW / BLOCK correctly.
It says nothing about any real model.
"""
import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from bdp_model_gate import GateConfig, ModelGate, StructuredGateContext
from bdp_model_gate.structured import default_structured_checks

warnings.filterwarnings("ignore")

DECISION_MAP = {"PASS": "PASS", "NEEDS_REVIEW": "REVIEW", "BLOCKED": "BLOCK"}
SEED = 7
SIGNAL = 1.8   # how strongly the features determine repayment
N_ROWS = 40000


def clean_detail(t, n=160):
    return " ".join(str(t).split())[:n]


def build_reason(report, max_items=5):
    if report.gate_status == "PASS":
        return "All checks passed."
    flags = [r for r in report.results if r.flag not in ("OK", "NOT_APPLICABLE")]
    if report.gate_status == "BLOCKED":
        flags = [r for r in flags if r.blocking]
    shown = " | ".join(f"{r.check_name}: {clean_detail(r.detail)}" for r in flags[:max_items])
    extra = len(flags) - max_items
    return shown + (f" | (+{extra} more)" if extra > 0 else "")


def make_data(scenario, n=N_ROWS, seed=SEED):
    """Synthetic loan-style data, ordered by time (month 0..23)."""
    rng = np.random.default_rng(seed)
    month = np.sort(rng.integers(0, 24, n))

    # --- protected attributes (never given to the model) ---
    gender = rng.choice(["female", "male"], n)
    age = np.clip(rng.normal(38, 11, n), 21, 70)

    # --- financial features ---
    income = rng.lognormal(10.8, 0.5, n)
    loan_amount = income * rng.uniform(0.2, 2.0, n)
    term = rng.choice([12, 24, 36, 48, 60], n)
    history = np.clip(rng.gamma(3, 2, n), 0, 30)
    debts = rng.poisson(1.5, n)
    util = rng.beta(2, 4, n)
    employment = rng.choice(["salaried", "self_employed", "contract"], n, p=[0.55, 0.30, 0.15])
    purpose = rng.choice(["business", "education", "home", "consumer"], n, p=[0.25, 0.15, 0.2, 0.4])

    # --- a sector feature: in 'review' it is a proxy for gender ---
    p_care = np.where(gender == "female", 0.70, 0.15) if scenario == "review" else np.full(n, 0.30)
    sector = np.where(rng.random(n) < p_care, "care_services", "other")

    # --- true repayment probability (stable over time) ---
    z = (
        0.4
        + 1.5 * (np.log(income) - 10.8)
        - 1.3 * (loan_amount / income)
        - 1.2 * util
        - 0.30 * debts
        + 0.10 * history
        - 0.010 * (term - 36)
        + np.where(employment == "salaried", 0.3, np.where(employment == "contract", -0.3, 0.0))
    )
    if scenario == "review":
        # historical label bias: same finances, lower repayment label for the proxy group
        z = z - 1.0 * (sector == "care_services")
    # stronger, centred signal so a decent model clears the default 0.80 AUC bar (~70% repay rate)
    z = 0.85 + SIGNAL * (z - z.mean())
    prob = 1 / (1 + np.exp(-z))
    y = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "income": income, "loan_amount": loan_amount, "loan_to_income": loan_amount / income,
        "term": term, "credit_history_years": history, "existing_debts": debts,
        "utilisation": util, "employment": employment, "purpose": purpose, "sector": sector,
    })
    df = pd.get_dummies(df, columns=["employment", "purpose", "sector"], dtype=float)
    protected = pd.DataFrame({
        "gender": gender,
        "age_band": pd.cut(age, [0, 29, 44, 120], labels=["under_30", "30_to_44", "45_plus"]).astype(str),
    })
    return df, pd.Series(y, name="repaid"), protected


def main(scenario):
    X, y, protected = make_data(scenario)

    # Out-of-time split: rows are time-ordered, so cut by position (no shuffle).
    cut = int(len(X) * 0.75)
    X_train, X_val = X.iloc[:cut], X.iloc[cut:]
    y_train, y_val = y.iloc[:cut], y.iloc[cut:]
    prot_val = protected.iloc[cut:]

    model = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.08, subsample=0.8, random_state=SEED
    ).fit(X_train, y_train)

    model_card = {
        # 'block' uses the high-risk label with no DPIA -> compliance must block it, for real.
        "use_case": "credit_scoring" if scenario == "block" else "ci_pipeline_test",
        "legal_basis": "Not applicable: synthetic data, no personal data",
        "data_minimization_justification": "Synthetic features only; protected attributes excluded from the model.",
        "training_data_source": "Synthetic CI fixture (ci_fixture_gate.py, seed 7); not real customer data",
        "dpia_completed": False,
        "influences_decision_about_person": True,
        "explainability_method": "SHAP",
        "validation_strategy": "out_of_time",
    }

    config = GateConfig()
    config.uncertainty.bootstrap_samples = 1500

    context = StructuredGateContext(
        model=model, X_train=X_train, X=X_val, y_true=y_val,
        y_pred=model.predict_proba(X_val)[:, 1],
        protected_df=prot_val, model_card=model_card, task="binary",
    )
    report = ModelGate(checks=default_structured_checks(config)).run(context)

    if "--debug" in sys.argv:
        for r in report.results:
            if r.flag not in ("OK", "NOT_APPLICABLE"):
                print(f"[{r.flag}] {r.check_name} blocking={r.blocking}: {clean_detail(r.detail, 220)}", file=sys.stderr)
        print("model_score:", report.model_score, file=sys.stderr)

    print(json.dumps({"decision": DECISION_MAP[report.gate_status], "reason": build_reason(report)}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["pass", "review", "block"], required=True)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    try:
        main(args.scenario)
    except Exception as e:  # a crash must never look like a pass
        print(json.dumps({"decision": "BLOCK", "reason": f"gate crashed: {clean_detail(e)}"}))