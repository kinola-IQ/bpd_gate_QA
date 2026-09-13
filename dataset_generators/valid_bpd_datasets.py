import numpy as np
import pandas as pd
from dataset_generator_utils import DatasetBundle, binary_target
def generate_fraud_dataset(
    n=5000,
    seed=42,
) -> DatasetBundle:

    rng = np.random.default_rng(seed)

    merchant_category = rng.choice(
        ["grocery", "electronics", "fashion", "gaming", "travel", "crypto"],
        n,
        p=[0.25, 0.18, 0.20, 0.10, 0.17, 0.10],
    )

    transaction_amount = rng.lognormal(
        mean=9.0,
        sigma=1.0,
        size=n,
    )

    transaction_hour = rng.integers(0, 24, n)

    account_age_days = rng.gamma(3, 250, n)

    device_change_count = rng.poisson(0.25, n)

    failed_login_count = rng.poisson(0.35, n)

    previous_chargebacks = rng.poisson(0.08, n)

    velocity_1h = rng.poisson(1.5, n)

    velocity_24h = velocity_1h + rng.poisson(4, n)

    merchant_risk = pd.Series(
        merchant_category
    ).map({
        "grocery": 0.1,
        "fashion": 0.15,
        "electronics": 0.30,
        "travel": 0.35,
        "gaming": 0.55,
        "crypto": 0.80,
    }).to_numpy()

    night_activity = (
        (transaction_hour <= 5) |
        (transaction_hour >= 23)
    ).astype(int)

    logit = (
        -5.0
        + 0.000006 * transaction_amount
        + 0.70 * device_change_count
        + 0.45 * failed_login_count
        + 1.30 * previous_chargebacks
        + 0.20 * velocity_1h
        + 0.08 * velocity_24h
        + 2.20 * merchant_risk
        + 0.80 * night_activity
        - 0.002 * account_age_days
    )

    y = binary_target(logit, rng)

    X = pd.DataFrame({
        "transaction_amount": transaction_amount.round(2),
        "transaction_hour": transaction_hour,
        "account_age_days": account_age_days.round(0),
        "device_change_count": device_change_count,
        "failed_login_count": failed_login_count,
        "previous_chargebacks": previous_chargebacks,
        "velocity_1h": velocity_1h,
        "velocity_24h": velocity_24h,
        "merchant_risk": merchant_risk.round(3),
    })

    return DatasetBundle(
        X=X,
        y=y,
        metadata={
            "problem": "fraud_detection",
            "positive_class": "fraud",
            "expected_imbalance": True,
        }
    )

def generate_churn_dataset(
    n=4000,
    seed=42,
) -> DatasetBundle:

    rng = np.random.default_rng(seed)

    monthly_bill = rng.lognormal(4.7, 0.45, n)

    contract_length = rng.choice(
        [1, 12, 24],
        n,
        p=[0.45, 0.35, 0.20],
    )

    support_tickets = rng.poisson(1.8, n)

    months_active = rng.integers(1, 96, n)

    data_usage_gb = rng.gamma(3, 5, n)

    payment_failures = rng.poisson(0.5, n)

    late_payment_count = rng.poisson(1.2, n)

    devices = rng.integers(1, 6, n)

    satisfaction = np.clip(
        rng.normal(6.5, 1.8, n),
        1,
        10,
    )

    logit = (
        -2.3
        + 0.35 * support_tickets
        - 0.025 * months_active
        - 0.85 * (contract_length == 24)
        - 0.35 * (contract_length == 12)
        + 0.50 * payment_failures
        + 0.18 * late_payment_count
        - 0.35 * satisfaction
        + 0.003 * monthly_bill
    )

    y = binary_target(logit, rng)

    X = pd.DataFrame({
        "monthly_bill": monthly_bill.round(2),
        "contract_length": contract_length,
        "support_tickets": support_tickets,
        "months_active": months_active,
        "data_usage_gb": data_usage_gb.round(2),
        "payment_failures": payment_failures,
        "late_payment_count": late_payment_count,
        "number_of_devices": devices,
        "satisfaction_score": satisfaction.round(2),
    })

    return DatasetBundle(
        X=X,
        y=y,
        metadata={"problem": "customer_churn"},
    )

def generate_credit_scoring_dataset(
    n=3000,
    seed=42,
) -> DatasetBundle:

    rng = np.random.default_rng(seed)

    region = rng.choice(
        ["Lagos", "Abuja", "Kano", "Port Harcourt"],
        n,
        p=[0.40, 0.25, 0.20, 0.15],
    )

    gender = rng.choice(
        ["F", "M"],
        n,
        p=[0.45, 0.55],
    )

    income = (
        rng.lognormal(11.6, 0.45, n)
        * pd.Series(region).map({
            "Lagos": 1.35,
            "Abuja": 1.20,
            "Port Harcourt": 1.00,
            "Kano": 0.70,
        }).to_numpy()
    )

    age = rng.integers(21, 65, n)

    months_employed = np.clip(
        rng.normal(48, 30, n),
        0,
        None,
    )

    existing_loans = rng.poisson(1.1, n)

    debt_to_income = np.clip(
        rng.beta(2, 5, n) * 1.4,
        0.01,
        0.95,
    )

    distance_to_branch = (
        pd.Series(region).map({
            "Lagos": 2.0,
            "Abuja": 3.5,
            "Port Harcourt": 6.0,
            "Kano": 14.0,
        }).to_numpy()
        + rng.normal(0, 1.1, n)
    )

    # Deliberate gender effect.
    logit = (
        -1.0
        + 3.0 * np.log(np.maximum(income, 1) / 50_000)
        - 6.0 * debt_to_income
        + 0.030 * months_employed
        - 0.50 * existing_loans
        + 0.90 * (gender == "M")
    )

    repaid = binary_target(logit, rng)

    X = pd.DataFrame({
        "monthly_income_ngn": income.round(2),
        "age": age.astype(float),
        "months_employed": months_employed.round(),
        "existing_loans": existing_loans.astype(float),
        "debt_to_income": debt_to_income.round(4),
        "distance_to_branch_km": distance_to_branch.round(2),
    })

    protected_df = pd.DataFrame({
        "gender": gender,
        "region": region,
    })

    return DatasetBundle(
        X=X,
        y=repaid,
        protected_df=protected_df,
        metadata={
            "problem": "credit_scoring",
            "proxy_feature": "distance_to_branch_km",
            "protected_attributes": ["gender", "region"],
            "ndpa_context": True,
        },
    )

def generate_insurance_claim_dataset(
    n=3500,
    seed=42,
) -> DatasetBundle:

    rng = np.random.default_rng(seed)

    claim_amount = rng.lognormal(10, 0.75, n)
    policy_age = rng.integers(1, 180, n)
    incident_type = rng.choice(
        ["collision", "theft", "fire", "weather"],
        n,
        p=[0.45, 0.25, 0.10, 0.20],
    )
    customer_tenure = rng.integers(1, 180, n)
    previous_claims = rng.poisson(0.7, n)
    vehicle_age = rng.integers(0, 20, n)
    repair_estimate = claim_amount * rng.uniform(0.6, 1.3, n)

    incident_risk = pd.Series(incident_type).map({
        "collision": 0.2,
        "theft": 0.5,
        "fire": 0.7,
        "weather": 0.3,
    }).to_numpy()

    logit = (
        1.0
        - 0.00001 * repair_estimate
        - 0.65 * previous_claims
        - 0.08 * incident_risk
        + 0.003 * customer_tenure
        - 0.02 * vehicle_age
    )

    y = binary_target(logit, rng)

    X = pd.DataFrame({
        "claim_amount": claim_amount.round(2),
        "policy_age_months": policy_age,
        "customer_tenure_months": customer_tenure,
        "previous_claims": previous_claims,
        "vehicle_age": vehicle_age,
        "repair_estimate": repair_estimate.round(2),
        "incident_risk": incident_risk.round(3),
    })

    protected_df = pd.DataFrame({
        "region": rng.choice(
            ["North", "Southwest", "Southeast"],
            n,
        ),
        "gender": rng.choice(["F", "M"], n),
    })

    return DatasetBundle(
        X=X,
        y=y,
        protected_df=protected_df,
        metadata={"problem": "insurance_claim_approval"},
    )

def generate_medical_diagnosis_dataset(
    n=5000,
    seed=42,
) -> DatasetBundle:

    rng = np.random.default_rng(seed)

    age = rng.integers(18, 90, n)

    blood_pressure = rng.normal(125, 20, n)
    cholesterol = rng.normal(190, 35, n)
    heart_rate = rng.normal(75, 12, n)
    glucose = rng.normal(105, 30, n)
    bmi = np.clip(rng.normal(27, 5, n), 15, 50)

    smoking_history = rng.binomial(1, 0.22, n)
    family_history = rng.binomial(1, 0.28, n)

    logit = (
        -7.0
        + 0.045 * age
        + 0.015 * blood_pressure
        + 0.008 * cholesterol
        + 0.010 * glucose
        + 0.08 * bmi
        + 0.9 * smoking_history
        + 1.1 * family_history
    )

    y = binary_target(logit, rng)

    X = pd.DataFrame({
        "age": age,
        "blood_pressure": blood_pressure.round(2),
        "cholesterol": cholesterol.round(2),
        "heart_rate": heart_rate.round(2),
        "glucose": glucose.round(2),
        "BMI": bmi.round(2),
        "smoking_history": smoking_history,
        "family_history": family_history,
    })

    protected_df = pd.DataFrame({
        "sex": rng.choice(["F", "M"], n),
        "age_group": pd.cut(
            age,
            bins=[0, 40, 60, 120],
            labels=["young", "middle", "older"],
        ),
    })

    return DatasetBundle(
        X=X,
        y=y,
        protected_df=protected_df,
        metadata={
            "problem": "medical_diagnosis",
            "critical_metric": "sensitivity",
        },
    )

def generate_employee_attrition_dataset(
    n=3000,
    seed=42,
) -> DatasetBundle:

    rng = np.random.default_rng(seed)

    salary = rng.lognormal(11.5, 0.45, n)

    years_at_company = rng.integers(0, 25, n)

    promotion_count = rng.poisson(
        np.maximum(years_at_company / 5, 0.1)
    )

    overtime_hours = rng.gamma(2, 8, n)

    satisfaction_score = np.clip(
        rng.normal(6.5, 1.8, n),
        1,
        10,
    )

    manager_changes = rng.poisson(1.4, n)

    training_hours = rng.gamma(2, 10, n)

    logit = (
        -1.8
        - 0.06 * years_at_company
        - 0.000003 * salary
        - 0.45 * satisfaction_score
        + 0.04 * overtime_hours
        + 0.16 * manager_changes
        - 0.08 * training_hours
        - 0.10 * promotion_count
    )

    y = binary_target(logit, rng)

    X = pd.DataFrame({
        "salary": salary.round(2),
        "years_at_company": years_at_company,
        "promotion_count": promotion_count,
        "overtime_hours": overtime_hours.round(2),
        "satisfaction_score": satisfaction_score.round(2),
        "manager_changes": manager_changes,
        "training_hours": training_hours.round(2),
    })

    protected_df = pd.DataFrame({
        "gender": rng.choice(["F", "M"], n),
        "age_group": rng.choice(
            ["18-29", "30-44", "45-60"],
            n,
        ),
        "region": rng.choice(
            ["Lagos", "Abuja", "Port Harcourt", "Kano"],
            n,
        ),
    })

    return DatasetBundle(
        X=X,
        y=y,
        protected_df=protected_df,
        metadata={"problem": "employee_attrition"},
    )

def generate_purchase_dataset(
    n=5000,
    seed=42,
) -> DatasetBundle:

    rng = np.random.default_rng(seed)

    session_duration = rng.gamma(2.5, 4, n)
    pages_viewed = rng.poisson(6, n) + 1
    cart_value = rng.lognormal(7, 1.0, n)
    previous_orders = rng.poisson(2, n)
    discount_seen = rng.binomial(1, 0.45, n)
    mobile_user = rng.binomial(1, 0.65, n)
    days_since_last_purchase = rng.integers(0, 720, n)

    logit = (
        -4
        + 0.08 * session_duration
        + 0.18 * pages_viewed
        + 0.00002 * cart_value
        + 0.45 * previous_orders
        + 0.35 * discount_seen
        - 0.002 * days_since_last_purchase
        - 0.15 * mobile_user
    )

    y = binary_target(logit, rng)

    X = pd.DataFrame({
        "session_duration": session_duration.round(2),
        "pages_viewed": pages_viewed,
        "cart_value": cart_value.round(2),
        "previous_orders": previous_orders,
        "discount_seen": discount_seen,
        "mobile_user": mobile_user,
        "days_since_last_purchase": days_since_last_purchase,
    })

    return DatasetBundle(
        X=X,
        y=y,
        metadata={"problem": "ecommerce_purchase_prediction"},
    )

def generate_recruitment_dataset(
    n=3000,
    seed=42,
) -> DatasetBundle:

    rng = np.random.default_rng(seed)

    years_experience = rng.integers(0, 15, n)
    technical_score = np.clip(rng.normal(70, 15, n), 0, 100)
    assessment_score = np.clip(rng.normal(68, 16, n), 0, 100)
    projects_count = rng.poisson(4, n)
    interview_score = np.clip(rng.normal(72, 13, n), 0, 100)
    certifications = rng.poisson(2, n)

    gender = rng.choice(["F", "M"], n)
    age_group = rng.choice(
        ["18-24", "25-34", "35-44", "45+"],
        n,
    )

    logit = (
        -10
        + 0.20 * years_experience
        + 0.055 * technical_score
        + 0.045 * assessment_score
        + 0.12 * projects_count
        + 0.045 * interview_score
        + 0.15 * certifications

        # Deliberately plant bias.
        + 0.55 * (gender == "M")
    )

    y = binary_target(logit, rng)

    X = pd.DataFrame({
        "years_experience": years_experience,
        "technical_score": technical_score.round(2),
        "assessment_score": assessment_score.round(2),
        "projects_count": projects_count,
        "interview_score": interview_score.round(2),
        "certifications": certifications,
    })

    protected_df = pd.DataFrame({
        "gender": gender,
        "age_group": age_group,
    })

    return DatasetBundle(
        X=X,
        y=y,
        protected_df=protected_df,
        metadata={
            "problem": "recruitment_screening",
            "fairness_sensitive": True,
        },
    )

def generate_spam_dataset(
    n=5000,
    seed=42,
) -> DatasetBundle:

    rng = np.random.default_rng(seed)

    spam_templates = [
        "Congratulations you won a prize claim now",
        "You have been selected for a reward",
        "Urgent account verification required",
        "Click this link to receive your bonus",
        "Exclusive offer available today",
        "You won free cash",
    ]

    ham_templates = [
        "Meeting moved to tomorrow morning",
        "Please send the project report",
        "Dinner is ready",
        "Your appointment is confirmed",
        "Can you review the document",
        "The package has arrived",
    ]

    labels = rng.binomial(1, 0.28, n)

    texts = []

    for label in labels:
        if label:
            base = rng.choice(spam_templates)
            extra = rng.choice([
                " click now",
                " urgent",
                " reply immediately",
                " verify now",
                " www.example.com",
            ])
        else:
            base = rng.choice(ham_templates)
            extra = rng.choice([
                " thanks",
                " see you then",
                " let me know",
                " regards",
                "",
            ])

        texts.append(base + extra)

    X = pd.DataFrame({
        "text": texts,
    })

    return DatasetBundle(
        X=X,
        y=pd.Series(labels, name="is_spam"),
        metadata={"problem": "spam_detection"},
    )

def generate_phishing_url_dataset(
    n=5000,
    seed=42,
) -> DatasetBundle:

    rng = np.random.default_rng(seed)

    legitimate_domains = [
        "google.com",
        "microsoft.com",
        "paypal.com",
        "amazon.com",
        "github.com",
    ]

    suspicious_domains = [
        "paypa1-security.com",
        "amazon-login-secure.com",
        "verify-account-now.net",
        "microsoft-support-login.xyz",
        "secure-banking-update.info",
    ]

    labels = rng.binomial(1, 0.25, n)

    urls = []

    for label in labels:
        if label:
            domain = rng.choice(suspicious_domains)

            url = (
                "http://"
                + domain
                + "/"
                + rng.choice([
                    "login",
                    "verify",
                    "account/update",
                    "secure/auth",
                ])
            )
        else:
            domain = rng.choice(legitimate_domains)
            url = f"https://{domain}/home"

        urls.append(url)

    url_length = np.array([len(x) for x in urls])
    subdomain_count = np.array([
        x.count(".") - 1
        for x in urls
    ])

    special_char_count = np.array([
        sum(c in "@?=&-%" for c in x)
        for x in urls
    ])

    digit_ratio = np.array([
        sum(c.isdigit() for c in x) / max(len(x), 1)
        for x in urls
    ])

    https_enabled = np.array([
        int(x.startswith("https"))
        for x in urls
    ])

    domain_age_days = np.where(
        labels == 1,
        rng.integers(1, 300, n),
        rng.integers(1000, 5000, n),
    )

    X = pd.DataFrame({
        "url": urls,
        "url_length": url_length,
        "subdomain_count": subdomain_count,
        "special_char_count": special_char_count,
        "digit_ratio": digit_ratio.round(4),
        "https_enabled": https_enabled,
        "domain_age_days": domain_age_days,
    })

    return DatasetBundle(
        X=X,
        y=pd.Series(labels, name="is_phishing"),
        metadata={"problem": "phishing_detection"},
    )

def generate_predictive_maintenance_dataset(
    n=10000,
    seed=42,
) -> DatasetBundle:

    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(
        "2025-01-01",
        periods=n,
        freq="h",
    )

    operating_hours = np.arange(n) / 24

    temperature = (
        60
        + 0.002 * operating_hours
        + rng.normal(0, 4, n)
    )

    vibration = (
        2
        + 0.0005 * operating_hours
        + rng.normal(0, 0.4, n)
    )

    pressure = rng.normal(100, 8, n)

    rpm = rng.normal(1800, 100, n)

    maintenance_age_days = (
        operating_hours % 180
    )

    logit = (
        -8
        + 0.05 * temperature
        + 0.9 * vibration
        + 0.02 * maintenance_age_days
        + 0.0001 * operating_hours
    )

    failure = binary_target(logit, rng)

    X = pd.DataFrame({
        "timestamp": timestamps,
        "temperature": temperature.round(3),
        "vibration": vibration.round(3),
        "pressure": pressure.round(2),
        "rpm": rpm.round(2),
        "operating_hours": operating_hours.round(2),
        "maintenance_age_days": maintenance_age_days.round(2),
    })

    return DatasetBundle(
        X=X,
        y=failure,
        metadata={
            "problem": "predictive_maintenance",
            "split_strategy": "temporal",
        },
    )

def generate_delivery_failure_dataset(
    n=5000,
    seed=42,
) -> DatasetBundle:

    rng = np.random.default_rng(seed)

    distance_km = rng.gamma(2, 4, n)

    estimated_duration_min = (
        distance_km * rng.normal(8, 2, n)
    )

    traffic_index = np.clip(
        rng.beta(2, 2, n),
        0,
        1,
    )

    weather_score = rng.uniform(0.4, 1.0, n)

    rider_experience_days = rng.integers(
        1,
        1500,
        n,
    )

    merchant_preparation_time = rng.gamma(2, 8, n)

    package_weight = rng.lognormal(1, 0.6, n)

    pickup_delay = rng.gamma(1.5, 5, n)

    logit = (
        -3
        + 0.08 * distance_km
        + 2.5 * traffic_index
        - 0.0005 * rider_experience_days
        + 0.04 * merchant_preparation_time
        + 0.015 * package_weight
        + 0.08 * pickup_delay
        - 0.9 * weather_score
    )

    y = binary_target(logit, rng)

    X = pd.DataFrame({
        "distance_km": distance_km.round(2),
        "estimated_duration_min":
            estimated_duration_min.round(2),
        "traffic_index": traffic_index.round(3),
        "weather_score": weather_score.round(3),
        "rider_experience_days": rider_experience_days,
        "merchant_preparation_time":
            merchant_preparation_time.round(2),
        "package_weight": package_weight.round(2),
        "pickup_delay": pickup_delay.round(2),
    })

    protected_df = pd.DataFrame({
        "rider_region": rng.choice(
            ["Lagos", "Ogun", "Oyo"],
            n,
        ),
        "merchant_type": rng.choice(
            ["restaurant", "retail", "grocery", "pharmacy"],
            n,
        ),
    })

    return DatasetBundle(
        X=X,
        y=y,
        protected_df=protected_df,
        metadata={"problem": "delivery_failure_prediction"},
    )

def generate_bnpl_dataset(
    n=4000,
    seed=42,
) -> DatasetBundle:

    rng = np.random.default_rng(seed)

    income = rng.lognormal(
        np.log(180_000),
        0.5,
        n,
    )

    purchase_amount = rng.lognormal(
        np.log(30_000),
        0.8,
        n,
    )

    previous_orders = rng.poisson(3, n)
    previous_defaults = rng.binomial(2, 0.15, n)
    account_age = rng.integers(1, 60, n)
    payment_frequency = rng.gamma(2, 3, n)

    debt_ratio = np.clip(
        rng.beta(2, 4, n),
        0,
        1,
    )

    logit = (
        -3
        + 0.00001 * purchase_amount
        - 0.000004 * income
        - 0.15 * previous_orders
        + 1.7 * previous_defaults
        + 0.10 * debt_ratio * 10
        - 0.02 * account_age
        - 0.06 * payment_frequency
    )

    y = binary_target(logit, rng)

    X = pd.DataFrame({
        "income": income.round(2),
        "purchase_amount": purchase_amount.round(2),
        "previous_orders": previous_orders,
        "previous_defaults": previous_defaults,
        "account_age_months": account_age,
        "payment_frequency": payment_frequency.round(2),
        "debt_ratio": debt_ratio.round(4),
    })

    protected_df = pd.DataFrame({
        "region": rng.choice(
            ["Lagos", "Kano", "Abuja", "Rivers"],
            n,
        ),
        "gender": rng.choice(["F", "M"], n),
    })

    return DatasetBundle(
        X=X,
        y=y,
        protected_df=protected_df,
        metadata={"problem": "bnpl_default"},
    )

def generate_proxy_dataset(
    n=3000,
    seed=42,
) -> DatasetBundle:

    rng = np.random.default_rng(seed)

    region = rng.choice(
        ["Lagos", "Abuja", "Kano", "PH"],
        n,
        p=[0.4, 0.25, 0.2, 0.15],
    )

    # This becomes the protected variable.
    gender = rng.choice(
        ["F", "M"],
        n,
    )

    # Proxy feature.
    region_code = pd.Series(region).map({
        "Lagos": 1,
        "Abuja": 2,
        "Kano": 3,
        "PH": 4,
    }).to_numpy()

    distance_to_branch = (
        region_code * 3
        + rng.normal(0, 0.2, n)
    )

    # Other legitimate-looking variables.
    income = rng.lognormal(
        np.log(180_000),
        0.5,
        n,
    )

    debt_ratio = rng.beta(2, 5, n)

    logit = (
        -2
        + 0.00001 * income
        - 4 * debt_ratio
        + 0.5 * region_code
    )

    y = binary_target(logit, rng)

    X = pd.DataFrame({
        "income": income.round(2),
        "debt_ratio": debt_ratio.round(4),
        "distance_to_branch_km":
            distance_to_branch.round(3),
    })

    protected_df = pd.DataFrame({
        "region": region,
        "gender": gender,
    })

    return DatasetBundle(
        X=X,
        y=y,
        protected_df=protected_df,
        metadata={
            "purpose": "proxy_correlation",
            "proxy": "distance_to_branch_km",
        },
    )

def generate_disparate_impact_dataset(
    n=4000,
    seed=42,
) -> DatasetBundle:

    rng = np.random.default_rng(seed)

    group = rng.choice(
        ["A", "B"],
        n,
        p=[0.5, 0.5],
    )

    qualification_score = np.clip(
        rng.normal(70, 15, n),
        0,
        100,
    )

    # Intentionally shift the outcome by group.
    group_effect = np.where(
        group == "A",
        1.0,
        -0.7,
    )

    logit = (
        -5
        + 0.08 * qualification_score
        + group_effect
    )

    y = binary_target(logit, rng)

    X = pd.DataFrame({
        "qualification_score":
            qualification_score.round(2),
    })

    protected_df = pd.DataFrame({
        "group": group,
    })

    return DatasetBundle(
        X=X,
        y=y,
        protected_df=protected_df,
        metadata={
            "purpose": "disparate_impact",
        },
    )

def generate_equalized_odds_dataset(
    n=5000,
    seed=42,
) -> DatasetBundle:

    rng = np.random.default_rng(seed)

    group = rng.choice(
        ["A", "B"],
        n,
    )

    signal = rng.normal(0, 1, n)

    # Group B has noisier measurements.
    noise_scale = np.where(
        group == "A",
        0.35,
        1.20,
    )

    observed_signal = (
        signal
        + rng.normal(0, noise_scale)
    )

    y = (signal > 0.25).astype(int)

    X = pd.DataFrame({
        "observed_signal":
            observed_signal.round(4),
    })

    protected_df = pd.DataFrame({
        "group": group,
    })

    return DatasetBundle(
        X=X,
        y=pd.Series(y, name="target"),
        protected_df=protected_df,
        metadata={
            "purpose": "equalized_odds",
            "group_specific_noise": True,
        },
    )

def generate_calibration_dataset(
    n=6000,
    seed=42,
) -> DatasetBundle:

    rng = np.random.default_rng(seed)

    x = rng.normal(0, 1, n)

    true_probability = sigmoid(
        -0.8 + 1.3 * x
    )

    y = rng.binomial(
        1,
        true_probability,
        n,
    )

    # Simulate a badly calibrated model output.
    predicted_probability = np.clip(
        0.5 * true_probability + 0.25,
        0,
        1,
    )

    X = pd.DataFrame({
        "signal": x.round(4),
    })

    X["true_probability"] = true_probability
    X["predicted_probability"] = predicted_probability

    return DatasetBundle(
        X=X,
        y=pd.Series(y, name="target"),
        metadata={
            "purpose": "calibration",
            "known_true_probability": True,
        },
    )


def generate_adversarial_dataset(
    n=5000,
    seed=42,
) -> DatasetBundle:

    rng = np.random.default_rng(seed)

    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)

    logit = (
        2.2 * x1
        + 1.4 * x2
        - 0.8 * x3
    )

    y = (logit > 0).astype(int)

    X = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "x3": x3,
    })

    return DatasetBundle(
        X=X,
        y=pd.Series(y, name="target"),
        metadata={
            "purpose": "adversarial_robustness",
            "perturbation_epsilon": 0.03,
        },
    )

def generate_ng_fintech_dataset(
    n=5000,
    seed=42,
) -> DatasetBundle:

    rng = np.random.default_rng(seed)

    region = rng.choice(
        [
            "Lagos",
            "Abuja",
            "Kano",
            "Rivers",
            "Oyo",
            "Enugu",
        ],
        n,
        p=[
            0.30,
            0.17,
            0.15,
            0.14,
            0.12,
            0.12,
        ],
    )

    gender = rng.choice(
        ["F", "M"],
        n,
        p=[0.48, 0.52],
    )

    age = rng.integers(18, 70, n)

    income = rng.lognormal(
        np.log(180_000),
        0.65,
        n,
    )

    months_employed = np.clip(
        rng.normal(55, 35, n),
        0,
        360,
    )

    existing_loans = rng.poisson(1.2, n)

    debt_to_income = np.clip(
        rng.beta(2, 5, n),
        0.01,
        0.95,
    )

    account_age = rng.integers(1, 120, n)

    previous_defaults = rng.binomial(
        3,
        0.08,
        n,
    )

    distance_to_branch = (
        pd.Series(region).map({
            "Lagos": 2,
            "Abuja": 3,
            "Kano": 14,
            "Rivers": 6,
            "Oyo": 9,
            "Enugu": 8,
        }).to_numpy()
        + rng.normal(0, 1.2, n)
    )

    # Deliberate bias.
    gender_effect = 0.65 * (gender == "M")

    # Deliberate nonlinear risk.
    logit = (
        -2
        + 2.5 * np.log(np.maximum(income, 1) / 100_000)
        - 5.5 * debt_to_income
        + 0.02 * months_employed
        - 0.45 * existing_loans
        - 0.015 * account_age
        - 1.2 * previous_defaults
        + gender_effect
    )

    default = binary_target(logit, rng)

    X = pd.DataFrame({
        "monthly_income_ngn": income.round(2),
        "age": age,
        "months_employed": months_employed.round(),
        "existing_loans": existing_loans,
        "debt_to_income": debt_to_income.round(4),
        "account_age_months": account_age,
        "previous_defaults": previous_defaults,
        "distance_to_branch_km":
            distance_to_branch.round(2),
    })

    protected_df = pd.DataFrame({
        "gender": gender,
        "region": region,
        "age_group": pd.cut(
            age,
            bins=[0, 25, 35, 50, 120],
            labels=[
                "18-25",
                "26-35",
                "36-50",
                "51+",
            ],
        ),
    })

    return DatasetBundle(
        X=X,
        y=default,
        protected_df=protected_df,
        metadata={
            "problem": "ng_fintech_credit_risk",
            "country": "Nigeria",
            "deliberate_proxy": "distance_to_branch_km",
            "deliberate_bias": "gender_effect",
        },
    )