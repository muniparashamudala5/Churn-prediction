"""
Generates a synthetic dataset that mirrors the schema of the Kaggle
"Telco Customer Churn" dataset (https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

This is ONLY a stand-in so the rest of the pipeline can be run and tested
end-to-end immediately. For your actual project / resume, download the
real dataset from Kaggle and save it as:

    data/raw/telco_churn.csv

The real file has the exact same columns this script generates, so no
other code needs to change.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 2000


def choice(options, p=None, size=N):
    return RNG.choice(options, size=size, p=p)


def generate():
    gender = choice(["Male", "Female"])
    senior = choice([0, 1], p=[0.84, 0.16])
    partner = choice(["Yes", "No"])
    dependents = choice(["Yes", "No"], p=[0.3, 0.7])
    tenure = RNG.integers(0, 73, size=N)

    phone_service = choice(["Yes", "No"], p=[0.9, 0.1])
    multiple_lines = np.where(
        phone_service == "No", "No phone service", choice(["Yes", "No"])
    )
    internet_service = choice(["DSL", "Fiber optic", "No"], p=[0.35, 0.44, 0.21])

    def dependent_on_internet(col_p_yes=0.5):
        out = choice(["Yes", "No"], p=[col_p_yes, 1 - col_p_yes])
        out = np.where(internet_service == "No", "No internet service", out)
        return out

    online_security = dependent_on_internet(0.35)
    online_backup = dependent_on_internet(0.4)
    device_protection = dependent_on_internet(0.4)
    tech_support = dependent_on_internet(0.35)
    streaming_tv = dependent_on_internet(0.45)
    streaming_movies = dependent_on_internet(0.45)

    contract = choice(["Month-to-month", "One year", "Two year"], p=[0.55, 0.24, 0.21])
    paperless_billing = choice(["Yes", "No"], p=[0.6, 0.4])
    payment_method = choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        p=[0.34, 0.23, 0.22, 0.21],
    )

    base_charge = np.where(internet_service == "Fiber optic", 70, np.where(internet_service == "DSL", 45, 20))
    addon_count = (
        (online_security == "Yes").astype(int)
        + (online_backup == "Yes").astype(int)
        + (device_protection == "Yes").astype(int)
        + (tech_support == "Yes").astype(int)
        + (streaming_tv == "Yes").astype(int)
        + (streaming_movies == "Yes").astype(int)
    )
    monthly_charges = base_charge + addon_count * 5 + RNG.normal(0, 5, N)
    monthly_charges = np.clip(monthly_charges, 18, 120).round(2)
    total_charges = (monthly_charges * tenure + RNG.normal(0, 20, N)).round(2)
    total_charges = np.clip(total_charges, 0, None)

    # Introduce a few blank strings in TotalCharges, like the real dataset does
    # for customers with tenure == 0.
    total_charges_str = total_charges.astype(str)
    total_charges_str[tenure == 0] = " "

    # Build churn probability from a logistic-ish combination of risk factors,
    # so the target is actually learnable and imbalanced like the real data.
    risk = (
        -0.04 * tenure
        + 0.015 * monthly_charges
        + np.where(contract == "Month-to-month", 1.2, 0)
        + np.where(contract == "One year", 0.3, 0)
        + np.where(payment_method == "Electronic check", 0.5, 0)
        + np.where(internet_service == "Fiber optic", 0.4, 0)
        + np.where(tech_support == "No", 0.3, 0)
        + np.where(dependents == "No", 0.15, 0)
        - 2.2
    )
    prob_churn = 1 / (1 + np.exp(-risk))
    churn = (RNG.random(N) < prob_churn).astype(int)
    churn_label = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame(
        {
            "customerID": [f"{i:04d}-SYNTH" for i in range(N)],
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges_str,
            "Churn": churn_label,
        }
    )
    return df


if __name__ == "__main__":
    df = generate()
    out_path = "data/raw/telco_churn.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df["Churn"].value_counts(normalize=True))
