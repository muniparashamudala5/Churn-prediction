"""Load and clean the Telco Customer Churn dataset."""
import pandas as pd

TARGET = "Churn"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw(path: str = "data/raw/telco_churn.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges arrives as a string and has blank entries for customers
    # with tenure == 0 (brand new customers). Coerce to numeric and fill
    # those with 0, since they haven't been billed yet.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Normalize target to 0/1 (handles both plain object dtype and pandas'
    # newer nullable "string" dtype, which doesn't compare equal to `object`).
    if set(pd.Series(df[TARGET]).unique()) <= {"Yes", "No"}:
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)

    df = df.drop(columns=["customerID"], errors="ignore")
    df = df.dropna(subset=[TARGET])
    return df


def load_clean(path: str = "data/raw/telco_churn.csv") -> pd.DataFrame:
    return clean(load_raw(path))


def split_X_y(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    return X, y
