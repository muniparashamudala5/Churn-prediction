"""
Train a churn prediction model end to end:
  1. Load + clean data
  2. Train/test split
  3. Preprocessing pipeline (scaling + one-hot encoding)
  4. Baseline: Logistic Regression
  5. Main model: XGBoost (falls back to sklearn's HistGradientBoostingClassifier
     if xgboost isn't installed, so this script always runs)
  6. Randomized hyperparameter search
  7. Evaluate on held-out test set
  8. Save the fitted pipeline to models/churn_pipeline.joblib
"""
import json
import sys

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, ".")
from src.data_prep import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    load_clean,
    split_X_y,
)

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_model(scale_pos_weight: float):
    if HAS_XGBOOST:
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
        )
    # Fallback used only when xgboost isn't available in this environment.
    # Swap this out for XGBClassifier above once you `pip install xgboost`.
    print("[warn] xgboost not installed — falling back to HistGradientBoostingClassifier")
    return HistGradientBoostingClassifier(random_state=42)


def param_distributions():
    if HAS_XGBOOST:
        return {
            "model__n_estimators": [100, 200, 300, 400],
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        }
    return {
        "model__max_iter": [100, 150, 200, 300],
        "model__max_depth": [None, 3, 5, 8],
        "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
        "model__l2_regularization": [0.0, 0.1, 0.5, 1.0],
    }


def evaluate(name, pipeline, X_test, y_test):
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, probs), 4),
    }
    print(f"\n--- {name} ---")
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, preds, target_names=["No churn", "Churn"]))
    return metrics


def main():
    df = load_clean("data/raw/telco_churn.csv")
    X, y = split_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor()

    # --- Baseline: Logistic Regression ---
    baseline = Pipeline(
        [("preprocess", preprocessor), ("model", LogisticRegression(max_iter=1000))]
    )
    baseline.fit(X_train, y_train)
    evaluate("Baseline: Logistic Regression", baseline, X_test, y_test)

    # --- Main model ---
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos

    main_pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", build_model(scale_pos_weight)),
        ]
    )

    search = RandomizedSearchCV(
        main_pipeline,
        param_distributions=param_distributions(),
        n_iter=15,
        scoring="f1",
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_
    print("\nBest params:", search.best_params_)

    metrics = evaluate("Tuned model (test set)", best_pipeline, X_test, y_test)

    joblib.dump(best_pipeline, "models/churn_pipeline.joblib")
    with open("models/metrics.json", "w") as f:
        json.dump(
            {
                "model_type": "XGBClassifier" if HAS_XGBOOST else "HistGradientBoostingClassifier",
                "best_params": search.best_params_,
                "test_metrics": metrics,
            },
            f,
            indent=2,
        )
    print("\nSaved pipeline to models/churn_pipeline.joblib")
    print("Saved metrics to models/metrics.json")


if __name__ == "__main__":
    main()
