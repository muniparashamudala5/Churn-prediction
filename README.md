# Customer Churn Prediction

An end-to-end ML project: data cleaning, feature engineering, model training
(XGBoost) with hyperparameter tuning, evaluation, and a FastAPI service for
real-time predictions — containerized with Docker.

## Problem

Predict whether a telecom customer will churn, using their account,
service, and billing attributes. Built on the
[Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(Kaggle) — 7,043 customers, 21 columns, ~26.5% churn rate (imbalanced).

## Project structure

```
churn-prediction/
├── data/
│   ├── raw/                      # place telco_churn.csv here
│   └── generate_sample_data.py   # makes a synthetic stand-in dataset
├── notebooks/
│   └── eda_notes.md              # EDA findings
├── src/
│   ├── data_prep.py              # loading + cleaning
│   └── train.py                  # training, tuning, evaluation
├── app/
│   ├── main.py                   # FastAPI app (/health, /predict)
│   └── schema.py                 # request/response schemas
├── models/                       # saved pipeline + metrics (generated)
├── tests/
│   └── test_api.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data

Download the real dataset from Kaggle and save it as `data/raw/telco_churn.csv`:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

No Kaggle account handy yet? Run this to generate a synthetic dataset with the
exact same schema, so you can run the whole pipeline immediately:

```bash
python data/generate_sample_data.py
```

Swap in the real CSV later — nothing else changes.

## Train

```bash
python src/train.py
```

This cleans the data, trains a logistic regression baseline, then trains and
tunes an XGBoost classifier with `RandomizedSearchCV` (5-fold CV, scoring on
F1 because of the class imbalance), evaluates it on a held-out test set, and
saves:
- `models/churn_pipeline.joblib` — the full fitted sklearn `Pipeline`
  (preprocessing + model), so inference just needs `pipeline.predict()`.
- `models/metrics.json` — the best hyperparameters and test metrics.

Sample output on the synthetic dataset (your numbers will differ on the real
Kaggle data):

```
Baseline: Logistic Regression
  precision 0.661 | recall 0.368 | f1 0.473 | roc_auc 0.781

Tuned model (test set)
  precision 0.488 | recall 0.368 | f1 0.419 | roc_auc 0.733
```

> **Note on xgboost:** if `xgboost` isn't installed, `train.py` automatically
> falls back to scikit-learn's `HistGradientBoostingClassifier` so the script
> never breaks your workflow. Run `pip install xgboost` (it's already in
> `requirements.txt`) to use the real XGBoost model for your resume project.

## Run the API locally

```bash
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs for interactive Swagger docs, or:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35, "TotalCharges": 140.70
      }'
```

Response:
```json
{"churn": true, "churn_probability": 0.71, "risk_level": "high"}
```

## Tests

```bash
pytest tests/ -v
```

## Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

Then hit the same endpoints at `http://localhost:8000`.

## Deploy

Push this repo to GitHub, then deploy the Docker image for free on either:
- **Render** — New → Web Service → connect the repo → it detects the
  Dockerfile automatically.
- **Hugging Face Spaces** — create a Space with the "Docker" SDK and push
  this repo to it.

## Next steps / ideas to extend this

- Swap in the real Kaggle dataset and re-tune.
- Add SHAP values to `/predict` for per-customer explainability.
- Log predictions and set up a simple drift-monitoring dashboard.
- Add a `/batch-predict` endpoint that accepts a CSV upload.
