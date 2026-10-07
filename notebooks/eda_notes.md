# EDA Notes — Telco Customer Churn

Turn this into a Jupyter notebook (`eda.ipynb`) once you're working with the
real Kaggle data — these are the checks and plots worth doing, with the
`pandas`/`matplotlib` snippets to produce them.

## 1. Shape and types

```python
import pandas as pd
df = pd.read_csv("data/raw/telco_churn.csv")
df.info()
df.head()
```

- 7,043 rows, 21 columns in the real dataset.
- `TotalCharges` loads as an **object/string** column — it has blank entries
  for the ~11 customers with `tenure == 0` (brand new customers who haven't
  been billed yet). Coerce to numeric and fill with 0.
- `SeniorCitizen` is 0/1 but is really categorical, not numeric — treat it
  as a category for plots, not as something to scale.

## 2. Target balance

```python
df["Churn"].value_counts(normalize=True).plot(kind="bar")
```

~26.5% churn, ~73.5% no churn → imbalanced. Use `scale_pos_weight` (XGBoost)
or `class_weight="balanced"` / stratified sampling, and **don't rely on
accuracy** — track precision, recall, F1, and ROC-AUC instead.

## 3. Churn by contract type

```python
pd.crosstab(df["Contract"], df["Churn"], normalize="index").plot(kind="bar", stacked=True)
```

Expect month-to-month customers to churn far more than one/two-year
contracts — contract length is usually the single strongest predictor.

## 4. Churn by tenure

```python
df.boxplot(column="tenure", by="Churn")
```

Churned customers skew toward low tenure — new customers are the highest
risk group, which matters for retention targeting.

## 5. Churn by monthly charges / internet service

```python
df.boxplot(column="MonthlyCharges", by="Churn")
pd.crosstab(df["InternetService"], df["Churn"], normalize="index").plot(kind="bar", stacked=True)
```

Fiber optic customers typically churn more than DSL, despite (or because of)
paying more — worth calling out in the write-up as a counter-intuitive
finding.

## 6. Correlation among numeric features

```python
df[["tenure", "MonthlyCharges", "TotalCharges"]].corr()
```

`TotalCharges` correlates strongly with `tenure` (it's roughly tenure ×
monthly charges), so watch for multicollinearity if you ever use a linear
model without regularization.

## 7. Payment method

```python
pd.crosstab(df["PaymentMethod"], df["Churn"], normalize="index").plot(kind="bar", stacked=True)
```

Electronic check tends to have the highest churn rate of the four payment
methods — a good candidate "risk factor" to mention in the project write-up.
