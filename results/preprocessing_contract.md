
# Shared Preprocessing Contract — ML Group 8

## 1. Dataset

- Dataset: UCI Default of Credit Card Clients (#350)
- Input file: data/raw/credit_default.csv
- Total records: 30,000
- Original columns: 25
- Predictive features: 23
- Target: default_next_month
- Positive class: 1 (Default)
- Negative class: 0 (No Default)

## 2. Data Quality and Cleaning Policy

The dataset contains no missing values and no duplicate IDs.

Duplicate investigation identified:

- 35 duplicate rows when ID is excluded.
- 56 duplicate predictor rows when ID and target are excluded.
- 21 predictor groups containing conflicting target labels.

Proposed policy: retain all 30,000 records because identical
predictor values do not necessarily represent the same client.

This policy is subject to team agreement.

ID is excluded from the model inputs, and the target column
is separated from the predictive features.

## 3. Feature Classification

### Numerical Features (14)

- LIMIT_BAL
- AGE
- BILL_AMT1 to BILL_AMT6
- PAY_AMT1 to PAY_AMT6

### Categorical Features (9)

- SEX
- EDUCATION
- MARRIAGE
- PAY_0
- PAY_2
- PAY_3
- PAY_4
- PAY_5
- PAY_6

Original categorical codes are retained, including special
education, marriage and repayment-status codes.

Repayment-status variables are treated as categorical features
to avoid imposing unsupported numerical distances between states.

## 4. Encoding and Scaling

Categorical features:

- OneHotEncoder(handle_unknown="ignore")
- sparse_output=False

Numerical features:

- Logistic Regression: StandardScaler
- Random Forest: Passthrough
- Gradient Boosting: Passthrough

No missing-value imputation is applied because the audited
dataset contains no missing values.

Dense transformed output supports all three base models.

## 5. Shared Train/Test Split

- Training proportion: 80%
- Testing proportion: 20%
- Random state: 42
- Stratification: Target variable

Verified split:

| Class | Training | Testing |
|-------|----------|---------|
|   0   |  18,691  |  4,673  |
|   1   |  5,309   |  1,327  |
| Total |  24,000  |  6,000  |

Training shape: (24000, 23)
Testing shape: (6000, 23)

Row index overlap: 0

Identical predictor overlap:
15 test records (0.25%) have identical predictor values
to at least one training record.

This overlap is documented as a potential evaluation limitation.

## 6. Cross-Validation

- StratifiedKFold
- n_splits=3
- shuffle=True
- random_state=42
- Primary selection metric: Positive-class F1

All model selection and hyperparameter tuning must use
training data only.

The held-out test set is reserved for M3's final evaluation.

## 7. Shared Pipeline Interfaces

Source file: src/preprocessing.py

Functions:

- load_data(csv_path="data/raw/credit_default.csv")
- split_data(df)
- build_pipeline(estimator, scale_numeric=True)
- get_cv()

split_data returns:

X_train, X_test, y_train, y_test

Pipeline step names:

- preprocess
- model

All preprocessing estimators are fitted within training
or individual cross-validation folds.

## 8. Usage Example

```python
from src.preprocessing import (
    load_data,
    split_data,
    build_pipeline,
    get_cv,
)

from sklearn.ensemble import RandomForestClassifier

df = load_data()

X_train, X_test, y_train, y_test = split_data(df)

pipeline = build_pipeline(
    RandomForestClassifier(random_state=42),
    scale_numeric=False,
)

cv = get_cv()
```

Use scale_numeric=True for Logistic Regression and
scale_numeric=False for tree-based models.

## 9. Integration Test Results

Test file: src/test_preprocessing.py

Training subset: (2000, 23)
Validation subset: (100, 23)

Transformed training subset: (2000, 74)

The transformed feature count may vary depending on
categories observed in the fitted training data.

Models tested:

- Logistic Regression
- Random Forest
- Gradient Boosting

All three models passed:

- Parameter compatibility
- Fit
- Predict
- Predict probability
- Serialization
- Model compatibility

Run from the project root:

```powershell
.\.venv\Scripts\python.exe -m src.test_preprocessing
```

Result: ALL INTEGRATION TESTS PASSED

These are integration smoke tests, not official model
performance results.

## 10. Development Environment

- Python: 3.12.10
- pandas: 3.1.1
- scikit-learn: 1.9.1

## 11. Handoff Status

Owner: M2

Local integration testing: Passed

Receivers: M1, M3, M4

Receiver verification and team approval: Pending