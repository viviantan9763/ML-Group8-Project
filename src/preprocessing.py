
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Shared experiment settings
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "default_next_month"
ID_COLUMN = "ID"


# Categorical features
CATEGORICAL_FEATURES = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]


# Numerical features
NUMERICAL_FEATURES = [
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]


# Load the common dataset
def load_data(csv_path="data/raw/credit_default.csv"):
    path = Path(csv_path)

    if not path.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        path = project_root / path

    df = pd.read_csv(path)

    required_columns = (
        [ID_COLUMN, TARGET_COLUMN]
        + CATEGORICAL_FEATURES
        + NUMERICAL_FEATURES
    )

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df[required_columns].isna().any().any():
        raise ValueError("Dataset contains missing values.")

    if df[ID_COLUMN].duplicated().any():
        raise ValueError("Dataset contains duplicate IDs.")

    if set(df[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError("Target must contain binary labels 0 and 1.")

    return df


# Shared stratified train/test split
def split_data(df):
    X = df.drop(columns=[ID_COLUMN, TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    expected_features = (
        CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    )

    if set(X.columns) != set(expected_features):
        raise ValueError("Unexpected predictor columns.")

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


# Shared cross-validation settings
def get_cv():
    return StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


# Build an unfitted preprocessing and model pipeline
def build_pipeline(estimator, scale_numeric=True):

    numeric_transformer = (
        StandardScaler()
        if scale_numeric
        else "passthrough"
    )

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )

    return pipeline
