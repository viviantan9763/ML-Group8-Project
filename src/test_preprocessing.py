
import io

import joblib
import numpy as np

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.model_selection import train_test_split

from src.preprocessing import (
    load_data,
    split_data,
    build_pipeline,
    get_cv,
)


def main():

    # Load shared dataset and split
    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df)

    # Use training data only for integration testing
    X_fit, X_check, y_fit, y_check = train_test_split(
        X_train,
        y_train,
        train_size=2000,
        test_size=100,
        stratify=y_train,
        random_state=42,
    )

    models = {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
            True,
            {"model__C": 1.0},
        ),

        "Random Forest": (
            RandomForestClassifier(
                n_estimators=10,
                random_state=42,
            ),
            False,
            {"model__n_estimators": 10},
        ),

        "Gradient Boosting": (
            GradientBoostingClassifier(
                n_estimators=10,
                random_state=42,
            ),
            False,
            {"model__n_estimators": 10},
        ),
    }

    print("=== Shared Preprocessing Integration Test ===")

    print("Training subset:", X_fit.shape)
    print("Validation subset:", X_check.shape)
    print("CV folds:", get_cv().get_n_splits())

    for name, (estimator, scaling, params) in models.items():

        print(f"\n--- {name} ---")

        pipeline = build_pipeline(
            estimator,
            scale_numeric=scaling,
        )

        assert list(pipeline.named_steps) == [
            "preprocess",
            "model",
        ]

        # Parameter and cloning compatibility
        assert all(
            key in pipeline.get_params()
            for key in params
        )

        pipeline.set_params(**params)
        pipeline = clone(pipeline)

        print("Parameter compatibility: PASS")

        pipeline.fit(X_fit, y_fit)
        print("Fit: PASS")

        predictions = pipeline.predict(X_check)

        assert len(predictions) == len(X_check)
        assert set(predictions).issubset({0, 1})

        print("Predict: PASS")

        probabilities = pipeline.predict_proba(X_check)

        assert probabilities.shape == (
            len(X_check), 2
        )

        assert np.allclose(
            probabilities.sum(axis=1), 1.0
        )

        print("Predict probability: PASS")

        transformed = pipeline.named_steps[
            "preprocess"
        ].transform(X_fit)

        assert np.isfinite(transformed).all()

        print("Transformed shape:", transformed.shape)

        # Test model serialization in memory
        buffer = io.BytesIO()

        joblib.dump(pipeline, buffer)
        buffer.seek(0)

        restored_pipeline = joblib.load(buffer)

        assert np.array_equal(
            predictions,
            restored_pipeline.predict(X_check),
        )

        print("Serialization: PASS")
        print("Model compatibility: PASS")

    print("\nALL INTEGRATION TESTS PASSED")


if __name__ == "__main__":
    main()
