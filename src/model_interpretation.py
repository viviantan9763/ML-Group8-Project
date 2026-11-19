
"""
M4 - Gradient Boosting Explainability

Provides:
1. Built-in Gradient Boosting feature importance.
2. Permutation importance using training data.

Uses the shared preprocessing pipeline from M2.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance

from src.preprocessing import load_data, split_data
from src.gradient_boosting import make_gb_pipeline


def get_feature_names(fitted_pipeline):
    """Get feature names after shared preprocessing."""

    preprocessor = fitted_pipeline.named_steps["preprocess"]

    return preprocessor.get_feature_names_out()


def get_builtin_importance(fitted_pipeline):
    """Extract built-in Gradient Boosting feature importance."""

    model = fitted_pipeline.named_steps["model"]
    feature_names = get_feature_names(fitted_pipeline)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False,
    )

    return importance_df.reset_index(drop=True)


def get_permutation_importance(
    fitted_pipeline,
    X,
    y,
):
    """Calculate permutation importance using F1 score."""

    result = permutation_importance(
        fitted_pipeline,
        X,
        y,
        scoring="f1",
        n_repeats=5,
        random_state=42,
        n_jobs=1,
    )

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })

    importance_df = importance_df.sort_values(
        by="importance_mean",
        ascending=False,
    )

    return importance_df.reset_index(drop=True)


def main():
    print("=== Loading Data ===")

    df = load_data()
    X_train, _, y_train, _ = split_data(df)

    # Load best parameters from previous tuning
    results_path = Path("results/gb_cv.json")

    with results_path.open("r", encoding="utf-8") as file:
        previous_results = json.load(file)

    best_params = previous_results["best_params"]

    # Build and fit the selected model using training data only
    pipeline = make_gb_pipeline()
    pipeline.set_params(**best_params)

    print("\n=== Fitting Gradient Boosting Model ===")
    pipeline.fit(X_train, y_train)

    # Built-in feature importance
    print("\n=== Built-in Feature Importance ===")

    builtin_df = get_builtin_importance(pipeline)

    print("\nTop 15 Features:")
    print(builtin_df.head(15).to_string(index=False))

    # Permutation importance
    print("\n=== Permutation Importance ===")

    permutation_df = get_permutation_importance(
        pipeline,
        X_train,
        y_train,
    )

    print("\nTop 15 Features:")
    print(permutation_df.head(15).to_string(index=False))

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    builtin_df.to_csv(
        output_dir / "gb_feature_importance.csv",
        index=False,
    )

    permutation_df.to_csv(
        output_dir / "gb_permutation_importance.csv",
        index=False,
    )

    print("\nResults saved:")
    print("results/gb_feature_importance.csv")
    print("results/gb_permutation_importance.csv")


if __name__ == "__main__":
    main()
