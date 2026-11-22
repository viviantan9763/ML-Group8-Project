
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

# ---------------------------------------------------------------------------
# FP / FN error analysis (added for Section 7.2 / 7.3)
# ---------------------------------------------------------------------------
import numpy as np

DEFAULT_ERROR_FEATURES = [
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1",
]


def error_analysis_table(predictions_df, raw_df, model_key, id_col="ID"):
    """
    Decompose frozen test predictions into TN / FP / FN / TP and summarise
    feature means and medians per error group.

    Returns (summary_df, counts_dict).
    """
    pred_col = f"pred_{model_key}"
    prob_col = f"prob_{model_key}"

    for c in (id_col, "y_true", pred_col, prob_col):
        if c not in predictions_df.columns:
            raise KeyError(f"predictions_df missing column: {c!r}")

    raw = raw_df.copy()
    if "default_next_month" not in raw.columns:
        raw = raw.rename(columns={raw.columns[-1]: "default_next_month"})

    keep = [id_col] + [c for c in raw.columns
                       if c not in (id_col, "default_next_month")]
    merged = predictions_df.merge(raw[keep], on=id_col, how="left",
                                  validate="one_to_one")

    y = merged["y_true"].astype(int).to_numpy()
    yhat = merged[pred_col].astype(int).to_numpy()

    counts = {
        "TN": int(((y == 0) & (yhat == 0)).sum()),
        "FP": int(((y == 0) & (yhat == 1)).sum()),
        "FN": int(((y == 1) & (yhat == 0)).sum()),
        "TP": int(((y == 1) & (yhat == 1)).sum()),
    }

    merged["error_group"] = np.select(
        [(y == 0) & (yhat == 0), (y == 0) & (yhat == 1),
         (y == 1) & (yhat == 0), (y == 1) & (yhat == 1)],
        ["TN", "FP", "FN", "TP"], default="?",
    )

    features = [f for f in DEFAULT_ERROR_FEATURES if f in merged.columns]
    summary = (merged.groupby("error_group")[features]
               .agg(["mean", "median"]).round(2))
    summary = summary.reindex([g for g in ["TN", "FP", "FN", "TP"]
                               if g in summary.index])
    return summary, counts


def save_error_analysis_figure(predictions_df, raw_df, model_key,
                               out_path, id_col="ID"):
    """Three-panel diagnostic figure used in Section 7.2."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_path = Path(out_path)
    pred_col = f"pred_{model_key}"
    prob_col = f"prob_{model_key}"

    raw = raw_df.copy()
    if "default_next_month" not in raw.columns:
        raw = raw.rename(columns={raw.columns[-1]: "default_next_month"})

    merged = predictions_df.merge(raw.drop(columns=["default_next_month"]),
                                  on=id_col, how="left", validate="one_to_one")
    y = merged["y_true"].astype(int)
    yhat = merged[pred_col].astype(int)
    merged["error_group"] = np.select(
        [(y == 0) & (yhat == 0), (y == 0) & (yhat == 1),
         (y == 1) & (yhat == 0), (y == 1) & (yhat == 1)],
        ["TN", "FP", "FN", "TP"], default="?",
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if "PAY_0" in merged.columns:
        sns.countplot(data=merged, x="PAY_0", hue="error_group",
                      order=sorted(merged["PAY_0"].unique()), ax=axes[0])
        axes[0].set_title(f"{model_key}: PAY_0 by error group")

    if "LIMIT_BAL" in merged.columns:
        sns.boxplot(data=merged, x="error_group", y="LIMIT_BAL",
                    order=["TN", "FP", "FN", "TP"], ax=axes[1])
        axes[1].set_title(f"{model_key}: LIMIT_BAL by error group")

    sns.histplot(data=merged, x=prob_col, hue="y_true", bins=40,
                 stat="density", common_norm=False, ax=axes[2])
    axes[2].axvline(0.5, color="red", linestyle="--", label="threshold=0.5")
    axes[2].set_title(f"{model_key}: predicted prob by true label")
    axes[2].legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path

if __name__ == "__main__":
    main()
