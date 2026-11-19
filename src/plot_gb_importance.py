
"""
Gradient Boosting Feature Importance Visualisation

Creates:
1. Built-in Feature Importance chart.
2. Permutation Importance chart.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_builtin_importance():
    """Plot built-in Gradient Boosting feature importance."""

    data_path = Path("results/gb_feature_importance.csv")
    output_path = Path("results/gb_feature_importance.png")

    df = pd.read_csv(data_path)
    top_features = df.head(15).sort_values(
        by="importance",
        ascending=True,
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_features["feature"],
        top_features["importance"],
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Gradient Boosting - Built-in Feature Importance")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Saved:", output_path)


def plot_permutation_importance():
    """Plot permutation importance based on F1 score."""

    data_path = Path("results/gb_permutation_importance.csv")
    output_path = Path("results/gb_permutation_importance.png")

    df = pd.read_csv(data_path)
    top_features = df.head(15).sort_values(
        by="importance_mean",
        ascending=True,
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_features["feature"],
        top_features["importance_mean"],
        xerr=top_features["importance_std"],
        capsize=3,
    )

    plt.xlabel("Mean F1 Score Decrease")
    plt.ylabel("Features")
    plt.title("Gradient Boosting - Permutation Importance")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Saved:", output_path)


def main():
    print("=== Creating Feature Importance Charts ===")

    plot_builtin_importance()
    plot_permutation_importance()

    print("\nAll charts created successfully.")


if __name__ == "__main__":
    main()
