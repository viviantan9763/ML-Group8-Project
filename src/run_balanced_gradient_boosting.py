
"""
Balanced Gradient Boosting Baseline, Tuning and Evaluation

Uses balanced sample weights inside each training fold.
The final test set is reserved.
"""

import json
import time
from pathlib import Path

from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
    cross_validate,
)

from src.preprocessing import (
    load_data,
    split_data,
    get_cv,
    RANDOM_STATE,
)

from src.gradient_boosting import make_gb_pipeline


def main():

    start_time = time.perf_counter()

    df = load_data()
    X_train, _, y_train, _ = split_data(df)

    print("Training data shape:", X_train.shape)
    print("Training target shape:", y_train.shape)
    print("Final test set reserved.")

    cv = get_cv()

    # Balanced GB baseline
    print("\n=== Balanced Gradient Boosting Baseline ===")

    baseline_pipeline = make_gb_pipeline(balanced=True)

    baseline_scores = cross_val_score(
        baseline_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
        n_jobs=1,
    )

    baseline_mean = float(baseline_scores.mean())
    baseline_std = float(baseline_scores.std())

    print("Baseline CV F1 scores:", baseline_scores)
    print("Baseline mean F1:", baseline_mean)
    print("Baseline std F1:", baseline_std)

    # Hyperparameter tuning
    print("\n=== Balanced GB Hyperparameter Tuning ===")

    search_space = {
        "model__n_estimators": [100, 150, 200],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [2, 3],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 5],
        "model__subsample": [0.8, 1.0],
    }

    search = RandomizedSearchCV(
        estimator=make_gb_pipeline(balanced=True),
        param_distributions=search_space,
        n_iter=12,
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=False,
        return_train_score=True,
    )

    tuning_start = time.perf_counter()
    search.fit(X_train, y_train)
    tuning_time = time.perf_counter() - tuning_start

    best_index = search.best_index_

    tuned_mean = float(
        search.cv_results_["mean_test_score"][best_index]
    )

    tuned_std = float(
        search.cv_results_["std_test_score"][best_index]
    )

    print("\n=== Balanced GB Tuning Results ===")
    print("Baseline F1:", baseline_mean)
    print("Tuned F1:", tuned_mean)
    print("Best parameters:", search.best_params_)
    print("Tuning time:", round(tuning_time, 2), "seconds")

    # Supporting evaluation
    print("\n=== Balanced GB Supporting Evaluation ===")

    tuned_pipeline = make_gb_pipeline(balanced=True)
    tuned_pipeline.set_params(**search.best_params_)

    scoring = {
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    evaluation_start = time.perf_counter()

    evaluation = cross_validate(
        tuned_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=True,
    )

    evaluation_time = time.perf_counter() - evaluation_start

    # Save results
    results = {
        "model": "gb_balanced",
        "class_weighting": "balanced sample weights",
        "cv_folds": 3,
        "random_state": RANDOM_STATE,
        "scoring": "f1",
        "baseline_f1_mean": baseline_mean,
        "baseline_f1_std": baseline_std,
        "tuned_f1_mean": tuned_mean,
        "tuned_f1_std": tuned_std,
        "best_params": search.best_params_,
        "search_method": "RandomizedSearchCV",
        "search_space": search_space,
        "n_iter": 12,
        "tuning_seconds": round(tuning_time, 2),
    }

    for metric in scoring:

        train_scores = evaluation[f"train_{metric}"]
        validation_scores = evaluation[f"test_{metric}"]

        results[f"train_{metric}_mean"] = float(
            train_scores.mean()
        )

        results[f"train_{metric}_std"] = float(
            train_scores.std()
        )

        results[f"validation_{metric}_mean"] = float(
            validation_scores.mean()
        )

        results[f"validation_{metric}_std"] = float(
            validation_scores.std()
        )

    results["f1_train_validation_gap"] = (
        results["train_f1_mean"]
        - results["validation_f1_mean"]
    )

    results["mean_fit_time_seconds"] = float(
        evaluation["fit_time"].mean()
    )

    results["mean_score_time_seconds"] = float(
        evaluation["score_time"].mean()
    )

    results["supporting_evaluation_seconds"] = round(
        evaluation_time, 2
    )

    results["elapsed_seconds"] = round(
        time.perf_counter() - start_time, 2
    )

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "gb_balanced_cv.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\n=== Balanced GB Evaluation Summary ===")

    for metric in scoring:
        print(
            f"{metric.upper()} - "
            f"Train: {results[f'train_{metric}_mean']:.4f}, "
            f"Validation: "
            f"{results[f'validation_{metric}_mean']:.4f}"
        )

    print(
        "F1 train-validation gap:",
        round(results["f1_train_validation_gap"], 4),
    )

    print(
        "Mean fit time:",
        round(results["mean_fit_time_seconds"], 2),
        "seconds",
    )

    print("\nResults saved to:", output_path)
    print("Total elapsed time:", results["elapsed_seconds"])


if __name__ == "__main__":
    main()
