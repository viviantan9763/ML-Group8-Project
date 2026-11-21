b import Path

import json
import pandas as pd
import time

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV

from src.preprocessing import (
    load_data,
    split_data,
    build_pipeline,
    get_cv,
    RANDOM_STATE,
)


def run_baseline(X_train, y_train):

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline(
        model,
        scale_numeric=True,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    start_time = time.perf_counter()

    results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=get_cv(),
        scoring=scoring,
        return_train_score=False,
        error_score="raise",
    )

    elapsed = time.perf_counter() - start_time

    print("\n=== Logistic Regression Baseline ===")

    for metric in scoring:
        scores = results[f"test_{metric}"]

        print(
            f"{metric.upper()}: "
            f"{scores.mean():.4f} "
            f"(SD: {scores.std():.4f})"
        )

    print(f"\nElapsed time: {elapsed:.2f} seconds")

    return results


def run_tuning(X_train, y_train):

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline(
        model,
        scale_numeric=True,
    )

    param_grid = {
        "model__C": [0.001, 0.01, 0.1, 1, 10],
        "model__solver": ["lbfgs", "liblinear"],
        "model__class_weight": [None, "balanced"],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=get_cv(),
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )

    start_time = time.perf_counter()

    search.fit(X_train, y_train)

    elapsed = time.perf_counter() - start_time

    best_index = search.best_index_
    best_std = search.cv_results_[
        "std_test_score"
    ][best_index]

    print("\n=== Logistic Regression Tuning ===")
    print("Best Parameters:", search.best_params_)
    print(f"Best CV F1: {search.best_score_:.4f}")
    print(f"Standard Deviation: {best_std:.4f}")
    print(f"Elapsed time: {elapsed:.2f} seconds")

    return search, elapsed


def run_preprocessing_experiment(X_train, y_train):

    experiments = {
        "With Scaling": True,
        "Without Scaling": False,
    }

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    rows = []

    print("\n=== Preprocessing Impact Experiment ===")

    for name, scale_numeric in experiments.items():

        model = LogisticRegression(
            C=0.1,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=RANDOM_STATE,
        )

        pipeline = build_pipeline(
            model,
            scale_numeric=scale_numeric,
        )

        start_time = time.perf_counter()

        results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=get_cv(),
            scoring=scoring,
            return_train_score=False,
            error_score="raise",
        )

        elapsed = time.perf_counter() - start_time

        row = {
            "experiment": name,
            "scale_numeric": scale_numeric,
        }

        print(f"\n{name}")

        for metric in scoring:

            scores = results[f"test_{metric}"]

            row[f"{metric}_mean"] = scores.mean()
            row[f"{metric}_std"] = scores.std()

            print(
                f"{metric.upper()}: "
                f"{scores.mean():.4f} "
                f"(SD: {scores.std():.4f})"
            )

        row["elapsed_seconds"] = elapsed

        print(f"Elapsed time: {elapsed:.2f} seconds")

        rows.append(row)

    output = pd.DataFrame(rows)

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output.to_csv(
        results_dir / "preprocessing_impact.csv",
        index=False,
    )

    print("\nSaved: results/preprocessing_impact.csv")

    return output


def save_lr_results(baseline_results, search, tuning_elapsed):

    results = {
        "model": "lr",
        "cv_folds": get_cv().get_n_splits(),
        "random_state": RANDOM_STATE,
        "scoring": "f1",

        "baseline_f1_mean": float(
            baseline_results["test_f1"].mean()
        ),

        "baseline_f1_std": float(
            baseline_results["test_f1"].std()
        ),

        "tuned_f1_mean": float(
            search.best_score_
        ),

        "tuned_f1_std": float(
            search.cv_results_["std_test_score"][
                search.best_index_
            ]
        ),

        "best_params": search.best_params_,

        "search_method": "GridSearchCV",

        "search_space": search.param_grid,

        "elapsed_seconds": round(tuning_elapsed, 4),
    }

    results_dir = (
        Path(__file__).resolve().parents[1] / "results"
    )

    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "lr_cv.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nSaved: results/lr_cv.json")

    return results


if __name__ == "__main__":

    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df)

    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    baseline_results = run_baseline(
        X_train, y_train
    )

    tuning_results, tuning_elapsed = run_tuning(
        X_train, y_train
    )

    preprocessing_results = run_preprocessing_experiment(
        X_train, y_train
    )

    save_lr_results(
        baseline_results,
        tuning_results,
        tuning_elapsed,
    )