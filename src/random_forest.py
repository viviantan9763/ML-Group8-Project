"""Compare and tune random forests on the shared training split."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_validate

from src.preprocessing import RANDOM_STATE, build_pipeline, get_cv, load_data, split_data


SCORING = {
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "balanced_accuracy": "balanced_accuracy",
}

SEARCH_SPACE = {
    "model__n_estimators": [100, 200, 400, 600],
    "model__max_depth": [6, 10, 16, 24, None],
    "model__min_samples_split": [2, 5, 10, 20],
    "model__min_samples_leaf": [1, 2, 4, 8, 16],
    "model__max_features": ["sqrt", "log2", 0.25, 0.5],
    "model__class_weight": [None, "balanced", "balanced_subsample"],
    "model__max_samples": [0.65, 0.85, None],
}


def make_random_forest(**params):
    """Create an unfitted pipeline for training or use in the ensemble."""
    options = {"random_state": RANDOM_STATE, "n_jobs": 1, **params}
    return build_pipeline(RandomForestClassifier(**options), scale_numeric=False)


def summarize_folds(scores):
    return {
        metric: {
            "folds": scores[metric].tolist(),
            "mean": float(scores[metric].mean()),
            "std": float(scores[metric].std()),
        }
        for metric in SCORING
    }


def run_experiment(n_iter, jobs, output_dir, model_path, search_verbose=1):
    X_train, _, y_train, _ = split_data(load_data())
    cv = list(get_cv().split(X_train, y_train))

    print(f"Evaluating baseline on {len(y_train):,} training rows.", flush=True)
    baseline = cross_validate(
        make_random_forest(), X_train, y_train, cv=cv, scoring=SCORING,
        n_jobs=jobs, error_score="raise",
    )
    baseline_metrics = summarize_folds({
        metric: baseline[f"test_{metric}"] for metric in SCORING
    })
    print(f"Baseline mean CV F1 = {baseline_metrics['f1']['mean']:.4f}", flush=True)

    search = RandomizedSearchCV(
        make_random_forest(), SEARCH_SPACE, n_iter=n_iter, scoring=SCORING,
        refit=False, cv=cv, n_jobs=jobs, random_state=RANDOM_STATE,
        error_score="raise", verbose=search_verbose,
    )
    search.fit(X_train, y_train)
    candidates = pd.DataFrame(search.cv_results_)
    candidates.insert(0, "draw", candidates.index + 1)
    best = candidates.loc[candidates["mean_test_f1"].idxmax()]
    tuned_params = {
        name.removeprefix("model__"): value for name, value in best["params"].items()
    }
    tuned_metrics = summarize_folds({
        metric: best[[f"split{i}_test_{metric}" for i in range(len(cv))]].to_numpy(dtype=float)
        for metric in SCORING
    })
    f1_gain = tuned_metrics["f1"]["mean"] - baseline_metrics["f1"]["mean"]
    selected = "tuned" if f1_gain > 0 else "baseline"
    selected_params = tuned_params if selected == "tuned" else {}

    print(f"Refitting the {selected} forest on all training rows.", flush=True)
    pipeline = make_random_forest(**selected_params).fit(X_train, y_train)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path, compress=3)

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates.sort_values("rank_test_f1", kind="stable").to_csv(
        output_dir / "random_forest_search.csv", index=False
    )
    summary = {
        "training_rows": len(y_train),
        "cv_folds": len(cv),
        "random_state": RANDOM_STATE,
        "selection_metric": "positive_class_f1",
        "candidates_evaluated": len(candidates),
        "search_space": SEARCH_SPACE,
        "baseline": baseline_metrics,
        "tuned": tuned_metrics,
        "tuned_params": tuned_params,
        "tuned_f1_gain": f1_gain,
        "selected_model": selected,
        "selected_params": selected_params,
    }
    (output_dir / "random_forest_cv.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Tuned mean CV F1 = {tuned_metrics['f1']['mean']:.4f}")
    print(f"Selected {selected}. Saved the fitted pipeline to {model_path}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-iter", type=int, default=48, help="Number of sampled settings")
    parser.add_argument("--jobs", type=int, default=2, help="Parallel cross-validation jobs")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/random_forest.joblib"))
    parser.add_argument("--live-progress", action="store_true", help="Stream candidate and fold scores")
    args = parser.parse_args()
    if args.n_iter < 1 or args.jobs == 0:
        parser.error("--n-iter must be positive and --jobs must be nonzero")
    if args.live_progress:
        with joblib.parallel_config(backend="threading"):
            run_experiment(args.n_iter, args.jobs, args.output_dir, args.model_path, search_verbose=10)
    else:
        run_experiment(args.n_iter, args.jobs, args.output_dir, args.model_path)


if __name__ == "__main__":
    main()
