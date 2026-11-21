"""M1: E2 weighted soft voting and E3 repeated-CV robustness analysis.

Place as src/ensemble_experiments.py in the existing team repository.
Run from the project root with the same interpreter as src.ensemble:

    python -m src.ensemble_experiments --e2
    python -m src.ensemble_experiments --e3 --repeats 2

This script does NOT load or evaluate the held-out test set. It does NOT
modify results/ensemble_cv.json or src/ensemble.py. E2 uses fixed weights
[2,1] chosen a priori, NOT a broad hyperparameter search. Repeated folds in
E3 are correlated and are a robustness diagnostic, not a significance test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from src.ensemble import build_tuned_models, preflight, read_handoffs
from src.preprocessing import get_cv, load_data, split_data

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SCORING = {
    "f1": "f1",  # the positive/default class is 1
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc",
    "neg_brier": "neg_brier_score",
}


def new_voting(models: dict, weights: list[int]) -> VotingClassifier:
    """Create a fresh, unfitted ensemble with full independent pipelines."""
    return VotingClassifier(
        estimators=[
            ("rf", clone(models["RF"])),
            ("gb_balanced", clone(models["GB"])),
        ],
        voting="soft",
        weights=weights,
        n_jobs=1,
    )


def candidate_models(models: dict) -> dict:
    """Do not change the tuned base parameters in any of these candidates."""
    return {
        "RF": clone(models["RF"]),
        "GB_balanced": clone(models["GB"]),
        "E1_equal_1to1": new_voting(models, [1, 1]),
        "E2_weighted_2to1": new_voting(models, [2, 1]),
    }


def evaluate_cv(label: str, estimator, X_train, y_train, folds: list) -> dict:
    print(f"Starting CV: {label} ({len(folds)} folds)", flush=True)
    result = cross_validate(
        estimator, X_train, y_train,
        cv=folds,
        scoring=SCORING,
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )
    payload = {"model": label, "fold_count": len(folds)}
    for metric in SCORING:
        vals = np.asarray(result[f"test_{metric}"], dtype=float)
        # neg_brier_score is a scorer: we report actual loss (lower is better).
        if metric == "neg_brier":
            vals = -vals
            output_key = "brier"
        else:
            output_key = metric
        payload[f"{output_key}_folds"] = vals.tolist()
        payload[f"{output_key}_mean"] = float(vals.mean())
        payload[f"{output_key}_std"] = float(vals.std(ddof=0))
    payload["mean_fit_seconds"] = float(np.mean(result["fit_time"]))
    payload["mean_score_seconds"] = float(np.mean(result["score_time"]))
    print(
        f"Completed {label}: F1={payload['f1_mean']:.4f} "
        f"± {payload['f1_std']:.4f}, recall={payload['recall_mean']:.4f}, "
        f"brier={payload['brier_mean']:.4f}",
        flush=True,
    )
    return payload


def make_rows(measured: dict) -> pd.DataFrame:
    rows = []
    for name, score in measured.items():
        rows.append({
            "model": name,
            "mean_f1": score["f1_mean"],
            "std_f1": score["f1_std"],
            "mean_precision": score["precision_mean"],
            "mean_recall": score["recall_mean"],
            "mean_roc_auc": score["roc_auc_mean"],
            "mean_brier_lower_better": score["brier_mean"],
            "mean_fit_seconds": score["mean_fit_seconds"],
            "folds": score["fold_count"],
        })
    return pd.DataFrame(rows)


def save_evidence(stem: str, report: dict, table: pd.DataFrame) -> None:
    RESULTS.mkdir(exist_ok=True, parents=True)
    json_path = RESULTS / f"{stem}.json"
    csv_path = RESULTS / f"{stem}.csv"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table.to_csv(csv_path, index=False)
    print(f"Saved: {json_path.relative_to(ROOT)}", flush=True)
    print(f"Saved: {csv_path.relative_to(ROOT)}", flush=True)


def paired_comparison(measured: dict, reference: str = "RF") -> dict:
    ref = np.asarray(measured[reference]["f1_folds"], dtype=float)
    out = {}
    for name, score in measured.items():
        if name == reference:
            continue
        deltas = np.asarray(score["f1_folds"], dtype=float) - ref
        out[name] = {
            "paired_f1_differences_vs_rf": deltas.tolist(),
            "mean_f1_difference_vs_rf": float(deltas.mean()),
            "folds_better_than_rf": int(np.sum(deltas > 0)),
            "folds_equal_to_rf": int(np.sum(deltas == 0)),
            "total_folds": len(deltas),
        }
    return out


def training_fingerprint(X_train, y_train) -> str:
    """Local evidence of the row/order/content used here, not teammates' hashes."""
    xhash = pd.util.hash_pandas_object(X_train, index=True).to_numpy().tobytes()
    yhash = pd.util.hash_pandas_object(y_train, index=True).to_numpy().tobytes()
    return hashlib.sha256(xhash + yhash).hexdigest()


def run_e2(models, X_train, y_train, digest: str) -> None:
    """E2: one pre-registered 2:1 weight alternative vs E1 and base models."""
    folds = list(get_cv().split(X_train, y_train))
    candidates = candidate_models(models)
    measured = {
        name: evaluate_cv(name, estimator, X_train, y_train, folds)
        for name, estimator in candidates.items()
    }
    report = {
        "experiment": "E2_fixed_weight_soft_voting",
        "hypothesis": "Giving RF twice GB's voting weight may increase F1.",
        "weights": {"E1_equal_1to1": [1, 1], "E2_weighted_2to1": [2, 1]},
        "protocol": {"cv": "get_cv()", "folds": len(folds),
                     "same_folds_for_all": True, "test_set_used": False,
                     "positive_class": 1, "primary_metric": "f1"},
        "training_fingerprint_local": digest,
        "measured": measured,
        "paired_differences": paired_comparison(measured),
        "limitations": [
            "2:1 was a predeclared single alternative, not a large parameter sweep.",
            "Training CV was already used for base-model tuning and model selection.",
            "This is not the independent held-out test-set performance.",
        ],
    }
    save_evidence("ensemble_e2_weight_comparison", report, make_rows(measured))
    print("E2 paired mean F1 difference (weighted minus equal): "
          f"{measured['E2_weighted_2to1']['f1_mean'] - measured['E1_equal_1to1']['f1_mean']:+.4f}")


def run_e3(models, X_train, y_train, digest: str, repeats: int) -> None:
    """E3: check ranking robustness on paired, repeated TRAINING CV splits."""
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=repeats, random_state=42)
    folds = list(cv.split(X_train, y_train))
    candidates = candidate_models(models)
    measured = {
        name: evaluate_cv(name, estimator, X_train, y_train, folds)
        for name, estimator in candidates.items()
    }
    comparisons = paired_comparison(measured)
    report = {
        "experiment": "E3_repeated_stratified_cv_robustness",
        "protocol": {"cv": "RepeatedStratifiedKFold", "splits": 3,
                     "repeats": repeats, "folds": len(folds), "random_state": 42,
                     "same_folds_for_all": True, "test_set_used": False,
                     "positive_class": 1, "primary_metric": "f1"},
        "training_fingerprint_local": digest,
        "measured": measured,
        "paired_differences": comparisons,
        "limitations": [
            "Repeated CV folds overlap: their scores are not independent samples.",
            "Do not interpret fold win counts or fold SD as statistical significance.",
            "Repeated training CV is a robustness diagnostic, not an independent test.",
            "M3 evaluates one frozen final model on the held-out test set.",
        ],
    }
    save_evidence("ensemble_e3_repeated_cv", report, make_rows(measured))
    print("\nPaired F1 differences vs RF across identical repeated folds:")
    for name, d in comparisons.items():
        print(f"  {name}: {d['mean_f1_difference_vs_rf']:+.4f}, "
              f"better on {d['folds_better_than_rf']}/{d['total_folds']} folds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_mutually_exclusive_group(required=True)
    choices.add_argument("--e2", action="store_true", help="Run fixed 2:1 soft voting experiment")
    choices.add_argument("--e3", action="store_true", help="Run repeated-CV robustness experiment")
    parser.add_argument("--repeats", type=int, default=2,
                        help="E3 repeats (default 2; use 3 for a larger experiment)")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2 for E3")

    print("Using team model handoffs; held-out test will NOT be scored.", flush=True)
    handoffs = read_handoffs()
    models = build_tuned_models(handoffs)
    X_train, _, y_train, _ = split_data(load_data())
    preflight(handoffs, models, X_train, y_train)
    digest = training_fingerprint(X_train, y_train)
    print("Local training-data fingerprint (for handoff):", digest, flush=True)
    if args.e2:
        run_e2(models, X_train, y_train, digest)
    else:
        run_e3(models, X_train, y_train, digest, args.repeats)
    print("Done. Original E1 evidence and src/ensemble.py were not modified.", flush=True)






if __name__ == "__main__":
    main()
