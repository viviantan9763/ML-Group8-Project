"""M1 / Chapter 6 -- RF + balanced GB ensemble, using team handoffs.

COPY THIS FILE into the repository as: src/ensemble.py
Run from repository root:
    python -m src.ensemble --check-only
    python -m src.ensemble --run

This script NEVER uses X_test or y_test to choose or evaluate the models.
It consumes actual tuned params from the shared results JSON files. It does NOT
claim that handoff CV values are independently verified until --run succeeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

from src.preprocessing import (
    RANDOM_STATE, build_pipeline, get_cv, load_data, split_data
)
from src.random_forest import make_random_forest
from src.gradient_boosting import make_gb_pipeline

ROOT = Path(__file__).resolve().parents[1]  # valid when copied to src/ensemble.py
RESULTS = ROOT / "results"


def load_json(filename: str) -> dict:
    path = RESULTS / filename
    if not path.is_file():
        raise FileNotFoundError(f"Expected handoff not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_handoffs() -> dict:
    return {
        "RF": load_json("rf_ensemble_handoff.json"),
        "GB": load_json("gb_balanced_cv.json"),
        "LR": load_json("lr_cv.json"),
    }


def reported_results(handoffs: dict) -> pd.DataFrame:
    rf, gb, lr = handoffs["RF"], handoffs["GB"], handoffs["LR"]
    data = [
        {"model": "RF", "tuned_cv_f1_mean": rf["cv_scores"]["f1"]["mean"],
         "tuned_cv_f1_std": rf["cv_scores"]["f1"]["std"], "source": "rf_ensemble_handoff.json"},
        {"model": "GB (balanced)", "tuned_cv_f1_mean": gb["tuned_f1_mean"],
         "tuned_cv_f1_std": gb["tuned_f1_std"], "source": "gb_balanced_cv.json"},
        {"model": "LR", "tuned_cv_f1_mean": lr["tuned_f1_mean"],
         "tuned_cv_f1_std": lr["tuned_f1_std"], "source": "lr_cv.json"},
    ]
    return pd.DataFrame(data).sort_values("tuned_cv_f1_mean", ascending=False).reset_index(drop=True)


def build_tuned_models(handoffs: dict) -> dict:
    # RF constructor takes sklearn-estimator parameters WITHOUT model__ prefixes.
    rf = make_random_forest(**handoffs["RF"]["selected_params"])

    # GB factory needs balanced=True. Changing it to False would invalidate the
    # 0.5397 F1 comparison and would NOT reproduce M4's balanced experiment.
    gb = make_gb_pipeline(balanced=True)
    gb.set_params(**handoffs["GB"]["best_params"])

    # LR has no dedicated factory in M2's file; rebuild the same base Pipeline
    # used in run_tuning(), then set the tuned model__ parameters.
    lr = build_pipeline(
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        scale_numeric=True,
    )
    lr.set_params(**handoffs["LR"]["best_params"])
    return {"RF": rf, "GB": gb, "LR": lr}


def make_ensemble():
    """Return an UNFITTED RF + balanced GB soft-voting ensemble for M3.

    scikit-learn fits each component pipeline again in every CV training fold.
    """
    models = build_tuned_models(read_handoffs())
    return VotingClassifier(
        estimators=[("rf", models["RF"]), ("gb_balanced", models["GB"])],
        voting="soft", weights=[1, 1], n_jobs=1,
    )


def make_weighted_ensemble():
    """
    Final E2 Ensemble:
    Random Forest + Balanced Gradient Boosting
    Weighted Soft Voting (2:1).
    """
    handoffs = read_handoffs()
    models = build_tuned_models(handoffs)

    return VotingClassifier(
        estimators=[
            ("rf", models["RF"]),
            ("gb_balanced", models["GB"])
        ],
        voting="soft",
        weights=[2, 1],
        n_jobs=1
    )


def preflight(handoffs: dict, models: dict, X_train, y_train) -> None:
    assert X_train.shape == (24000, 23), X_train.shape
    assert len(y_train) == 24000
    assert "ID" not in X_train.columns
    assert "default_next_month" not in X_train.columns
    assert set(y_train.unique()) == {0, 1}
    assert int(y_train.sum()) == 5309
    for name, model in models.items():
        clone(model)  # Check that scikit-learn can build independent CV estimators
        assert model.named_steps.get("preprocess") is not None
        assert model.named_steps.get("model") is not None
    assert get_cv().get_n_splits() == 3
    assert handoffs["LR"]["cv_folds"] == 3 and handoffs["GB"]["cv_folds"] == 3
    assert handoffs["RF"]["cv_protocol"]["folds"] == 3
    assert handoffs["RF"]["cv_protocol"]["seed"] == 42
    assert handoffs["GB"]["random_state"] == handoffs["LR"]["random_state"] == 42

    print("Preflight passed: training shape, class counts, CV settings, pipeline clone.")
    print("CAVEAT: M2/M4 JSON files do not prove identical dataset hashes or rows.")
    try:
        from src.random_forest import SEARCH_SPACE
        rf_params = handoffs["RF"]["selected_params"]
        if ("model__max_depth" in SEARCH_SPACE
                and rf_params.get("max_depth") not in SEARCH_SPACE["model__max_depth"]):
            print("PROVENANCE WARNING: RF handoff max_depth was not searched by current RF source.")
        if "ccp_alpha" in rf_params and "model__ccp_alpha" not in SEARCH_SPACE:
            print("PROVENANCE WARNING: RF handoff ccp_alpha was not searched by current RF source.")
    except ImportError:
        print("Note: RF SEARCH_SPACE not exported; verify the original tuning provenance manually.")


def cv_evaluate(name: str, estimator, X_train, y_train, folds: list) -> dict:
    scores = cross_validate(
        estimator, X_train, y_train, cv=folds, n_jobs=1,
        scoring={"f1": "f1", "precision": "precision", "recall": "recall",
                 "roc_auc": "roc_auc", "neg_brier": "neg_brier_score"},
        return_train_score=True, error_score="raise",
    )
    report = {"model": name, "cv_folds": len(folds)}
    for metric in ("f1", "precision", "recall", "roc_auc", "neg_brier"):
        test = np.asarray(scores[f"test_{metric}"], dtype=float)
        train = np.asarray(scores[f"train_{metric}"], dtype=float)
        report[f"cv_{metric}_folds"] = [float(x) for x in test]
        report[f"cv_{metric}_mean"] = float(test.mean())
        report[f"cv_{metric}_std"] = float(test.std())
        report[f"train_{metric}_mean"] = float(train.mean())
    report["cv_brier_loss_mean"] = -report["cv_neg_brier_mean"]
    report["cv_fit_seconds_mean"] = float(np.mean(scores["fit_time"]))
    report["cv_predict_seconds_mean"] = float(np.mean(scores["score_time"]))
    print(f"{name:<15} F1 = {report['cv_f1_mean']:.4f} +/- {report['cv_f1_std']:.4f} "
          f"| recall = {report['cv_recall_mean']:.4f} "
          f"| Brier = {report['cv_brier_loss_mean']:.4f}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Check imports, handoffs, and split; do not fit models")
    parser.add_argument("--run", action="store_true", help="Actually run training-only CV and save evidence")
    args = parser.parse_args()
    if not args.run and not args.check_only:
        parser.error("choose --check-only or --run")

    handoffs = read_handoffs()
    ranking = reported_results(handoffs)
    print("\nRanking from teammate-supplied tuned training CV:\n", ranking.to_string(index=False))
    assert ranking.iloc[:2]["model"].tolist() == ["RF", "GB (balanced)"], \
        "The model ranking changed! Reconsider model selection before this run."

    models = build_tuned_models(handoffs)
    df = load_data()
    X_train, _, y_train, _ = split_data(df)  # held-out test is not used
    preflight(handoffs, models, X_train, y_train)
    if args.check_only:
        print("\nCheck-only succeeded. No CV run and no performance claims generated.")
        return

    folds = list(get_cv().split(X_train, y_train))  # identical folds for all models
    measured = {}
    for name, model in models.items():
        measured[name] = cv_evaluate(name, model, X_train, y_train, folds)
    ensemble = make_ensemble()
    measured["RF+BalancedGB Soft Voting"] = cv_evaluate(
        "RF+BalancedGB Soft Voting", ensemble, X_train, y_train, folds
    )
    RESULTS.mkdir(exist_ok=True, parents=True)
    ranking.to_csv(RESULTS / "model_selection.csv", index=False)

    # A report-ready comparison from receiver-validated training CV, not a test result.
    rows = []
    for name, metrics in measured.items():
        rows.append({
            "model": name,
            "cv_f1_mean": metrics["cv_f1_mean"],
            "cv_f1_std": metrics["cv_f1_std"],
            "cv_precision_mean": metrics["cv_precision_mean"],
            "cv_recall_mean": metrics["cv_recall_mean"],
            "cv_roc_auc_mean": metrics["cv_roc_auc_mean"],
            "cv_brier_loss_mean": metrics["cv_brier_loss_mean"],
            "cv_fit_seconds_mean": metrics["cv_fit_seconds_mean"],
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS / "ensemble_cv_comparison.csv", index=False)

    # Figure is clearly labelled training CV; never confuse it with final test.
    import matplotlib.pyplot as plt
    FIGURES = ROOT / "figures"
    FIGURES.mkdir(exist_ok=True, parents=True)
    colors = ["#4E79A7", "#9467BD", "#7B61A8", "#D97706"]
    fig, ax = plt.subplots(figsize=(10, 5.3))
    bars = ax.bar(
        comparison["model"], comparison["cv_f1_mean"],
        yerr=comparison["cv_f1_std"], capsize=5,
        color=colors[:len(comparison)], edgecolor="white",
    )
    for bar, value in zip(bars, comparison["cv_f1_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.014,
                f"{value:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Base Models vs Ensemble — Training-only 3-Fold CV F1")
    ax.set_ylabel("Positive-class F1-score (mean ± fold SD)")
    ax.set_ylim(0, max(.67, float(comparison["cv_f1_mean"].max()) + .10))
    ax.tick_params(axis="x", labelrotation=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "ensemble_training_cv_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    ensemble_delta_rf = (measured["RF+BalancedGB Soft Voting"]["cv_f1_mean"]
                         - measured["RF"]["cv_f1_mean"])
    print(f"\nEnsemble minus RF CV F1: {ensemble_delta_rf:+.4f} (training CV only)")
    report = {
        "method": "equal_weight_soft_voting",
        "models_selected": ["RF", "GB (balanced)"],
        "excluded_model": "LR",
        "selection_source": "teammate_tuned_training_cv_reports",
        "cv_protocol": {"folds": 3, "shuffle": True, "random_state": 42,
                        "f1_positive_class": 1, "test_set_used": False},
        "original_reported_model_ranking": ranking.to_dict(orient="records"),
        "receiver_cv": measured,
        "ensemble_f1_delta_vs_rf": float(ensemble_delta_rf),
        "note": "Repeated training CV is for reproducibility and comparative validation, "
                "not an independent unbiased estimate after hyperparameter selection; "
                "M3 must evaluate once on the untouched test set."
    }
    (RESULTS / "ensemble_cv.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("\nSaved: results/model_selection.csv, results/ensemble_cv.json,")
    print("       results/ensemble_cv_comparison.csv, figures/ensemble_training_cv_comparison.png")
    print("M3 handoff: from src.ensemble import make_weighted_ensemble; fit on shared training split; then final test evaluation.")


if __name__ == "__main__":
    main()
