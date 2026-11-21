"""M1 -> M3 final handoff: E2 Weighted Soft Voting (RF:Balanced GB = 2:1).

Use inside the team's existing ML Group project, from the project root:

    from src.ensemble_final import make_weighted_ensemble
    model = make_weighted_ensemble()    # UNFITTED
    model.fit(X_train, y_train)         # train data only

Safe integration check (NO training / NO held-out test):
    python -m src.ensemble_final --check-only

Requirements already present in the team repository:
    src.random_forest.make_random_forest(**params)
    src.gradient_boosting.make_gb_pipeline(balanced=True)
    results/rf_ensemble_handoff.json, results/gb_balanced_cv.json

This file contains only E2; it does not construct E1, tune weights, or score test data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.base import clone
from sklearn.ensemble import VotingClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def _load_config(filename: str, parameter_key: str) -> dict:
    path = RESULTS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Copy the teammate's verified JSON into results/."
        )
    try:
        with path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}") from exc
    params = config.get(parameter_key)
    if not isinstance(params, dict) or not params:
        raise ValueError(
            f"{filename} must contain a nonempty '{parameter_key}' dictionary."
        )
    return config


def _load_verified_handoffs() -> tuple[dict, dict]:
    rf = _load_config("rf_ensemble_handoff.json", "selected_params")
    gb = _load_config("gb_balanced_cv.json", "best_params")

    # Explicitly guard against accidentally using a different GB handoff.
    if gb.get("model") not in (None, "gb_balanced"):
        raise ValueError("GB config is not 'gb_balanced'; use gb_balanced_cv.json.")
    if gb.get("class_weighting") not in (None, "balanced sample weights"):
        raise ValueError("GB handoff is not the balanced sample-weight version.")

    # Training protocol checks. They do NOT prove that teammate row hashes match.
    rf_cv = rf.get("cv_protocol", {})
    if rf_cv.get("folds") not in (None, 3) or rf_cv.get("seed") not in (None, 42):
        raise ValueError("RF config does not match the expected 3-fold / seed 42 protocol.")
    if gb.get("cv_folds") not in (None, 3) or gb.get("random_state") not in (None, 42):
        raise ValueError("GB config does not match the expected 3-fold / seed 42 protocol.")
    return rf, gb


def make_weighted_ensemble() -> VotingClassifier:
    """Return an UNFITTED E2 model, using the verified team RF & balanced GB.

    Components are complete pipelines, so each is fitted on exactly the supplied
    training data by VotingClassifier.fit. Predictions use sklearn's default
    classification decision rule; do not tune threshold on held-out test data.
    """
    # Import inside factory to give M3 a clear error if teammates' src modules
    # are missing; no alternative model or preprocessing is silently invented.
    from src.random_forest import make_random_forest
    from src.gradient_boosting import make_gb_pipeline

    rf_cfg, gb_cfg = _load_verified_handoffs()

    rf = make_random_forest(**rf_cfg["selected_params"])
    gb = make_gb_pipeline(balanced=True)
    gb.set_params(**gb_cfg["best_params"])

    # Guard against a wrong factory returning a bare estimator rather than the
    # preprocessing-inclusive pipeline used for training-CV comparisons.
    for name, model in (("RF", rf), ("Balanced GB", gb)):
        if not hasattr(model, "named_steps"):
            raise TypeError(f"{name} must be the team's full sklearn Pipeline.")
        if "preprocess" not in model.named_steps or "model" not in model.named_steps:
            raise ValueError(
                f"{name} pipeline must contain 'preprocess' and 'model' steps."
            )
        if not hasattr(model.named_steps["model"], "predict_proba"):
            raise TypeError(f"{name} model must support predict_proba for soft voting.")

    return VotingClassifier(
        estimators=[("rf", rf), ("gb_balanced", gb)],
        voting="soft",
        weights=[2, 1],
        n_jobs=1,
    )


def check_only() -> None:
    """Verify factory imports, config, and clonability. Never fit or score data."""
    model = make_weighted_ensemble()
    clone(model)
    assert model.voting == "soft" and list(model.weights) == [2, 1]
    assert [name for name, _ in model.estimators] == ["rf", "gb_balanced"]
    rf_cfg, gb_cfg = _load_verified_handoffs()
    print("PASS: E2 Weighted Soft Voting, RF:Balanced GB = 2:1")
    print("PASS: RF and balanced GB configuration read successfully")
    print("PASS: full pipelines can be cloned by scikit-learn")
    print("RF tuning run:", rf_cfg.get("run_id", "not recorded"))
    print("GB parameters:", gb_cfg["best_params"])
    print("NO model fitting. NO training CV. NO held-out test evaluation.")
    print("NEXT: M3 may fit the frozen E2 on the shared X_train, y_train.")


def main() -> None:
    parser = argparse.ArgumentParser(description="E2-only frozen ensemble factory")
    parser.add_argument("--check-only", action="store_true", help="Check setup; never fit/test")
    args = parser.parse_args()
    if not args.check_only:
        parser.print_help()
        return
    check_only()


if __name__ == "__main__":
    main()
