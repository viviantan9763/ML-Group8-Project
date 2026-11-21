import json

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest, t as student_t
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from streamlit.testing.v1 import AppTest

from m3_workspace.engine.artifact_manager import ROOT
from src import final_evaluation as fe
from src.gradient_boosting import BalancedGradientBoostingClassifier
from src.preprocessing import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, build_pipeline
from src.random_forest import make_random_forest


def _synthetic_split(n_train=256, n_test=64, seed=42):
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test

    data = {}
    data["SEX"] = rng.choice([1, 2], size=n_total)
    data["EDUCATION"] = rng.choice([1, 2, 3], size=n_total)
    data["MARRIAGE"] = rng.choice([1, 2], size=n_total)
    pay0 = rng.choice([-1, 0, 1, 2], size=n_total, p=[0.25, 0.45, 0.15, 0.15])
    data["PAY_0"] = pay0
    for col in ("PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"):
        data[col] = rng.choice([-1, 0, 2], size=n_total, p=[0.25, 0.55, 0.20])

    limit_bal = rng.normal(150000, 50000, size=n_total).clip(10000, 400000)
    data["LIMIT_BAL"] = limit_bal
    data["AGE"] = rng.integers(22, 65, size=n_total)
    for col in [c for c in NUMERICAL_FEATURES if c not in ("LIMIT_BAL", "AGE")]:
        data[col] = rng.normal(25000, 12000, size=n_total)

    df = pd.DataFrame(data)[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    logits = -1.3 + 2.1 * (pay0 >= 2) - 0.8 * ((limit_bal - 150000) / 50000)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs).astype(int)

    X_train = df.iloc[:n_train].copy()
    X_test = df.iloc[n_train:].copy()
    y_train = y[:n_train]
    y_test = y[n_train:]
    test_ids = list(range(1000, 1000 + n_test))
    return X_train, X_test, y_train, y_test, test_ids


def _lightweight_models():
    gb = build_pipeline(
        BalancedGradientBoostingClassifier(
            n_estimators=6,
            learning_rate=0.1,
            max_depth=2,
            min_samples_split=5,
            min_samples_leaf=3,
            subsample=0.8,
            random_state=42,
        ),
        scale_numeric=False,
    )
    lr = build_pipeline(
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=500, random_state=42),
        scale_numeric=True,
    )
    rf = make_random_forest(
        n_estimators=6,
        max_depth=5,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=42,
    )
    e2 = VotingClassifier(
        estimators=[("rf", rf), ("gb_balanced", gb)],
        voting="soft",
        weights=[2, 1],
        n_jobs=1,
    )
    return {
        "gb_balanced": gb,
        "logistic_regression": lr,
        "random_forest": rf,
        "ensemble_e2": e2,
    }


def test_mcnemar_exact_chi2_and_zero_discordant():
    y = np.array([1, 1, 0, 0, 1, 0])
    res_tie = fe.mcnemar_test(y, y.copy(), y.copy())
    assert res_tie["discordant"] == 0
    assert res_tie["p_value"] == pytest.approx(1.0)
    assert res_tie["significant_0_05"] is False

    # Build exact n10=12, n01=2 -> discordant=14 (< 25 triggers exact binomial)
    y_small = np.ones(20, dtype=int)
    pred_a_small = np.ones(20, dtype=int)
    pred_b_small = np.ones(20, dtype=int)
    pred_b_small[:12] = 0
    pred_a_small[12:14] = 0
    res_small = fe.mcnemar_test(y_small, pred_a_small, pred_b_small)
    assert res_small["method"] == "exact_binomial"
    assert res_small["discordant"] == 14
    assert res_small["net_advantage_a"] == 10
    assert res_small["p_value"] == pytest.approx(binomtest(2, 14, 0.5).pvalue)
    assert res_small["significant_0_05"] is True

    # Build n10=35, n01=10 -> discordant=45 (>= 25 triggers chi2 continuity-corrected)
    y_large = np.ones(60, dtype=int)
    pred_a_large = np.ones(60, dtype=int)
    pred_b_large = np.ones(60, dtype=int)
    pred_b_large[:35] = 0
    pred_a_large[35:45] = 0
    res_large = fe.mcnemar_test(y_large, pred_a_large, pred_b_large)
    assert res_large["method"] == "chi2_continuity_corrected"
    assert res_large["discordant"] == 45
    assert res_large["statistic"] == pytest.approx(((abs(35 - 10) - 1.0) ** 2) / 45.0)
    assert res_large["significant_0_05"] is True


def test_nadeau_bengio_corrected_ttest():
    a = np.array([0.550, 0.542, 0.558, 0.560, 0.545, 0.553])
    b = np.array([0.540, 0.533, 0.546, 0.547, 0.535, 0.544])
    res = fe.nadeau_bengio_corrected_ttest(a, b, n_train=16000, n_test=8000)

    diff = a - b
    expected_var = (1.0 / 6.0 + 8000.0 / 16000.0) * np.var(diff, ddof=1)
    expected_t = diff.mean() / np.sqrt(expected_var)
    expected_p = 2.0 * student_t.sf(abs(expected_t), df=5)
    assert res["t_stat"] == pytest.approx(expected_t)
    assert res["p_value"] == pytest.approx(expected_p)
    assert res["folds_a_better"] == 6

    res_equal = fe.nadeau_bengio_corrected_ttest(a, a, n_train=16000, n_test=8000)
    assert res_equal["t_stat"] == 0.0
    assert res_equal["p_value"] == pytest.approx(1.0)
    assert res_equal["significant_0_05"] is False


def test_paired_bootstrap_ci_determinism():
    rng = np.random.default_rng(7)
    y = rng.binomial(1, 0.25, size=300)
    prob_b = np.where(y == 1, 0.52, 0.42) + rng.normal(0, 0.06, size=300)
    prob_a = np.where(y == 1, 0.78, 0.22) + rng.normal(0, 0.06, size=300)
    prob_a = np.clip(prob_a, 0.01, 0.99)
    prob_b = np.clip(prob_b, 0.01, 0.99)
    pred_a = (prob_a >= 0.5).astype(int)
    pred_b = (prob_b >= 0.5).astype(int)

    res1 = fe.paired_bootstrap_ci(y, pred_a, prob_a, pred_b, prob_b, n_boot=100, seed=42)
    res2 = fe.paired_bootstrap_ci(y, pred_a, prob_a, pred_b, prob_b, n_boot=100, seed=42)
    assert res1 == res2
    assert res1["f1"]["ci_low"] <= res1["f1"]["observed_diff"] <= res1["f1"]["ci_high"]
    assert res1["roc_auc"]["significant_0_05"] is True


def test_preflight_check_official_remains_untouched():
    info = fe.preflight_check()
    assert info["checks_passed"] is True
    assert info["final_result_exists"] is False
    assert not fe.FINAL_EVAL_JSON.exists()
    assert info["dataset"]["train_rows"] == 24000
    assert info["dataset"]["test_rows"] == 6000
    assert list(info["models"].keys()) == list(fe.MODEL_ORDER)
    assert "E2 vs RF" in info["training_cv"]["nadeau_bengio_e3"]


def test_end_to_end_synthetic_evaluation_and_lock(tmp_path):
    res_dir = tmp_path / "results"
    art_dir = tmp_path / "artifacts"
    data = _synthetic_split()
    models = _lightweight_models()

    report = fe.run_final_evaluation(
        results_dir=res_dir,
        artifacts_dir=art_dir,
        force=False,
        perm_repeats=2,
        n_boot=60,
        data_override=data,
        models_override=models,
    )

    assert not fe.FINAL_EVAL_JSON.exists()
    eval_path = res_dir / "final_evaluation.json"
    comp_path = res_dir / "final_model_comparison.csv"
    pred_path = res_dir / "final_test_predictions.csv"
    joblib_path = art_dir / "final_ensemble_e2.joblib"

    assert eval_path.is_file()
    assert comp_path.is_file()
    assert pred_path.is_file()
    assert joblib_path.is_file()

    parsed = json.loads(eval_path.read_text(encoding="utf-8"))
    assert set(parsed["models"].keys()) == set(fe.MODEL_ORDER)
    assert set(parsed["dummies"].keys()) == {"dummy_prior", "dummy_all_positive"}
    assert len(parsed["statistical_tests"]["pairwise_held_out"]) == 6
    assert len(parsed["critical_discussion"]) == 5
    assert parsed["models"]["logistic_regression"]["internals"]["family"] == "linear"
    assert parsed["models"]["random_forest"]["internals"]["n_trees"] == 6

    _, X_test, _, _, _ = data
    pred_df = pd.read_csv(pred_path)
    loaded_e2 = joblib.load(joblib_path)
    assert np.allclose(loaded_e2.predict_proba(X_test)[:, 1], pred_df["prob_ensemble_e2"].to_numpy())

    with pytest.raises(FileExistsError, match="locked"):
        fe.run_final_evaluation(
            results_dir=res_dir,
            artifacts_dir=art_dir,
            force=False,
            data_override=data,
            models_override=models,
        )


def test_streamlit_final_eval_app_locked_and_loaded(tmp_path, monkeypatch):
    # 1. Verify default locked Pre-Flight state does not touch X_test and disables execution button
    app_locked = AppTest.from_file(ROOT / "m3_workspace/app/final_eval_app.py", default_timeout=30).run()
    assert not app_locked.exception
    assert not fe.FINAL_EVAL_JSON.exists()
    assert app_locked.button(key="run_official_eval_btn").disabled is True

    app_locked.radio(key="partg_section").set_value("2 · Metrics & Confusion Matrices").run()
    assert not app_locked.exception
    assert not fe.FINAL_EVAL_JSON.exists()

    # 2. Generate synthetic artifacts in tmp_path and verify all 7 GUI sections render cleanly
    res_dir = tmp_path / "results"
    art_dir = tmp_path / "artifacts"
    fe.run_final_evaluation(
        results_dir=res_dir,
        artifacts_dir=art_dir,
        force=False,
        perm_repeats=2,
        n_boot=40,
        data_override=_synthetic_split(),
        models_override=_lightweight_models(),
    )

    import m3_workspace.app.final_eval_app as final_eval_app

    monkeypatch.setattr(final_eval_app, "FINAL_EVAL_JSON", res_dir / "final_evaluation.json")
    monkeypatch.setattr(final_eval_app, "FINAL_COMPARISON_CSV", res_dir / "final_model_comparison.csv")
    monkeypatch.setattr(final_eval_app, "FINAL_PREDICTIONS_CSV", res_dir / "final_test_predictions.csv")
    monkeypatch.setattr(final_eval_app, "FINAL_ENSEMBLE_JOBLIB", art_dir / "final_ensemble_e2.joblib")

    app_loaded = AppTest.from_file(ROOT / "m3_workspace/app/final_eval_app.py", default_timeout=30).run()
    assert not app_loaded.exception
    for section in final_eval_app.SECTIONS:
        app_loaded.radio(key="partg_section").set_value(section).run()
        assert not app_loaded.exception
        if section.startswith("6"):
            app_loaded.radio(key="interp_metric").set_value("ROC-AUC Drop").run()
            assert not app_loaded.exception
