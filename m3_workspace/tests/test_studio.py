import importlib
import sys
import time

import joblib
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from m3_workspace.app.dashboards import png_plot
from m3_workspace.engine import experiment_manager as manager
from m3_workspace.engine.artifact_manager import ROOT, history, read_json, run_folder
from m3_workspace.engine.tuning_engine import GROUP_BASELINE, search_grids, validate_model
from src.evaluation import evaluate


@pytest.fixture
def config():
    return {
        "dataset": "sandbox", "model": GROUP_BASELINE | {"n_estimators": 4},
        "cv": {"folds": 3, "shuffle": True, "seed": 42, "primary": "f1", "secondary": ["recall"]},
        "fit_limit": 100,
    }


@pytest.mark.parametrize("params", [
    {"bootstrap": False, "oob_score": True}, {"bootstrap": False, "max_samples": .5},
    {"n_jobs": 999}, {"max_depth": 0}, {"n_estimators": 0}, {"ccp_alpha": -.1},
])
def test_invalid_baseline(params):
    with pytest.raises(ValueError):
        validate_model(params)


def test_conditional_search():
    grids = search_grids(GROUP_BASELINE, {"bootstrap": [True, False], "max_samples": [None, .5]})
    assert len(grids) == 3
    assert {"model__bootstrap": [False], "model__max_samples": [None]} in grids
    with pytest.raises(ValueError, match="no valid"):
        search_grids(GROUP_BASELINE, {"bootstrap": [False], "max_samples": [.5]})
    with pytest.raises(ValueError):
        search_grids(GROUP_BASELINE, {"n_estimators": []})


def test_baseline_and_export(tmp_path, config):
    result = manager.execute(config, tmp_path)
    assert len(result["cv"]) == 2
    assert len(pd.read_csv(tmp_path / "folds.csv")) == 3
    assert sum(map(sum, result["metrics"]["confusion_matrix"])) == 320
    assert read_json(tmp_path / "summary.json") == result
    assert png_plot(tmp_path, result).startswith(b"\x89PNG")
    model = joblib.load(tmp_path / "model.joblib")
    assert len(model.named_steps["model"].estimators_) == 4
    assert model.predict_proba(manager.dataset("sandbox")[0][0].head()).shape == (5, 2)


@pytest.mark.parametrize("method,refit", [("random", "primary"), ("grid", "manual")])
def test_search_records_best_candidate(method, refit, tmp_path, config):
    config["search"] = {
        "method": method, "iterations": 3, "jobs": 2, "verbosity": 0, "seed": 42,
        "errors": "raise", "refit": refit, "space": {"n_estimators": [3, 4], "max_depth": [2, None]},
    }
    result = manager.execute(config, tmp_path)
    candidates = pd.read_csv(tmp_path / "cv_results.csv")
    assert len(candidates) == (3 if method == "random" else 4)
    assert result["winning_draw"] == candidates.iloc[0]["draw"]
    assert result["cv"]["f1"]["mean"] == pytest.approx(candidates.iloc[0]["mean_test_f1"])
    assert result["best_params"]["n_estimators"] == candidates.iloc[0]["param_model__n_estimators"]


def test_correct_metrics():
    scores = evaluate([0, 0, 1, 1], [0, 1, 1, 1], [.1, .8, .7, .9])
    assert scores["confusion_matrix"] == [[1, 1], [0, 2]]
    assert scores["precision"] == pytest.approx(2 / 3)
    assert scores["f1"] == pytest.approx(.8)
    assert scores["roc_auc"] == pytest.approx(.75)


def test_pipeline_unavailable(monkeypatch):
    import src
    monkeypatch.delattr(src, "preprocessing", raising=False)
    monkeypatch.setitem(sys.modules, "src.preprocessing", None)
    with pytest.raises(manager.DatasetBlocked, match="M2"):
        manager.dataset("official")
    assert manager.dataset("sandbox")[1]["rows"] == 400


def test_official_pipeline_reuses_m2():
    try:
        shared = importlib.import_module("src.preprocessing")
    except ImportError:
        pytest.skip("M2 preprocessing unavailable")
    model = manager.make_pipeline("official", {"n_estimators": 4})
    preprocess = model.named_steps["preprocess"]
    assert preprocess.transformers[0][1] == "passthrough"
    assert preprocess.transformers[1][2] == shared.CATEGORICAL_FEATURES


def test_run_history_and_duplicate_lock(tmp_path, config, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(manager.subprocess, "Popen", lambda *args, **kwargs: None)
        record = manager.launch(config, tmp_path)
        assert history(tmp_path)[0]["status"] == "READY"
        with pytest.raises(ValueError, match="already running"):
            manager.launch(config, tmp_path)
    manager.worker(run_folder(record, tmp_path), tmp_path / "active.lock")
    completed = history(tmp_path)[0]
    assert completed["status"] == "COMPLETED"
    assert completed["ended_at"]
    assert not (tmp_path / "active.lock").exists()


def test_actual_worker(tmp_path, config):
    record = manager.launch(config, tmp_path)
    path = run_folder(record, tmp_path)
    deadline = time.monotonic() + 40
    while (tmp_path / "active.lock").exists() and time.monotonic() < deadline:
        time.sleep(.1)
    assert read_json(path / "record.json")["status"] == "COMPLETED", (path / "worker.log").read_text()


def test_streamlit_builder_controls(monkeypatch):
    import m3_workspace.app.builder as builder
    real_dataset = builder.dataset
    monkeypatch.setattr(builder, "dataset", lambda _: real_dataset("sandbox"))
    app = AppTest.from_file(ROOT / "m3_workspace/app/main.py", default_timeout=15).run()
    assert not app.exception
    app.radio(key="mode").set_value("sandbox").run()
    app.selectbox(key="preset").set_value("Quick Test").run()
    assert app.number_input(key="iterations").value == 4
    assert app.text_input(key="grid_n_estimators").value == "[50, 100]"
    assert app.session_state["model_n_estimators"] == "100"
    assert len(app.tabs) == 8
    assert not app.exception
