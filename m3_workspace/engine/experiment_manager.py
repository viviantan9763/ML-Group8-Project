"""Run and persist baseline and tuned forests, independently of Streamlit."""

import argparse
import hashlib
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread
from time import perf_counter
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from m3_workspace.engine.artifact_manager import HISTORY, ROOT, read_json, reserve, run_folder, write_json
from m3_workspace.engine.tuning_engine import effective_workers, validate_config
from src.evaluation import curve, evaluate, out_of_fold, predict


OFFICIAL_DATA = ROOT / "data/raw/credit_default.csv"
FREEZE = ROOT / "m3_workspace" / "final_protocol.json"


class DatasetBlocked(ValueError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def shared_pipeline():
    try:
        from src import preprocessing
        for name in ("load_data", "split_data", "build_pipeline", "get_cv", "TARGET_COLUMN", "ID_COLUMN"):
            getattr(preprocessing, name)
    except (ImportError, AttributeError) as error:
        raise DatasetBlocked(f"M2's shared preprocessing is unavailable: {error}") from error
    return preprocessing


def dataset(mode):
    if mode == "official":
        shared = shared_pipeline()
        if not OFFICIAL_DATA.is_file():
            raise DatasetBlocked(f"Official dataset is missing at {OFFICIAL_DATA}.")
        df = shared.load_data(OFFICIAL_DATA)
        split = shared.split_data(df)
        target = shared.TARGET_COLUMN
        fingerprint = digest(OFFICIAL_DATA.read_bytes())
        pipeline = "M2 shared pipeline: one-hot categorical predictors, numeric passthrough, ID excluded"
        pipeline_hash = digest(Path(shared.__file__).read_bytes())
        title = "UCI Default of Credit Card Clients (#350)"
    elif mode == "sandbox":
        X, y = make_classification(
            n_samples=400, n_features=8, n_informative=5, weights=[0.78, 0.22], random_state=42,
        )
        df = pd.DataFrame(X, columns=[f"synthetic_{i}" for i in range(X.shape[1])])
        target = "synthetic_target"
        df[target] = y
        split = train_test_split(df.drop(columns=target), df[target], test_size=0.20, stratify=y, random_state=42)
        fingerprint = digest(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
        pipeline = "Sandbox pipeline: numeric predictors passed directly to the forest"
        pipeline_hash = "sandbox-numeric-v1"
        title = "Synthetic sandbox"
    else:
        raise ValueError("Unknown dataset mode.")
    X_train, X_test, y_train, y_test = split
    metadata = {
        "name": title, "fingerprint": fingerprint, "pipeline": pipeline, "pipeline_hash": pipeline_hash,
        "split_id": digest(np.asarray(X_train.index, dtype="<i8").tobytes()),
        "rows": len(df), "columns": df.shape[1], "features": X_train.shape[1],
        "target": target, "positive_class": 1, "missing": int(df.isna().sum().sum()),
        "train_rows": len(y_train), "test_rows": len(y_test),
        "distribution": {str(key): int(value) for key, value in df[target].value_counts().items()},
        "train_distribution": {str(key): int(value) for key, value in y_train.value_counts().items()},
        "test_distribution": {str(key): int(value) for key, value in y_test.value_counts().items()},
    }
    return split, metadata


def make_pipeline(mode, params):
    if mode == "official":
        shared_pipeline()
        from src.random_forest import make_random_forest
        return make_random_forest(**params)
    return Pipeline([("model", RandomForestClassifier(**params))])


def prepare(config):
    split, metadata = dataset(config["dataset"])
    params, cv, scores, grids, cost = validate_config(config, split[2])
    folds = list(cv.split(split[0], split[2]))
    smallest_training_fold = min(len(train) for train, _ in folds)
    samples = (config.get("search") or {}).get("space", {}).get("max_samples", [params["max_samples"]])
    if any(type(value) is int and value > smallest_training_fold for value in samples):
        raise ValueError(f"max_samples cannot exceed {smallest_training_fold} rows in a CV training fold.")
    return split, metadata, params, folds, scores, grids, cost


def execute(config, folder, expected_dataset=None):
    start = perf_counter()
    (X, _, y, _), metadata, params, folds, scores, grids, cost = prepare(config)
    if expected_dataset is not None and metadata != expected_dataset:
        raise ValueError("Dataset or preprocessing changed after the run was queued.")
    folder.mkdir(parents=True, exist_ok=True)
    fitted_params = {**params, "n_jobs": effective_workers(params["n_jobs"])}
    search = config.get("search")
    winner = None
    best_estimator = None
    if search:
        fitted_params["n_jobs"] = 1
        pipeline = make_pipeline(config["dataset"], fitted_params)
        automatic = search["refit"] == "primary"
        options = {
            "scoring": scores, "refit": config["cv"]["primary"] if automatic else False,
            "cv": folds, "n_jobs": cost["workers"], "verbose": search["verbosity"],
            "pre_dispatch": cost["workers"], "return_train_score": False,
            "error_score": "raise" if search["errors"] == "raise" else np.nan,
        }
        if search["method"] == "grid":
            estimator = GridSearchCV(pipeline, grids, **options)
        else:
            estimator = RandomizedSearchCV(
                pipeline, grids, n_iter=cost["candidates"], random_state=search["seed"], **options
            )
        print(f"Searching {cost['candidates']} configurations across {len(folds)} folds.", flush=True)
        # Threads share the dense input and forward completed-fit logs to the worker log.
        with joblib.parallel_config(backend="threading"):
            estimator.fit(X, y)
        candidates = pd.DataFrame(estimator.cv_results_)
        candidates.insert(0, "draw", candidates.index + 1)
        primary = f"mean_test_{config['cv']['primary']}"
        valid = candidates[primary].replace([np.inf, -np.inf], np.nan).dropna()
        if valid.empty:
            raise ValueError("Every candidate failed to produce a valid primary score.")
        winner = int(valid.idxmax())
        candidates.sort_values(primary, ascending=False, kind="stable").to_csv(folder / "cv_results.csv", index=False)
        fitted_params.update({
            key.removeprefix("model__"): value for key, value in candidates.loc[winner, "params"].items()
        })
        if automatic and estimator.best_index_ == winner:
            best_estimator = estimator.best_estimator_
    pipeline = make_pipeline(config["dataset"], fitted_params)
    print("Collecting out-of-fold predictions and fold metrics.", flush=True)
    fold_cv = cross_validate(
        pipeline, X, y, cv=folds, scoring=scores, return_estimator=True, error_score="raise"
    )
    results = pd.DataFrame({
        "fold": range(1, len(folds) + 1), "fit_seconds": fold_cv["fit_time"],
        **{metric: fold_cv[f"test_{metric}"] for metric in scores},
    })
    results.to_csv(folder / "folds.csv", index=False)
    if not search:
        results.to_csv(folder / "cv_results.csv", index=False)
    metrics, pr, prediction_seconds = out_of_fold(X, y, folds, fold_cv["estimator"])
    pr.to_csv(folder / "precision_recall.csv", index=False)
    if best_estimator is None:
        print("Fitting selected forest on all training rows.", flush=True)
        best_estimator = pipeline.fit(X, y)
    joblib.dump(best_estimator, folder / "model.joblib", compress=3)
    forest = best_estimator.named_steps["model"]
    summary = {
        "dataset": metadata, "cost": cost, "best_params": fitted_params, "winning_draw": None if winner is None else winner + 1,
        "metrics": metrics, "cv": {metric: {
            "mean": float(results[metric].mean()), "std": float(results[metric].std(ddof=0))
        } for metric in scores},
        "runtime_seconds": perf_counter() - start, "fold_fit_seconds": float(results["fit_seconds"].sum()),
        "prediction_seconds": prediction_seconds, "mean_tree_depth": float(np.mean([t.get_depth() for t in forest.estimators_])),
        "mean_tree_leaves": float(np.mean([t.get_n_leaves() for t in forest.estimators_])),
        "training_oob_accuracy": getattr(forest, "oob_score_", None),
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "pandas": pd.__version__},
        "artifacts": ["config.json", "worker.log", "folds.csv", "cv_results.csv", "precision_recall.csv", "model.joblib", "summary.json"],
    }
    write_json(folder / "summary.json", summary)
    return summary


def launch(config, root=HISTORY):
    run_id = uuid4().hex
    record = {"id": run_id, "config": config, "type": "tuning" if config.get("search") else "baseline",
              "status": "READY", "started_at": now(), "ended_at": None, "summary": None}
    folder = run_folder(record, root)
    lock = root / "active.lock"
    try:
        reserve(lock, {"run_id": run_id, "dataset": config["dataset"]})
    except FileExistsError as error:
        raise ValueError("An experiment is already running. Follow it in Export & History.") from error
    try:
        write_json(folder / "config.json", config)
        write_json(folder / "record.json", record)
        with (folder / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", "-m", "m3_workspace.engine.experiment_manager", str(folder), str(lock)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUTF8": "1"},
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        if process is not None:
            Thread(target=watch_worker, args=(process, folder, lock), daemon=True).start()
    except Exception as error:
        write_json(folder / "record.json", record | {"status": "FAILED", "ended_at": now(), "error": str(error)})
        lock.unlink(missing_ok=True)
        raise
    return record


def watch_worker(process, folder, lock):
    code = process.wait()
    record = read_json(folder / "record.json")
    if record["status"] in ("READY", "RUNNING"):
        record.update(status="FAILED", ended_at=now(), error=f"Worker exited unexpectedly with code {code}.")
        write_json(folder / "record.json", record)
        lock.unlink(missing_ok=True)


def worker(folder, lock):
    record = read_json(folder / "record.json")
    start = perf_counter()
    try:
        record.update(status="RUNNING", worker_pid=os.getpid())
        write_json(folder / "record.json", record)
        record.update(summary=execute(
            record["config"], folder, record["config"].get("dataset_snapshot")
        ), status="COMPLETED")
    except DatasetBlocked as error:
        record.update(status="BLOCKED", error=str(error))
    except Exception as error:
        traceback.print_exc()
        record.update(status="FAILED", error=str(error))
    finally:
        record.update(ended_at=now(), runtime_seconds=perf_counter() - start)
        try:
            write_json(folder / "record.json", record)
        finally:
            lock.unlink(missing_ok=True)


def compatible(a, b):
    if a["config"]["dataset"] != b["config"]["dataset"]:
        return False
    first, second = a["summary"]["dataset"], b["summary"]["dataset"]
    if any(first[key] != second[key] for key in ("fingerprint", "split_id", "pipeline_hash")):
        return False
    cv_keys = ("folds", "shuffle")
    a_cv, b_cv = a["config"]["cv"], b["config"]["cv"]
    return all(a_cv[key] == b_cv[key] for key in cv_keys) and (
        not a_cv["shuffle"] or a_cv["seed"] == b_cv["seed"]
    )


def final_evaluate(record, root=HISTORY):
    if record["status"] != "COMPLETED" or record["config"]["dataset"] != "official" or not FREEZE.exists():
        raise ValueError("Final evaluation requires a completed official run and an approved frozen protocol.")
    freeze = read_json(FREEZE)
    if (freeze.get("run_id") != record["id"] or freeze.get("approved") is not True
            or not str(freeze.get("protocol", "")).strip()):
        raise ValueError("The protocol must identify and approve this exact frozen run.")
    (_, X, _, y), metadata = dataset("official")
    if metadata != record["summary"]["dataset"] or freeze.get("dataset_fingerprint") != metadata["fingerprint"]:
        raise ValueError("Frozen protocol, run, and current dataset do not match.")
    output = root / "final" / metadata["fingerprint"]
    reserve(output / "attempt.lock", {"run_id": record["id"], "started_at": now()})
    fitted = joblib.load(run_folder(record, root) / "model.joblib")
    labels, probabilities, seconds = predict(fitted, X)
    result = {"run_id": record["id"], "finished_at": now(), "metrics": evaluate(y, labels, probabilities),
              "prediction_seconds": seconds}
    curve(y, probabilities).to_csv(output / "precision_recall.csv", index=False)
    write_json(output / "evaluation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Validate and queue a Studio configuration JSON")
    parser.add_argument("worker_paths", nargs="*", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.config is not None and not args.worker_paths:
        configuration = read_json(args.config)
        _, snapshot, *_ = prepare(configuration)
        record = launch(configuration | {"dataset_snapshot": snapshot})
        print(run_folder(record))
    elif args.config is None and len(args.worker_paths) == 2:
        worker(*args.worker_paths)
    else:
        parser.error("Specify --config with an exported configuration JSON.")
