"""Configuration, dataset inspection, and training controls."""

import json

import pandas as pd
import streamlit as st

from m3_workspace.app.dashboards import comparison, evidence, exports, legacy_evidence, live_history
from m3_workspace.app.visualizer import visualizer
from m3_workspace.engine.artifact_manager import HISTORY, history
from m3_workspace.engine.experiment_manager import dataset, launch, prepare
from m3_workspace.engine.tuning_engine import CPUS, DEFAULTS, GROUP_BASELINE, HELP, METRICS, PRESETS, SPACE


S = st.session_state
LABELS = {
    "n_estimators": "Number of trees", "max_depth": "Maximum depth", "max_features": "Features per split",
    "random_state": "Forest seed", "n_jobs": "Forest workers", "oob_score": "Out-of-bag scoring",
}


def label(key):
    return LABELS.get(key, key.replace("_", " ").capitalize())


def set_model(params):
    for key, value in params.items():
        if key in HELP:
            S[f"model_{key}"] = value if key in ("bootstrap", "oob_score") else json.dumps(value)


def set_space(space):
    for key in SPACE:
        S[f"use_{key}"] = key in space
        S[f"grid_{key}"] = json.dumps(space.get(key, SPACE[key]))


def apply_preset():
    name = S.preset
    if name in PRESETS:
        iterations, space = PRESETS[name]
        S.iterations = iterations
        set_space(space)


def import_configuration():
    try:
        configuration = json.loads(S.config_upload.getvalue())
        prepare(configuration)
        S.mode = configuration["dataset"]
        S.limit = configuration["fit_limit"]
        set_model(configuration["model"])
        for key, value in configuration["cv"].items():
            S[f"cv_{key}"] = value
        if configuration.get("search"):
            search = configuration["search"]
            S.preset = "Custom Search"
            for key in ("method", "iterations", "jobs", "verbosity", "errors", "refit"):
                S[key] = search[key]
            S.search_seed = search["seed"]
            set_space(search["space"])
        S.config_notice = ("success", "Configuration loaded.")
    except (AttributeError, ValueError, TypeError, KeyError) as error:
        S.config_notice = ("error", f"Could not load configuration. {error}")


def initialise():
    if "builder_initialised" in S:
        return
    set_model(GROUP_BASELINE)
    S.preset = "Balanced Search"
    apply_preset()
    S.builder_initialised = True


def scalar_widget(key, container):
    if key in ("bootstrap", "oob_score"):
        container.checkbox(label(key), key=f"model_{key}", help=HELP[key])
    elif key == "criterion":
        container.selectbox(label(key), ['"gini"', '"entropy"', '"log_loss"'], key=f"model_{key}",
                            format_func=json.loads, help=HELP[key])
    else:
        container.text_input(label(key), key=f"model_{key}", help=HELP[key])


def cv_controls():
    st.subheader("Validation protocol")
    st.caption("The group protocol uses three shuffled stratified folds, seed 42 and positive-class F1.")
    a, b, c = st.columns(3)
    folds = a.number_input("Stratified folds", min_value=2, max_value=100, value=3, key="cv_folds")
    shuffle = b.checkbox("Shuffle", value=True, key="cv_shuffle")
    seed = c.number_input("CV seed", min_value=0, max_value=2**32 - 1, value=42, key="cv_seed",
                          disabled=not shuffle)
    primary = a.selectbox("Primary metric", METRICS, key="cv_primary")
    b.multiselect("Secondary metrics", METRICS, default=["precision", "recall", "roc_auc", "accuracy"],
                  key="cv_secondary")
    secondary = [metric for metric in S.cv_secondary if metric != primary]
    st.caption("Each fold fits its own preprocessing. Out-of-fold plots combine predictions from the validation partitions.")
    st.number_input("Maximum total fits", 1, 5000, 350, key="limit",
                    help="Includes search, validation of the selected configuration and final training refit.")
    return {"folds": int(folds), "shuffle": shuffle, "seed": int(seed), "primary": primary, "secondary": secondary}


def submit(kind, config, metadata, parsing_error=None):
    error = parsing_error
    if not metadata:
        error = "The active dataset is blocked. See Dataset & Pipeline."
    if not error:
        try:
            *_, cost = prepare(config)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            error = str(exc)
    if error:
        st.error(error)
    else:
        a, b, c, d = st.columns(4)
        a.metric("Candidates", cost["candidates"])
        b.metric("Folds", config["cv"]["folds"])
        c.metric("Total fits", cost["fits"])
        d.metric("Workers", cost["workers"])
        st.caption("Concurrent fits retain transformed fold data and trees in memory. More or larger trees increase memory use. Reduce workers if memory is limited.")
    a, b, c = st.columns(3)
    if a.button(f"Validate {kind.lower()}", disabled=bool(error)):
        st.success("Configuration and training-split checks passed.")
    c.download_button(f"Save {kind.lower()} configuration", json.dumps(config, indent=2),
                      f"{kind.lower()}_configuration.json", "application/json", disabled=bool(error))
    if b.button(f"Run {kind.lower()}", type="primary", disabled=bool(error) or (HISTORY / "active.lock").exists()):
        try:
            record = launch(config | {"dataset_snapshot": metadata})
            st.success(f"Run {record['id'][:8]} queued. Follow it in Export & History.")
        except (ValueError, OSError) as exc:
            st.error(str(exc))


def show_dataset(metadata, error):
    if error:
        st.error(error)
        return
    st.subheader(metadata["name"])
    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{metadata['rows']:,}")
    b.metric("Columns", metadata["columns"])
    c.metric("Predictors", metadata["features"])
    d.metric("Missing values", metadata["missing"])
    st.write(metadata["pipeline"])
    st.write(f"Target **{metadata['target']}** · Positive class **1** · Shared 80/20 stratified split, seed 42")
    st.dataframe(pd.DataFrame({
        "Full dataset": metadata["distribution"], "Training": metadata["train_distribution"],
        "Held out": metadata["test_distribution"],
    }).rename_axis("Class"), width="stretch")
    with st.expander("Reproducibility identifiers"):
        st.json({key: metadata[key] for key in ("fingerprint", "split_id", "pipeline_hash")})


def baseline_controls():
    st.subheader("Baseline configuration")
    a, b = st.columns(2)
    a.button("Load group baseline", on_click=set_model, args=(GROUP_BASELINE,),
             help="Uses the project's default forest with seed 42 and one worker.")
    b.button("Load scikit-learn defaults", on_click=set_model, args=(DEFAULTS,),
             help="Includes random_state=null. Set an integer seed to make results repeatable.")
    st.caption("Use JSON values in flexible fields. Enter null for an unset option and quote text such as \"sqrt\".")
    a, b, c = st.columns(3)
    for key, column in zip(("n_estimators", "max_depth", "max_features", "criterion", "bootstrap", "random_state"),
                           (a, b, c, a, b, c)):
        scalar_widget(key, column)
    with st.expander("Advanced forest settings"):
        columns = st.columns(3)
        for i, key in enumerate(k for k in HELP if k not in (
            "n_estimators", "max_depth", "max_features", "criterion", "bootstrap", "random_state"
        )):
            scalar_widget(key, columns[i % 3])
    try:
        return {key: S[f"model_{key}"] if key in ("bootstrap", "oob_score")
                else json.loads(S[f"model_{key}"]) for key in HELP}, None
    except json.JSONDecodeError:
        return {}, "One or more baseline fields contain invalid JSON values."


def tuning_controls():
    st.subheader("Hyperparameter search")
    st.caption("Parameters outside the search space take their values from Baseline Builder.")
    a, b, c = st.columns(3)
    a.selectbox("Preset", [*PRESETS, "Custom Search"], key="preset", on_change=apply_preset)
    method = b.selectbox("Search method", ["random", "grid"], key="method",
                         format_func={"random": "RandomizedSearchCV", "grid": "GridSearchCV"}.get)
    iterations = c.number_input("Random draws", 1, 5000, key="iterations", disabled=method == "grid")
    values = {}
    error = None
    for key in SPACE:
        a, b = st.columns([1, 2])
        enabled = a.checkbox(label(key), key=f"use_{key}", help=HELP[key])
        text = b.text_input(f"Values for {label(key)}", key=f"grid_{key}", disabled=not enabled)
        if enabled:
            try:
                values[key] = json.loads(text)
            except json.JSONDecodeError:
                error = f"{key} needs a JSON list such as [5, 10, null]."
    with st.expander("Search-space preview"):
        st.json(values)
        st.caption("The engine filters invalid bootstrap and sample-size combinations before calculating the candidate count.")
    with st.expander("Search execution settings"):
        a, b, c = st.columns(3)
        workers = a.number_input("Parallel CV workers", 1, CPUS, min(4, CPUS), key="jobs")
        verbosity = b.number_input("Log verbosity", 0, 10, 3, key="verbosity")
        seed = c.number_input("Random search seed", 0, 2**32 - 1, 42, key="search_seed")
        errors = a.selectbox("Candidate errors", ["raise", "nan"], key="errors",
                             format_func={"raise": "Stop on error", "nan": "Record failed scores"}.get)
        refit = b.selectbox("Refit strategy", ["primary", "manual"], key="refit",
                            format_func={"primary": "Refit inside search", "manual": "Refit after ranking"}.get)
    return {"space": values, "iterations": int(iterations), "method": method, "jobs": int(workers),
            "verbosity": int(verbosity), "seed": int(seed), "errors": errors, "refit": refit}, error


def render():
    initialise()
    st.markdown("""
        <style>
        .block-container { max-width: 1260px; padding-top: 2.4rem; }
        [data-testid="stMetric"] { border-top: 2px solid #458d8b; padding-top: .55rem; }
        [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
        [data-testid="stTabs"] [role="tablist"] { gap: .75rem; }
        </style>
    """, unsafe_allow_html=True)
    st.title("Random Forest Builder")
    st.caption("Credit card default prediction · M3 experiment studio")
    controls, upload = st.columns([2, 1])
    mode = controls.radio("Dataset mode", ["official", "sandbox"], horizontal=True, key="mode",
                          format_func={"official": "Official credit data", "sandbox": "Synthetic sandbox"}.get)
    with upload.expander("Load configuration"):
        st.file_uploader("Configuration JSON", type="json", key="config_upload")
        st.button("Load configuration", on_click=import_configuration, disabled=S.config_upload is None)
    if "config_notice" in S:
        kind, notice = S.pop("config_notice")
        getattr(st, kind)(notice)
    try:
        _, metadata = dataset(mode)
        dataset_error = None
    except (ValueError, OSError, ImportError) as exc:
        metadata, dataset_error = None, str(exc)
    if (HISTORY / "active.lock").exists():
        st.info("An experiment is running. Progress and logs update in Export & History.")
    completed = [row for row in history() if row["status"] == "COMPLETED" and row["config"]["dataset"] == mode]
    tabs = st.tabs(["Dataset & Pipeline", "Baseline Builder", "Hyperparameter Tuning", "Cross-Validation",
                    "Results & Evaluation", "Experiment Comparison", "Random Forest Visualizer", "Export & History"])
    with tabs[3]:
        protocol = cv_controls()
    with tabs[0]:
        show_dataset(metadata, dataset_error)
    with tabs[1]:
        params, model_error = baseline_controls()
        config = {"dataset": mode, "model": params, "cv": protocol, "fit_limit": int(S.limit)}
        submit("Baseline", config, metadata, model_error)
    with tabs[2]:
        search, search_error = tuning_controls()
        submit("Tuning", config | {"search": search}, metadata, model_error or search_error)
    with tabs[4]:
        evidence(completed)
        if mode == "official":
            legacy_evidence()
    with tabs[5]:
        comparison(completed)
    with tabs[6]:
        visualizer(completed)
    with tabs[7]:
        live_history(mode)
        exports(completed)
