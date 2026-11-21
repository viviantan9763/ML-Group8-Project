r"""Standalone Streamlit Dashboard for Part G: Full 4-Model Final Evaluation.

Visually, architecturally, and operationally separated from M3 Random Forest
Builder (m3_workspace/app/main.py). Evaluates all 4 frozen group models side-by-side:
  - Base Model 1: Balanced Gradient Boosting (M4)
  - Base Model 2: Balanced Logistic Regression (M2)
  - Base Model 3: Tuned Random Forest (M3)
  - Ensemble Model: E2 Weighted Soft Voting RF:GB = 2:1 (M1)

Direct launch (opens browser automatically on port 8502):
    .\.venv\Scripts\python.exe m3_workspace/app/final_eval_app.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[2]


def _in_streamlit_context() -> bool:
    from streamlit import runtime
    from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

    return runtime.exists() or get_script_run_ctx(suppress_warning=True) is not None


def _pick_port(start: int = 8502, attempts: int = 20) -> int:
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start


def _open_browser_when_ready(port: int) -> None:
    health_url = f"http://127.0.0.1:{port}/_stcore/health"
    app_url = f"http://localhost:{port}"
    for _ in range(100):
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as resp:
                if resp.status == 200:
                    webbrowser.open(app_url)
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.25)


if (
    __name__ == "__main__"
    and not _in_streamlit_context()
    and os.environ.get("PARTG_EVAL_BOOTSTRAPPED") != "1"
):
    port = _pick_port(start=8502)
    env = {**os.environ, "PARTG_EVAL_BOOTSTRAPPED": "1"}
    print(
        f"Launching Part G Final 4-Model Evaluation Dashboard at http://localhost:{port} ...",
        flush=True,
    )
    threading.Thread(target=_open_browser_when_ready, args=(port,), daemon=True).start()
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL))


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src import final_evaluation as fe

FINAL_EVAL_JSON = fe.FINAL_EVAL_JSON
FINAL_COMPARISON_CSV = fe.FINAL_COMPARISON_CSV
FINAL_PREDICTIONS_CSV = fe.FINAL_PREDICTIONS_CSV
FINAL_ENSEMBLE_JOBLIB = fe.FINAL_ENSEMBLE_JOBLIB

SECTIONS = [
    "1 · Pre-Flight & Execution Gate",
    "2 · Metrics & Confusion Matrices",
    "3 · ROC, PR & Calibration Curves",
    "4 · Overfitting & Underfitting",
    "5 · Computational Efficiency",
    "6 · Model Interpretability",
    "7 · Statistical Significance & Exports",
]


def _eval_paths() -> tuple[Path, Path, Path, Path]:
    """Resolve output paths while honoring test monkeypatches across AppTest reruns."""
    mod = sys.modules.get("m3_workspace.app.final_eval_app")
    if mod is not None:
        return (
            Path(getattr(mod, "FINAL_EVAL_JSON")),
            Path(getattr(mod, "FINAL_COMPARISON_CSV")),
            Path(getattr(mod, "FINAL_PREDICTIONS_CSV")),
            Path(getattr(mod, "FINAL_ENSEMBLE_JOBLIB")),
        )
    return (
        Path(FINAL_EVAL_JSON),
        Path(FINAL_COMPARISON_CSV),
        Path(FINAL_PREDICTIONS_CSV),
        Path(FINAL_ENSEMBLE_JOBLIB),
    )


def _apply_theme_and_hero(is_complete: bool) -> None:
    st.markdown(
        """
        <style>
        .block-container { max-width: 1320px; padding-top: 1.4rem; padding-bottom: 2.5rem; }
        .partg-hero {
            background: linear-gradient(125deg, #0f172a 0%, #1e293b 52%, #312e81 100%);
            border-left: 6px solid #f59e0b;
            border-bottom: 3px solid #f59e0b;
            border-radius: 12px;
            padding: 1.35rem 1.65rem;
            color: #f8fafc;
            margin-bottom: 1.25rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.16);
        }
        .partg-pill {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            background: rgba(245, 158, 11, 0.18);
            color: #fcd34d;
            border: 1px solid rgba(251, 191, 36, 0.45);
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
            margin-bottom: 0.55rem;
        }
        .partg-hero h1 {
            margin: 0.15rem 0 0.35rem 0;
            font-size: 1.75rem;
            font-weight: 750;
            color: #ffffff;
        }
        .partg-hero p {
            margin: 0;
            font-size: 0.94rem;
            color: #cbd5e1;
        }
        .partg-callout {
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            border-left: 4px solid #4f46e5;
            border-radius: 8px;
            padding: 0.85rem 1.05rem;
            margin: 0.65rem 0 1.0rem 0;
            color: #0f172a;
        }
        .partg-lock-card {
            background: #fffbeb;
            border: 1px solid #fcd34d;
            border-left: 5px solid #d97706;
            border-radius: 10px;
            padding: 1.1rem 1.25rem;
            color: #451a03;
            margin-top: 0.5rem;
        }
        .partg-sealed-card {
            background: #ecfdf5;
            border: 1px solid #6ee7b7;
            border-left: 5px solid #059669;
            border-radius: 10px;
            padding: 1.0rem 1.2rem;
            color: #064e3b;
            margin-top: 0.5rem;
        }
        .model-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 0.9rem 1.0rem;
            min-height: 158px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }
        [data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-top: 3px solid #4f46e5;
            border-radius: 8px;
            padding: 0.65rem 0.85rem;
        }
        [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    status_label = "OFFICIAL TEST SEALED (READ-ONLY)" if is_complete else "HELD-OUT TEST LOCKED (UNTOUCHED)"
    st.markdown(
        f"""
        <div class="partg-hero">
            <div class="partg-pill">GROUP 8 · PART G FULL 4-MODEL EVALUATION · SEPARATE FROM M3 RANDOM FOREST BUILDER</div>
            <h1>⚖️ Part G — Final 4-Model Comparative Evaluation & Statistical Analysis</h1>
            <p>
                Independent held-out comparison of <strong>Base 1: Gradient Boosting (M4)</strong>,
                <strong>Base 2: Logistic Regression (M2)</strong>, <strong>Base 3: Tuned Random Forest (M3)</strong>,
                and <strong>Ensemble: E2 Soft Voting 2:1 (M1)</strong> · Status: <strong>{status_label}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(is_complete: bool) -> str:
    st.sidebar.markdown("### ⚖️ Part G Command Rail")
    st.sidebar.info(
        "**Clear Scope Distinction**\n\n"
        "- **This App (`final_eval_app.py`)**: Full **4-model** Part G comparative evaluation.\n"
        "- **M3 RF Studio (`main.py`)**: Individual **Random Forest** CV builder & tuning lab."
    )
    if is_complete:
        st.sidebar.success("✅ HELD-OUT TEST COMPLETE (READ-ONLY)")
    else:
        st.sidebar.warning("🔒 HELD-OUT TEST LOCKED (UNTOUCHED)")

    selected = st.sidebar.radio("Part G Rubric Sections", SECTIONS, key="partg_section")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**4 Frozen Models Legend**")
    for key in fe.MODEL_ORDER:
        spec = fe.MODEL_SPECS[key]
        st.sidebar.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>"
            f"<span style='display:inline-block;width:12px;height:12px;border-radius:3px;background:{spec['color']};'></span>"
            f"<span style='font-size:0.86rem;'><strong>{spec['short']}</strong> ({spec['owner']})</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    return selected


def _require_report(eval_json_path: Path) -> dict | None:
    if not eval_json_path.is_file():
        st.markdown(
            """
            <div class="partg-lock-card">
                <h4 style="margin-top:0;">🔒 Held-Out Test Split Still Untouched</h4>
                <p style="margin-bottom:0.35rem;">
                    To enforce strict zero test leakage (Rule 1), this page stays locked until you inspect
                    <strong>1 · Pre-Flight & Execution Gate</strong>, tick the freeze confirmation box, and run the
                    one-time evaluation button.
                </p>
                <p style="margin-bottom:0;">
                    All held-out metrics, confusion matrices, ROC/PR curves, overfitting gaps, timings, permutation
                    importances, and McNemar / Bootstrap statistical tests populate automatically once unlocked.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return None
    with eval_json_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _format_params_brief(key: str, cfg: dict) -> str:
    params = cfg.get("params", {})
    if key == "gb_balanced":
        return (
            f"n_estimators={params.get('model__n_estimators')}, "
            f"lr={params.get('model__learning_rate')}, "
            f"max_depth={params.get('model__max_depth')}, "
            f"subsample={params.get('model__subsample')}"
        )
    if key == "logistic_regression":
        return (
            f"C={params.get('model__C')}, "
            f"solver={params.get('model__solver')}, "
            f"class_weight={params.get('model__class_weight')}"
        )
    if key == "random_forest":
        return (
            f"run={cfg.get('run_id', '64f4bcdb')[:8]}, "
            f"trees={params.get('n_estimators')}, "
            f"depth={params.get('max_depth')}, leaf={params.get('min_samples_leaf')}, "
            f"class_weight={params.get('class_weight')}"
        )
    return "voting=soft, weights=[RF:2, GB:1], excludes LR"


def _render_section_1(eval_json_path: Path, joblib_path: Path) -> None:
    st.subheader("1 · Pre-Flight Verification & One-Time Execution Gate")
    info = fe.preflight_check(final_eval_path=eval_json_path)
    ds = info["dataset"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Training Split (80%)", f"{ds['train_rows']:,} rows", f"{ds['train_positive_rate']:.2%} defaults")
    k2.metric("Held-Out Test (20%)", f"{ds['test_rows']:,} rows", f"{ds['test_positive_rate']:.2%} defaults")
    k3.metric("Test Dummy F1 Floor", f"{ds['dummy_all_positive_f1_floor_test']:.4f}", "All-positive 2p/(1+p)")
    k4.metric("Split ID / Overlap", ds["split_id"][:10], f"{ds['train_test_duplicate_predictor_rows']} duplicate rows")

    st.markdown("#### Frozen Model Lineup (Cloned & Verified Pre-Run)")
    cols = st.columns(4)
    per_cv = info["training_cv"].get("per_model_3fold", {})
    for idx, key in enumerate(fe.MODEL_ORDER):
        m = info["models"][key]
        cv_f1 = per_cv.get(key, {}).get("cv_f1_mean")
        cv_sd = per_cv.get(key, {}).get("cv_f1_std")
        cv_text = f"{cv_f1:.4f} ± {cv_sd:.4f}" if cv_f1 is not None and cv_sd is not None else "N/A"
        brief = _format_params_brief(key, m)
        cols[idx].markdown(
            f"""
            <div class="model-card" style="border-top:4px solid {m['color']};">
                <div style="font-size:0.75rem;font-weight:700;color:{m['color']};text-transform:uppercase;">
                    {m['role']} · Owner {m['owner']}
                </div>
                <div style="font-size:1.0rem;font-weight:700;color:#0f172a;margin:0.2rem 0;">{m['short']}</div>
                <div style="font-size:0.82rem;color:#334155;margin-bottom:0.45rem;">
                    <strong>3-Fold CV F1:</strong> {cv_text}
                </div>
                <div style="font-size:0.76rem;color:#475569;line-height:1.35;">{brief}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Pre-Test Training CV Baseline & Repeated CV (E3) Evidence")
    cv_rows = []
    for key in fe.MODEL_ORDER:
        block = per_cv.get(key, {})
        cv_rows.append(
            {
                "Model": fe.MODEL_SPECS[key]["label"],
                "Role": fe.MODEL_SPECS[key]["role"],
                "3-Fold CV F1": _fmt_mean_std(block.get("cv_f1_mean"), block.get("cv_f1_std")),
                "CV Precision": _fmt_num(block.get("cv_precision_mean")),
                "CV Recall": _fmt_num(block.get("cv_recall_mean")),
                "CV ROC-AUC": _fmt_mean_std(block.get("cv_roc_auc_mean"), block.get("cv_roc_auc_std")),
                "CV Brier": _fmt_num(block.get("cv_brier_mean")),
                "Train F1 (3-Fold)": _fmt_num(block.get("train_f1_mean")),
            }
        )
    st.dataframe(pd.DataFrame(cv_rows), hide_index=True, width="stretch")

    nadeau_e3 = info["training_cv"].get("nadeau_bengio_e3", {})
    if nadeau_e3:
        with st.expander("M1 3×2 Repeated Stratified CV (E3) — Nadeau & Bengio Corrected Comparison (Pre-Test)"):
            nb_rows = []
            for pair_label, res in nadeau_e3.items():
                for metric_key, metric_label in (("f1", "F1-Score"), ("roc_auc", "ROC-AUC")):
                    mres = res[metric_key]
                    nb_rows.append(
                        {
                            "Comparison": pair_label,
                            "Metric": metric_label,
                            "Mean Diff": f"{mres['mean_diff']:+.4f}",
                            "Corrected t": f"{mres['t_stat']:+.3f}",
                            "p-value (df=5)": f"{mres['p_value']:.4f}",
                            "Sig (α=0.05)": "Yes" if mres["significant_0_05"] else "No",
                            "Folds A Better": f"{mres['folds_a_better']}/{mres['k']}",
                        }
                    )
            st.dataframe(pd.DataFrame(nb_rows), hide_index=True, width="stretch")

    st.markdown("#### Held-Out Test Execution Gate")
    if not eval_json_path.is_file():
        st.markdown(
            """
            <div class="partg-lock-card">
                <strong>Rule 1 Zero Test Leakage Gate:</strong> Only click below after reviewing all 4 frozen pipelines
                above and confirming no further hyperparameter or threshold changes will be made.
            </div>
            """,
            unsafe_allow_html=True,
        )
        confirmed = st.checkbox(
            "I confirm all 4 models (GB, LR, RF, E2) are frozen and ready for the one-time official held-out evaluation on the 6,000-row test split.",
            key="confirm_final_lock",
        )
        if st.button(
            "▶ Run Official Part G Held-Out Evaluation (One-Time)",
            type="primary",
            disabled=not confirmed,
            key="run_official_eval_btn",
        ):
            bar = st.progress(0.0)
            with st.status("Executing one-time official Part G held-out evaluation...", expanded=True) as status:
                def _on_progress(frac: float, msg: str) -> None:
                    bar.progress(float(np.clip(frac, 0.0, 1.0)))
                    st.write(msg)

                fe.run_final_evaluation(
                    results_dir=eval_json_path.parent,
                    artifacts_dir=joblib_path.parent,
                    force=False,
                    progress_callback=_on_progress,
                )
                status.update(label="Official Part G evaluation complete and sealed!", state="complete", expanded=False)
            st.success("All 4 models evaluated once on X_test. Artifacts saved and sealed.")
            st.rerun()
    else:
        with eval_json_path.open("r", encoding="utf-8") as fh:
            sealed = json.load(fh)
        vers = sealed.get("versions", {})
        st.markdown(
            f"""
            <div class="partg-sealed-card">
                <strong>✅ Official Evaluation Sealed (Read-Only)</strong><br/>
                Generated: <code>{sealed.get('generated_at', 'N/A')}</code> ·
                Split ID: <code>{sealed.get('dataset', {}).get('split_id', '')[:14]}</code> ·
                Environment: Python {vers.get('python')} / sklearn {vers.get('scikit_learn')} / pandas {vers.get('pandas')}<br/>
                To protect test integrity, re-running is disabled in the UI; browse Sections 2–7 in the left rail.
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_section_2(report: dict) -> None:
    st.subheader("2 · Held-Out Classification Metrics & Confusion Matrices")
    st.markdown(
        """
        <div class="partg-callout">
            <strong>Part G Rubric Scope Note:</strong> <code>default_next_month</code> (0/1) is a binary supervised
            classification task. All 6 required Classification metrics (<strong>Accuracy, Precision, Recall, F1-Score,
            ROC-AUC, Confusion Matrix</strong>) plus supporting PR-AUC, MCC, Balanced Accuracy, and Brier Loss are
            reported identically at the frozen default 0.50 decision threshold. <em>Regression (MAE/RMSE/MSE/R²) and
            Clustering (Silhouette/Davies-Bouldin/Calinski-Harabasz) metrics are N/A for this supervised task.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )

    models = report["models"]
    dummies = report["dummies"]

    best_f1 = max(fe.MODEL_ORDER, key=lambda k: models[k]["test_metrics"]["f1"])
    best_roc = max(fe.MODEL_ORDER, key=lambda k: models[k]["test_metrics"]["roc_auc"])
    best_rec = max(fe.MODEL_ORDER, key=lambda k: models[k]["test_metrics"]["recall"])
    best_brier = min(fe.MODEL_ORDER, key=lambda k: models[k]["test_metrics"]["brier"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top Held-Out F1", fe.MODEL_SPECS[best_f1]["short"], f"{models[best_f1]['test_metrics']['f1']:.4f}")
    c2.metric("Top ROC-AUC", fe.MODEL_SPECS[best_roc]["short"], f"{models[best_roc]['test_metrics']['roc_auc']:.4f}")
    c3.metric("Highest Recall", fe.MODEL_SPECS[best_rec]["short"], f"{models[best_rec]['test_metrics']['recall']:.4f}")
    c4.metric("Lowest Brier Loss", fe.MODEL_SPECS[best_brier]["short"], f"{models[best_brier]['test_metrics']['brier']:.4f}")

    table_rows = []
    for dkey in ("dummy_prior", "dummy_all_positive"):
        dm = dummies[dkey]
        tm = dm["test_metrics"]
        table_rows.append(
            {
                "Model": dm["label"],
                "Role": "Baseline Floor",
                "Accuracy": round(tm["accuracy"], 4),
                "Precision": round(tm["precision"], 4),
                "Recall": round(tm["recall"], 4),
                "F1-Score": round(tm["f1"], 4),
                "ROC-AUC": round(tm["roc_auc"], 4),
                "PR-AUC": round(tm["average_precision"], 4),
                "MCC": round(tm["mcc"], 4),
                "Balanced Acc": round(tm["balanced_accuracy"], 4),
                "Brier Loss": round(tm["brier"], 4),
            }
        )
    for key in fe.MODEL_ORDER:
        tm = models[key]["test_metrics"]
        spec = fe.MODEL_SPECS[key]
        table_rows.append(
            {
                "Model": spec["label"],
                "Role": f"{spec['role']} ({spec['owner']})",
                "Accuracy": round(tm["accuracy"], 4),
                "Precision": round(tm["precision"], 4),
                "Recall": round(tm["recall"], 4),
                "F1-Score": round(tm["f1"], 4),
                "ROC-AUC": round(tm["roc_auc"], 4),
                "PR-AUC": round(tm["average_precision"], 4),
                "MCC": round(tm["mcc"], 4),
                "Balanced Acc": round(tm["balanced_accuracy"], 4),
                "Brier Loss": round(tm["brier"], 4),
            }
        )
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")

    core_metrics = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-Score"),
        ("roc_auc", "ROC-AUC"),
    ]
    bar_fig = go.Figure()
    for key in fe.MODEL_ORDER:
        spec = fe.MODEL_SPECS[key]
        tm = models[key]["test_metrics"]
        bar_fig.add_trace(
            go.Bar(
                name=spec["short"],
                x=[label for _, label in core_metrics],
                y=[tm[mkey] for mkey, _ in core_metrics],
                marker_color=spec["color"],
                text=[f"{tm[mkey]:.3f}" for mkey, _ in core_metrics],
                textposition="outside",
            )
        )
    floor_f1 = report["dataset"]["dummy_all_positive_f1_floor_test"]
    bar_fig.add_hline(
        y=floor_f1,
        line_dash="dot",
        line_color="#dc2626",
        annotation_text=f"All-Positive F1 Floor ({floor_f1:.3f})",
        annotation_position="top right",
    )
    bar_fig.update_layout(
        title="Held-Out Core Classification Metrics Across All 4 Models",
        barmode="group",
        yaxis=dict(range=[0.0, 1.02], title="Score"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=430,
        margin=dict(t=80, b=40, l=50, r=20),
    )
    st.plotly_chart(bar_fig, width="stretch", key="partg_core_bar")

    st.markdown("#### 2×2 Confusion Matrices (6,000 Held-Out Test Clients)")
    cm_fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[fe.MODEL_SPECS[k]["label"] for k in fe.MODEL_ORDER],
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )
    fp_fn_rows = []
    for idx, key in enumerate(fe.MODEL_ORDER):
        r = idx // 2 + 1
        c = idx % 2 + 1
        cm = np.asarray(models[key]["test_metrics"]["confusion_matrix"], dtype=int)
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
        pct = cm / row_sums * 100.0
        # Display row 1=Actual 0 at top and row 2=Actual 1 at bottom cleanly via reversed y order
        z_vals = np.array([[tn, fp], [fn, tp]], dtype=float)
        text_vals = np.array(
            [
                [f"TN: {tn:,}<br>({pct[0, 0]:.1f}%)", f"FP: {fp:,}<br>({pct[0, 1]:.1f}%)"],
                [f"FN: {fn:,}<br>({pct[1, 0]:.1f}%)", f"TP: {tp:,}<br>({pct[1, 1]:.1f}%)"],
            ]
        )
        cm_fig.add_trace(
            go.Heatmap(
                z=z_vals,
                x=["Pred: No Default (0)", "Pred: Default (1)"],
                y=["Actual: No Default (0)", "Actual: Default (1)"],
                text=text_vals,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            ),
            row=r,
            col=c,
        )
        cm_fig.update_yaxes(autorange="reversed", row=r, col=c)
        fp_fn_rows.append(
            {
                "Model": fe.MODEL_SPECS[key]["short"],
                "True Negatives (TN)": tn,
                "False Positives (FP · False Alarm)": fp,
                "False Negatives (FN · Missed Default)": fn,
                "True Positives (TP · Caught Default)": tp,
                "Total Predicted Defaults": fp + tp,
                "Default Capture Rate (Recall)": round(tp / max(1, tp + fn), 4),
            }
        )
    cm_fig.update_layout(height=620, margin=dict(t=55, b=35, l=65, r=25))
    st.plotly_chart(cm_fig, width="stretch", key="partg_cm_grid")
    st.dataframe(pd.DataFrame(fp_fn_rows), hide_index=True, width="stretch")


def _render_section_3(report: dict) -> None:
    st.subheader("3 · ROC, Precision–Recall, Calibration & Prediction Agreement")
    models = report["models"]
    pos_rate = float(report["dataset"]["test_positive_rate"])

    left, right = st.columns(2)
    roc_fig = go.Figure()
    for key in fe.MODEL_ORDER:
        spec = fe.MODEL_SPECS[key]
        roc_df = pd.DataFrame(models[key]["curves"]["roc"])
        auc_val = models[key]["test_metrics"]["roc_auc"]
        roc_fig.add_trace(
            go.Scatter(
                x=roc_df["fpr"],
                y=roc_df["tpr"],
                mode="lines",
                name=f"{spec['short']} (AUC={auc_val:.4f})",
                line=dict(color=spec["color"], width=2.5),
            )
        )
    roc_fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Chance (AUC=0.50)",
            line=dict(color="#94a3b8", dash="dash"),
        )
    )
    roc_fig.update_layout(
        title="Held-Out ROC Curves (Threshold-Free Discrimination)",
        xaxis_title="False Positive Rate (1 − Specificity)",
        yaxis_title="True Positive Rate (Recall)",
        legend=dict(y=0.05, x=0.45, bgcolor="rgba(255,255,255,0.85)"),
        height=430,
    )
    left.plotly_chart(roc_fig, width="stretch", key="partg_roc_overlay")

    pr_fig = go.Figure()
    for key in fe.MODEL_ORDER:
        spec = fe.MODEL_SPECS[key]
        pr_df = pd.DataFrame(models[key]["curves"]["pr"])
        ap_val = models[key]["test_metrics"]["average_precision"]
        pr_fig.add_trace(
            go.Scatter(
                x=pr_df["recall"],
                y=pr_df["precision"],
                mode="lines",
                name=f"{spec['short']} (PR-AUC={ap_val:.4f})",
                line=dict(color=spec["color"], width=2.5),
            )
        )
    pr_fig.add_hline(
        y=pos_rate,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text=f"Chance Floor (p={pos_rate:.4f})",
        annotation_position="top right",
    )
    pr_fig.update_layout(
        title="Held-Out Precision–Recall Curves",
        xaxis_title="Recall (Sensitivity)",
        yaxis_title="Precision (PPV)",
        legend=dict(y=0.05, x=0.05, bgcolor="rgba(255,255,255,0.85)"),
        height=430,
    )
    right.plotly_chart(pr_fig, width="stretch", key="partg_pr_overlay")

    c_left, c_right = st.columns(2)
    cal_fig = go.Figure()
    for key in fe.MODEL_ORDER:
        spec = fe.MODEL_SPECS[key]
        cal_df = pd.DataFrame(models[key]["curves"]["calibration"])
        brier_val = models[key]["test_metrics"]["brier"]
        cal_fig.add_trace(
            go.Scatter(
                x=cal_df["mean_predicted"],
                y=cal_df["fraction_positive"],
                mode="lines+markers",
                name=f"{spec['short']} (Brier={brier_val:.4f})",
                line=dict(color=spec["color"], width=2.2),
            )
        )
    cal_fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color="#334155", dash="dash"),
        )
    )
    cal_fig.update_layout(
        title="Probability Calibration (10 Quantile Bins)",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Observed Default Fraction",
        height=420,
    )
    c_left.plotly_chart(cal_fig, width="stretch", key="partg_cal_overlay")

    pa = report["pairwise_agreement"]
    labels = pa["labels"]
    agree_mat = np.asarray(pa["agreement_rate"], dtype=float) * 100.0
    kappa_mat = np.asarray(pa["cohen_kappa"], dtype=float)
    agree_fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Prediction Agreement (%)", "Cohen's κ"],
        horizontal_spacing=0.14,
    )
    agree_fig.add_trace(
        go.Heatmap(
            z=agree_mat,
            x=labels,
            y=labels,
            text=[[f"{v:.1f}%" for v in row] for row in agree_mat],
            texttemplate="%{text}",
            colorscale="Teal",
            showscale=False,
        ),
        row=1,
        col=1,
    )
    agree_fig.add_trace(
        go.Heatmap(
            z=kappa_mat,
            x=labels,
            y=labels,
            text=[[f"{v:.3f}" for v in row] for row in kappa_mat],
            texttemplate="%{text}",
            colorscale="Purples",
            showscale=False,
        ),
        row=1,
        col=2,
    )
    agree_fig.update_yaxes(autorange="reversed", row=1, col=1)
    agree_fig.update_yaxes(autorange="reversed", row=1, col=2)
    agree_fig.update_layout(title="4×4 Inter-Model Prediction Agreement & Cohen's κ", height=420)
    c_right.plotly_chart(agree_fig, width="stretch", key="partg_agreement_heatmap")


def _render_section_4(report: dict) -> None:
    st.subheader("4 · Overfitting & Underfitting Analysis (Train vs. CV vs. Held-Out Test)")
    models = report["models"]

    chart_cols = st.columns(2)
    for col, (metric_key, metric_label) in zip(chart_cols, (("f1", "F1-Score"), ("roc_auc", "ROC-AUC"))):
        cv_key = f"cv_{metric_key}_mean"
        x_labels = [fe.MODEL_SPECS[k]["short"] for k in fe.MODEL_ORDER]
        train_vals = [models[k]["train_metrics"][metric_key] for k in fe.MODEL_ORDER]
        cv_vals = [models[k]["generalization"].get(cv_key) for k in fe.MODEL_ORDER]
        test_vals = [models[k]["test_metrics"][metric_key] for k in fe.MODEL_ORDER]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Full Train (In-Sample)",
                x=x_labels,
                y=train_vals,
                marker_color="#94a3b8",
                text=[f"{v:.3f}" for v in train_vals],
                textposition="outside",
            )
        )
        if any(v is not None for v in cv_vals):
            fig.add_trace(
                go.Bar(
                    name="3-Fold Training CV Mean",
                    x=x_labels,
                    y=[0.0 if v is None else v for v in cv_vals],
                    marker_color="#0ea5e9",
                    text=["N/A" if v is None else f"{v:.3f}" for v in cv_vals],
                    textposition="outside",
                )
            )
        fig.add_trace(
            go.Bar(
                name="Held-Out Test",
                x=x_labels,
                y=test_vals,
                marker_color="#4f46e5",
                text=[f"{v:.3f}" for v in test_vals],
                textposition="outside",
            )
        )
        fig.update_layout(
            title=f"{metric_label}: Train vs. 3-Fold CV vs. Held-Out Test",
            barmode="group",
            yaxis=dict(range=[0.0, 1.02], title=metric_label),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=420,
            margin=dict(t=80, b=40),
        )
        col.plotly_chart(fig, width="stretch", key=f"partg_gap_{metric_key}")

    gap_rows = []
    for key in fe.MODEL_ORDER:
        m = models[key]
        gen = m["generalization"]
        gap_rows.append(
            {
                "Model": fe.MODEL_SPECS[key]["label"],
                "Train F1": round(m["train_metrics"]["f1"], 4),
                "3-Fold CV F1": _fmt_num(gen.get("cv_f1_mean")),
                "Held-Out Test F1": round(m["test_metrics"]["f1"], 4),
                "Train − Test Gap": f"{gen['f1_train_test_gap']:+.4f}",
                "CV − Test Shift": _fmt_signed(gen.get("f1_cv_test_gap")),
                "Train − Test AUC Gap": f"{gen['roc_auc_train_test_gap']:+.4f}",
                "Bias / Variance Diagnosis": gen["regime"],
            }
        )
    st.dataframe(pd.DataFrame(gap_rows), hide_index=True, width="stretch")

    st.markdown(
        f"""
        <div class="partg-callout">
            <strong>Critical Overfitting & Underfitting Assessment:</strong><br/>
            {report["critical_discussion"]["overfitting_underfitting"]}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_section_5(report: dict) -> None:
    st.subheader("5 · Computational Efficiency (Fit Time, Latency, Throughput & Size)")
    models = report["models"]

    fastest_fit = min(fe.MODEL_ORDER, key=lambda k: models[k]["efficiency"]["fit_seconds"])
    fastest_pred = min(fe.MODEL_ORDER, key=lambda k: models[k]["efficiency"]["predict_ms_total"])
    smallest_kb = min(fe.MODEL_ORDER, key=lambda k: models[k]["efficiency"]["artifact_size_kb"])
    e2_fit = models["ensemble_e2"]["efficiency"]["fit_seconds"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Fastest Fit Time", fe.MODEL_SPECS[fastest_fit]["short"], f"{models[fastest_fit]['efficiency']['fit_seconds']:.2f}s")
    k2.metric("Fastest 6k Inference", fe.MODEL_SPECS[fastest_pred]["short"], f"{models[fastest_pred]['efficiency']['predict_ms_total']:.1f} ms")
    k3.metric("Smallest Model File", fe.MODEL_SPECS[smallest_kb]["short"], f"{models[smallest_kb]['efficiency']['artifact_size_kb']:.1f} KB")
    k4.metric("E2 Ensemble Total Fit", f"{e2_fit:.2f}s", "n_jobs=1 hardware parity")

    names = [fe.MODEL_SPECS[k]["short"] for k in fe.MODEL_ORDER]
    colors = [fe.MODEL_SPECS[k]["color"] for k in fe.MODEL_ORDER]
    effs = [models[k]["efficiency"] for k in fe.MODEL_ORDER]

    sub = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Full Training Fit Time (seconds, lower better)",
            "Held-Out Scoring Latency (ms for test split, lower better)",
            "Inference Throughput (rows/sec, higher better)",
            "Compressed .joblib Footprint (KB, lower better)",
        ],
        vertical_spacing=0.22,
        horizontal_spacing=0.12,
    )
    specs_panels = [
        (1, 1, [e["fit_seconds"] for e in effs], lambda v: f"{v:.2f}s"),
        (1, 2, [e["predict_ms_total"] for e in effs], lambda v: f"{v:.1f}ms"),
        (2, 1, [e["throughput_rows_per_sec"] for e in effs], lambda v: f"{v:,.0f}/s"),
        (2, 2, [e["artifact_size_kb"] for e in effs], lambda v: f"{v:,.1f} KB"),
    ]
    for r, c, vals, fmt in specs_panels:
        sub.add_trace(
            go.Bar(
                x=names,
                y=vals,
                marker_color=colors,
                text=[fmt(v) for v in vals],
                textposition="outside",
                showlegend=False,
            ),
            row=r,
            col=c,
        )
    sub.update_layout(height=590, margin=dict(t=55, b=40))
    st.plotly_chart(sub, width="stretch", key="partg_eff_subplots")

    scatter_fig = go.Figure()
    for key in fe.MODEL_ORDER:
        eff = models[key]["efficiency"]
        tm = models[key]["test_metrics"]
        spec = fe.MODEL_SPECS[key]
        scatter_fig.add_trace(
            go.Scatter(
                x=[eff["fit_seconds"]],
                y=[tm["f1"]],
                mode="markers+text",
                name=spec["short"],
                text=[spec["short"]],
                textposition="top center",
                marker=dict(
                    size=max(14, min(56, np.sqrt(eff["artifact_size_kb"]) * 1.1)),
                    color=spec["color"],
                    line=dict(width=1.5, color="#0f172a"),
                ),
            )
        )
    scatter_fig.update_layout(
        title="Accuracy–Compute Trade-Off: Held-Out F1 vs. Fit Time (Bubble Area = Model Size KB)",
        xaxis_title="Training Fit Time (seconds, n_jobs=1)",
        yaxis_title="Held-Out F1-Score",
        height=390,
        showlegend=True,
    )
    st.plotly_chart(scatter_fig, width="stretch", key="partg_tradeoff_scatter")

    eff_rows = []
    for key in fe.MODEL_ORDER:
        eff = models[key]["efficiency"]
        eff_rows.append(
            {
                "Model": fe.MODEL_SPECS[key]["label"],
                "Fit Time (s)": round(eff["fit_seconds"], 3),
                "Predict Time (ms)": round(eff["predict_ms_total"], 2),
                "ms / 1k Rows": round(eff["predict_ms_per_1k_rows"], 2),
                "Throughput (rows/s)": round(eff["throughput_rows_per_sec"], 1),
                "Compressed Size (KB)": round(eff["artifact_size_kb"], 1),
            }
        )
    st.dataframe(pd.DataFrame(eff_rows), hide_index=True, width="stretch")
    st.markdown(
        f"<div class='partg-callout'><strong>Efficiency Summary:</strong> {report['critical_discussion']['computational_efficiency']}</div>",
        unsafe_allow_html=True,
    )


def _render_section_6(report: dict) -> None:
    st.subheader("6 · Model Interpretability & Out-of-Sample Feature Attribution")
    models = report["models"]

    spec_rows = []
    for key in fe.MODEL_ORDER:
        meta = fe.MODEL_SPECS[key]
        p_top = models[key]["permutation_importance"][0] if models[key]["permutation_importance"] else {"feature": "N/A", "f1_mean": 0.0}
        spec_rows.append(
            {
                "Model": meta["label"],
                "Interpretability Tier": meta["interpretability"],
                "Top Raw Feature (F1 Perm Drop)": p_top["feature"],
                "Top F1 Drop": round(p_top["f1_mean"], 4),
            }
        )
    st.dataframe(pd.DataFrame(spec_rows), hide_index=True, width="stretch")

    criterion = st.radio(
        "Permutation Importance Criterion (Held-Out Test Split)",
        ["F1 Drop", "ROC-AUC Drop"],
        horizontal=True,
        key="interp_metric",
    )
    m_field = "f1_mean" if criterion == "F1 Drop" else "roc_auc_mean"
    s_field = "f1_std" if criterion == "F1 Drop" else "roc_auc_std"

    e2_top10 = [row["feature"] for row in models["ensemble_e2"]["permutation_importance"][:10]]
    ordered_top = list(reversed(e2_top10))

    perm_fig = go.Figure()
    for key in fe.MODEL_ORDER:
        lookup = {row["feature"]: row for row in models[key]["permutation_importance"]}
        vals = [lookup.get(feat, {}).get(m_field, 0.0) for feat in ordered_top]
        stds = [lookup.get(feat, {}).get(s_field, 0.0) for feat in ordered_top]
        perm_fig.add_trace(
            go.Bar(
                name=fe.MODEL_SPECS[key]["short"],
                y=ordered_top,
                x=vals,
                error_x=dict(type="data", array=stds, visible=True),
                orientation="h",
                marker_color=fe.MODEL_SPECS[key]["color"],
            )
        )
    perm_fig.update_layout(
        title=f"Top 10 Raw Features by Out-of-Sample Permutation Importance ({criterion})",
        barmode="group",
        xaxis_title=f"Mean Score Drop When Feature Shuffled ({criterion})",
        height=520,
        margin=dict(l=130, t=60, b=40),
    )
    st.plotly_chart(perm_fig, width="stretch", key="partg_perm_compare")

    grp_fig = go.Figure()
    e2_groups = [row["group"] for row in models["ensemble_e2"]["grouped_permutation_importance"]]
    ordered_groups = list(reversed(e2_groups))
    for key in fe.MODEL_ORDER:
        glookup = {row["group"]: row for row in models[key]["grouped_permutation_importance"]}
        vals = [glookup.get(g, {}).get(m_field, 0.0) for g in ordered_groups]
        grp_fig.add_trace(
            go.Bar(
                name=fe.MODEL_SPECS[key]["short"],
                y=ordered_groups,
                x=vals,
                orientation="h",
                marker_color=fe.MODEL_SPECS[key]["color"],
            )
        )
    grp_fig.update_layout(
        title=f"Domain-Grouped Block Permutation Importance ({criterion}, mitigates collinearity)",
        barmode="group",
        xaxis_title=f"Mean Score Drop When Entire Domain Block Shuffled ({criterion})",
        height=390,
        margin=dict(l=210, t=60, b=40),
    )
    st.plotly_chart(grp_fig, width="stretch", key="partg_grouped_perm")

    with st.expander("White-Box Inspection: Logistic Regression Top 15 Standardized Log-Odds Coefficients"):
        lr_int = models["logistic_regression"]["internals"]
        top_coefs = lr_int.get("top_coefficients", [])
        if top_coefs:
            cdf = pd.DataFrame(top_coefs).iloc[::-1]
            cfig = go.Figure(
                go.Bar(
                    x=cdf["coef"],
                    y=cdf["raw_name"],
                    orientation="h",
                    marker_color=["#dc2626" if v > 0 else "#0284c7" for v in cdf["coef"]],
                )
            )
            cfig.update_layout(
                title=f"Logistic Regression Standardized Log-Odds (Intercept = {lr_int['intercept']:+.4f})",
                xaxis_title="Coefficient (positive = higher default risk)",
                height=440,
            )
            st.plotly_chart(cfig, width="stretch", key="partg_lr_coefs")
            st.dataframe(pd.DataFrame(top_coefs)[["raw_name", "coef", "odds_ratio"]], hide_index=True, width="stretch")

    with st.expander("Tree & Ensemble Internals Audit (RF, GB & E2 Voting Structure)"):
        rf_i = models["random_forest"]["internals"]
        gb_i = models["gb_balanced"]["internals"]
        e2_i = models["ensemble_e2"]["internals"]
        audit_rows = [
            {
                "Model": "RF (Tuned)",
                "Units": f"{rf_i['n_trees']} trees",
                "Mean Depth": round(rf_i["mean_depth"], 2),
                "Max Depth": rf_i["max_depth"],
                "Mean Leaves/Tree": round(rf_i["mean_leaves"], 1),
                "Total Nodes": f"{rf_i['total_nodes']:,}",
                "Key Note": f"Singleton leaf fraction = {rf_i['singleton_leaf_fraction']:.2%}",
            },
            {
                "Model": "GB (Balanced)",
                "Units": f"{gb_i['n_stages']} stages",
                "Mean Depth": round(gb_i["mean_depth"], 2),
                "Max Depth": gb_i["max_depth"],
                "Mean Leaves/Tree": round(gb_i["mean_leaves"], 1),
                "Total Nodes": f"{gb_i['total_nodes']:,}",
                "Key Note": f"lr={gb_i['learning_rate']}, subsample={gb_i['subsample']}",
            },
            {
                "Model": "E2 Ensemble (2:1)",
                "Units": "RF + GB (soft)",
                "Mean Depth": "Blended",
                "Max Depth": "Blended",
                "Mean Leaves/Tree": "Blended",
                "Total Nodes": f"{rf_i['total_nodes'] + gb_i['total_nodes']:,}",
                "Key Note": e2_i["probability_formula"],
            },
        ]
        st.dataframe(pd.DataFrame(audit_rows), hide_index=True, width="stretch")

    st.markdown(
        f"<div class='partg-callout'><strong>Interpretability Assessment:</strong> {report['critical_discussion']['model_interpretability']}</div>",
        unsafe_allow_html=True,
    )


def _render_section_7(
    report: dict,
    eval_json_path: Path,
    comp_csv_path: Path,
    pred_csv_path: Path,
    joblib_path: Path,
) -> None:
    st.subheader("7 · Statistical Significance Tests, Critical Verdict & M4 Exports")
    sig = report["statistical_tests"]
    disc = report["critical_discussion"]

    st.info(f"**Ensemble Significance Verdict:** {disc['ensemble_significance_verdict']}")

    st.markdown("#### Pairwise Held-Out McNemar's Test (6,000 Predictions)")
    mc_rows = []
    for pair in sig["pairwise_held_out"].values():
        mc = pair["mcnemar"]
        mc_rows.append(
            {
                "Comparison (A vs B)": f"{pair['label_a']} vs {pair['label_b']}",
                "Primary E2 Check": "Yes" if pair["is_primary_ensemble_comparison"] else "Base vs Base",
                "n10 (A Right, B Wrong)": mc["n10_a_right_b_wrong"],
                "n01 (A Wrong, B Right)": mc["n01_a_wrong_b_right"],
                "Discordant": mc["discordant"],
                "Net Adv (A)": f"{mc['net_advantage_a']:+d}",
                "Method": mc["method"],
                "Statistic": round(mc["statistic"], 4),
                "p-value": round(mc["p_value"], 4),
                "Sig (α=0.05)": "Yes" if mc["significant_0_05"] else "No",
            }
        )
    st.dataframe(pd.DataFrame(mc_rows), hide_index=True, width="stretch")

    n_boot = sig.get("bootstrap_resamples", 2000)
    st.markdown(f"#### Paired Stratified Bootstrap 95% Confidence Intervals ({n_boot:,} Resamples)")
    primary_pairs = [p for p in sig["pairwise_held_out"].values() if p["is_primary_ensemble_comparison"]]
    forest_fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["ΔF1 (E2 − Base, 95% CI)", "ΔROC-AUC (E2 − Base, 95% CI)"],
        horizontal_spacing=0.14,
    )
    for col_idx, metric_key in ((1, "f1"), (2, "roc_auc")):
        y_names = [f"E2 − {p['label_b']}" for p in primary_pairs][::-1]
        diffs = [p["bootstrap"][metric_key]["observed_diff"] for p in primary_pairs][::-1]
        err_hi = [p["bootstrap"][metric_key]["ci_high"] - p["bootstrap"][metric_key]["observed_diff"] for p in primary_pairs][::-1]
        err_lo = [p["bootstrap"][metric_key]["observed_diff"] - p["bootstrap"][metric_key]["ci_low"] for p in primary_pairs][::-1]
        sigs = [p["bootstrap"][metric_key]["significant_0_05"] for p in primary_pairs][::-1]

        forest_fig.add_trace(
            go.Scatter(
                x=diffs,
                y=y_names,
                mode="markers",
                marker=dict(
                    size=12,
                    color=["#059669" if s else "#64748b" for s in sigs],
                ),
                error_x=dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo, thickness=2),
                showlegend=False,
            ),
            row=1,
            col=col_idx,
        )
        forest_fig.add_vline(x=0.0, line_dash="dash", line_color="#dc2626", row=1, col=col_idx)
    forest_fig.update_layout(height=320, margin=dict(t=50, b=35, l=135))
    st.plotly_chart(forest_fig, width="stretch", key="partg_boot_forest")

    boot_rows = []
    for pair in sig["pairwise_held_out"].values():
        b_f1 = pair["bootstrap"]["f1"]
        b_roc = pair["bootstrap"]["roc_auc"]
        b_acc = pair["bootstrap"]["accuracy"]
        b_ap = pair["bootstrap"]["average_precision"]
        boot_rows.append(
            {
                "Comparison (A − B)": f"{pair['label_a']} − {pair['label_b']}",
                "ΔF1 [95% CI]": f"{b_f1['observed_diff']:+.4f} [{b_f1['ci_low']:+.4f}, {b_f1['ci_high']:+.4f}]",
                "F1 Sig": "Yes" if b_f1["significant_0_05"] else "No",
                "ΔROC-AUC [95% CI]": f"{b_roc['observed_diff']:+.4f} [{b_roc['ci_low']:+.4f}, {b_roc['ci_high']:+.4f}]",
                "AUC Sig": "Yes" if b_roc["significant_0_05"] else "No",
                "ΔAccuracy": f"{b_acc['observed_diff']:+.4f}",
                "ΔPR-AUC": f"{b_ap['observed_diff']:+.4f}",
            }
        )
    st.dataframe(pd.DataFrame(boot_rows), hide_index=True, width="stretch")

    nadeau = sig.get("nadeau_bengio_repeated_cv", {})
    if nadeau:
        st.markdown("#### Nadeau & Bengio Variance-Corrected Repeated CV t-Test (M1 3×2 CV)")
        nb_rows = []
        for comp_name, block in nadeau.items():
            for mk, ml in (("f1", "F1-Score"), ("roc_auc", "ROC-AUC")):
                r = block[mk]
                nb_rows.append(
                    {
                        "Comparison": comp_name,
                        "Metric": ml,
                        "Mean Diff": f"{r['mean_diff']:+.4f}",
                        "Corrected t": f"{r['t_stat']:+.3f}",
                        "p-value": round(r["p_value"], 4),
                        "Sig (α=0.05)": "Yes" if r["significant_0_05"] else "No",
                        "Folds A Better": f"{r['folds_a_better']}/{r['k']}",
                    }
                )
        st.dataframe(pd.DataFrame(nb_rows), hide_index=True, width="stretch")

    st.markdown("#### Part G 5-Point Critical Discussion Synthesis")
    for title, key in (
        ("1. Performance Differences", "performance_differences"),
        ("2. Overfitting & Underfitting", "overfitting_underfitting"),
        ("3. Computational Efficiency", "computational_efficiency"),
        ("4. Model Interpretability", "model_interpretability"),
        ("5. Ensemble Significance Verdict", "ensemble_significance_verdict"),
    ):
        st.markdown(f"**{title}:** {disc[key]}")

    st.markdown("#### Report & M4 Explainability Handoff Exports")
    d1, d2, d3, d4 = st.columns(4)
    if comp_csv_path.is_file():
        d1.download_button(
            "Download Comparison CSV",
            comp_csv_path.read_bytes(),
            file_name="final_model_comparison.csv",
            mime="text/csv",
            key="dl_final_comp_csv",
        )
    if eval_json_path.is_file():
        d2.download_button(
            "Download Evaluation JSON",
            eval_json_path.read_bytes(),
            file_name="final_evaluation.json",
            mime="application/json",
            key="dl_final_eval_json",
        )
    if pred_csv_path.is_file():
        d3.download_button(
            "Download Test Predictions CSV",
            pred_csv_path.read_bytes(),
            file_name="final_test_predictions.csv",
            mime="text/csv",
            key="dl_final_pred_csv",
        )
    if joblib_path.is_file():
        kb = joblib_path.stat().st_size / 1024.0
        d4.success(f"Frozen E2 Saved\n`{joblib_path.name}` ({kb:,.1f} KB)")
    else:
        d4.warning(f"Missing `{joblib_path.name}`")


def _fmt_num(val) -> str:
    return "N/A" if val is None else f"{float(val):.4f}"


def _fmt_signed(val) -> str:
    return "N/A" if val is None else f"{float(val):+.4f}"


def _fmt_mean_std(mean_val, std_val) -> str:
    if mean_val is None:
        return "N/A"
    if std_val is None:
        return f"{float(mean_val):.4f}"
    return f"{float(mean_val):.4f} ± {float(std_val):.4f}"


def render() -> None:
    st.set_page_config(
        page_title="Part G · Final 4-Model Comparative Evaluation",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    eval_json_path, comp_csv_path, pred_csv_path, joblib_path = _eval_paths()
    is_complete = eval_json_path.is_file()

    _apply_theme_and_hero(is_complete)
    section = _render_sidebar(is_complete)

    if section == SECTIONS[0]:
        _render_section_1(eval_json_path, joblib_path)
        return

    report = _require_report(eval_json_path)
    if report is None:
        return

    if section == SECTIONS[1]:
        _render_section_2(report)
    elif section == SECTIONS[2]:
        _render_section_3(report)
    elif section == SECTIONS[3]:
        _render_section_4(report)
    elif section == SECTIONS[4]:
        _render_section_5(report)
    elif section == SECTIONS[5]:
        _render_section_6(report)
    elif section == SECTIONS[6]:
        _render_section_7(report, eval_json_path, comp_csv_path, pred_csv_path, joblib_path)


if __name__ == "__main__":
    render()
