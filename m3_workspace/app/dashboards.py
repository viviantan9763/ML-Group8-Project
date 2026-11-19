"""Evidence, run comparisons, and local artifact downloads."""

import io
import json
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from matplotlib.figure import Figure

from m3_workspace.engine.artifact_manager import HISTORY, ROOT, history, read_json, run_folder
from m3_workspace.engine.experiment_manager import FREEZE, compatible, final_evaluate


def run_label(record):
    return f"{record['type'].capitalize()} · {record['id'][:8]} · {record['started_at'][:16]} UTC"


def choose(records, label, key):
    if not records:
        st.info("Complete an experiment with the active dataset to view its results.")
        return None
    by_id = {row["id"]: row for row in records}
    selected = st.selectbox(label, list(by_id), key=key, format_func=lambda key: run_label(by_id[key]))
    return by_id[selected]


def charts(metrics, pr, label, key):
    a, b = st.columns(2)
    matrix = pd.DataFrame(metrics["confusion_matrix"], index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
    heatmap = px.imshow(matrix, text_auto=True, color_continuous_scale="Teal", title=f"{label} confusion matrix")
    line = px.line(pr, x="recall", y="precision", range_x=[0, 1], range_y=[0, 1.02], title=f"{label} precision–recall")
    a.plotly_chart(heatmap, key=f"{key}_cm")
    b.plotly_chart(line, key=f"{key}_pr")


def evidence(records):
    selected = choose(records, "Experiment to inspect", "evidence_run")
    if selected is None:
        return
    folder = run_folder(selected)
    summary = selected["summary"]
    st.subheader("Out-of-fold evaluation")
    st.caption("Predictions come from validation folds. Tuned configuration scores reuse the folds that selected it and can be optimistic.")
    metrics = summary["metrics"]
    for column, key in zip(st.columns(5), ("accuracy", "precision", "recall", "f1", "roc_auc")):
        column.metric(key.replace("_", " ").upper() if key in ("f1", "roc_auc") else key.title(), f"{metrics[key]:.4f}")
    charts(metrics, pd.read_csv(folder / "precision_recall.csv"), "Out-of-fold", "oof")
    a, b = st.columns(2)
    a.metric("Total experiment runtime", f"{summary['runtime_seconds']:.2f} s")
    b.metric("Fold training time (sum)", f"{summary['fold_fit_seconds']:.2f} s")
    if summary.get("training_oob_accuracy") is not None:
        st.caption(f"Refitted forest training OOB accuracy: {summary['training_oob_accuracy']:.4f}")
    st.dataframe(pd.read_csv(folder / "folds.csv"), hide_index=True, width="stretch")
    st.dataframe(pd.DataFrame(summary["cv"]).T.rename_axis("CV metric"), width="stretch")
    with st.expander("Selected configuration"):
        st.json(summary["best_params"])
    if selected["type"] == "tuning":
        primary = selected["config"]["cv"]["primary"]
        column = f"mean_test_{primary}"
        candidates = pd.read_csv(folder / "cv_results.csv")
        ranked = candidates.sort_values(column, ascending=False, kind="stable")
        st.subheader("Candidate rankings")
        by_draw = candidates.sort_values("draw")
        fig = go.Figure()
        fig.add_scatter(x=by_draw["draw"], y=by_draw[column], mode="markers", name="Candidate")
        fig.add_scatter(x=by_draw["draw"], y=by_draw[column].cummax(), mode="lines",
                        line_shape="hv", name="Best so far")
        fig.update_layout(xaxis_title="Search draw", yaxis_title=f"Mean CV {primary}",
                          height=320, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, key="score_history")
        st.write(f"Winning draw **{summary['winning_draw']}**")
        display = ["draw", f"rank_test_{primary}", column, f"std_test_{primary}", "params"]
        st.dataframe(ranked[display], hide_index=True, width="stretch")
        draw = st.selectbox("Inspect a candidate draw", ranked["draw"].tolist(), key=f"draw_{selected['id']}")
        row = ranked.loc[ranked["draw"].eq(draw)].iloc[0]
        st.code(row["params"], language="python")
        fold_columns = [key for key in row.index if key.startswith("split") and key.endswith(f"_test_{primary}")]
        st.dataframe(row[fold_columns].rename("Validation score").to_frame(), width="stretch")
    with st.expander("Final held-out evaluation"):
        show_final(selected)


def legacy_evidence():
    folder = ROOT / "results"
    summary_path = folder / "random_forest_cv.json"
    scores_path = folder / "random_forest_search.csv"
    if not (summary_path.is_file() and scores_path.is_file()):
        return
    summary = read_json(summary_path)
    with st.expander("Original 48-candidate report"):
        st.caption("Archived training CV experiment. Its records are separate from Studio runs.")
        a, b = st.columns(2)
        a.metric("Baseline mean F1", f"{summary['baseline']['f1']['mean']:.4f}")
        b.metric("Tuned mean F1", f"{summary['tuned']['f1']['mean']:.4f}",
                 f"{summary['tuned_f1_gain'] * 100:+.2f} pp")
        st.json(summary["tuned_params"])
        st.download_button("Export archived candidate CSV", scores_path.read_bytes(),
                           "random_forest_search.csv", "text/csv")


def show_final(record):
    meta = record["summary"]["dataset"]
    output = HISTORY / "final" / meta["fingerprint"]
    path = output / "evaluation.json"
    if path.exists():
        result = read_json(path)
        if result["run_id"] == record["id"]:
            st.json(result["metrics"])
            charts(result["metrics"], pd.read_csv(output / "precision_recall.csv"), "Final test", "final")
        else:
            st.info("The dataset’s one-time final evaluation has already been used by the frozen run.")
        return
    if (output / "attempt.lock").exists():
        st.warning("A final evaluation was already attempted. Review its status with the group before taking any further action.")
        return
    if record["config"]["dataset"] != "official" or not FREEZE.exists():
        st.info("Locked until the group freezes an exact official run and approves the final evaluation protocol.")
        return
    try:
        freeze = read_json(FREEZE)
        allowed = (freeze.get("approved") is True and freeze.get("run_id") == record["id"]
                   and freeze.get("dataset_fingerprint") == meta["fingerprint"]
                   and bool(str(freeze.get("protocol", "")).strip()))
    except (OSError, ValueError):
        allowed = False
    confirm = st.checkbox("I confirm this is the approved one-time test evaluation.", disabled=not allowed)
    if st.button("Evaluate frozen model on final test", disabled=not (confirm and allowed)):
        try:
            final_evaluate(record)
            st.rerun()
        except (ValueError, OSError) as error:
            st.error(str(error))


def comparison(records):
    first = choose(records, "Reference experiment", "compare_reference")
    if first is None:
        return
    choices = [row for row in records if compatible(first, row)]
    by_id = {row["id"]: row for row in choices}
    selected = st.multiselect("Add compatible experiments", [key for key in by_id if key != first["id"]],
                              format_func=lambda key: run_label(by_id[key]), key="compare_runs")
    if not selected:
        st.caption("Comparison requires another completed run with the same dataset fingerprint, split, pipeline and validation folds.")
        return
    runs = [first, *(by_id[key] for key in selected)]
    metric = st.selectbox("Rank by out-of-fold metric", ["f1", "recall", "precision", "roc_auc", "accuracy"])
    entries = []
    for run in runs:
        summary = run["summary"]
        primary = run["config"]["cv"]["primary"]
        entries.append({
            "Run": run_label(run), **{key: summary["metrics"][key] for key in ("f1", "recall", "precision", "roc_auc", "accuracy")},
            "Primary CV metric": primary, "CV mean": summary["cv"][primary]["mean"],
            "CV SD": summary["cv"][primary]["std"], "Runtime (s)": summary["runtime_seconds"],
            "OOF prediction (s)": summary["prediction_seconds"],
        })
    frame = pd.DataFrame(entries)
    best = frame.loc[frame[metric].idxmax()]
    st.success(f"Highest out-of-fold {metric} among selected runs: {best['Run']} ({best[metric]:.4f}).")
    st.dataframe(frame, hide_index=True, width="stretch")
    params = pd.DataFrame({run_label(run): run["summary"]["best_params"] for run in runs})
    differences = params.loc[params.astype(str).nunique(axis=1).gt(1)].astype(str)
    with st.expander("Hyperparameter differences", expanded=True):
        if differences.empty:
            st.write("These runs use identical forest parameters.")
        else:
            st.dataframe(differences, width="stretch")
    frame = frame.merge(params.T.rename_axis("Run").reset_index(), on="Run", validate="1:1")
    st.download_button("Export comparison CSV", frame.to_csv(index=False), "rf_comparison.csv", "text/csv")


@st.fragment(run_every=2)
def live_history(mode):
    st.subheader("Experiment history")
    records = [row for row in history() if row["config"]["dataset"] == mode]
    if not records:
        st.caption("Queued and completed experiments will appear here.")
        return
    st.dataframe(pd.DataFrame([{
        "Run": row["id"], "Type": row["type"], "Status": row["status"], "Started": row["started_at"],
        "Ended": row.get("ended_at"), "Seconds": row.get("runtime_seconds"),
    } for row in records]), hide_index=True, width="stretch")
    record = choose(records, "Run log", "history_log")
    path = run_folder(record) / "worker.log"
    if path.exists():
        log = path.read_text(encoding="utf-8", errors="replace")
        total = re.search(r"^Searching (\d+) configurations across (\d+) folds", log, re.M)
        if total:
            expected = int(total[1]) * int(total[2])
            actual = len(re.findall(r"^\[CV[^\]]*\] END", log, re.M))
            if actual:
                st.progress(min(1.0, actual / expected), text=f"{actual:,} of {expected:,} search fold fits reported")
        st.code(log[-18000:] or "Waiting for worker output…", language="text")
    if record.get("error"):
        st.error(record["error"])
    if st.button("Refresh completed results", key="refresh_results"):
        st.rerun()


def png_plot(folder, summary):
    fig = Figure(figsize=(9, 4), layout="constrained")
    ax, curve = fig.subplots(1, 2)
    matrix = summary["metrics"]["confusion_matrix"]
    ax.imshow(matrix, cmap="Blues")
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            ax.text(x, y, value, ha="center", va="center",
                    color="white" if value > max(map(max, matrix)) / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted class", ylabel="Actual class", title="OOF confusion matrix")
    points = pd.read_csv(folder / "precision_recall.csv")
    curve.plot(points["recall"], points["precision"], color="#007d79")
    curve.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02), title="OOF precision–recall")
    target = io.BytesIO()
    fig.savefig(target, format="png", dpi=160)
    return target.getvalue()


def exports(records):
    st.subheader("Export artifacts")
    st.caption("Official rf_cv.json export is blocked until M1 confirms the exact handoff schema.")
    record = choose(records, "Experiment to export", "export_run")
    if record is None:
        return
    folder = run_folder(record)
    summary = record["summary"]
    a, b, c = st.columns(3)
    a.download_button(f"Export {record['type']} configuration JSON", (folder / "config.json").read_bytes(),
                      f"{record['type']}_configuration.json", "application/json")
    b.download_button("Export CV results CSV", (folder / "cv_results.csv").read_bytes(), "cv_results.csv", "text/csv")
    c.download_button("Export best parameters JSON", json.dumps(summary["best_params"], indent=2),
                      "best_params.json", "application/json")
    a.download_button("Export experiment summary", (folder / "record.json").read_bytes(), "experiment.json", "application/json")
    if b.checkbox("Prepare plots PNG", key="export_png"):
        b.download_button("Export plots PNG", png_plot(folder, summary), "rf_validation.png", "image/png")
    if c.checkbox("Prepare model download", key="export_model"):
        c.download_button("Export trained model joblib", (folder / "model.joblib").read_bytes(),
                          "random_forest.joblib", "application/octet-stream")
