"""Interactive inspection of persisted forests and measured run differences."""

import io

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from matplotlib.figure import Figure
from sklearn.tree import plot_tree

from m3_workspace.app.dashboards import choose, run_label
from m3_workspace.engine.artifact_manager import run_folder
from m3_workspace.engine.experiment_manager import compatible, dataset


@st.cache_resource(max_entries=2, show_spinner="Loading the fitted forest")
def fitted_model(path):
    return joblib.load(path)


def sequence(size, key):
    state = st.session_state
    cursor, playing = f"{key}_cursor", f"{key}_playing"
    state.setdefault(cursor, 1)
    state.setdefault(playing, False)
    state[cursor] = min(size, max(1, state[cursor]))
    if state[playing]:
        state[cursor] = min(size, state[cursor] + 1)
        if state[cursor] == size:
            state[playing] = False

    def play():
        if state[cursor] == size:
            state[cursor] = 1
        state[playing] = True

    def move(delta):
        state[playing] = False
        state[cursor] = min(size, max(1, state[cursor] + delta))

    a, b, c, d, e = st.columns(5)
    a.button("Play", key=f"{key}_play", on_click=play)
    b.button("Pause", key=f"{key}_pause", on_click=move, args=(0,))
    c.button("Previous", key=f"{key}_prev", on_click=move, args=(-1,))
    d.button("Next", key=f"{key}_next", on_click=move, args=(1,))
    e.button("Reset", key=f"{key}_reset", on_click=move, args=(-size,))
    st.progress(state[cursor] / size, text=f"{state[cursor]} of {size}")
    return state[cursor]


def inspect_bootstrap(forest, tree_index, training_rows, pace, key):
    if not forest.bootstrap:
        st.info("This forest used all training rows for each tree because bootstrap sampling was disabled.")
        return
    # estimators_samples_ materializes all draws. Cache only the relevant tree's draw.
    indices = bootstrap_indices(forest, tree_index, key)
    distinct, counts = np.unique(indices, return_counts=True)
    st.write(f"Tree seed **{forest.estimators_[tree_index].random_state}** · Sampling with replacement")
    a, b, c = st.columns(3)
    a.metric("Total draws", len(indices))
    b.metric("Repeated draws", len(indices) - len(distinct))
    c.metric("Out-of-bag rows", training_rows - len(distinct))
    st.caption("The replay shows the first 60 draws. Indexes refer to positions in the saved training split.")

    @st.fragment(run_every=pace)
    def animate():
        step = sequence(min(60, len(indices)), key)
        shown = pd.DataFrame({"Draw": range(1, step + 1), "Training position": indices[:step]})
        shown["Selection"] = np.where(shown["Training position"].duplicated(), "Repeated", "First appearance")
        fig = px.scatter(shown, x="Draw", y="Training position", color="Selection", height=300)
        fig.update_xaxes(range=[0, min(60, len(indices)) + 1])
        fig.update_yaxes(range=[0, training_rows])
        st.plotly_chart(fig)

    animate()
    with st.expander("Inspect repeated and out-of-bag positions"):
        a, b = st.columns(2)
        a.dataframe(pd.DataFrame({"Repeated position": distinct[counts > 1], "Times drawn": counts[counts > 1]}),
                    hide_index=True, height=230)
        b.dataframe(pd.DataFrame({"Out-of-bag position": np.setdiff1d(np.arange(training_rows), distinct)}),
                    hide_index=True, height=230)


@st.cache_data(max_entries=12)
def bootstrap_indices(_forest, tree_index, model_identity):
    return _forest.estimators_samples_[tree_index]


def inspect_tree(tree, transformed, names):
    a, b, c = st.columns(3)
    a.metric("Actual depth", tree.get_depth())
    b.metric("Leaves", tree.get_n_leaves())
    shown_depth = c.selectbox("Rendered depth", [1, 2, 3], index=1)
    if st.checkbox("Render tree diagram"):
        fig = Figure(figsize=(14, 5))
        plot_tree(tree, max_depth=shown_depth, feature_names=names, class_names=["0", "1"],
                  filled=True, fontsize=7, ax=fig.subplots())
        image = io.BytesIO()
        fig.savefig(image, format="png", bbox_inches="tight", dpi=140)
        st.image(image.getvalue())
    path = tree.decision_path(transformed).indices
    leaf = tree.apply(transformed)[0]
    split = []
    for node in path:
        if node == leaf:
            probabilities = tree.predict_proba(transformed)[0]
            st.write(f"Leaf **{leaf}** · Predicted class **{int(tree.predict(transformed)[0])}** · P(default) **{probabilities[1]:.3f}**")
            break
        feature = tree.tree_.feature[node]
        cutoff = tree.tree_.threshold[node]
        value = transformed[0, feature]
        split.append({"Node": node, "Feature": names[feature], "Value": value, "Threshold": cutoff,
                      "Branch": "Left (≤)" if value <= cutoff else "Right (>)"})
    st.dataframe(pd.DataFrame(split), hide_index=True)


def inspect_votes(pipeline, X, transformed, pace, key):
    forest = pipeline.named_steps["model"]
    probabilities = np.array([tree.predict_proba(transformed)[0, 1] for tree in forest.estimators_])
    votes = (probabilities > 0.5).astype(int)
    p_default = pipeline.predict_proba(X)[0, 1]
    a, b, c = st.columns(3)
    a.metric("Forest class", int(pipeline.predict(X)[0]))
    b.metric("Forest P(class 1)", f"{p_default:.4f}")
    c.metric("Hard majority class", int(votes.sum() > len(votes) / 2))
    st.caption("The classifier predicts from the mean of tree probabilities. An unweighted count of hard tree votes can give a different class. Ties resolve to class 0.")
    data = pd.DataFrame({"Tree": np.arange(1, len(votes) + 1), "Class 1 votes": votes.cumsum(),
                         "Class 0 votes": (1 - votes).cumsum(), "Predicted class": votes,
                         "P(class 1)": probabilities})

    @st.fragment(run_every=pace)
    def animate():
        position = sequence(len(data), key)
        shown = data.iloc[:position]
        fig = px.line(shown, x="Tree", y=["Class 0 votes", "Class 1 votes"], markers=True, height=300)
        fig.update_xaxes(range=[0, len(data) + 1])
        fig.update_yaxes(range=[0, max(1, len(data))])
        st.plotly_chart(fig)
        st.dataframe(shown.tail(15), hide_index=True, height=205)

    animate()
    with st.expander("All tree predictions"):
        st.dataframe(data, hide_index=True, height=230)


def impact(records, reference):
    key = st.selectbox("Parameter to compare", ["n_estimators", "max_depth", "min_samples_leaf", "max_features"])
    metric = st.selectbox("Measured outcome", ["F1", "Mean tree depth", "Mean tree leaves", "Runtime (s)"])
    rows = []
    for run in records:
        if compatible(reference, run):
            result = run["summary"]
            rows.append({
                "Run": run_label(run), key: str(result["best_params"][key]), "F1": result["metrics"]["f1"],
                "Mean tree depth": result["mean_tree_depth"], "Mean tree leaves": result["mean_tree_leaves"],
                "Runtime (s)": result["runtime_seconds"],
            })
    st.caption("Measured results from comparable completed runs. Other settings can differ, so this comparison does not isolate the selected parameter.")
    st.plotly_chart(px.scatter(pd.DataFrame(rows), x=key, y=metric, hover_name="Run", height=320))


def visualizer(records):
    run = choose(records, "Fitted forest to explore", "visualizer_run")
    if run is None:
        return
    view = st.radio("Explore", ["Bootstrap draws", "Decision tree", "Forest voting", "Hyperparameter impact"],
                    horizontal=True, key="visual_mode")
    if view == "Hyperparameter impact":
        impact(records, run)
        return
    if not st.checkbox("Open fitted forest", key="open_forest"):
        return
    path = run_folder(run) / "model.joblib"
    pipeline = fitted_model(path)
    (training, _, _, _), meta = dataset(run["config"]["dataset"])
    if meta != run["summary"]["dataset"]:
        st.error("The dataset or shared pipeline has changed since this run.")
        return
    forest = pipeline.named_steps["model"]
    c1, c2, c3 = st.columns(3)
    tree_index = int(c1.number_input("Tree", 1, len(forest.estimators_), 1,
                                     key=f"tree_{run['id']}")) - 1 if view != "Forest voting" else 0
    if view != "Decision tree":
        speed = c3.select_slider("Replay speed", ["Slow", "Normal", "Fast"], value="Normal", key="viz_speed")
        pace = {"Slow": 1.5, "Normal": 0.65, "Fast": 0.25}[speed]
    if view == "Bootstrap draws":
        inspect_bootstrap(forest, tree_index, len(training), pace, f"sample_{run['id']}_{tree_index}")
        return
    position = int(c2.number_input("Training observation position", 0, len(training) - 1, 0,
                                   key=f"observation_{run['id']}"))
    X = training.iloc[[position]]
    with st.expander("Observation values"):
        st.dataframe(X)
    preprocess = pipeline.named_steps.get("preprocess")
    transformed = np.asarray(preprocess.transform(X) if preprocess else X, dtype=np.float32)
    if view == "Decision tree":
        names = list(preprocess.get_feature_names_out()) if preprocess else list(training.columns)
        inspect_tree(forest.estimators_[tree_index], transformed, names)
    else:
        inspect_votes(pipeline, X, transformed, pace, f"vote_{run['id']}_{position}")
