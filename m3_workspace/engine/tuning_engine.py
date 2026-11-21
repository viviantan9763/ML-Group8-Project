"""Validated experiment settings and conditional scikit-learn search spaces."""

import math
import os
from itertools import product

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


CPUS = min(32, os.cpu_count() or 1)
METRICS = ("f1", "precision", "recall", "roc_auc", "accuracy", "average_precision", "balanced_accuracy")
SPACE = {
    "n_estimators": [100, 200, 400, 600],
    "max_depth": [6, 10, 16, 24, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8, 16],
    "max_features": ["sqrt", "log2", 0.25, 0.5],
    "criterion": ["gini", "entropy", "log_loss"],
    "bootstrap": [True, False],
    "class_weight": [None, "balanced", "balanced_subsample"],
    "max_leaf_nodes": [None, 32, 128],
    "max_samples": [0.65, 0.85, None],
    "ccp_alpha": [0.0, 0.0001, 0.001],
}
HELP = {
    "n_estimators": "Number of trees. More trees usually take longer to fit.",
    "criterion": "Measure used to choose a split.",
    "max_depth": "Maximum levels below the root. null allows unlimited depth.",
    "min_samples_split": "Minimum number or fraction of observations required before splitting.",
    "min_samples_leaf": "Minimum number or fraction of observations allowed in a leaf.",
    "max_features": "Number, fraction, sqrt or log2 of features tried at each split. null uses all features.",
    "max_leaf_nodes": "Maximum number of leaves per tree. null removes the limit.",
    "min_impurity_decrease": "Minimum improvement in impurity required to split.",
    "bootstrap": "Sample training rows with replacement for each tree.",
    "oob_score": "Estimate accuracy with rows omitted from each tree’s bootstrap sample.",
    "class_weight": "Weight the target classes. null gives equal weight.",
    "max_samples": "Number or fraction of rows drawn per tree when bootstrapping. null uses the full row count.",
    "ccp_alpha": "Cost-complexity pruning strength. Zero disables pruning.",
    "random_state": "Seed for forest training. An integer enables reproducibility.",
    "n_jobs": "Parallel tree workers for a baseline. The search uses one tree worker per CV worker.",
}
DEFAULTS = {key: value for key, value in RandomForestClassifier().get_params().items() if key in HELP}
GROUP_BASELINE = {**DEFAULTS, "random_state": 42, "n_jobs": 1}
PRESETS = {
    "Quick Test": (4, {"n_estimators": [50, 100], "max_depth": [8, None]}),
    "Balanced Search": (24, {key: SPACE[key] for key in (
        "n_estimators", "max_depth", "min_samples_leaf", "max_features", "class_weight"
    )}),
    "Thorough Search": (48, {key: SPACE[key] for key in (
        "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
        "max_features", "class_weight", "max_samples"
    )}),
}


def validate_model(params):
    if not isinstance(params, dict) or set(params) - set(HELP):
        raise ValueError("Configuration contains unsupported forest parameters.")
    options = GROUP_BASELINE | params
    # Use the estimator's own parameter constraints before launching any fit.
    RandomForestClassifier(**options)._validate_params()
    workers = options["n_jobs"]
    if workers not in (None, -1) and not 1 <= workers <= CPUS:
        raise ValueError(f"Forest workers must be within 1–{CPUS}, null, or -1 (bounded).")
    if not options["bootstrap"] and (options["oob_score"] or options["max_samples"] is not None):
        raise ValueError("OOB scoring and max_samples require bootstrap.")
    return options


def effective_workers(workers):
    return CPUS if workers == -1 else workers or 1


def search_grids(params, space):
    if not space or set(space) - set(SPACE):
        raise ValueError("Enable at least one supported search parameter.")
    for key, values in space.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} needs a nonempty JSON list.")
        if len({repr(value) for value in values}) != len(values):
            raise ValueError(f"{key} contains duplicate values.")
    conditional = ("bootstrap", "max_samples")
    grids = []
    for bootstrap, samples in product(*(space.get(key, [params[key]]) for key in conditional)):
        if not bootstrap and (samples is not None or params["oob_score"]):
            continue
        fixed = {"bootstrap": bootstrap, "max_samples": samples}
        validate_model(params | fixed)
        grid = {key: values for key, values in space.items() if key not in conditional}
        for key, values in grid.items():
            for value in values:
                validate_model(params | fixed | {key: value})
        grid.update({key: [value] for key, value in fixed.items() if key in space})
        grids.append({f"model__{key}": values for key, values in grid.items()})
    if not grids:
        raise ValueError("The search space contains no valid bootstrap combinations.")
    return grids


def validate_config(config, y):
    params = validate_model(config["model"])
    if params["random_state"] is None:
        raise ValueError("Enter an integer forest seed to make the experiment reproducible.")
    cv = config["cv"]
    folds = cv["folds"]
    if type(folds) is not int or not 2 <= folds <= int(y.value_counts().min()):
        raise ValueError("Folds must be between 2 and the minority-class training count.")
    scores = [cv["primary"], *cv["secondary"]]
    if len(scores) != len(set(scores)) or set(scores) - set(METRICS):
        raise ValueError("Select supported, distinct CV metrics.")
    search = config.get("search")
    grids = search_grids(params, search["space"]) if search else None
    count, total, workers = 1, 1, effective_workers(params["n_jobs"])
    if search:
        if search["method"] not in ("random", "grid") or search["refit"] not in ("primary", "manual"):
            raise ValueError("Unknown search method or refit strategy.")
        if search["errors"] not in ("raise", "nan") or not 1 <= search["jobs"] <= CPUS:
            raise ValueError("Invalid search error policy or worker count.")
        if not 0 <= search["verbosity"] <= 10:
            raise ValueError("Verbosity must be within 0–10.")
        total = sum(math.prod(map(len, grid.values())) for grid in grids)
        count = search["iterations"] if search["method"] == "random" else total
        if type(count) is not int or not 1 <= count <= total:
            raise ValueError(f"Choose between 1 and {total} random draws.")
        workers = search["jobs"]
    # Tuned candidates receive a second CV pass for OOF predictions and one final fit.
    fits = count * folds + (folds if search else 0) + 1
    limit = config["fit_limit"]
    if not 1 <= limit <= 5000 or fits > limit:
        raise ValueError(f"{fits:,} fits exceed the configured budget of {limit:,} (maximum 5,000).")
    splitter = StratifiedKFold(
        n_splits=folds, shuffle=cv["shuffle"], random_state=cv["seed"] if cv["shuffle"] else None
    )
    return params, splitter, scores, grids, {"candidates": count, "space_size": total, "fits": fits, "workers": workers}
