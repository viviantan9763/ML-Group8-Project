"""Binary classification evidence for M3 experiments."""

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)


def predict(estimator, X):
    start = perf_counter()
    probabilities = estimator.predict_proba(X)
    predictions = estimator.classes_[probabilities.argmax(axis=1)]
    return predictions, probabilities[:, list(estimator.classes_).index(1)], perf_counter() - start


def evaluate(y, predictions, probabilities):
    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
        "average_precision": average_precision_score(y, probabilities),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
    }


def curve(y, probability):
    precision, recall, _ = precision_recall_curve(y, probability)
    return pd.DataFrame({"recall": recall, "precision": precision})


def out_of_fold(X, y, folds, estimators):
    predictions = np.empty(len(y), dtype=int)
    probabilities = np.empty(len(y))
    elapsed = 0
    for (_, validation), estimator in zip(folds, estimators, strict=True):
        predictions[validation], probabilities[validation], seconds = predict(estimator, X.iloc[validation])
        elapsed += seconds
    return evaluate(y, predictions, probabilities), curve(y, probabilities), elapsed
