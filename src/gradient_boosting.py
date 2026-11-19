
"""
Gradient Boosting Model

Uses the shared preprocessing pipeline.
Supports optional balanced sample weighting.
"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.preprocessing import build_pipeline


class BalancedGradientBoostingClassifier(GradientBoostingClassifier):
    """
    Gradient Boosting with balanced sample weights.

    Weights are calculated inside each model fit,
    using only the training data of that fold.
    """

    def fit(self, X, y, sample_weight=None, monitor=None):

        if sample_weight is None:
            sample_weight = compute_sample_weight(
                class_weight="balanced",
                y=y,
            )

        return super().fit(
            X,
            y,
            sample_weight=sample_weight,
            monitor=monitor,
        )


def make_gb_pipeline(balanced=False):
    """
    Create an unfitted Gradient Boosting pipeline.

    Args:
        balanced (bool):
            False = Original Gradient Boosting.
            True = Balanced sample weighting.

    Returns:
        sklearn.pipeline.Pipeline
    """

    model_class = (
        BalancedGradientBoostingClassifier
        if balanced
        else GradientBoostingClassifier
    )

    estimator = model_class(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=1.0,
        random_state=42,
    )

    pipeline = build_pipeline(
        estimator,
        scale_numeric=False,
    )

    return pipeline
