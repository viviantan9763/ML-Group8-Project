
"""
Gradient Boosting Model

Uses the shared preprocessing pipeline.
"""

from sklearn.ensemble import GradientBoostingClassifier

from src.preprocessing import build_pipeline


def make_gb_pipeline():
    """
    Create an unfitted Gradient Boosting pipeline.

    Returns:
        sklearn.pipeline.Pipeline:
            Shared preprocessing + Gradient Boosting model.
    """

    estimator = GradientBoostingClassifier(
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
