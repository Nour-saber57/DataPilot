"""
Model registry — central catalog of candidate models per task type.

Design decisions:
- Only scikit-learn estimators to keep the dependency graph small.
- Gradient Boosting included for competitive performance; XGBoost intentionally
  excluded to avoid C-extension install issues during demo.
- Hyperparameters are conservative defaults optimized for fast training
  on datasets under 100k rows.
"""

from __future__ import annotations

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression


def get_models(task_type: str) -> dict:
    """Return a name→estimator mapping for the given task type.

    Raises ValueError for unknown task types to fail fast.
    """
    if task_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, solver="lbfgs", n_jobs=-1
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=150, max_depth=12, random_state=42, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
            ),
        }

    if task_type == "regression":
        return {
            "Linear Regression": LinearRegression(n_jobs=-1),
            "Random Forest": RandomForestRegressor(
                n_estimators=150, max_depth=12, random_state=42, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
            ),
        }

    raise ValueError(f"Unknown task type: {task_type!r}. Expected 'classification' or 'regression'.")
