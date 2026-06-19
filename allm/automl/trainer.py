"""
Model trainer — fits each registered model and captures timing metadata.

Design decisions:
- Each model trains independently with its own try/except so a single model
  failure doesn't kill the entire run.
- Training time is recorded per model for leaderboard display and Gemini context.
- Returns a list of result dicts consumed by the evaluator.
"""

from __future__ import annotations

import time
import logging
from typing import Any

import numpy as np
from sklearn.base import clone

from automl.model_registry import get_models

logger = logging.getLogger(__name__)


def train_models(
    task_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> list[dict[str, Any]]:
    """Train all registered models for the task type.

    Returns a list of dicts:
        {name, model, training_time_sec, error}
    Models that fail to fit are included with error != None so the evaluator
    can surface diagnostics without crashing.
    """
    models = get_models(task_type)
    results: list[dict[str, Any]] = []

    for name, estimator in models.items():
        entry: dict[str, Any] = {
            "name": name,
            "model": None,
            "training_time_sec": 0.0,
            "error": None,
        }
        try:
            model = clone(estimator)
            start = time.perf_counter()
            model.fit(X_train, y_train)
            entry["training_time_sec"] = round(time.perf_counter() - start, 3)
            entry["model"] = model
        except Exception as exc:
            logger.warning("Model %s failed to train: %s", name, exc)
            entry["error"] = str(exc)

        results.append(entry)

    return results
