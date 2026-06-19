"""
Model evaluator — computes metrics and produces the ranked leaderboard.

Scoring strategy:
- Classification: weighted F1 is the primary ranking metric because it handles
  class imbalance better than accuracy. Accuracy is still reported for interpretability.
- Regression: RMSE is the primary ranking metric (lower is better). MAE and R² are
  secondary metrics for context.

Extended analysis (v2):
- Cross-validation scores for statistical robustness.
- Per-class precision/recall/F1 breakdown for classification.
- Overfitting detection via train-vs-test score gap.
- Data quality profiling for the report.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score


# ── Core metric functions ────────────────────────────────────────────────────

def _evaluate_classification(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "weighted_precision": round(
            precision_score(y_true, y_pred, average="weighted", zero_division=0), 4
        ),
        "weighted_recall": round(
            recall_score(y_true, y_pred, average="weighted", zero_division=0), 4
        ),
    }


def _evaluate_regression(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    y_range = y_true.max() - y_true.min()
    nrmse = rmse / y_range if y_range > 0 else 0.0
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "nrmse": round(nrmse, 4),
        "mape_pct": round(mape, 2),
    }


# ── Per-class breakdown ──────────────────────────────────────────────────────

def per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> pd.DataFrame:
    """Produce per-class precision, recall, F1 for classification tasks.

    Returns a DataFrame with one row per class plus macro/weighted averages.
    """
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    rows = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            rows.append({
                "Class": str(label),
                "Precision": round(metrics["precision"], 4),
                "Recall": round(metrics["recall"], 4),
                "F1-Score": round(metrics["f1-score"], 4),
                "Support": int(metrics["support"]),
            })
    return pd.DataFrame(rows)


# ── Cross-validation ─────────────────────────────────────────────────────────

def cross_validate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    cv_folds: int = 5,
) -> dict[str, float]:
    """Run stratified k-fold CV and return mean ± std of the primary metric.

    Uses weighted F1 for classification, neg_root_mean_squared_error for regression.
    Falls back gracefully if CV fails (e.g., too few samples per class).
    """
    try:
        if task_type == "classification":
            scoring = "f1_weighted"
        else:
            scoring = "neg_root_mean_squared_error"

        scores = cross_val_score(
            model, X_train, y_train, cv=cv_folds, scoring=scoring, n_jobs=-1
        )

        if task_type == "regression":
            scores = -scores

        return {
            "cv_mean": round(float(scores.mean()), 4),
            "cv_std": round(float(scores.std()), 4),
            "cv_folds": cv_folds,
            "cv_scores": [round(float(s), 4) for s in scores],
        }
    except Exception:
        return {"cv_mean": None, "cv_std": None, "cv_folds": cv_folds, "cv_scores": []}


# ── Overfitting detection ────────────────────────────────────────────────────

def detect_overfitting(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
) -> dict[str, Any]:
    """Compare train score vs test score to flag potential overfitting.

    Heuristic thresholds:
    - Gap > 0.10 for classification F1 → likely overfitting.
    - Train R² near 1.0 but test R² much lower → likely overfitting.
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    if task_type == "classification":
        train_score = f1_score(y_train, y_train_pred, average="weighted")
        test_score = f1_score(y_test, y_test_pred, average="weighted")
        metric_name = "Weighted F1"
    else:
        train_score = r2_score(y_train, y_train_pred)
        test_score = r2_score(y_test, y_test_pred)
        metric_name = "R²"

    gap = abs(train_score - test_score)

    if gap > 0.15:
        risk = "high"
        verdict = "Strong signs of overfitting. Consider regularization or more data."
    elif gap > 0.08:
        risk = "moderate"
        verdict = "Mild overfitting detected. Cross-validation scores should be checked."
    else:
        risk = "low"
        verdict = "Model generalizes well. Train and test scores are close."

    return {
        "metric_name": metric_name,
        "train_score": round(train_score, 4),
        "test_score": round(test_score, 4),
        "gap": round(gap, 4),
        "risk_level": risk,
        "verdict": verdict,
    }


# ── Data quality profiling ───────────────────────────────────────────────────

def profile_data_quality(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    """Produce a data quality profile for the dataset.

    Checks for: missing values per column, high-cardinality categoricals,
    near-zero-variance features, duplicate rows, and class imbalance.
    """
    total = len(df)
    missing = df.isnull().sum()
    missing_pct = (missing / total * 100).round(2)
    columns_with_missing = missing[missing > 0].to_dict()

    # High cardinality categoricals (>50 unique or >80% unique)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    high_cardinality = []
    for col in cat_cols:
        n_unique = df[col].nunique()
        if n_unique > 50 or (n_unique / total) > 0.8:
            high_cardinality.append({"column": col, "unique_values": n_unique})

    # Near-zero-variance numeric features
    num_cols = df.select_dtypes(include="number").columns
    low_variance = []
    for col in num_cols:
        if col == target_col:
            continue
        cv = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
        if abs(cv) < 0.01:
            low_variance.append(col)

    # Duplicate rows
    n_duplicates = df.duplicated().sum()

    # Class imbalance (classification only)
    class_balance = None
    if target_col in df.columns:
        target = df[target_col]
        if target.dtype == "object" or target.nunique() <= 20:
            counts = target.value_counts()
            imbalance_ratio = counts.max() / counts.min() if counts.min() > 0 else float("inf")
            class_balance = {
                "distribution": counts.to_dict(),
                "imbalance_ratio": round(imbalance_ratio, 2),
                "is_imbalanced": imbalance_ratio > 3,
            }

    return {
        "total_rows": total,
        "total_columns": len(df.columns),
        "columns_with_missing": columns_with_missing,
        "missing_pct": {k: v for k, v in missing_pct.items() if v > 0},
        "high_cardinality_columns": high_cardinality,
        "low_variance_features": low_variance,
        "duplicate_rows": int(n_duplicates),
        "class_balance": class_balance,
    }


# ── Main evaluation pipeline ────────────────────────────────────────────────

def evaluate_models(
    trained_models: list[dict[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
) -> list[dict[str, Any]]:
    """Score every successfully trained model against the holdout set.

    Includes cross-validation scores and overfitting detection for each model.
    Skips models with training errors.
    """
    scored: list[dict[str, Any]] = []

    for entry in trained_models:
        if entry["error"] is not None:
            scored.append({
                **entry,
                "metrics": {},
                "predictions": None,
                "cv_results": {},
                "overfit_analysis": {},
                "per_class": None,
            })
            continue

        model = entry["model"]
        y_pred = model.predict(X_test)

        if task_type == "classification":
            metrics = _evaluate_classification(y_test, y_pred)
            per_class = per_class_metrics(y_test, y_pred)
        else:
            metrics = _evaluate_regression(y_test, y_pred)
            per_class = None

        cv_results = cross_validate_model(model, X_train, y_train, task_type)

        overfit = detect_overfitting(
            model, X_train, y_train, X_test, y_test, task_type
        )

        scored.append({
            **entry,
            "metrics": metrics,
            "predictions": y_pred,
            "cv_results": cv_results,
            "overfit_analysis": overfit,
            "per_class": per_class,
        })

    return scored


def build_leaderboard(
    scored_models: list[dict[str, Any]],
    task_type: str,
) -> pd.DataFrame:
    """Build a sorted leaderboard DataFrame from scored model results.

    Now includes CV scores and overfitting risk alongside test metrics.
    """
    rows = []
    for entry in scored_models:
        if entry["error"] is not None:
            continue

        row = {"Model": entry["name"]}

        if task_type == "classification":
            row["Accuracy"] = entry["metrics"]["accuracy"]
            row["F1 (weighted)"] = entry["metrics"]["weighted_f1"]
            row["Precision"] = entry["metrics"]["weighted_precision"]
            row["Recall"] = entry["metrics"]["weighted_recall"]
        else:
            row["RMSE"] = entry["metrics"]["rmse"]
            row["MAE"] = entry["metrics"]["mae"]
            row["R²"] = entry["metrics"]["r2"]
            row["MAPE %"] = entry["metrics"]["mape_pct"]

        cv = entry.get("cv_results", {})
        if cv.get("cv_mean") is not None:
            row["CV Mean"] = cv["cv_mean"]
            row["CV Std"] = cv["cv_std"]

        overfit = entry.get("overfit_analysis", {})
        row["Overfit Risk"] = overfit.get("risk_level", "N/A")
        row["Train Time (s)"] = entry["training_time_sec"]

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if task_type == "classification":
        df = df.sort_values("F1 (weighted)", ascending=False).reset_index(drop=True)
    else:
        df = df.sort_values("RMSE", ascending=True).reset_index(drop=True)

    df.index = df.index + 1
    df.index.name = "Rank"
    return df


def select_best_model(
    scored_models: list[dict[str, Any]],
    task_type: str,
) -> dict[str, Any] | None:
    """Return the top-ranked model entry, or None if nothing trained."""
    valid = [m for m in scored_models if m["error"] is None]
    if not valid:
        return None

    if task_type == "classification":
        return max(valid, key=lambda m: m["metrics"].get("weighted_f1", 0))
    return min(valid, key=lambda m: m["metrics"].get("rmse", float("inf")))
