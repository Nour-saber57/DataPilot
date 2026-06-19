"""
Explainability module — comprehensive visualizations for AutoML analysis.

Generates eight types of visuals organized into three categories:

EDA (Exploratory Data Analysis):
1. Correlation heatmap — numeric feature relationships.
2. Target distribution — class balance or value spread.
3. Missing values chart — per-column missing data overview.

Model Diagnostics:
4. Feature importance (bar chart) — tree-based, coef_, or permutation fallback.
5. Confusion matrix heatmap (classification).
6. ROC curve with AUC (classification, binary/multiclass).
7. Prediction vs Actual scatter (regression).
8. Residual distribution (regression).

Model Comparison:
9. Metric comparison bar chart across all trained models.

Design decisions:
- All plots use a consistent dark color palette for cohesive report output.
- Figures are returned as matplotlib Figure objects for Streamlit st.pyplot()
  AND saved to disk for the Markdown report.
- Feature importance capped at top 15 for readability.
- ROC curve handles both binary and multiclass via one-vs-rest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

PALETTE = {
    "primary": "#6366f1",
    "secondary": "#a78bfa",
    "accent": "#f472b6",
    "bg": "#0f172a",
    "surface": "#1e293b",
    "text": "#e2e8f0",
    "grid": "#334155",
    "positive": "#34d399",
    "negative": "#f87171",
    "warning": "#fbbf24",
}

MODEL_COLORS = ["#6366f1", "#f472b6", "#34d399", "#fbbf24", "#f87171", "#a78bfa"]

MAX_FEATURES_DISPLAYED = 15


# ── Shared styling ───────────────────────────────────────────────────────────

def _apply_base_style(fig: plt.Figure, ax: plt.Axes) -> None:
    """Apply the shared dark theme to a figure."""
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["surface"])
    ax.tick_params(colors=PALETTE["text"], labelsize=9)
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.title.set_color(PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])


def _style_colorbar(ax: plt.Axes) -> None:
    """Fix colorbar text color on dark backgrounds."""
    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.ax.yaxis.set_tick_params(color=PALETTE["text"])
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["text"])


def save_plot(fig: plt.Figure, filepath: str | Path) -> str:
    """Persist a figure to disk as PNG. Returns the absolute path."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(filepath.resolve())


# ── EDA plots ────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, target_col: str) -> plt.Figure:
    """Correlation matrix heatmap for numeric features."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return _empty_figure("Not enough numeric features for correlation analysis")

    corr = numeric_df.corr()

    # Limit to 20 features max for readability
    if corr.shape[0] > 20:
        # Keep features most correlated with target
        if target_col in corr.columns:
            top_features = corr[target_col].abs().nlargest(20).index.tolist()
            corr = corr.loc[top_features, top_features]

    size = max(6, min(12, corr.shape[0] * 0.55))
    fig, ax = plt.subplots(figsize=(size, size))
    _apply_base_style(fig, ax)

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True if corr.shape[0] <= 12 else False,
        fmt=".2f" if corr.shape[0] <= 12 else "",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.3,
        linecolor=PALETTE["grid"],
        cbar_kws={"shrink": 0.8},
        ax=ax,
        annot_kws={"size": 8, "color": PALETTE["text"]},
    )

    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    _style_colorbar(ax)
    fig.tight_layout()
    return fig


def plot_target_distribution(
    df: pd.DataFrame, target_col: str, task_type: str
) -> plt.Figure:
    """Visualize the target variable distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    _apply_base_style(fig, ax)

    target = df[target_col]

    if task_type == "classification":
        counts = target.value_counts().sort_index()
        bars = ax.bar(
            [str(c) for c in counts.index],
            counts.values,
            color=MODEL_COLORS[:len(counts)],
            edgecolor=PALETTE["grid"],
            linewidth=0.5,
        )
        for bar, val in zip(bars, counts.values):
            pct = val / len(target) * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.02,
                f"{val}\n({pct:.1f}%)",
                ha="center", va="bottom",
                color=PALETTE["text"], fontsize=9,
            )
        ax.set_xlabel("Class", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"Target Distribution — {target_col}", fontsize=12, fontweight="bold")
    else:
        ax.hist(
            target.dropna(),
            bins=min(50, max(15, len(target) // 10)),
            color=PALETTE["primary"],
            edgecolor=PALETTE["secondary"],
            linewidth=0.5,
            alpha=0.85,
        )
        ax.axvline(
            target.median(), color=PALETTE["accent"], linestyle="--",
            linewidth=1.5, label=f"Median: {target.median():,.2f}",
        )
        ax.axvline(
            target.mean(), color=PALETTE["positive"], linestyle="--",
            linewidth=1.5, label=f"Mean: {target.mean():,.2f}",
        )
        ax.set_xlabel(target_col, fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(f"Target Distribution — {target_col}", fontsize=12, fontweight="bold")
        ax.legend(
            facecolor=PALETTE["surface"], edgecolor=PALETTE["grid"],
            labelcolor=PALETTE["text"], fontsize=9,
        )

    fig.tight_layout()
    return fig


def plot_missing_values(df: pd.DataFrame) -> plt.Figure | None:
    """Bar chart of missing values per column. Returns None if no missing data."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)

    if missing.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, max(3, len(missing) * 0.35)))
    _apply_base_style(fig, ax)

    bars = ax.barh(
        missing.index.astype(str),
        missing.values,
        color=PALETTE["warning"],
        edgecolor=PALETTE["grid"],
        linewidth=0.5,
        height=0.6,
    )

    total = len(df)
    for bar, val in zip(bars, missing.values):
        pct = val / total * 100
        ax.text(
            bar.get_width() + total * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val} ({pct:.1f}%)",
            va="center", ha="left",
            color=PALETTE["text"], fontsize=8,
        )

    ax.set_xlabel("Missing Count", fontsize=10)
    ax.set_title("Missing Values by Column", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Feature importance ───────────────────────────────────────────────────────

def compute_feature_importance(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Extract feature importances with fallback chain.

    Priority: tree importances → |coef_| → permutation importance.
    Returns a DataFrame sorted by importance descending, normalized to 0–1.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
        if len(importances) != len(feature_names):
            importances = np.abs(model.coef_).mean(axis=0)
    else:
        result = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        importances = result.importances_mean

    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)

    max_imp = fi_df["Importance"].max()
    if max_imp > 0:
        fi_df["Importance"] = fi_df["Importance"] / max_imp

    return fi_df


def plot_feature_importance(fi_df: pd.DataFrame, model_name: str) -> plt.Figure:
    """Horizontal bar chart of the top features with gradient coloring."""
    top = fi_df.head(MAX_FEATURES_DISPLAYED).copy()
    top = top.sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.4)))
    _apply_base_style(fig, ax)

    # Gradient colors based on importance
    colors = plt.cm.cool(np.linspace(0.2, 0.8, len(top)))

    bars = ax.barh(
        top["Feature"],
        top["Importance"],
        color=colors,
        edgecolor=PALETTE["secondary"],
        linewidth=0.5,
        height=0.6,
    )

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center", ha="left",
            color=PALETTE["text"], fontsize=8,
        )

    ax.set_xlabel("Relative Importance", fontsize=10)
    ax.set_title(f"Feature Importance — {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1.15)
    fig.tight_layout()
    return fig


# ── Classification diagnostics ───────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str
) -> plt.Figure:
    """Annotated heatmap confusion matrix with percentages."""
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

    # Percentage matrix for annotation
    cm_pct = cm.astype(float) / cm.sum() * 100

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.2), max(4, len(labels) * 1.0)))
    _apply_base_style(fig, ax)

    annot_text = np.array([
        [f"{count}\n({pct:.1f}%)" for count, pct in zip(row_c, row_p)]
        for row_c, row_p in zip(cm, cm_pct)
    ])

    sns.heatmap(
        cm,
        annot=annot_text,
        fmt="",
        cmap="rocket_r",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor=PALETTE["grid"],
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold")
    _style_colorbar(ax)
    fig.tight_layout()
    return fig


def plot_roc_curve(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> plt.Figure | None:
    """ROC curve with AUC. Handles binary and multiclass (OvR)."""
    if not hasattr(model, "predict_proba"):
        return None

    try:
        y_proba = model.predict_proba(X_test)
    except Exception:
        return None

    classes = np.unique(y_test)
    fig, ax = plt.subplots(figsize=(7, 6))
    _apply_base_style(fig, ax)

    if len(classes) == 2:
        # Binary
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        auc = roc_auc_score(y_test, y_proba[:, 1])
        ax.plot(fpr, tpr, color=PALETTE["primary"], linewidth=2,
                label=f"AUC = {auc:.4f}")
        ax.fill_between(fpr, tpr, alpha=0.15, color=PALETTE["primary"])
    else:
        # Multiclass OvR
        y_bin = label_binarize(y_test, classes=classes)
        for i, cls in enumerate(classes):
            if y_bin.shape[1] <= i:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
            color = MODEL_COLORS[i % len(MODEL_COLORS)]
            ax.plot(fpr, tpr, color=color, linewidth=1.5,
                    label=f"Class {cls} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color=PALETTE["grid"], linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=12, fontweight="bold")
    ax.legend(
        facecolor=PALETTE["surface"], edgecolor=PALETTE["grid"],
        labelcolor=PALETTE["text"], fontsize=9, loc="lower right",
    )
    fig.tight_layout()
    return fig


# ── Regression diagnostics ───────────────────────────────────────────────────

def plot_prediction_vs_actual(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str
) -> plt.Figure:
    """Scatter plot of predicted vs actual with perfect prediction line."""
    fig, ax = plt.subplots(figsize=(7, 6))
    _apply_base_style(fig, ax)

    ax.scatter(
        y_true, y_pred,
        alpha=0.5, s=20,
        color=PALETTE["primary"],
        edgecolors=PALETTE["secondary"],
        linewidths=0.3,
    )

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.05
    ax.plot(
        [min_val - margin, max_val + margin],
        [min_val - margin, max_val + margin],
        color=PALETTE["accent"], linestyle="--", linewidth=1.5,
        label="Perfect Prediction",
    )

    ax.set_xlabel("Actual Values", fontsize=10)
    ax.set_ylabel("Predicted Values", fontsize=10)
    ax.set_title(f"Predicted vs Actual — {model_name}", fontsize=12, fontweight="bold")
    ax.legend(
        facecolor=PALETTE["surface"], edgecolor=PALETTE["grid"],
        labelcolor=PALETTE["text"], fontsize=9,
    )
    fig.tight_layout()
    return fig


def plot_residual_distribution(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str
) -> plt.Figure:
    """Residual histogram + KDE to check for normality of errors."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1 = axes[0]
    _apply_base_style(fig, ax1)
    ax1.hist(
        residuals, bins=30,
        color=PALETTE["primary"], edgecolor=PALETTE["secondary"],
        linewidth=0.5, alpha=0.85,
    )
    ax1.axvline(0, color=PALETTE["accent"], linestyle="--", linewidth=1.5)
    ax1.set_xlabel("Residual (Actual − Predicted)", fontsize=10)
    ax1.set_ylabel("Frequency", fontsize=10)
    ax1.set_title("Residual Distribution", fontsize=12, fontweight="bold")

    # Residual vs predicted
    ax2 = axes[1]
    _apply_base_style(fig, ax2)
    ax2.scatter(
        y_pred, residuals,
        alpha=0.5, s=20,
        color=PALETTE["primary"],
        edgecolors=PALETTE["secondary"],
        linewidths=0.3,
    )
    ax2.axhline(0, color=PALETTE["accent"], linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Predicted Values", fontsize=10)
    ax2.set_ylabel("Residuals", fontsize=10)
    ax2.set_title(f"Residuals vs Predicted — {model_name}", fontsize=12, fontweight="bold")

    fig.tight_layout()
    return fig


# ── Model comparison ─────────────────────────────────────────────────────────

def plot_model_comparison(
    scored_models: list[dict[str, Any]],
    task_type: str,
) -> plt.Figure:
    """Grouped bar chart comparing all models on key metrics."""
    valid = [m for m in scored_models if m["error"] is None]
    if not valid:
        return _empty_figure("No models to compare")

    names = [m["name"] for m in valid]

    if task_type == "classification":
        metrics_to_plot = ["accuracy", "weighted_f1", "weighted_precision", "weighted_recall"]
        metric_labels = ["Accuracy", "F1", "Precision", "Recall"]
    else:
        # Normalize regression metrics to 0-1 for visual comparison
        metrics_to_plot = ["r2"]
        metric_labels = ["R²"]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 2.5), 5))
    _apply_base_style(fig, ax)

    x = np.arange(len(names))
    width = 0.8 / len(metrics_to_plot)

    for i, (metric_key, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        values = [m["metrics"].get(metric_key, 0) for m in valid]
        offset = (i - len(metrics_to_plot) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, values,
            width=width,
            label=label,
            color=MODEL_COLORS[i % len(MODEL_COLORS)],
            edgecolor=PALETTE["grid"],
            linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom",
                color=PALETTE["text"], fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Model Comparison", fontsize=12, fontweight="bold")
    ax.legend(
        facecolor=PALETTE["surface"], edgecolor=PALETTE["grid"],
        labelcolor=PALETTE["text"], fontsize=9,
    )
    if task_type == "classification":
        ax.set_ylim(0, 1.12)

    fig.tight_layout()
    return fig


# ── Utilities ────────────────────────────────────────────────────────────────

def _empty_figure(message: str) -> plt.Figure:
    """Create a minimal figure with a centered message for edge cases."""
    fig, ax = plt.subplots(figsize=(6, 3))
    _apply_base_style(fig, ax)
    ax.text(0.5, 0.5, message, ha="center", va="center",
            color=PALETTE["text"], fontsize=12, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


# ── Main generation pipeline ────────────────────────────────────────────────

def generate_plots(
    df: pd.DataFrame,
    target_col: str,
    best_model_entry: dict[str, Any],
    scored_models: list[dict[str, Any]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    task_type: str,
    output_dir: str | Path = "outputs/plots",
) -> dict[str, Any]:
    """Produce all analysis artifacts.

    Returns a dict with DataFrames, Figure objects, and file paths for
    every plot. Figures are returned live for Streamlit and also saved
    to disk for the report.
    """
    output_dir = Path(output_dir)
    model = best_model_entry["model"]
    model_name = best_model_entry["name"]
    y_pred = best_model_entry["predictions"]

    result: dict[str, Any] = {}

    # ── EDA plots ──
    corr_fig = plot_correlation_heatmap(df, target_col)
    result["correlation_fig"] = corr_fig
    result["correlation_path"] = save_plot(
        plot_correlation_heatmap(df, target_col), output_dir / "correlation_heatmap.png"
    )

    dist_fig = plot_target_distribution(df, target_col, task_type)
    result["target_distribution_fig"] = dist_fig
    result["target_distribution_path"] = save_plot(
        plot_target_distribution(df, target_col, task_type),
        output_dir / "target_distribution.png",
    )

    missing_fig = plot_missing_values(df)
    if missing_fig is not None:
        result["missing_values_fig"] = missing_fig
        result["missing_values_path"] = save_plot(
            plot_missing_values(df), output_dir / "missing_values.png"
        )

    # ── Feature importance ──
    fi_df = compute_feature_importance(model, X_test, y_test, feature_names)
    result["feature_importance_df"] = fi_df
    result["feature_importance_fig"] = plot_feature_importance(fi_df, model_name)
    result["feature_importance_path"] = save_plot(
        plot_feature_importance(fi_df, model_name),
        output_dir / "feature_importance.png",
    )

    # ── Task-specific diagnostics ──
    if task_type == "classification":
        result["diagnostic_fig"] = plot_confusion_matrix(y_test, y_pred, model_name)
        result["diagnostic_path"] = save_plot(
            plot_confusion_matrix(y_test, y_pred, model_name),
            output_dir / "confusion_matrix.png",
        )

        roc_fig = plot_roc_curve(model, X_test, y_test, model_name)
        if roc_fig is not None:
            result["roc_fig"] = roc_fig
            result["roc_path"] = save_plot(
                plot_roc_curve(model, X_test, y_test, model_name),
                output_dir / "roc_curve.png",
            )
    else:
        result["diagnostic_fig"] = plot_prediction_vs_actual(y_test, y_pred, model_name)
        result["diagnostic_path"] = save_plot(
            plot_prediction_vs_actual(y_test, y_pred, model_name),
            output_dir / "prediction_vs_actual.png",
        )

        result["residual_fig"] = plot_residual_distribution(y_test, y_pred, model_name)
        result["residual_path"] = save_plot(
            plot_residual_distribution(y_test, y_pred, model_name),
            output_dir / "residual_distribution.png",
        )

    # ── Model comparison ──
    result["model_comparison_fig"] = plot_model_comparison(scored_models, task_type)
    result["model_comparison_path"] = save_plot(
        plot_model_comparison(scored_models, task_type),
        output_dir / "model_comparison.png",
    )

    return result
