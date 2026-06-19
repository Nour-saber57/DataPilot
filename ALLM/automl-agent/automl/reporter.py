"""
Report generator — produces a comprehensive Markdown report.

The report aggregates all analysis from the AutoML run into a structured
document covering data quality, preprocessing, model comparison, best model
deep-dive, feature importance, diagnostic plots, overfitting analysis,
and actionable recommendations.

Design decisions:
- Markdown output for version-control friendliness and Streamlit rendering.
- Leaderboard separately saved as CSV for notebook interoperability.
- Plot images referenced via relative paths so outputs/ is self-contained.
- Recommendations are context-aware — different advice for overfitting,
  class imbalance, high cardinality, etc.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


# ── Section formatters ───────────────────────────────────────────────────────

def _metric_table(metrics: dict[str, float]) -> str:
    if not metrics:
        return "_No metrics available._"
    lines = ["| Metric | Value |", "|--------|-------|"]
    for key, val in metrics.items():
        display = key.replace("_", " ").replace("pct", "%").title()
        lines.append(f"| {display} | {val} |")
    return "\n".join(lines)


def _feature_table(fi_df: pd.DataFrame | None, top_n: int = 10) -> str:
    if fi_df is None or fi_df.empty:
        return "_No feature importance data available._"
    top = fi_df.head(top_n)
    lines = ["| Rank | Feature | Importance |", "|------|---------|------------|"]
    for idx, row in enumerate(top.itertuples(), start=1):
        lines.append(f"| {idx} | {row.Feature} | {row.Importance:.4f} |")
    return "\n".join(lines)


def _embed_plot(plot_paths: dict, key: str, output_dir: Path, alt: str) -> str:
    path = plot_paths.get(key)
    if not path:
        return ""
    rel = os.path.relpath(path, output_dir)
    return f"![{alt}]({rel})\n"


# ── Data quality section ─────────────────────────────────────────────────────

def _build_data_quality_section(quality: dict[str, Any]) -> list[str]:
    lines = ["## 2. Data Quality Analysis\n"]

    # Missing values
    cols_missing = quality.get("columns_with_missing", {})
    if cols_missing:
        lines.append("### Missing Values\n")
        lines.append("| Column | Missing | % |")
        lines.append("|--------|---------|---|")
        pcts = quality.get("missing_pct", {})
        for col, count in cols_missing.items():
            lines.append(f"| {col} | {count} | {pcts.get(col, 0):.1f}% |")
        lines.append("")
    else:
        lines.append("**Missing Values:** None detected ✅\n")

    # Duplicates
    dupes = quality.get("duplicate_rows", 0)
    if dupes > 0:
        lines.append(f"**Duplicate Rows:** {dupes} ({dupes / quality['total_rows'] * 100:.1f}%)\n")
    else:
        lines.append("**Duplicate Rows:** None ✅\n")

    # Class imbalance
    balance = quality.get("class_balance")
    if balance:
        lines.append("### Class Balance\n")
        lines.append("| Class | Count |")
        lines.append("|-------|-------|")
        for cls, count in balance["distribution"].items():
            lines.append(f"| {cls} | {count} |")
        lines.append("")
        ratio = balance["imbalance_ratio"]
        if balance["is_imbalanced"]:
            lines.append(
                f"⚠️ **Imbalance ratio: {ratio:.1f}x** — consider resampling "
                f"(SMOTE) or class weighting.\n"
            )
        else:
            lines.append(f"**Imbalance ratio: {ratio:.1f}x** — within acceptable range ✅\n")

    # High cardinality
    high_card = quality.get("high_cardinality_columns", [])
    if high_card:
        lines.append("### High-Cardinality Categoricals\n")
        for entry in high_card:
            lines.append(f"- `{entry['column']}`: {entry['unique_values']} unique values")
        lines.append("\n> These may cause feature explosion after one-hot encoding.\n")

    # Low variance
    low_var = quality.get("low_variance_features", [])
    if low_var:
        lines.append(f"### Near-Zero Variance Features\n")
        lines.append(f"Features with very low variance: `{'`, `'.join(low_var)}`\n")
        lines.append("> Consider removing these — they contribute minimal signal.\n")

    return lines


# ── Per-class metrics section ────────────────────────────────────────────────

def _build_per_class_section(per_class_df: pd.DataFrame | None) -> list[str]:
    if per_class_df is None or per_class_df.empty:
        return []
    lines = ["### Per-Class Breakdown\n"]
    lines.append(per_class_df.to_markdown(index=False))
    lines.append("")

    # Flag weak classes
    detail_classes = per_class_df[~per_class_df["Class"].isin([
        "accuracy", "macro avg", "weighted avg"
    ])]
    if not detail_classes.empty:
        weak = detail_classes[detail_classes["F1-Score"] < 0.7]
        if not weak.empty:
            labels = ", ".join(weak["Class"].tolist())
            lines.append(
                f"⚠️ Classes with F1 < 0.70: **{labels}** — "
                f"these may need more training samples or feature engineering.\n"
            )

    return lines


# ── Overfitting section ──────────────────────────────────────────────────────

def _build_overfit_section(
    scored_models: list[dict[str, Any]],
) -> list[str]:
    lines = ["## 5. Overfitting Analysis\n"]
    lines.append("| Model | Train Score | Test Score | Gap | Risk |")
    lines.append("|-------|-----------|-----------|-----|------|")

    for entry in scored_models:
        if entry.get("error"):
            continue
        oa = entry.get("overfit_analysis", {})
        if not oa:
            continue
        name = entry["name"]
        risk_emoji = {"low": "🟢", "moderate": "🟡", "high": "🔴"}.get(
            oa.get("risk_level", ""), ""
        )
        lines.append(
            f"| {name} | {oa.get('train_score', 'N/A')} "
            f"| {oa.get('test_score', 'N/A')} "
            f"| {oa.get('gap', 'N/A')} "
            f"| {risk_emoji} {oa.get('risk_level', 'N/A').title()} |"
        )

    lines.append("")

    # Verdict from best model
    for entry in scored_models:
        if entry.get("error"):
            continue
        oa = entry.get("overfit_analysis", {})
        verdict = oa.get("verdict")
        if verdict:
            metric = oa.get("metric_name", "Score")
            lines.append(f"> **{entry['name']}** ({metric}): {verdict}\n")

    return lines


# ── Cross-validation section ────────────────────────────────────────────────

def _build_cv_section(
    scored_models: list[dict[str, Any]],
    task_type: str,
) -> list[str]:
    lines = ["## 6. Cross-Validation Results\n"]

    metric_label = "Weighted F1" if task_type == "classification" else "RMSE"
    lines.append(f"Metric: **{metric_label}** (5-fold CV)\n")
    lines.append("| Model | CV Mean | CV Std | Fold Scores |")
    lines.append("|-------|---------|--------|-------------|")

    for entry in scored_models:
        if entry.get("error"):
            continue
        cv = entry.get("cv_results", {})
        if cv.get("cv_mean") is None:
            continue
        fold_str = ", ".join(f"{s:.4f}" for s in cv.get("cv_scores", []))
        lines.append(
            f"| {entry['name']} | {cv['cv_mean']} "
            f"| ±{cv['cv_std']} "
            f"| {fold_str} |"
        )

    lines.append("")
    lines.append(
        "> Low CV std indicates stable performance across folds. "
        "High std suggests sensitivity to data splits.\n"
    )
    return lines


# ── Smart recommendations ────────────────────────────────────────────────────

def _build_recommendations(
    task_type: str,
    quality: dict[str, Any],
    scored_models: list[dict[str, Any]],
    best_metrics: dict[str, float],
) -> list[str]:
    lines = ["## 9. Recommendations\n"]
    recs: list[str] = []

    # Overfitting-aware recs
    for entry in scored_models:
        if entry.get("error"):
            continue
        oa = entry.get("overfit_analysis", {})
        if oa.get("risk_level") == "high":
            recs.append(
                "Apply stronger regularization (increase C penalty for Logistic "
                "Regression, reduce max_depth for tree models)."
            )
            recs.append("Use cross-validation-based hyperparameter tuning (Optuna, GridSearchCV).")
            break

    # Class imbalance recs
    balance = quality.get("class_balance")
    if balance and balance.get("is_imbalanced"):
        recs.append("Apply SMOTE or class_weight='balanced' to address class imbalance.")

    # Missing value recs
    missing = quality.get("columns_with_missing", {})
    if missing:
        pcts = quality.get("missing_pct", {})
        heavy = [c for c, p in pcts.items() if p > 30]
        if heavy:
            recs.append(
                f"Columns with >30% missing data ({', '.join(heavy)}): "
                f"consider dropping or using domain-specific imputation."
            )

    # High cardinality recs
    if quality.get("high_cardinality_columns"):
        recs.append("Reduce high-cardinality categoricals via target encoding or frequency encoding.")

    # Performance-based recs
    if task_type == "classification":
        f1 = best_metrics.get("weighted_f1", 0)
        if f1 < 0.80:
            recs.append("F1 below 0.80 — consider feature engineering, additional models, or more training data.")
        elif f1 >= 0.95:
            recs.append("Very high F1 — verify no data leakage (target information in features).")
    else:
        r2 = best_metrics.get("r2", 0)
        if r2 < 0.70:
            recs.append("R² below 0.70 — dataset may need additional features or nonlinear transformations.")
        elif r2 >= 0.99:
            recs.append("Near-perfect R² — verify no data leakage or trivially correlated features.")

    # Standard recs
    recs.extend([
        "Run SHAP analysis for deeper, model-agnostic feature explanations.",
        "Apply Optuna hyperparameter tuning on the top 2 models.",
        "Evaluate stacking or blending ensembles for marginal gains.",
        "Test on fully held-out validation data before production deployment.",
    ])

    for i, rec in enumerate(recs, 1):
        lines.append(f"{i}. {rec}")

    lines.append("")
    return lines


# ── Main report generator ───────────────────────────────────────────────────

def generate_report(
    dataset_summary: dict[str, Any],
    task_type: str,
    leaderboard: pd.DataFrame,
    best_model_name: str,
    best_metrics: dict[str, float],
    feature_importance_df: pd.DataFrame | None = None,
    plot_paths: dict[str, str] | None = None,
    data_quality: dict[str, Any] | None = None,
    per_class_df: pd.DataFrame | None = None,
    scored_models: list[dict[str, Any]] | None = None,
    gemini_analysis: str | None = None,
    output_dir: str | Path = "outputs",
) -> str:
    """Generate the full Markdown report.

    Sections:
    1. Dataset Summary
    2. Data Quality Analysis
    3. Model Leaderboard
    4. Best Model Deep-Dive
    5. Overfitting Analysis
    6. Cross-Validation Results
    7. Feature Importance
    8. Diagnostic Plots
    9. Recommendations
    10. AI Analysis (optional)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if plot_paths is None:
        plot_paths = {}
    if scored_models is None:
        scored_models = []
    if data_quality is None:
        data_quality = {}

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    primary_metric = "Weighted F1" if task_type == "classification" else "RMSE"
    primary_value = best_metrics.get(
        "weighted_f1" if task_type == "classification" else "rmse", "N/A"
    )

    s: list[str] = []

    # ── Header ──
    s.append("# AutoML Experiment Report\n")
    s.append(f"**Author:** Sara Musalim  ")
    s.append(f"**Generated:** {timestamp}  ")
    s.append(f"**Task Type:** {task_type.title()}  ")
    s.append(f"**Best Model:** {best_model_name}  ")
    s.append(f"**{primary_metric}:** {primary_value}\n")
    s.append("---\n")

    # ── 1. Dataset Summary ──
    s.append("## 1. Dataset Summary\n")
    ds = dataset_summary
    s.append(f"| Property | Value |")
    s.append(f"|----------|-------|")
    s.append(f"| Rows | {ds.get('rows', 'N/A'):,} |")
    s.append(f"| Columns | {ds.get('columns', 'N/A')} |")
    s.append(f"| Features | {ds.get('features', 'N/A')} |")
    s.append(f"| Target Column | `{ds.get('target', 'N/A')}` |")
    s.append(f"| Total Missing Values | {ds.get('missing_total', 'N/A')} |")

    num_cols = ds.get("numeric_columns", [])
    cat_cols = ds.get("categorical_columns", [])
    s.append(f"| Numeric Features | {len(num_cols)} |")
    s.append(f"| Categorical Features | {len(cat_cols)} |")
    s.append("")

    # Target distribution plot
    s.append(_embed_plot(plot_paths, "target_distribution_path", output_dir, "Target Distribution"))

    # ── 2. Data Quality ──
    if data_quality:
        s.extend(_build_data_quality_section(data_quality))
        s.append(_embed_plot(plot_paths, "missing_values_path", output_dir, "Missing Values"))
        s.append(_embed_plot(plot_paths, "correlation_path", output_dir, "Correlation Heatmap"))

    # ── 3. Leaderboard ──
    s.append("## 3. Model Leaderboard\n")
    if not leaderboard.empty:
        s.append(leaderboard.to_markdown())
    else:
        s.append("_No models were successfully trained._")
    s.append("")
    s.append(_embed_plot(plot_paths, "model_comparison_path", output_dir, "Model Comparison"))

    # ── 4. Best Model ──
    s.append("## 4. Best Model — Deep Dive\n")
    s.append(f"**{best_model_name}** achieved the best {primary_metric} of **{primary_value}**.\n")
    s.append("### All Metrics\n")
    s.append(_metric_table(best_metrics))
    s.append("")

    # Per-class breakdown
    s.extend(_build_per_class_section(per_class_df))

    # ── 5. Overfitting ──
    if scored_models:
        s.extend(_build_overfit_section(scored_models))

    # ── 6. Cross-Validation ──
    if scored_models:
        s.extend(_build_cv_section(scored_models, task_type))

    # ── 7. Feature Importance ──
    s.append("## 7. Feature Importance\n")
    s.append(_feature_table(feature_importance_df))
    s.append("")
    s.append(_embed_plot(plot_paths, "feature_importance_path", output_dir, "Feature Importance"))

    # ── 8. Diagnostic Plots ──
    s.append("## 8. Diagnostic Plots\n")
    if task_type == "classification":
        s.append("### Confusion Matrix\n")
        s.append(_embed_plot(plot_paths, "diagnostic_path", output_dir, "Confusion Matrix"))
        s.append("### ROC Curve\n")
        s.append(_embed_plot(plot_paths, "roc_path", output_dir, "ROC Curve"))
    else:
        s.append("### Predicted vs Actual\n")
        s.append(_embed_plot(plot_paths, "diagnostic_path", output_dir, "Predicted vs Actual"))
        s.append("### Residual Analysis\n")
        s.append(_embed_plot(plot_paths, "residual_path", output_dir, "Residual Distribution"))

    # ── 9. Recommendations ──
    s.extend(_build_recommendations(task_type, data_quality, scored_models, best_metrics))

    # ── 10. Gemini Analysis (optional) ──
    if gemini_analysis:
        s.append("## 10. AI Analysis (Gemini)\n")
        s.append(gemini_analysis)
        s.append("")

    report_content = "\n".join(s)
    report_path = output_dir / "report.md"
    report_path.write_text(report_content, encoding="utf-8")

    return str(report_path.resolve())


# ── File serializers ─────────────────────────────────────────────────────────

def save_leaderboard_csv(
    leaderboard: pd.DataFrame,
    output_dir: str | Path = "outputs",
) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "leaderboard.csv"
    leaderboard.to_csv(path, index=True)
    return str(path.resolve())


def save_best_model(
    model: Any,
    output_dir: str | Path = "outputs",
) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "best_model.joblib"
    joblib.dump(model, path)
    return str(path.resolve())
