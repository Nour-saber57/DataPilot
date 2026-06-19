"""
Orchestrator — single entry-point that wires preprocessing → training →
evaluation → explainability → report.

Design rationale:
- Returns an AutoMLResult dataclass so every downstream consumer (Streamlit,
  Gemini prompt builder, reporter) has a single typed contract to depend on.
- The orchestrator doesn't import Streamlit, keeping it testable as a pure
  backend function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from automl.evaluator import (
    build_leaderboard,
    evaluate_models,
    profile_data_quality,
    select_best_model,
)
from automl.explainer import generate_plots
from automl.preprocessor import PreprocessedData, detect_task_type, preprocess
from automl.reporter import (
    generate_report,
    save_best_model,
    save_leaderboard_csv,
)
from automl.trainer import train_models


@dataclass
class AutoMLResult:
    """Immutable result container for the entire pipeline run."""

    task_type: str
    dataset_summary: dict[str, Any]
    data_quality: dict[str, Any]
    leaderboard: pd.DataFrame
    best_model_name: str
    best_model: Any
    best_metrics: dict[str, float]
    best_overfit: dict[str, Any]
    best_cv: dict[str, Any]
    per_class_df: pd.DataFrame | None
    feature_importance_df: pd.DataFrame | None
    plot_paths: dict[str, str]
    plot_figures: dict[str, Any]
    model_path: str
    leaderboard_path: str
    report_path: str
    scored_models: list[dict[str, Any]] = field(default_factory=list)


def run_automl(
    df: pd.DataFrame,
    target_col: str,
    task_type: str | None = None,
    output_dir: str = "outputs",
) -> AutoMLResult:
    """Execute the full AutoML pipeline.

    Args:
        df: Raw user-uploaded DataFrame.
        target_col: Name of the target column.
        task_type: 'classification' or 'regression'. Auto-detected if None.
        output_dir: Base directory for saved artifacts.

    Returns:
        AutoMLResult with all metrics, plots, and file paths.

    Raises:
        ValueError: If no models could be trained.
    """
    # 1. Task detection
    if task_type is None:
        task_type = detect_task_type(df[target_col])

    # 2. Data quality profiling
    quality = profile_data_quality(df, target_col)

    # 3. Preprocess
    data: PreprocessedData = preprocess(df, target_col, task_type)

    # 4. Train
    trained = train_models(task_type, data.X_train, data.y_train)

    # 5. Evaluate (now includes CV and overfitting detection)
    scored = evaluate_models(
        trained, data.X_train, data.y_train, data.X_test, data.y_test, task_type
    )
    leaderboard = build_leaderboard(scored, task_type)
    best = select_best_model(scored, task_type)

    if best is None:
        raise ValueError(
            "All models failed to train. Check dataset quality and preprocessing logs."
        )

    # 6. Explainability (now includes EDA and model comparison plots)
    plots_dir = str(Path(output_dir) / "plots")
    plot_artifacts = generate_plots(
        df=df,
        target_col=target_col,
        best_model_entry=best,
        scored_models=scored,
        X_test=data.X_test,
        y_test=data.y_test,
        feature_names=data.feature_names,
        task_type=task_type,
        output_dir=plots_dir,
    )

    # 7. Save model and leaderboard
    model_path = save_best_model(best["model"], output_dir)
    lb_path = save_leaderboard_csv(leaderboard, output_dir)

    # 8. Collect all plot paths for the report
    all_plot_paths = {
        k: v for k, v in plot_artifacts.items() if k.endswith("_path")
    }

    # 9. Generate report
    report_path = generate_report(
        dataset_summary=data.dataset_summary,
        task_type=task_type,
        leaderboard=leaderboard,
        best_model_name=best["name"],
        best_metrics=best["metrics"],
        feature_importance_df=plot_artifacts["feature_importance_df"],
        plot_paths=all_plot_paths,
        data_quality=quality,
        per_class_df=best.get("per_class"),
        scored_models=scored,
        output_dir=output_dir,
    )

    # Collect all figure objects for Streamlit
    all_figures = {
        k.replace("_fig", ""): v
        for k, v in plot_artifacts.items()
        if k.endswith("_fig") and v is not None
    }

    return AutoMLResult(
        task_type=task_type,
        dataset_summary=data.dataset_summary,
        data_quality=quality,
        leaderboard=leaderboard,
        best_model_name=best["name"],
        best_model=best["model"],
        best_metrics=best["metrics"],
        best_overfit=best.get("overfit_analysis", {}),
        best_cv=best.get("cv_results", {}),
        per_class_df=best.get("per_class"),
        feature_importance_df=plot_artifacts["feature_importance_df"],
        plot_paths=all_plot_paths,
        plot_figures=all_figures,
        model_path=model_path,
        leaderboard_path=lb_path,
        report_path=report_path,
        scored_models=scored,
    )
