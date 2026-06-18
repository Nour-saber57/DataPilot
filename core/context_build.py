def build_context(
    profile,
    health_score,
    task_type,
    strengths,
    weaknesses,
    models,
    best_model,
    results
):

    context = {
        "dataset_profile": profile or {},
        "health_score": float(health_score) if health_score is not None else 0,
        "task_type": task_type or "Unknown",
        "strengths": strengths or [],
        "weaknesses": weaknesses or [],
        "recommended_models": models or [],
        "best_model": best_model or "Not determined",
        "training_results": results or {}
    }

    return context


## dummy data for testing
context = build_context(
    profile={"rows": 1000, "columns": 12},
    health_score=78,
    task_type="Regression",
    strengths=["Low missing values"],
    weaknesses=["Outliers in target"],
    models=["Random Forest", "XGBoost"],
    best_model="Random Forest",
    results={"RF": 0.86, "XGB": 0.84}
)