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

    return {
        "dataset_profile": profile,
        "health_score": health_score,
        "task_type": task_type,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "recommended_models": models,
        "best_model": best_model,
        "training_results": results
    }