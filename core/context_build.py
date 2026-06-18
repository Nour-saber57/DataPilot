def build_context(
    dataset_profile,
    health_score,
    strengths,
    weaknesses,
    recommended_models,
    best_model,
    training_results
):
    return {
        "dataset_profile": dataset_profile,
        "health_score": health_score,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "recommended_models": recommended_models,
        "best_model": best_model,
        "training_results": training_results
    }
    