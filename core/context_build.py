from Backend.model_registry import evaluate_classification, evaluate_regression, get_all_models, identify_task
from Backend.preprocessor import identify_feature_types

def build_context(df, target, y_true, y_pred):
    numeric_columns, categorical_columns = identify_feature_types(df, target)

    task_type = identify_task(df, target)

    # Model information
    best_estimators, best_params = get_all_models(task_type)

    estimator_names = {
        model_name: estimator.__class__.__name__
        for model_name, estimator in best_estimators.items()
    }

    if task_type == "classification":
        metrics = evaluate_classification(y_true, y_pred)
    else:
        metrics = evaluate_regression(y_true, y_pred)

    context = {
        "dataset_profile": {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "target_column": target,

            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,

            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum())
        },

        "task_type": task_type,

        "model_recommendations": {
            "available_models": list(best_estimators.keys()),
            "best_estimators": estimator_names,
            "best_params": best_params
        },

        "evaluation_metrics": metrics
    }

    return context