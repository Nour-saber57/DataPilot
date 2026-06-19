## Getters for chatbot integration
from Backend.model_registry import evaluate_classification, evaluate_regression, get_all_models, identify_task
from Backend.preprocessor import identify_feature_types


df = 
target =
y_true =
y_pred =

numeric_columns, categorical_columns = identify_feature_types(df, target)

task_type = identify_task(df, target)

best_estimators, best_params = get_all_models(task_type)

classification_metrics = evaluate_classification(y_true, y_pred)

regression_metrics = evaluate_regression(y_true, y_pred)
