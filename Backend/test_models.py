import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from model_registry import (
    identify_task, get_classification_models, get_regression_models,
    train_model, evaluate_classification, evaluate_regression
)

print("=" * 70)
print("TESTING ALL MODELS WITH GRIDSEARCHCV")
print("=" * 70)

# ==================== CLASSIFICATION TEST ====================
print("\n" + "=" * 70)
print("CLASSIFICATION TASK - Testing All Models")
print("=" * 70)

# Create sample classification data
X_class, y_class = make_classification(n_samples=200, n_features=20, n_informative=10, 
                                        n_classes=3, random_state=42)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

classification_models = get_classification_models()
print(f"\nAvailable Classification Models: {list(classification_models.keys())}")

results_class = {}

for model_name in classification_models.keys():
    print(f"\n{'─' * 70}")
    print(f"Training: {model_name.upper()}")
    print(f"{'─' * 70}")
    
    try:
        # Train model with GridSearchCV
        best_model, best_params = train_model(model_name, "classification", X_train_class, y_train_class)
        
        # Make predictions
        y_pred = best_model.predict(X_test_class)
        
        # Evaluate
        metrics = evaluate_classification(y_test_class, y_pred)
        results_class[model_name] = metrics
        
        print(f"✓ {model_name} trained successfully!")
        print(f"  Best Parameters: {best_params}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        
    except Exception as e:
        print(f"✗ Error training {model_name}: {str(e)}")


# ==================== REGRESSION TEST ====================
print("\n\n" + "=" * 70)
print("REGRESSION TASK - Testing All Models")
print("=" * 70)

# Create sample regression data
X_reg, y_reg = make_regression(n_samples=200, n_features=20, n_informative=10, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

regression_models = get_regression_models()
print(f"\nAvailable Regression Models: {list(regression_models.keys())}")

results_reg = {}

for model_name in regression_models.keys():
    print(f"\n{'─' * 70}")
    print(f"Training: {model_name.upper()}")
    print(f"{'─' * 70}")
    
    try:
        # Train model with GridSearchCV
        best_model, best_params = train_model(model_name, "regression", X_train_reg, y_train_reg)
        
        # Make predictions
        y_pred = best_model.predict(X_test_reg)
        
        # Evaluate
        metrics = evaluate_regression(y_test_reg, y_pred)
        results_reg[model_name] = metrics
        
        print(f"✓ {model_name} trained successfully!")
        print(f"  Best Parameters: {best_params}")
        print(f"  MSE:  {metrics['mse']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  R²:   {metrics['r2']:.4f}")
        
    except Exception as e:
        print(f"✗ Error training {model_name}: {str(e)}")


# ==================== SUMMARY ====================
print("\n\n" + "=" * 70)
print("SUMMARY - Best Performers")
print("=" * 70)

print("\nCLASSIFICATION RESULTS (sorted by F1-Score):")
print("─" * 70)
if results_class:
    sorted_class = sorted(results_class.items(), key=lambda x: x[1]['f1'], reverse=True)
    for i, (model_name, metrics) in enumerate(sorted_class, 1):
        print(f"{i}. {model_name:20s} - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

print("\nREGRESSION RESULTS (sorted by R²):")
print("─" * 70)
if results_reg:
    sorted_reg = sorted(results_reg.items(), key=lambda x: x[1]['r2'], reverse=True)
    for i, (model_name, metrics) in enumerate(sorted_reg, 1):
        print(f"{i}. {model_name:20s} - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")

print("\n" + "=" * 70)
print("✓ All tests completed successfully!")
print("=" * 70)

### Getters
from trainer import (
    get_models,
    get_best_model,
    get_results,
    get_task_type,
    get_health_score
)

print("=" * 50)
print("TESTING TRAINER GETTERS")
print("=" * 50)

models = ["Logistic Regression", "Random Forest", "Decision Tree"]
best_model = "Random Forest"
results = {
    "Random Forest": {
        "accuracy": 0.95,
        "precision": 0.94
    }
}
task_type = "Classification"
health_score = 92.5

print("\nTesting get_models()")
print(get_models(models))

print("\nTesting get_best_model()")
print(get_best_model(best_model))

print("\nTesting get_results()")
print(get_results(results))

print("\nTesting get_task_type()")
print(get_task_type(task_type))

print("\nTesting get_health_score()")
print(get_health_score(health_score))

print("\nAll tests completed successfully.")