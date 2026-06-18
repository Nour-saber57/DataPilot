# MODEL REGISTRY WITH GRIDSEARCHCV - QUICK START GUIDE

## Overview
A comprehensive model registry implementing 14+ machine learning models with automatic hyperparameter tuning using GridSearchCV.

## Features
- ✅ **Automatic Model Selection**: Classification or Regression detection
- ✅ **GridSearchCV**: Built-in hyperparameter tuning for all models
- ✅ **Individual Functions**: Each model has its own tuning function
- ✅ **Unified Interface**: Simple API to train any model
- ✅ **Metrics**: Pre-built evaluation functions

---

## Available Models

### Classification (7 models)
```python
from model_registry import get_classification_models

models = get_classification_models()
# Available: 
# - logistic_regression
# - random_forest
# - svm
# - knn
# - decision_tree
# - gradient_boosting
# - naive_bayes
```

### Regression (7 models)
```python
from model_registry import get_regression_models

models = get_regression_models()
# Available:
# - linear_regression
# - ridge
# - lasso
# - random_forest
# - svm
# - gradient_boosting
# - knn
```

---

## Usage Examples

### Basic Training (Classification)
```python
from model_registry import train_model, evaluate_classification

# Train a model with GridSearchCV
best_model, best_params = train_model('random_forest', 'classification', X_train, y_train)

print(f"Best Parameters: {best_params}")

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate
metrics = evaluate_classification(y_test, y_pred)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1-Score: {metrics['f1']:.4f}")
```

### Basic Training (Regression)
```python
from model_registry import train_model, evaluate_regression

# Train a model with GridSearchCV
best_model, best_params = train_model('gradient_boosting', 'regression', X_train, y_train)

print(f"Best Parameters: {best_params}")

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate
metrics = evaluate_regression(y_test, y_pred)
print(f"R² Score: {metrics['r2']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")
```

### Train All Models
```python
from model_registry import get_all_models, train_model, evaluate_classification

# Get all classification models
models = get_all_models('classification')

results = {}
for model_name in models.keys():
    best_model, best_params = train_model(model_name, 'classification', X_train, y_train)
    y_pred = best_model.predict(X_test)
    metrics = evaluate_classification(y_test, y_pred)
    results[model_name] = metrics
    print(f"{model_name}: F1={metrics['f1']:.4f}")
```

---

## Hyperparameter Grids

Each model has custom hyperparameters optimized for performance:

### Logistic Regression
- C: [0.001, 0.01, 0.1, 1, 10, 100]
- penalty: ['l1', 'l2']
- solver: ['liblinear', 'saga']
- max_iter: [100, 200, 500]

### Random Forest (Classification)
- n_estimators: [50, 100, 200]
- max_depth: [5, 10, 20, None]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]
- max_features: ['sqrt', 'log2']

### Gradient Boosting
- n_estimators: [50, 100, 200]
- learning_rate: [0.01, 0.05, 0.1]
- max_depth: [3, 5, 7]
- min_samples_split: [2, 5, 10]
- subsample: [0.8, 0.9, 1.0]

### SVM (Classification)
- C: [0.1, 1, 10, 100]
- kernel: ['linear', 'rbf', 'poly']
- gamma: ['scale', 'auto']
- degree: [2, 3, 4]

*See model_registry.py for complete grids*

---

## Evaluation Metrics

### Classification
```python
metrics = evaluate_classification(y_true, y_pred)
# Returns: accuracy, precision, recall, f1
```

### Regression
```python
metrics = evaluate_regression(y_true, y_pred)
# Returns: mse, rmse, mae, r2
```

---

## API Reference

### Functions

#### `identify_task(df, target_column)`
Automatically detect if task is classification or regression

#### `get_all_models(task)`
Get dictionary of all models for given task

#### `train_model(model_name, task, X_train, y_train)`
Train a model with GridSearchCV
- Returns: (best_model, best_params)

#### `evaluate_classification(y_true, y_pred)`
Evaluate classification model

#### `evaluate_regression(y_true, y_pred)`
Evaluate regression model

---

## Integration with Preprocessing

Works seamlessly with preprocessor.py:

```python
from preprocessor import (
    identify_feature_types, 
    split_data, 
    create_preprocessing_pipeline
)
from model_registry import train_model, evaluate_classification

# Load and preprocess data
X_train, X_test, y_train, y_test = split_data(df, target)
pipeline = create_preprocessing_pipeline(df, target)
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)

# Train model
best_model, params = train_model('random_forest', 'classification', 
                                  X_train_processed, y_train)
y_pred = best_model.predict(X_test_processed)

# Evaluate
metrics = evaluate_classification(y_test, y_pred)
```

---

## Files
- `model_registry.py` - Main module with all models and GridSearchCV
- `test_models.py` - Comprehensive test suite
- `preprocessor.py` - Data preprocessing pipeline

## Next Steps
1. Use `train_model()` to train models on your data
2. Run `test_models.py` to benchmark all models
3. Integrate with your FastAPI/Streamlit application
4. Deploy best performing model
