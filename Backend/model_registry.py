import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score, mean_absolute_error


def identify_task(df, target_column):
    if df[target_column].dtype == 'object':
        if len(df[target_column].unique()) <= 20:
            return "classification"
        else:
            return "regression"
    elif df[target_column].dtype in [np.float64, np.int64]:
        if len(df[target_column].unique()) <= 20:
            return "classification"
        else:
            return "regression"
    else:
        return "regression"


# ==================== CLASSIFICATION MODELS ====================

def logistic_regression_model(X_train, y_train):
    """Logistic Regression with GridSearchCV"""
    model = LogisticRegression()
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [100, 200, 500]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def random_forest_classifier_model(X_train, y_train):
    """Random Forest Classifier with GridSearchCV"""
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def svm_classifier_model(X_train, y_train):
    """Support Vector Machine Classifier with GridSearchCV"""
    model = SVC()
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto'],
        'degree': [2, 3, 4]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def knn_classifier_model(X_train, y_train):
    """K-Nearest Neighbors Classifier with GridSearchCV"""
    model = KNeighborsClassifier()
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def decision_tree_classifier_model(X_train, y_train):
    """Decision Tree Classifier with GridSearchCV"""
    model = DecisionTreeClassifier(random_state=42)
    param_grid = {
        'max_depth': [3, 5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def gradient_boosting_classifier_model(X_train, y_train):
    """Gradient Boosting Classifier with GridSearchCV"""
    model = GradientBoostingClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.8, 0.9, 1.0]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def naive_bayes_model(X_train, y_train):
    """Naive Bayes Classifier with GridSearchCV"""
    model = GaussianNB()
    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


# ==================== REGRESSION MODELS ====================

def linear_regression_model(X_train, y_train):
    """Linear Regression with GridSearchCV"""
    model = LinearRegression()
    param_grid = {
        'fit_intercept': [True, False]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def ridge_regression_model(X_train, y_train):
    """Ridge Regression with GridSearchCV"""
    model = Ridge()
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'saga']
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def lasso_regression_model(X_train, y_train):
    """Lasso Regression with GridSearchCV"""
    model = Lasso(random_state=42)
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1, 10],
        'max_iter': [1000, 5000, 10000]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def random_forest_regressor_model(X_train, y_train):
    """Random Forest Regressor with GridSearchCV"""
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def svm_regressor_model(X_train, y_train):
    """Support Vector Machine Regressor with GridSearchCV"""
    model = SVR()
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto'],
        'epsilon': [0.01, 0.1, 0.2]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def gradient_boosting_regressor_model(X_train, y_train):
    """Gradient Boosting Regressor with GridSearchCV"""
    model = GradientBoostingRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.8, 0.9, 1.0]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def knn_regressor_model(X_train, y_train):
    """K-Nearest Neighbors Regressor with GridSearchCV"""
    model = KNeighborsRegressor()
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


# ==================== MODEL SELECTION ====================

def get_classification_models():
    """Returns dictionary of all classification models"""
    return {
        'logistic_regression': logistic_regression_model,
        'random_forest': random_forest_classifier_model,
        'svm': svm_classifier_model,
        'knn': knn_classifier_model,
        'decision_tree': decision_tree_classifier_model,
        'gradient_boosting': gradient_boosting_classifier_model,
        'naive_bayes': naive_bayes_model
    }


def get_regression_models():
    """Returns dictionary of all regression models"""
    return {
        'linear_regression': linear_regression_model,
        'ridge': ridge_regression_model,
        'lasso': lasso_regression_model,
        'random_forest': random_forest_regressor_model,
        'svm': svm_regressor_model,
        'gradient_boosting': gradient_boosting_regressor_model,
        'knn': knn_regressor_model
    }


def get_all_models(task):
    """Get all models for the specified task"""
    if task == "classification":
        return get_classification_models()
    elif task == "regression":
        return get_regression_models()
    else:
        raise ValueError(f"Unknown task: {task}")


def train_model(model_name, task, X_train, y_train):
    """Train a specific model with GridSearchCV"""
    models = get_all_models(task)
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")
    
    model_func = models[model_name]
    best_model, best_params = model_func(X_train, y_train)
    
    return best_model, best_params


# ==================== EVALUATION ====================

def evaluate_classification(y_true, y_pred):
    """Evaluate classification model"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }


def evaluate_regression(y_true, y_pred):
    """Evaluate regression model"""
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
# ================ Add getters ================
### Getters
def get_models(models: list) -> list:
    return models
def get_best_model(best_model: str) -> str:
    return best_model
def get_results(results: dict) -> dict:
    return results
def get_task_type(task_type: str) -> str:
    return task_type
def get_health_score(score: float) -> float:
    return score