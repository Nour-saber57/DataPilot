"""
Advanced ML Analytics & Diagnostics
Extracts insights from trained models and data for comprehensive analysis
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance


def extract_feature_importance(model, X_test, y_test, feature_names, task_type):
    """
    Extract feature importance from a trained model
    
    Returns: DataFrame with feature importance scores
    """
    importance_dict = {}
    
    # Try to get feature importances (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        importance_dict = dict(zip(feature_names, model.feature_importances_))
    # Try permutation importance as fallback
    elif hasattr(model, 'predict'):
        try:
            perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
            importance_dict = dict(zip(feature_names, perm_importance.importances_mean))
        except:
            importance_dict = {name: 0.0 for name in feature_names}
    else:
        importance_dict = {name: 0.0 for name in feature_names}
    
    # Convert to DataFrame sorted by importance
    importance_df = pd.DataFrame([
        {"Feature": name, "Importance": score}
        for name, score in importance_dict.items()
    ]).sort_values("Importance", ascending=False)
    
    return importance_df


def calculate_overfitting_metrics(model, X_train, X_test, y_train, y_test, task_type):
    """
    Calculate overfitting analysis: train vs test performance gap
    
    Returns: Dictionary with overfitting risk level and metrics
    """
    if task_type == "classification":
        from sklearn.metrics import f1_score
        train_score = f1_score(y_train, model.predict(X_train), average='weighted', zero_division=0)
        test_score = f1_score(y_test, model.predict(X_test), average='weighted', zero_division=0)
        metric_name = "F1 Score"
    else:
        from sklearn.metrics import r2_score
        train_score = r2_score(y_train, model.predict(X_train))
        test_score = r2_score(y_test, model.predict(X_test))
        metric_name = "R² Score"
    
    gap = abs(train_score - test_score)
    
    # Determine risk level
    if gap < 0.05:
        risk_level = "low"
    elif gap < 0.15:
        risk_level = "moderate"
    else:
        risk_level = "high"
    
    verdict = ""
    if train_score > test_score and gap > 0.1:
        verdict = "⚠️ Model is likely overfitting. Consider regularization or reducing complexity."
    elif test_score > train_score:
        verdict = "✓ Model generalizes well. Test performance exceeds training performance."
    else:
        verdict = "✓ Model shows balanced performance across train and test sets."
    
    return {
        "train_score": round(train_score, 4),
        "test_score": round(test_score, 4),
        "gap": round(gap, 4),
        "risk_level": risk_level,
        "metric_name": metric_name,
        "verdict": verdict,
    }


def calculate_cv_metrics(model, X, y, task_type, cv=5):
    """
    Calculate cross-validation metrics
    
    Returns: Dictionary with CV mean, std, and fold scores
    """
    if task_type == "classification":
        scoring = "f1_weighted"
    else:
        scoring = "r2"
    
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    return {
        "cv_mean": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
        "cv_scores": [round(float(s), 4) for s in cv_scores],
    }


def analyze_per_class_metrics(y_true, y_pred, labels=None):
    """
    Generate per-class classification metrics
    
    Returns: DataFrame with per-class precision, recall, f1-score, support
    """
    try:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Extract per-class metrics (exclude 'accuracy', 'macro avg', 'weighted avg')
        per_class_data = []
        for label, metrics in report.items():
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                per_class_data.append({
                    "Class": str(label),
                    "Precision": round(metrics['precision'], 4),
                    "Recall": round(metrics['recall'], 4),
                    "F1-Score": round(metrics['f1-score'], 4),
                    "Support": int(metrics['support']),
                })
        
        return pd.DataFrame(per_class_data) if per_class_data else None
    except:
        return None


def get_confusion_matrix_data(y_true, y_pred):
    """
    Get confusion matrix as DataFrame for visualization
    
    Returns: Confusion matrix as pandas DataFrame
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        cm_df = pd.DataFrame(
            cm,
            index=[f"True {label}" for label in unique_labels],
            columns=[f"Pred {label}" for label in unique_labels],
        )
        return cm_df
    except:
        return None
