"""
Visualization Module
Create plots for EDA, model diagnostics, and analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_target_distribution(df, target_column, task_type):
    """
    Plot target variable distribution
    
    Returns: matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if task_type == "classification":
        counts = df[target_column].value_counts()
        colors = sns.color_palette("husl", len(counts))
        counts.plot(kind='bar', ax=ax, color=colors)
        ax.set_title(f"Target Distribution: {target_column}", fontsize=14, fontweight='bold')
        ax.set_ylabel("Count", fontsize=12)
        ax.set_xlabel("Class", fontsize=12)
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.hist(df[target_column], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_title(f"Target Distribution: {target_column}", fontsize=14, fontweight='bold')
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_xlabel(target_column, fontsize=12)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_correlation_matrix(df, target_column):
    """
    Plot correlation heatmap for numeric features
    
    Returns: matplotlib figure
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    corr = numeric_df.corr()
    
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_missing_values(df, target_column):
    """
    Plot missing value distribution
    
    Returns: matplotlib figure or None if no missing values
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 5))
    missing.plot(kind='barh', ax=ax, color='coral')
    ax.set_title("Missing Values per Feature", fontsize=14, fontweight='bold')
    ax.set_xlabel("Count", fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_feature_importance(importance_df, top_n=15):
    """
    Plot feature importance
    
    Args:
        importance_df: DataFrame with 'Feature' and 'Importance' columns
        top_n: Number of top features to display
    
    Returns: matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    top_features = importance_df.head(top_n).sort_values('Importance', ascending=True)
    colors = sns.color_palette("viridis", len(top_features))
    
    ax.barh(top_features['Feature'], top_features['Importance'], color=colors)
    ax.set_title(f"Top {top_n} Feature Importance", fontsize=14, fontweight='bold')
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred):
    """
    Plot confusion matrix heatmap for classification
    
    Returns: matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=unique_labels, yticklabels=unique_labels,
                cbar_kws={"shrink": 0.8})
    
    ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_pred_proba, classes=None):
    """
    Plot ROC curve for binary/multiclass classification
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        classes: List of class names (optional)
    
    Returns: matplotlib figure
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Handle binary classification
        if len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
        else:
            # For multiclass, plot one-vs-rest ROC curves
            unique_classes = np.unique(y_true)
            colors = sns.color_palette("husl", len(unique_classes))
            
            for i, class_label in enumerate(unique_classes):
                y_true_binary = (y_true == class_label).astype(int)
                fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                
                class_name = classes[i] if classes else f"Class {class_label}"
                ax.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.3f})',
                       color=colors[i])
            
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error plotting ROC curve: {e}")
        return None


def plot_regression_diagnostics(y_true, y_pred):
    """
    Plot regression diagnostic plots: predicted vs actual, residuals
    
    Returns: matplotlib figure with subplots
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Predicted vs Actual
    axes[0].scatter(y_true, y_pred, alpha=0.6, s=50, color='steelblue')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual', fontsize=12)
    axes[0].set_ylabel('Predicted', fontsize=12)
    axes[0].set_title('Predicted vs Actual', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Residuals
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, s=50, color='steelblue')
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Values', fontsize=12)
    axes[1].set_ylabel('Residuals', fontsize=12)
    axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_model_comparison(leaderboard):
    """
    Plot comparison of model performances
    
    Args:
        leaderboard: List of dicts with 'model' and 'score' keys
    
    Returns: matplotlib figure
    """
    df = pd.DataFrame(leaderboard).sort_values('score', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(df))
    
    ax.barh(df['model'], df['score'], color=colors)
    ax.set_xlabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add score labels on bars
    for i, (model, score) in enumerate(zip(df['model'], df['score'])):
        ax.text(score + 0.01, i, f'{score:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig
