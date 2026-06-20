"""
Data Profiling Module
Analyze data quality, class balance, missing values, cardinality
"""

import pandas as pd
import numpy as np


def profile_dataset(df, target_column, task_type):
    """
    Generate comprehensive data quality profile
    
    Args:
        df: Input DataFrame
        target_column: Target column name
        task_type: 'classification' or 'regression'
    
    Returns: Dictionary with data quality metrics
    """
    
    profile = {}
    
    # Basic info
    profile['rows'] = len(df)
    profile['columns'] = len(df.columns)
    profile['features'] = len(df.columns) - 1
    
    # Missing values
    missing_count = df.isnull().sum().sum()
    profile['missing_values'] = int(missing_count)
    profile['missing_percentage'] = round((missing_count / (len(df) * len(df.columns))) * 100, 2)
    
    # Duplicates
    duplicate_rows = df.duplicated().sum()
    profile['duplicate_rows'] = int(duplicate_rows)
    
    # Data types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    profile['numeric_features'] = len(numeric_cols)
    profile['categorical_features'] = len(categorical_cols)
    
    # High cardinality check
    high_cardinality = []
    for col in categorical_cols:
        if col != target_column:
            unique_count = df[col].nunique()
            cardinality_ratio = unique_count / len(df)
            if cardinality_ratio > 0.5:  # More than 50% unique values
                high_cardinality.append({
                    "column": col,
                    "unique_values": unique_count,
                    "cardinality_ratio": round(cardinality_ratio, 2)
                })
    profile['high_cardinality_columns'] = high_cardinality
    
    # Class balance (for classification)
    if task_type == "classification":
        class_distribution = df[target_column].value_counts()
        profile['class_distribution'] = class_distribution.to_dict()
        
        # Check for imbalance
        class_counts = class_distribution.values
        imbalance_ratio = max(class_counts) / min(class_counts) if len(class_counts) > 1 else 1.0
        
        is_imbalanced = imbalance_ratio > 3.0  # More than 3:1 ratio
        profile['class_balance'] = {
            "is_imbalanced": is_imbalanced,
            "imbalance_ratio": round(imbalance_ratio, 2),
            "min_class_samples": int(min(class_counts)),
            "max_class_samples": int(max(class_counts)),
        }
    
    return profile


def get_data_warnings(profile, task_type):
    """
    Generate list of data quality warnings
    
    Returns: List of warning strings
    """
    warnings = []
    
    if profile.get('duplicate_rows', 0) > 0:
        warnings.append(f"⚠️ {profile['duplicate_rows']} duplicate rows found")
    
    if profile.get('missing_values', 0) > 0:
        warnings.append(f"⚠️ {profile['missing_values']} missing values ({profile['missing_percentage']}% of data)")
    
    if task_type == "classification":
        class_balance = profile.get('class_balance', {})
        if class_balance.get('is_imbalanced'):
            ratio = class_balance.get('imbalance_ratio', 1.0)
            warnings.append(f"⚠️ Class imbalance detected (ratio: {ratio:.1f}x)")
    
    if profile.get('high_cardinality_columns'):
        n = len(profile['high_cardinality_columns'])
        warnings.append(f"⚠️ {n} high-cardinality categorical column(s)")
    
    return warnings


def get_numeric_summary(df):
    """
    Get summary statistics for numeric features
    
    Returns: DataFrame with summary statistics
    """
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.describe().T


def get_categorical_summary(df):
    """
    Get summary for categorical features
    
    Returns: Dictionary with categorical feature info
    """
    categorical_df = df.select_dtypes(include=['object'])
    summary = {}
    
    for col in categorical_df.columns:
        summary[col] = {
            "unique_values": categorical_df[col].nunique(),
            "top_value": categorical_df[col].value_counts().index[0],
            "top_value_count": int(categorical_df[col].value_counts().values[0]),
            "missing_count": categorical_df[col].isnull().sum(),
        }
    
    return summary
