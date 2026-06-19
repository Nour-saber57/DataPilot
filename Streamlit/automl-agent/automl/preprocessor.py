"""
Data preprocessing pipeline for tabular datasets.

Handles missing value imputation, numeric scaling, and categorical encoding.
Produces train/test splits ready for model consumption.

Design decisions:
- SimpleImputer with median for numerics (robust to outliers vs mean).
- StandardScaler chosen over MinMaxScaler for better gradient-based model performance.
- OneHotEncoder with handle_unknown='ignore' to survive unseen categories at inference.
- 80/20 split with stratification for classification tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreprocessedData:
    """Container for preprocessed train/test arrays and metadata."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    transformer: ColumnTransformer
    dataset_summary: dict[str, Any] = field(default_factory=dict)


def detect_task_type(y: pd.Series) -> str:
    """Infer whether the target is classification or regression.

    Heuristic: if dtype is object/category or the ratio of unique values
    to total values is below 5%, treat as classification.
    """
    if y.dtype == "object" or pd.api.types.is_categorical_dtype(y):
        return "classification"
    unique_ratio = y.nunique() / len(y)
    if unique_ratio < 0.05 or y.nunique() <= 20:
        return "classification"
    return "regression"


def build_dataset_summary(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    """Produce a human-readable dataset summary dict for downstream use."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "features": len(df.columns) - 1,
        "target": target_col,
        "missing_total": int(df.isnull().sum().sum()),
        "missing_per_column": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist(),
    }


def preprocess(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreprocessedData:
    """Run the full preprocessing pipeline.

    Steps:
    1. Separate features / target.
    2. Identify numeric vs categorical columns.
    3. Build a ColumnTransformer (impute → scale / encode).
    4. Fit on train, transform both splits.
    """
    df = df.copy()
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Drop columns with all missing values — they carry no signal.
    all_missing = X.columns[X.isnull().all()].tolist()
    if all_missing:
        X = X.drop(columns=all_missing)

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Build transformers
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    # Encode target for classification if it's categorical
    if task_type == "classification" and y.dtype == "object":
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target_col)

    stratify = y if task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    X_train_transformed = column_transformer.fit_transform(X_train)
    X_test_transformed = column_transformer.transform(X_test)

    # Reconstruct feature names post-transform
    feature_names = numeric_cols.copy()
    if categorical_cols:
        encoder = column_transformer.named_transformers_["cat"].named_steps["encoder"]
        cat_feature_names = encoder.get_feature_names_out(categorical_cols).tolist()
        feature_names.extend(cat_feature_names)

    dataset_summary = build_dataset_summary(df, target_col)

    return PreprocessedData(
        X_train=X_train_transformed,
        X_test=X_test_transformed,
        y_train=y_train.values,
        y_test=y_test.values,
        feature_names=feature_names,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        transformer=column_transformer,
        dataset_summary=dataset_summary,
    )
