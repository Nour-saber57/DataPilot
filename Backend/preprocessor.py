import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


def identify_feature_types(df, target):
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    if target in numeric_columns:
        numeric_columns.remove(target)
    if target in categorical_columns:
        categorical_columns.remove(target)

    return numeric_columns, categorical_columns

def split_data(df, target, test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def create_preprocessing_pipeline(df, target):
    numeric_columns, categorical_columns = identify_feature_types(df, target)

    
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', StandardScaler()),
    ])

    
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', add_indicator=True)),  
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ])

    pipeline = ColumnTransformer(transformers=[
        ('numeric', numeric_pipeline, numeric_columns),
        ('categorical', categorical_pipeline, categorical_columns),
    ])

    return pipeline 

