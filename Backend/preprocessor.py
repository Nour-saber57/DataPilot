import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

### 1. Clean Column Names
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Removes leading/trailing whitespaces and replaces internal spaces with underscores."""
    df_cleaned = df.copy()
    df_cleaned.columns = df_cleaned.columns.str.strip().str.replace(' ', '_')
    return df_cleaned

### 2. Data Validation
def validate_data(df: pd.DataFrame, target: str, min_rows: int = 5) -> None:
    """Validates dataframe existence, target column presence, and checks for minimum row requirements."""
    if df is None or df.empty:
        raise ValueError("The DataFrame is empty or None!")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in the DataFrame!")
    if len(df) < min_rows:
        raise ValueError(f"Insufficient data ({len(df)} rows). A minimum of {min_rows} rows is required.")

### 3. Optional Outlier Handling
def handle_outliers_iqr(df: pd.DataFrame, numeric_cols: list, factor: float = 1.5) -> pd.DataFrame:
    """Optional function to handle outliers using the IQR method, skipping constant columns (IQR == 0)."""
    df_out = df.copy()
    for col in numeric_cols:
        if col in df_out.columns:
            Q1 = df_out[col].quantile(0.25)
            Q3 = df_out[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            df_out[col] = np.clip(df_out[col], lower_bound, upper_bound)
    return df_out

### 4. Improved Feature Detection
def identify_feature_types(df: pd.DataFrame, target: str, ordinal_cols: list = None) -> tuple:
    """Identifies feature types using 'number' for scalability, routing specified columns to ordinal and the rest to OHE."""
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if target in numeric_columns: numeric_columns.remove(target)
    if target in categorical_columns: categorical_columns.remove(target)
    
    ord_cols = [c for c in ordinal_cols if c in categorical_columns] if ordinal_cols else []
    ohe_cols = [c for c in categorical_columns if c not in ord_cols]
    
    return numeric_columns, ohe_cols, ord_cols

### 5. Data Splitting
def split_data(df: pd.DataFrame, target: str, test_size=0.2, random_state=42) -> tuple:
    """Splits the dataset into training and testing sets."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

### 6. Advanced Preprocessing Pipeline
def create_preprocessing_pipeline(df: pd.DataFrame, target: str, ordinal_cols: list = None) -> ColumnTransformer:
    """Creates a comprehensive ColumnTransformer preprocessing pipeline, dropping any unprocessed columns for safety."""
    numeric_columns, ohe_cols, ord_cols = identify_feature_types(df, target, ordinal_cols)
    transformers = []
    
    if numeric_columns:
        numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('scaler', StandardScaler()),
        ])
        transformers.append(('numeric', numeric_pipeline, numeric_columns))
        
    if ohe_cols:
        ohe_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent', add_indicator=True)),  
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ])
        transformers.append(('cat_ohe', ohe_pipeline, ohe_cols))
        
    if ord_cols:
        ord_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ])
        transformers.append(('cat_ord', ord_pipeline, ord_cols))
        
    return ColumnTransformer(transformers=transformers, remainder='drop')