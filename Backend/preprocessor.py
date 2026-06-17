def identify_feature_types(df, target):
    # Identify numeric and categorical columns in the DataFrame, excluding the target column
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Remove the target column from each list if it exists
    if target in numeric_columns:
        numeric_columns.remove(target)
    if target in categorical_columns:
        categorical_columns.remove(target)

    return numeric_columns, categorical_columns
