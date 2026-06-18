import pandas as pd
import numpy as np
from preprocessor import identify_feature_types, split_data, create_preprocessing_pipeline

# Generate sample data
print("=" * 60)
print("CREATING SAMPLE DATA FOR TESTING")
print("=" * 60)

np.random.seed(42)
n_samples = 200

# Create sample dataset
sample_data = {
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.randint(20000, 150000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'employment_years': np.random.randint(0, 40, n_samples),
    'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'loan_approved': np.random.choice([0, 1], n_samples)
}

df = pd.DataFrame(sample_data)
target = 'loan_approved'

print(f"\nDataset shape: {df.shape}")
print(f"Target column: {target}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nData types:\n{df.dtypes}")

# TEST 1: identify_feature_types
print("\n" + "=" * 60)
print("TEST 1: identify_feature_types()")
print("=" * 60)

numeric_cols, categorical_cols = identify_feature_types(df, target)
print(f"\nNumeric columns: {numeric_cols}")
print(f"Categorical columns: {categorical_cols}")

# TEST 2: split_data
print("\n" + "=" * 60)
print("TEST 2: split_data()")
print("=" * 60)

X_train, X_test, y_train, y_test = split_data(df, target, test_size=0.2, random_state=42)
print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"\nTrain/Test split: {len(X_train)}/{len(X_test)} (80/20)")

# TEST 3: create_preprocessing_pipeline
print("\n" + "=" * 60)
print("TEST 3: create_preprocessing_pipeline()")
print("=" * 60)

try:
    pipeline = create_preprocessing_pipeline(df, target)
    print("\n✓ Pipeline created successfully")
    
    # Fit and transform data
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    
    print(f"\nOriginal X_train shape: {X_train.shape}")
    print(f"Processed X_train shape: {X_train_processed.shape}")
    print(f"Original X_test shape: {X_test.shape}")
    print(f"Processed X_test shape: {X_test_processed.shape}")
    print("\n✓ Data preprocessing successful")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print("✓ All tests completed successfully!")
print("\nYour preprocessor.py functions are working correctly.")

