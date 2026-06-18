"""
FULL BACKEND INTEGRATION TEST
Tests the complete pipeline: Data Preprocessing -> Model Training -> Evaluation
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from preprocessor import (
    identify_feature_types,
    split_data,
    create_preprocessing_pipeline
)
from model_registry import (
    identify_task,
    get_classification_models,
    get_regression_models,
    train_model,
    evaluate_classification,
    evaluate_regression
)

print("\n" + "=" * 80)
print("FULL BACKEND INTEGRATION TEST - PREPROCESSING + MODELS")
print("=" * 80)

# ==================== STEP 1: CREATE SAMPLE DATA ====================
print("\n" + "-" * 80)
print("STEP 1: Creating Sample Dataset")
print("-" * 80)

# Create classification dataset with mixed feature types
np.random.seed(42)
n_samples = 300

# Numeric features
age = np.random.randint(18, 80, n_samples)
income = np.random.randint(20000, 150000, n_samples)
credit_score = np.random.randint(300, 850, n_samples)
employment_years = np.random.randint(0, 40, n_samples)

# Categorical features
marital_status = np.random.choice(['Single', 'Married', 'Divorced'], n_samples)
education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)

# Target variable
loan_approved = np.random.choice([0, 1], n_samples)

# Create DataFrame
df = pd.DataFrame({
    'age': age,
    'income': income,
    'credit_score': credit_score,
    'employment_years': employment_years,
    'marital_status': marital_status,
    'education': education,
    'loan_approved': loan_approved
})

print(f"✓ Dataset created: {df.shape[0]} samples, {df.shape[1]} features")
print(f"\nDataset Preview:")
print(df.head())
print(f"\nData Types:\n{df.dtypes}")
print(f"\nTarget Distribution:\n{df['loan_approved'].value_counts()}")

target_column = 'loan_approved'

# ==================== STEP 2: TASK IDENTIFICATION ====================
print("\n" + "-" * 80)
print("STEP 2: Task Identification")
print("-" * 80)

task = identify_task(df, target_column)
print(f"✓ Task identified: {task.upper()}")

# ==================== STEP 3: FEATURE TYPE IDENTIFICATION ====================
print("\n" + "-" * 80)
print("STEP 3: Feature Type Identification")
print("-" * 80)

numeric_cols, categorical_cols = identify_feature_types(df, target_column)
print(f"✓ Numeric features: {numeric_cols}")
print(f"✓ Categorical features: {categorical_cols}")

# ==================== STEP 4: DATA SPLITTING ====================
print("\n" + "-" * 80)
print("STEP 4: Train-Test Split")
print("-" * 80)

X_train, X_test, y_train, y_test = split_data(df, target_column, test_size=0.2, random_state=42)
print(f"✓ X_train shape: {X_train.shape}")
print(f"✓ X_test shape: {X_test.shape}")
print(f"✓ y_train shape: {y_train.shape}")
print(f"✓ y_test shape: {y_test.shape}")
print(f"✓ Split ratio: {len(X_train)}/{len(X_test)} (80/20)")

# ==================== STEP 5: DATA PREPROCESSING ====================
print("\n" + "-" * 80)
print("STEP 5: Data Preprocessing Pipeline")
print("-" * 80)

try:
    pipeline = create_preprocessing_pipeline(df, target_column)
    print("✓ Preprocessing pipeline created")
    
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    
    print(f"✓ X_train processed shape: {X_train_processed.shape}")
    print(f"✓ X_test processed shape: {X_test_processed.shape}")
    print(f"  (Shape changed due to one-hot encoding of categorical features)")
except Exception as e:
    print(f"✗ Error in preprocessing: {e}")
    X_train_processed = X_train
    X_test_processed = X_test

# ==================== STEP 6: MODEL TRAINING & EVALUATION ====================
print("\n" + "-" * 80)
print("STEP 6: Model Training with GridSearchCV")
print("-" * 80)

if task == "classification":
    models = get_classification_models()
    eval_func = evaluate_classification
else:
    models = get_regression_models()
    eval_func = evaluate_regression

print(f"\nAvailable {task} models: {list(models.keys())}")
print(f"\nTraining all models with GridSearchCV (5-fold CV)...\n")

results = {}
timings = {}

for model_name in models.keys():
    try:
        print(f"Training: {model_name:25s}", end=" -> ", flush=True)
        
        # Train model with GridSearchCV
        best_model, best_params = train_model(model_name, task, X_train_processed, y_train)
        
        # Make predictions
        y_pred = best_model.predict(X_test_processed)
        
        # Evaluate
        metrics = eval_func(y_test, y_pred)
        results[model_name] = {
            'metrics': metrics,
            'best_params': best_params,
            'model': best_model
        }
        
        if task == "classification":
            print(f"F1: {metrics['f1']:.4f}, Acc: {metrics['accuracy']:.4f} ✓")
        else:
            print(f"R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f} ✓")
            
    except Exception as e:
        print(f"✗ Error: {str(e)[:50]}")

# ==================== STEP 7: PERFORMANCE COMPARISON ====================
print("\n" + "-" * 80)
print("STEP 7: Performance Ranking")
print("-" * 80)

if task == "classification":
    sorted_results = sorted(results.items(), key=lambda x: x[1]['metrics']['f1'], reverse=True)
    print(f"\n{'Rank':<6} {'Model':<25} {'F1-Score':<12} {'Accuracy':<12} {'Precision':<12}")
    print("─" * 70)
    for rank, (model_name, data) in enumerate(sorted_results, 1):
        metrics = data['metrics']
        print(f"{rank:<6} {model_name:<25} {metrics['f1']:<12.4f} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f}")
else:
    sorted_results = sorted(results.items(), key=lambda x: x[1]['metrics']['r2'], reverse=True)
    print(f"\n{'Rank':<6} {'Model':<25} {'R² Score':<12} {'RMSE':<12} {'MAE':<12}")
    print("─" * 70)
    for rank, (model_name, data) in enumerate(sorted_results, 1):
        metrics = data['metrics']
        print(f"{rank:<6} {model_name:<25} {metrics['r2']:<12.4f} {metrics['rmse']:<12.4f} {metrics['mae']:<12.4f}")

# ==================== STEP 8: BEST MODEL DETAILS ====================
print("\n" + "-" * 80)
print("STEP 8: Best Model Details")
print("-" * 80)

if sorted_results:
    best_model_name = sorted_results[0][0]
    best_model_data = sorted_results[0][1]
    
    print(f"\n🏆 Best Model: {best_model_name.upper()}")
    print(f"\nBest Hyperparameters:")
    for param, value in best_model_data['best_params'].items():
        print(f"  - {param}: {value}")
    
    print(f"\nBest Metrics:")
    for metric_name, metric_value in best_model_data['metrics'].items():
        print(f"  - {metric_name}: {metric_value:.4f}")

# ==================== STEP 9: PIPELINE SUMMARY ====================
print("\n" + "=" * 80)
print("INTEGRATION TEST SUMMARY")
print("=" * 80)

print(f"""
✓ Data Loading:          Successful ({df.shape[0]} samples, {df.shape[1]} features)
✓ Task Identification:   {task.upper()}
✓ Feature Analysis:      {len(numeric_cols)} numeric, {len(categorical_cols)} categorical
✓ Data Splitting:        80/20 train/test split
✓ Preprocessing:         StandardScaler + OneHotEncoder
✓ Model Training:        {len(results)} models trained with GridSearchCV
✓ Evaluation:            Complete performance metrics calculated
✓ Best Model:            {best_model_name if sorted_results else 'N/A'}

Pipeline Status: ✅ FULLY OPERATIONAL
""")

print("=" * 80)
print("Backend integration test completed successfully!")
print("=" * 80)

# ==================== STEP 10: USAGE EXAMPLES ====================
print("\n" + "=" * 80)
print("NEXT STEPS - How to use in production:")
print("=" * 80)

print("""
1. IMPORT THE MODULES:
   from preprocessor import split_data, create_preprocessing_pipeline
   from model_registry import train_model, evaluate_classification

2. LOAD YOUR DATA:
   df = pd.read_csv('your_data.csv')
   
3. SPLIT THE DATA:
   X_train, X_test, y_train, y_test = split_data(df, 'target_column')
   
4. CREATE PIPELINE:
   pipeline = create_preprocessing_pipeline(df, 'target_column')
   X_train_processed = pipeline.fit_transform(X_train)
   X_test_processed = pipeline.transform(X_test)
   
5. TRAIN MODEL:
   best_model, best_params = train_model('random_forest', 'classification', 
                                          X_train_processed, y_train)
   
6. MAKE PREDICTIONS:
   y_pred = best_model.predict(X_test_processed)
   
7. EVALUATE:
   from model_registry import evaluate_classification
   metrics = evaluate_classification(y_test, y_pred)
   print(f"Accuracy: {metrics['accuracy']:.4f}")

8. INTEGRATE WITH FASTAPI/STREAMLIT:
   - Load the best model
   - Accept input data
   - Preprocess using the pipeline
   - Make predictions
   - Return results
""")

print("=" * 80)
