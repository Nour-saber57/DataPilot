"""
FastAPI backend - AutoML API
Connects to preprocessing and model registry for end-to-end ML pipeline
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import sys
import os

# Add Backend path to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Backend'))

from preprocessor import identify_feature_types, split_data, create_preprocessing_pipeline
from model_registry import (
    identify_task, get_all_models, train_model, 
    evaluate_classification, evaluate_regression
)

# Initialize FastAPI
app = FastAPI(
    title="AutoML Backend API",
    description="Machine Learning Pipeline with Preprocessing and Model Training",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store current state
current_data = None
current_task = None
train_results = {}


@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "success",
        "message": "AutoML Backend API is running",
        "version": "1.0.0"
    }


@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...), target_column: str = None):
    """Upload CSV file and analyze it"""
    global current_data, current_task
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        if target_column is None:
            return {
                "status": "error",
                "message": "target_column is required",
                "columns": df.columns.tolist()
            }
        
        if target_column not in df.columns:
            return {
                "status": "error",
                "message": f"Target column '{target_column}' not found in CSV"
            }
        
        current_data = df
        current_task = identify_task(df, target_column)
        
        numeric_cols, categorical_cols = identify_feature_types(df, target_column)
        
        return {
            "status": "success",
            "message": f"Data loaded successfully. Task: {current_task}",
            "data_shape": df.shape,
            "columns": df.columns.tolist(),
            "target_column": target_column,
            "task": current_task,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict()
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/available-models")
def get_available_models(task: str = None):
    """Get available models for the task"""
    try:
        if task is None:
            task = current_task
        
        if task is None:
            return {
                "status": "error",
                "message": "No task identified. Upload data first."
            }
        
        models = get_all_models(task)
        return {
            "status": "success",
            "task": task,
            "models": list(models.keys()),
            "count": len(models)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/train-model")
def train_single_model(model_name: str, target_column: str, test_size: float = 0.2):
    """Train a single model with GridSearchCV"""
    try:
        if current_data is None:
            return {
                "status": "error",
                "message": "No data loaded. Upload data first."
            }
        
        df = current_data
        task = identify_task(df, target_column)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(df, target_column, test_size=test_size)
        
        # Preprocess
        pipeline = create_preprocessing_pipeline(df, target_column)
        X_train_processed = pipeline.fit_transform(X_train)
        X_test_processed = pipeline.transform(X_test)
        
        # Train model
        best_model, best_params = train_model(model_name, task, X_train_processed, y_train)
        
        # Predict
        y_pred = best_model.predict(X_test_processed)
        
        # Evaluate
        if task == "classification":
            metrics = evaluate_classification(y_test, y_pred)
        else:
            metrics = evaluate_regression(y_test, y_pred)
        
        result = {
            "status": "success",
            "model_name": model_name,
            "task": task,
            "best_params": best_params,
            "metrics": metrics,
            "test_size": test_size,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }
        
        train_results[model_name] = result
        
        return result
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/train-all-models")
def train_all_models_endpoint(target_column: str, test_size: float = 0.2):
    """Train all models and compare performance"""
    try:
        if current_data is None:
            return {
                "status": "error",
                "message": "No data loaded. Upload data first."
            }
        
        df = current_data
        task = identify_task(df, target_column)
        models_dict = get_all_models(task)
        
        # Split and preprocess data once
        X_train, X_test, y_train, y_test = split_data(df, target_column, test_size=test_size)
        pipeline = create_preprocessing_pipeline(df, target_column)
        X_train_processed = pipeline.fit_transform(X_train)
        X_test_processed = pipeline.transform(X_test)
        
        results = {}
        
        for model_name in models_dict.keys():
            try:
                best_model, best_params = train_model(model_name, task, X_train_processed, y_train)
                y_pred = best_model.predict(X_test_processed)
                
                if task == "classification":
                    metrics = evaluate_classification(y_test, y_pred)
                    score_key = "f1"
                else:
                    metrics = evaluate_regression(y_test, y_pred)
                    score_key = "r2"
                
                results[model_name] = {
                    "status": "success",
                    "metrics": metrics,
                    "best_params": best_params,
                    "score": metrics[score_key]
                }
            except Exception as e:
                results[model_name] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Sort by performance
        sorted_results = sorted(
            [(k, v["score"]) for k, v in results.items() if v["status"] == "success"],
            key=lambda x: x[1],
            reverse=True
        )
        
        train_results.update(results)
        
        return {
            "status": "success",
            "task": task,
            "total_models": len(models_dict),
            "successful": len([r for r in results.values() if r["status"] == "success"]),
            "failed": len([r for r in results.values() if r["status"] == "error"]),
            "leaderboard": [{"model": name, "score": score} for name, score in sorted_results],
            "detailed_results": results
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/results")
def get_results(model_name: str = None):
    """Get training results"""
    if model_name:
        if model_name in train_results:
            return train_results[model_name]
        else:
            return {
                "status": "error",
                "message": f"No results found for model: {model_name}"
            }
    
    return {
        "status": "success",
        "trained_models": list(train_results.keys()),
        "results": train_results
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)