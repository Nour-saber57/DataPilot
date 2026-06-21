"""
FastAPI backend - AutoML API
Connects to preprocessing and model registry for end-to-end ML pipeline.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "Backend"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from model_registry import (  # noqa: E402
    evaluate_classification,
    evaluate_regression,
    get_all_models,
    identify_task,
    train_model,
)
from preprocessor import (  # noqa: E402
    create_preprocessing_pipeline,
    identify_feature_types,
    split_data,
)
from core.chat_service import generate_dataset_response


app = FastAPI(
    title="AutoML Backend API",
    description="Machine Learning Pipeline with Preprocessing and Model Training",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


datasets: dict[str, dict[str, Any]] = {}
jobs: dict[str, dict[str, Any]] = {}
latest_dataset_id: str | None = None


def make_error(status_code: int, message: str) -> None:
    raise HTTPException(status_code=status_code, detail={"status": "error", "message": message})


def normalize(value: Any) -> Any:
    return jsonable_encoder(value)


def validate_test_size(test_size: float) -> None:
    if not 0.05 <= test_size <= 0.5:
        make_error(400, "test_size must be between 0.05 and 0.5")


def get_dataset(dataset_id: str | None = None) -> dict[str, Any]:
    selected_id = dataset_id or latest_dataset_id
    if not selected_id or selected_id not in datasets:
        make_error(404, "No dataset loaded. Upload data first.")
    return datasets[selected_id]


def build_upload_summary(df: pd.DataFrame, target_column: str, dataset_id: str) -> dict[str, Any]:
    task = identify_task(df, target_column)
    numeric_cols, categorical_cols = identify_feature_types(df, target_column)

    return normalize(
        {
            "status": "success",
            "message": f"Data loaded successfully. Task: {task}",
            "dataset_id": dataset_id,
            "data_shape": df.shape,
            "columns": df.columns.tolist(),
            "target_column": target_column,
            "task": task,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
        }
    )


# ── Feature importance extraction ─────────────────────────────────────────────

def _get_feature_names(pipeline, df: pd.DataFrame, target_column: str) -> list[str]:
    """
    Get feature names after preprocessing.
    Tries pipeline.get_feature_names_out() first (sklearn >= 1.0),
    falls back to building names manually.
    """
    try:
        raw_names = pipeline.get_feature_names_out()
        # Strip transformer prefix (e.g. "numeric__age" → "age")
        cleaned = []
        for name in raw_names:
            if "__" in name:
                cleaned.append(name.split("__", 1)[1])
            else:
                cleaned.append(name)
        return cleaned
    except Exception:
        # Fallback: numeric cols + generic names for categoricals
        numeric_cols, categorical_cols = identify_feature_types(df, target_column)
        names = list(numeric_cols)
        for col in categorical_cols:
            names.append(col)
        return names


def extract_feature_importance(
    model,
    X_test_processed: np.ndarray,
    y_test,
    feature_names: list[str],
    task: str,
) -> dict[str, float]:
    """
    Extract feature importance from a trained model.

    Priority:
      1. model.feature_importances_  → tree-based models (RF, DT, GBM)
      2. model.coef_                 → linear models (LR, Ridge, Lasso)
      3. permutation importance      → KNN, SVM, NaiveBayes (fallback)

    Returns a dict {feature_name: importance_score} sorted by importance desc.
    """
    importance_values: np.ndarray | None = None
    n_features = X_test_processed.shape[1]

    # 1. Tree-based: direct feature_importances_
    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_

    # 2. Linear models: absolute value of coefficients
    elif hasattr(model, "coef_"):
        coef = np.array(model.coef_)
        if coef.ndim > 1:
            # Multiclass: average across classes
            coef = np.mean(np.abs(coef), axis=0)
        importance_values = np.abs(coef)

    # 3. Fallback: permutation importance (works for any model)
    else:
        try:
            from sklearn.inspection import permutation_importance
            from sklearn.metrics import f1_score, r2_score

            scoring = (
                lambda est, X, y: f1_score(y, est.predict(X), average="weighted", zero_division=0)
                if task == "classification"
                else r2_score(y, est.predict(X))
            )
            perm = permutation_importance(
                model, X_test_processed, y_test,
                n_repeats=5, random_state=42,
                scoring=scoring,
            )
            importance_values = perm.importances_mean
        except Exception:
            importance_values = np.zeros(n_features)

    # Align length with feature_names (transformer can expand features via OHE)
    if importance_values is not None and len(importance_values) != len(feature_names):
        # Trim or pad to match
        if len(importance_values) > len(feature_names):
            importance_values = importance_values[: len(feature_names)]
        else:
            importance_values = np.pad(
                importance_values,
                (0, len(feature_names) - len(importance_values)),
                constant_values=0.0,
            )

    if importance_values is None:
        importance_values = np.zeros(len(feature_names))

    # Normalise to [0, 1] so all models are comparable
    total = importance_values.sum()
    if total > 0:
        importance_values = importance_values / total

    result = {
        name: round(float(score), 6)
        for name, score in zip(feature_names, importance_values)
    }

    # Sort descending by score
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))


# ── Core training function ────────────────────────────────────────────────────

def train_selected_models(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    model_names: list[str] | None = None,
) -> dict[str, Any]:
    validate_test_size(test_size)

    if target_column not in df.columns:
        make_error(400, f"Target column '{target_column}' not found in dataset")

    task = identify_task(df, target_column)
    models_dict = get_all_models(task)

    if model_names is None:
        model_names = list(models_dict.keys())

    unknown_models = [name for name in model_names if name not in models_dict]
    if unknown_models:
        make_error(400, f"Unknown model(s): {', '.join(unknown_models)}")

    X_train, X_test, y_train, y_test = split_data(df, target_column, test_size=test_size)
    pipeline = create_preprocessing_pipeline(df, target_column)
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    # Get feature names after preprocessing (for importance labelling)
    feature_names = _get_feature_names(pipeline, df, target_column)

    results: dict[str, Any] = {}

    for model_name in model_names:
        try:
            best_model, best_params = train_model(model_name, task, X_train_processed, y_train)
            y_pred = best_model.predict(X_test_processed)

            if task == "classification":
                metrics = evaluate_classification(y_test, y_pred)
                score_key = "f1"

                # Predict probabilities for ROC curve (not all models support it)
                y_pred_proba: list | None = None
                if hasattr(best_model, "predict_proba"):
                    try:
                        y_pred_proba = best_model.predict_proba(X_test_processed).tolist()
                    except Exception:
                        y_pred_proba = None

            else:
                metrics = evaluate_regression(y_test, y_pred)
                score_key = "r2"
                y_pred_proba = None

            # Train score for overfitting analysis
            y_train_pred = best_model.predict(X_train_processed)
            if task == "classification":
                from sklearn.metrics import f1_score
                train_score = float(f1_score(y_train, y_train_pred, average="weighted", zero_division=0))
            else:
                from sklearn.metrics import r2_score
                train_score = float(r2_score(y_train, y_train_pred))

            # Cross-validation scores
            try:
                from sklearn.model_selection import cross_val_score
                import scipy.sparse as sp
                X_full = pipeline.transform(df.drop(columns=[target_column]))
                y_full = df[target_column]
                cv_scoring = "f1_weighted" if task == "classification" else "r2"
                cv_scores = cross_val_score(
                    best_model, X_full, y_full, cv=5, scoring=cv_scoring, n_jobs=-1
                ).tolist()
            except Exception:
                cv_scores = []

            # Feature importance
            importance = extract_feature_importance(
                best_model, X_test_processed, y_test, feature_names, task
            )

            result_entry = normalize(
                {
                    "status": "success",
                    "model_name": model_name,
                    "task": task,
                    "best_params": best_params,
                    "metrics": metrics,
                    "score": metrics[score_key],
                    "test_size": test_size,
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "y_true": y_test.tolist(),
                    "y_pred": y_pred.tolist(),
                    "y_pred_proba": y_pred_proba,
                    "train_score": train_score,
                    "cv_scores": cv_scores,
                    "feature_importance": importance,
                }
            )
            results[model_name] = result_entry

        except Exception as exc:
            results[model_name] = {
                "status": "error",
                "model_name": model_name,
                "message": str(exc),
            }

    leaderboard = sorted(
        [
            {"model": name, "score": result["score"]}
            for name, result in results.items()
            if result.get("status") == "success"
        ],
        key=lambda item: item["score"],
        reverse=True,
    )

    return normalize(
        {
            "status": "success",
            "task": task,
            "total_models": len(model_names),
            "successful": sum(1 for r in results.values() if r.get("status") == "success"),
            "failed": sum(1 for r in results.values() if r.get("status") == "error"),
            "leaderboard": leaderboard,
            "detailed_results": results,
        }
    )


# ── Background job runner ─────────────────────────────────────────────────────

def run_training_job(
    job_id: str,
    dataset_id: str,
    target_column: str,
    test_size: float,
    model_names: list[str] | None,
) -> None:
    jobs[job_id]["status"] = "running"
    try:
        dataset = get_dataset(dataset_id)
        result = train_selected_models(dataset["df"], target_column, test_size, model_names)
        jobs[job_id].update({"status": "success", "result": result})
        dataset["results"].update(result["detailed_results"])
    except HTTPException as exc:
        jobs[job_id].update(
            {
                "status": "error",
                "message": exc.detail["message"] if isinstance(exc.detail, dict) else str(exc.detail),
            }
        )
    except Exception as exc:
        jobs[job_id].update({"status": "error", "message": str(exc)})


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root() -> dict[str, Any]:
    return {
        "status": "success",
        "message": "AutoML Backend API is running",
        "version": "1.2.0",
        "datasets_loaded": len(datasets),
        "jobs_created": len(jobs),
    }


@app.post("/upload-data")
async def upload_data(
    file: UploadFile = File(...),
    target_column: str | None = Query(default=None),
) -> dict[str, Any]:
    global latest_dataset_id

    if not file.filename or not file.filename.lower().endswith(".csv"):
        make_error(400, "Only CSV files are supported")

    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8-sig")))
    except UnicodeDecodeError:
        make_error(400, "Could not decode CSV as UTF-8")
    except Exception as exc:
        make_error(400, f"Could not read CSV file: {exc}")

    if df.empty:
        make_error(400, "Uploaded CSV is empty")

    if target_column is None:
        return {
            "status": "error",
            "message": "target_column is required",
            "columns": df.columns.tolist(),
        }

    if target_column not in df.columns:
        make_error(400, f"Target column '{target_column}' not found in CSV")

    dataset_id = str(uuid4())
    task = identify_task(df, target_column)
    datasets[dataset_id] = {
        "df": df,
        "filename": file.filename,
        "target_column": target_column,
        "task": task,
        "results": {},
    }
    latest_dataset_id = dataset_id

    return build_upload_summary(df, target_column, dataset_id)


@app.get("/available-models")
def get_available_models(
    task: str | None = None,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    if task is None:
        dataset = get_dataset(dataset_id)
        task = dataset["task"]

    try:
        models = get_all_models(task)
    except ValueError as exc:
        make_error(400, str(exc))

    return {
        "status": "success",
        "task": task,
        "models": list(models.keys()),
        "count": len(models),
    }


@app.post("/train-model")
def train_single_model(
    background_tasks: BackgroundTasks,
    model_name: str,
    target_column: str | None = None,
    test_size: float = 0.2,
    dataset_id: str | None = None,
    async_training: bool = False,
) -> dict[str, Any]:
    dataset = get_dataset(dataset_id)
    resolved_target = target_column or dataset["target_column"]

    if async_training:
        job_id = str(uuid4())
        jobs[job_id] = {
            "status": "queued",
            "dataset_id": dataset_id or latest_dataset_id,
            "model_names": [model_name],
        }
        background_tasks.add_task(
            run_training_job,
            job_id,
            jobs[job_id]["dataset_id"],
            resolved_target,
            test_size,
            [model_name],
        )
        return {"status": "queued", "job_id": job_id}

    result = train_selected_models(dataset["df"], resolved_target, test_size, [model_name])
    single_result = result["detailed_results"][model_name]
    dataset["results"][model_name] = single_result
    return single_result


@app.post("/train-all-models")
def train_all_models_endpoint(
    background_tasks: BackgroundTasks,
    target_column: str | None = None,
    test_size: float = 0.2,
    dataset_id: str | None = None,
    async_training: bool = False,
) -> dict[str, Any]:
    dataset = get_dataset(dataset_id)
    resolved_target = target_column or dataset["target_column"]

    if async_training:
        job_id = str(uuid4())
        jobs[job_id] = {
            "status": "queued",
            "dataset_id": dataset_id or latest_dataset_id,
            "model_names": None,
        }
        background_tasks.add_task(
            run_training_job,
            job_id,
            jobs[job_id]["dataset_id"],
            resolved_target,
            test_size,
            None,
        )
        return {"status": "queued", "job_id": job_id}

    result = train_selected_models(dataset["df"], resolved_target, test_size)
    dataset["results"].update(result["detailed_results"])
    return result


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    if job_id not in jobs:
        make_error(404, f"No job found for job_id: {job_id}")
    return jobs[job_id]


@app.get("/results")
def get_results(
    model_name: str | None = None,
    dataset_id: str | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    if job_id:
        return get_job(job_id)

    dataset = get_dataset(dataset_id)
    results = dataset["results"]

    if model_name:
        if model_name not in results:
            make_error(404, f"No results found for model: {model_name}")
        return results[model_name]

    return {
        "status": "success",
        "dataset_id": dataset_id or latest_dataset_id,
        "trained_models": list(results.keys()),
        "results": results,
    }


@app.post("/chat")
def chat_with_dataset(message: str, dataset_id: str | None = None, model_name: str | None = None):
    dataset = get_dataset(dataset_id)
    results = dataset["results"]

    if not results:
        make_error(400, "No trained model results found. Train a model first.")

    if model_name is None:
        best_model = max(
            results.items(),
            key=lambda item: item[1].get("score", float("-inf")),
        )
        model_name, model_result = best_model
    else:
        if model_name not in results:
            make_error(404, f"No results found for model: {model_name}")
        model_result = results[model_name]

    if "y_true" not in model_result or "y_pred" not in model_result:
        make_error(400, "This model result does not include y_true/y_pred. Retrain the model first.")

    response = generate_dataset_response(
        message=message,
        df=dataset["df"],
        target=dataset["target_column"],
        y_true=model_result["y_true"],
        y_pred=model_result["y_pred"],
    )

    return {
        "status": "success",
        "model_name": model_name,
        "response": response,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
