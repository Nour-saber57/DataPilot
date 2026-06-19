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
    version="1.1.0",
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

    results: dict[str, Any] = {}

    for model_name in model_names:
        try:
            best_model, best_params = train_model(model_name, task, X_train_processed, y_train)
            y_pred = best_model.predict(X_test_processed)

            if task == "classification":
                metrics = evaluate_classification(y_test, y_pred)
                score_key = "f1"
            else:
                metrics = evaluate_regression(y_test, y_pred)
                score_key = "r2"

            results[model_name] = normalize(
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
                }
            )
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
            if result["status"] == "success"
        ],
        key=lambda item: item["score"],
        reverse=True,
    )

    return normalize(
        {
            "status": "success",
            "task": task,
            "total_models": len(model_names),
            "successful": sum(1 for result in results.values() if result["status"] == "success"),
            "failed": sum(1 for result in results.values() if result["status"] == "error"),
            "leaderboard": leaderboard,
            "detailed_results": results,
        }
    )


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
        jobs[job_id].update(
            {
                "status": "success",
                "result": result,
            }
        )
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


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "status": "success",
        "message": "AutoML Backend API is running",
        "version": "1.1.0",
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
            key=lambda item: item[1].get("score", float("-inf"))
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
