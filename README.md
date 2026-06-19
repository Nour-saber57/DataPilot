# Data_Pilot
An Explainable Intelligent AutoML Recommendation System


# 🧪 AutoML Agent + Gemini Chat
**Made by Sara Musalim**

A Streamlit-based AutoML application that automates the machine learning pipeline: upload a CSV, select a target, train models, view results, and chat with Gemini about the experiment.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Gemini API key

Copy the example env file and add your key:

```bash
cp .env.example .env
# Edit .env and replace 'your_gemini_api_key_here' with your actual key
```

Alternatively, set it as a Streamlit secret in `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your_key_here"
```

### 3. Run the app

```bash
streamlit run app.py
```

### 4. Demo workflow

1. Upload `data/heart.csv` (classification) or `data/housing.csv` (regression)
2. Select the target column (`target` or `price`)
3. Click **Run AutoML**
4. Explore the Results, Explainability, and Gemini Chat tabs
5. Download the report, leaderboard, or trained model from the sidebar

## Project Structure

```
automl-agent/
├── app.py                    # Streamlit UI
├── requirements.txt
├── .env.example
├── automl/
│   ├── preprocessor.py       # Data cleaning and feature engineering
│   ├── model_registry.py     # Candidate model catalog
│   ├── trainer.py            # Model fitting with timing
│   ├── evaluator.py          # Metrics and leaderboard
│   ├── explainer.py          # Feature importance and diagnostic plots
│   ├── reporter.py           # Markdown report and artifact serialization
│   └── orchestrator.py       # Pipeline coordinator
├── llm/
│   ├── gemini_client.py      # Gemini API integration
│   └── prompts.py            # Prompt templates
├── data/
│   ├── heart.csv             # Classification demo dataset
│   └── housing.csv           # Regression demo dataset
└── outputs/                  # Generated artifacts (auto-created)
    ├── plots/
    ├── leaderboard.csv
    ├── report.md
    └── best_model.joblib
```

## Models

| Task | Models | Primary Metric |
|------|--------|----------------|
| Classification | Logistic Regression, Random Forest, Gradient Boosting | Weighted F1 |
| Regression | Linear Regression, Random Forest, Gradient Boosting | RMSE |

## Team Contributions

| Member | Role | Files |
|--------|------|-------|
| Member 1 | Streamlit UI | `app.py` |
| Member 2 | AutoML Backend | `preprocessor.py`, `model_registry.py`, `trainer.py` |
| Member 3 | Evaluation & Plots | `evaluator.py`, `explainer.py`, `reporter.py` |
| Member 4 | Gemini Integration | `gemini_client.py`, `prompts.py`, `orchestrator.py` |
