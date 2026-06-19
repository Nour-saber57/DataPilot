"""
Prompt templates for the Gemini ML assistant.

The system prompt constrains Gemini to only reference actual experiment data.
This prevents hallucinated metrics or fabricated model comparisons.
"""

SYSTEM_PROMPT = """You are an ML assistant inside an AutoML application.
You must answer only using the experiment results below.
Do not invent scores, models, or features.
If something is not available, say that it is not available.
Explain in simple language and suggest practical improvements."""


def build_experiment_context(
    dataset_summary: dict,
    task_type: str,
    leaderboard_str: str,
    best_model_name: str,
    metrics: dict,
    feature_importance_str: str,
) -> str:
    """Format experiment results into a structured context block for the LLM."""
    return f"""Dataset summary:
{_format_summary(dataset_summary)}

Task type:
{task_type}

Leaderboard:
{leaderboard_str}

Best model:
{best_model_name}

Metrics:
{_format_metrics(metrics)}

Feature importance:
{feature_importance_str}"""


def build_user_prompt(context: str, question: str) -> str:
    """Combine the experiment context with the user's question."""
    return f"""{context}

User question:
{question}"""


SUGGESTED_QUESTIONS = [
    "Why did the best model win?",
    "Which features matter most and why?",
    "How could I improve the results?",
    "Is the model overfitting?",
    "What does the confusion matrix tell us?",
]


def _format_summary(summary: dict) -> str:
    lines = [f"  {k}: {v}" for k, v in summary.items() if k != "missing_per_column"]
    return "\n".join(lines)


def _format_metrics(metrics: dict) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in metrics.items())
