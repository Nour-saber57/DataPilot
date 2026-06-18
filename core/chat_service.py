import json
from pathlib import Path
from core.chatbot import ask_gemini

SYSTEM_PROMPT = Path("prompts/system_prompt.txt").read_text()


def generate_response(message, context):

    context_str = json.dumps(context, indent=2)

    prompt = f"""
{SYSTEM_PROMPT}

DATA CONTEXT:
{context_str}

USER QUESTION:
{message}
"""

    return ask_gemini(prompt)