from chatbot import ask_gemini
from context_build import context   # adjust import path

def build_prompt(user_message, context):

    return f"""
You are DataPilot AI, a machine learning assistant.

Use ONLY the provided context.

Context:
{context}

User Question:
{user_message}
"""

# TEST QUESTION
question = "Why is Random Forest the best model?"

prompt = build_prompt(question, context)

response = ask_gemini(prompt)

print(response)