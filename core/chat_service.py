from core.chatbot import ask_gemini

def chat(message, context):

    prompt = f"""
You are the AI assistant for DataPilot.

Dataset Context:
{context}

User Question:
{message}
"""

    return ask_gemini(prompt)