import json
from pathlib import Path
from core.chatbot import ask_gemini
from core.context_build import build_context

SYSTEM_PROMPT = Path("prompts/system_prompt.txt").read_text()

chat_history = []

MAX_HISTORY = 3



def add_to_history(role, message):
    chat_history.append({
        "role": role,
        "message": message
    })

    if len(chat_history) > MAX_HISTORY * 2:
        chat_history.pop(0)


def get_history_text():
    history_text = ""

    for msg in chat_history:
        history_text += (
            f"{msg['role'].upper()}: "
            f"{msg['message']}\n"
        )

    return history_text

def generate_response(message, context):

    add_to_history("user", message)

    context_str = json.dumps(
        context,
        indent=2,
        default=str
    )

    history_text = get_history_text()

    prompt = f"""{SYSTEM_PROMPT}\nDATA CONTEXT:{context_str}\nCHAT HISTORY:{history_text}\nCURRENT USER QUESTION:{message}"""

    response = ask_gemini(prompt)

    add_to_history("assistant", response)

    return response

df = 
target =
y_true =
y_pred =

context = build_context(df, target, y_true, y_pred)

user_input_message = 

generate_response(user_input_message, context)