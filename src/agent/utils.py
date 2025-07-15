from schemas import MessageHistory

def convert_message_history(user_query: str, message_history: MessageHistory) -> list[dict]:
    history = [{"role": message.role, "content": message.text} for message in message_history.messages]
    if user_query != "":
        history.append({ "role": "user", "content": user_query })
    return history

def convert_message_history_to_str(message_history: MessageHistory):
    return "\n".join([f"{message.role.capitalize()}: {message.text}" for message in message_history.messages])