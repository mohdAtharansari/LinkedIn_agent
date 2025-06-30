def format_recent_messages(messages: list) -> str:
    """Helper function to format the last few messages for the prompt."""

    recent_messages = messages[-4:]
    formatted_history = ""
    for msg in recent_messages:
        role = "User" if msg.type == "human" else "AI"
        formatted_history += f"{role}: {msg.content}\n"
    return formatted_history.strip()