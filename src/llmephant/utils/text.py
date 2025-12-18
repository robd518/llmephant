def get_last_user_message(messages):
    users = [m for m in messages if m.role == "user"]
    return users[-1].content if users else ""


def extract_user_only_conversation(messages, assistant_reply):
    convo = []
    for m in messages:
        if m.role == "user":
            convo.append({"role": "user", "content": f"USER: {m.content}"})
    convo.append({"role": "assistant", "content": assistant_reply})
    return convo