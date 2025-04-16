def get_chat_template(content: str, role: str="user") -> dict:
    return {"role": role, "content": content}
