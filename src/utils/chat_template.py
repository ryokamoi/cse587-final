def get_chat_template(content: str, role: str="user") -> dict[str, str]:
    return {"role": role, "content": content}
