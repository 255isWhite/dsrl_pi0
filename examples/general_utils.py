def dict_pretty_str(d: dict) -> str:
    max_key_len = max(len(str(k)) for k in d.keys())
    lines = [f"{k:<{max_key_len}} : {v}" for k, v in d.items()]
    return "\n".join(lines)
