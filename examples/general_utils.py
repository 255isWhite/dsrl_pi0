def dict_pretty_str(d: dict) -> str:
    max_key_len = max(len(str(k)) for k in d.keys())
    lines = [f"{k:<{max_key_len}} : {v}" for k, v in d.items()]
    return "\n".join(lines)

import hashlib

def safe_group_name(name: str, max_len: int = 120) -> str:
    """确保 group_name 不超过指定长度，并避免截断导致的重复"""
    if len(name) <= max_len:
        return name
    # 保留前 max_len-9 字符 + '_' + 8位hash
    prefix = name[:max_len - 9]
    suffix = hashlib.md5(name.encode()).hexdigest()[:8]
    return f"{prefix}_{suffix}"
