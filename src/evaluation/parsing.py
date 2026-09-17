import re

_NUM = re.compile(r"\b\d+(\.\d+)?\b")


def parse_score(text: str, default: float = 4.5) -> float:
    match = _NUM.search(text)
    if not match:
        return default
    return max(0.0, min(9.0, float(match.group())))