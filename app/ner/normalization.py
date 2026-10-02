import re


def normalize_entity(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    return text.lower()