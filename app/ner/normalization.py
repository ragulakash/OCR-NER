import re


def normalize_entity(text: str) -> str:
    """
    Normalize entity text for matching.

    Example:

        "  Kryptos   Infosys "

    becomes:

        "kryptos infosys"
    """

    text = text.strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.lower()