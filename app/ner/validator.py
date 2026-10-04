import re


# ---------------------------------------------------------
# GENERIC WORDS THAT SHOULD NOT BECOME LEARNED ENTITIES
# ---------------------------------------------------------

GENERIC_WORDS = {
    "panel",
    "preset",
    "output",
    "input",
    "workflow",
    "guide",
    "step",
    "steps",
    "effect",
    "effects",
    "employee",
    "document",
    "documents",
    "file",
    "files",
    "page",
    "pages",
    "text",
    "area",
    "place",
    "region",
    "location",
    "company",
    "organization",
    "business",
    "department",
    "team",
    "person",
    "people",
    "workspace",
    "scan",
    "scans",
    "minutes",
    "minute",
    "hours",
    "hour",
    "time",
    "day",
    "days",
}


EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9._%+-]+@"
    r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
)


PHONE_PATTERN = re.compile(
    r"^(?:\+?\d[\d\s().-]{7,}\d)$"
)


DATE_PATTERNS = [
    re.compile(
        r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$"
    ),
    re.compile(
        r"^\d{4}[/-]\d{1,2}[/-]\d{1,2}$"
    ),
    re.compile(
        r"^(?:"
        r"January|February|March|April|May|June|"
        r"July|August|September|October|November|December"
        r")\s+\d{1,2},?\s+\d{4}$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\d{1,2}\s+"
        r"(?:"
        r"January|February|March|April|May|June|"
        r"July|August|September|October|November|December"
        r")"
        r"\s+\d{4}$",
        re.IGNORECASE,
    ),
]


def _clean(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text.strip(),
    )


def _is_generic_word(text: str) -> bool:
    normalized = text.lower().strip()

    if normalized in GENERIC_WORDS:
        return True

    words = normalized.split()

    if len(words) == 1:
        return normalized in GENERIC_WORDS

    return False


def _valid_date(text: str) -> bool:
    text = _clean(text)

    return any(
        pattern.fullmatch(text)
        for pattern in DATE_PATTERNS
    )


def _valid_email(text: str) -> bool:
    return bool(
        EMAIL_PATTERN.fullmatch(
            _clean(text)
        )
    )


def _valid_phone(text: str) -> bool:
    cleaned = _clean(text)

    if not PHONE_PATTERN.fullmatch(cleaned):
        return False

    digits = re.sub(
        r"\D",
        "",
        cleaned,
    )

    return 8 <= len(digits) <= 15


def _valid_person(text: str, score: float) -> bool:
    normalized = _clean(text)

    if _is_generic_word(normalized):
        return False

    words = re.findall(
        r"[A-Za-z]+",
        normalized,
    )

    # Normal PERSON entities should have
    # at least two name-like words.
    if len(words) >= 2:
        return True

    # Allow a single proper-looking name only
    # when GLiNER confidence is very high.
    if (
        len(words) == 1
        and score >= 0.90
        and words[0][0].isupper()
    ):
        return True

    return False


def _valid_organization(
    text: str,
    score: float,
) -> bool:
    normalized = _clean(text)

    if _is_generic_word(normalized):
        return False

    words = normalized.split()

    # Single generic lowercase words should not
    # become organizations.
    if len(words) == 1:
        word = words[0]

        if (
            word.isalpha()
            and word.islower()
            and score < 0.90
        ):
            return False

    return True


def _valid_location(
    text: str,
    score: float,
) -> bool:
    normalized = _clean(text)

    if _is_generic_word(normalized):
        return False

    words = normalized.split()

    if len(words) == 1:
        word = words[0]

        if (
            word.isalpha()
            and word.islower()
            and score < 0.90
        ):
            return False

    return True


def _valid_technology(
    text: str,
    score: float,
) -> bool:
    normalized = _clean(text)

    if _is_generic_word(normalized):
        return False

    # Avoid long natural-language phrases being
    # learned as TECHNOLOGY.
    words = normalized.split()

    if len(words) > 4:
        return False

    # Low-confidence generic-looking lowercase
    # phrases should not enter learning.
    if (
        len(words) <= 2
        and all(word.islower() for word in words)
        and score < 0.75
    ):
        return False

    return True


def _valid_product(
    text: str,
    score: float,
) -> bool:
    normalized = _clean(text)

    if _is_generic_word(normalized):
        return False

    words = normalized.split()

    # Prevent sentence fragments from becoming products.
    if len(words) > 4:
        return False

    if (
        len(words) == 1
        and normalized.isalpha()
        and normalized.islower()
        and score < 0.80
    ):
        return False

    return True


def _valid_database(
    text: str,
    score: float,
) -> bool:
    normalized = _clean(text)

    if _is_generic_word(normalized):
        return False

    return True


def validate_entity(
    entity: dict,
    text: str = "",
) -> bool:
    """
    Validate a GLiNER entity before it is allowed
    into candidate memory or training data.
    """

    entity_text = _clean(
        str(entity.get("text", ""))
    )

    entity_type = str(
        entity.get("type", "")
    ).upper()

    try:
        score = float(
            entity.get("score", 0.0)
        )
    except (
        TypeError,
        ValueError,
    ):
        score = 0.0

    if not entity_text:
        return False

    if len(entity_text) < 2:
        return False

    if not re.search(
        r"[A-Za-z0-9]",
        entity_text,
    ):
        return False

    # -----------------------------------------------------
    # STRUCTURED TYPES
    # -----------------------------------------------------

    if entity_type == "DATE":
        return _valid_date(entity_text)

    if entity_type == "EMAIL":
        return _valid_email(entity_text)

    if entity_type == "PHONE":
        return _valid_phone(entity_text)

    # -----------------------------------------------------
    # GENERIC REJECTION
    # -----------------------------------------------------

    if _is_generic_word(entity_text):
        return False

    # -----------------------------------------------------
    # ENTITY-SPECIFIC VALIDATION
    # -----------------------------------------------------

    if entity_type == "PERSON":
        return _valid_person(
            entity_text,
            score,
        )

    if entity_type == "ORGANIZATION":
        return _valid_organization(
            entity_text,
            score,
        )

    if entity_type == "LOCATION":
        return _valid_location(
            entity_text,
            score,
        )

    if entity_type == "TECHNOLOGY":
        return _valid_technology(
            entity_text,
            score,
        )

    if entity_type == "PRODUCT":
        return _valid_product(
            entity_text,
            score,
        )

    if entity_type == "DATABASE":
        return _valid_database(
            entity_text,
            score,
        )

    # Unknown entity types are allowed only when
    # they have reasonably strong confidence.
    return score >= 0.75