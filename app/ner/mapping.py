import re

from app.ner.normalization import (
    normalize_entity,
)


# ============================================================
# KNOWN ENTITIES
# ============================================================

KNOWN_ENTITIES = {

    # --------------------------------------------------------
    # PERSON
    # --------------------------------------------------------

    "ragul akash": {
        "canonical": "Ragul Akash",
        "type": "PERSON",
    },


    # --------------------------------------------------------
    # ORGANIZATION
    # --------------------------------------------------------

    "kryptos infosys": {
        "canonical": "Kryptos Infosys",
        "type": "ORGANIZATION",
    },

    "google": {
        "canonical": "Google",
        "type": "ORGANIZATION",
    },


    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    "chennai": {
        "canonical": "Chennai",
        "type": "LOCATION",
    },


    # --------------------------------------------------------
    # TECHNOLOGY
    # --------------------------------------------------------

    "python": {
        "canonical": "Python",
        "type": "TECHNOLOGY",
    },

    "fastapi": {
        "canonical": "FastAPI",
        "type": "TECHNOLOGY",
    },


    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    "postgresql": {
        "canonical": "PostgreSQL",
        "type": "DATABASE",
    },
}


# ============================================================
# EXACT ENTITY MAPPING
# ============================================================

def map_known_entity(text: str):

    normalized = normalize_entity(
        text
    )

    return KNOWN_ENTITIES.get(
        normalized
    )


# ============================================================
# BUILD REGEX FOR KNOWN PHRASE
# ============================================================

def _build_phrase_pattern(
    normalized_name: str,
):

    words = normalized_name.split()

    escaped_words = [

        re.escape(word)

        for word in words
    ]

    pattern = r"\s+".join(
        escaped_words
    )

    return re.compile(

        rf"(?<!\w)"
        rf"{pattern}"
        rf"(?!\w)",

        re.IGNORECASE,
    )


# ============================================================
# FIND KNOWN PHRASES
# ============================================================

def find_known_phrases(
    text: str,
):

    matches = []


    for normalized_name, entity in (
        KNOWN_ENTITIES.items()
    ):

        pattern = (
            _build_phrase_pattern(
                normalized_name
            )
        )


        for match in pattern.finditer(
            text
        ):

            start = match.start()

            end = match.end()

            actual_text = text[
                start:end
            ]


            matches.append({

                "text":
                    actual_text,

                "normalized":
                    normalize_entity(
                        actual_text
                    ),

                "type":
                    entity["type"],

                "canonical":
                    entity["canonical"],

                "score":
                    1.0,

                "start":
                    start,

                "end":
                    end,

                "status":
                    "KNOWN",

                "source":
                    "known_phrase_match",
            })


    return matches