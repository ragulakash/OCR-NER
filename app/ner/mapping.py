from app.ner.normalization import normalize_entity


KNOWN_ENTITIES = {
    "python": {
        "canonical": "Python",
        "type": "TECHNOLOGY",
    },
    "fastapi": {
        "canonical": "FastAPI",
        "type": "TECHNOLOGY",
    },
    "postgresql": {
        "canonical": "PostgreSQL",
        "type": "DATABASE",
    },
    "kryptos infosys": {
        "canonical": "Kryptos Infosys",
        "type": "ORGANIZATION",
    },
    "chennai": {
        "canonical": "Chennai",
        "type": "LOCATION",
    },
}


def map_known_entity(text: str):

    normalized = normalize_entity(text)

    return KNOWN_ENTITIES.get(normalized)