import json
from pathlib import Path
from datetime import datetime
from typing import Optional


# ============================================================
# FILE CONFIGURATION
# ============================================================

CANDIDATE_FILE = Path("data/candidates/candidates.json")


# ============================================================
# AUTOMATIC TYPE CONFIRMATION CONFIGURATION
# ============================================================

MIN_OCCURRENCES_FOR_CONFIRMATION = 3

MIN_AVERAGE_CONFIDENCE = 0.80

MIN_TYPE_CONSISTENCY = 0.70


# ============================================================
# STORAGE
# ============================================================

def _ensure_storage():
    CANDIDATE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not CANDIDATE_FILE.exists():
        with open(
            CANDIDATE_FILE,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                {},
                file,
                indent=2,
                ensure_ascii=False
            )


def _load_candidates():
    _ensure_storage()

    try:
        with open(
            CANDIDATE_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        if not isinstance(data, dict):
            return {}

        return data

    except (
        json.JSONDecodeError,
        OSError,
        TypeError
    ):
        return {}


def _save_candidates(data):
    CANDIDATE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_file = CANDIDATE_FILE.with_suffix(".tmp")

    with open(
        temp_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )

    temp_file.replace(CANDIDATE_FILE)


# ============================================================
# PUBLIC STORAGE FUNCTIONS
# ============================================================

def load_candidates():
    return _load_candidates()


def save_candidates(data):
    _save_candidates(data)


# ============================================================
# NORMALIZATION
# ============================================================

def _normalize(text: str) -> str:
    return " ".join(
        text.strip().lower().split()
    )


# ============================================================
# CONTEXT COMPATIBILITY
# ============================================================

def _get_existing_context_texts(contexts):
    """
    Supports both the old and new context formats.

    Old format:

        ["context one", "context two"]

    New format:

        [
            {
                "context": "context one",
                "start": 0,
                "end": 10
            }
        ]
    """

    result = set()

    if not isinstance(contexts, list):
        return result

    for item in contexts:

        if isinstance(item, dict):

            value = item.get("context")

            if value:
                result.add(str(value))

        elif isinstance(item, str):

            result.add(item)

    return result


# ============================================================
# ADD / UPDATE CANDIDATE
# ============================================================

def add_candidate(
    text: str,
    proposed_type: Optional[str],
    context: str,
    start: int,
    end: int,
    score: float,
):
    """
    Store a discovered entity.

    GLiNER's proposed_type is only a hypothesis.
    It is NOT the confirmed entity type.
    """

    normalized = _normalize(text)

    if not normalized:
        return None

    candidates = _load_candidates()

    now = datetime.utcnow().isoformat()

    # ========================================================
    # CREATE NEW ENTITY
    # ========================================================

    if normalized not in candidates:

        candidates[normalized] = {
            "text": text,
            "normalized": normalized,
            "type": None,
            "type_confidence": 0.0,
            "type_consistency": 0.0,
            "status": "NEW",
            "occurrences": 0,
            "contexts": [],
            "evidence": [],
            "type_history": [],
            "type_votes": {},
            "created_at": now,
            "updated_at": now,
        }

    candidate = candidates[normalized]

    # ========================================================
    # ENSURE NEW FIELDS EXIST
    # ========================================================

    candidate.setdefault(
        "type",
        None
    )

    candidate.setdefault(
        "type_confidence",
        0.0
    )

    candidate.setdefault(
        "type_consistency",
        0.0
    )

    candidate.setdefault(
        "status",
        "LEARNING"
    )

    candidate.setdefault(
        "occurrences",
        0
    )

    candidate.setdefault(
        "contexts",
        []
    )

    candidate.setdefault(
        "evidence",
        []
    )

    candidate.setdefault(
        "type_history",
        []
    )

    candidate.setdefault(
        "type_votes",
        {}
    )

    # ========================================================
    # OCCURRENCE
    # ========================================================

    candidate["occurrences"] += 1

    # ========================================================
    # CONTEXT
    # ========================================================

    existing_contexts = _get_existing_context_texts(
        candidate["contexts"]
    )

    if context not in existing_contexts:

        candidate["contexts"].append(
            {
                "context": context,
                "start": start,
                "end": end,
                "timestamp": now,
            }
        )

    # ========================================================
    # GLiNER EVIDENCE
    # ========================================================

    candidate["evidence"].append(
        {
            "gliner_type": proposed_type,
            "gliner_score": round(
                float(score),
                4
            ),
            "context": context,
            "timestamp": now,
        }
    )

    # ========================================================
    # STATUS
    # ========================================================

    if candidate.get("status") != "CONFIRMED":
        candidate["status"] = "LEARNING"

    candidate["updated_at"] = now

    candidates[normalized] = candidate

    _save_candidates(candidates)

    return candidate


# ============================================================
# GET CANDIDATE
# ============================================================

def get_candidate(text: str):
    normalized = _normalize(text)

    candidates = _load_candidates()

    return candidates.get(normalized)


# ============================================================
# GET LEARNING CANDIDATES
# ============================================================

def get_learning_candidates():

    candidates = _load_candidates()

    return [
        candidate
        for candidate in candidates.values()
        if candidate.get("status")
        in {
            "NEW",
            "LEARNING"
        }
    ]


# ============================================================
# GET CONFIRMED CANDIDATES
# ============================================================

def get_confirmed_candidates():

    candidates = _load_candidates()

    return [
        candidate
        for candidate in candidates.values()
        if candidate.get("status") == "CONFIRMED"
    ]


# ============================================================
# UPDATE TYPE PREDICTION
# ============================================================

def update_type_prediction(
    text: str,
    predicted_type: str,
    confidence: float,
):
    """
    Add a BERT type prediction and recalculate
    accumulated evidence.
    """

    normalized = _normalize(text)

    candidates = _load_candidates()

    candidate = candidates.get(normalized)

    if candidate is None:
        return None

    now = datetime.utcnow().isoformat()

    confidence = max(
        0.0,
        min(
            1.0,
            float(confidence)
        )
    )

    # ========================================================
    # ENSURE REQUIRED FIELDS
    # ========================================================

    candidate.setdefault(
        "type_history",
        []
    )

    candidate.setdefault(
        "type_votes",
        {}
    )

    # ========================================================
    # STORE PREDICTION
    # ========================================================

    candidate["type_history"].append(
        {
            "type": predicted_type,
            "confidence": round(
                confidence,
                4
            ),
            "timestamp": now,
        }
    )

    # ========================================================
    # UPDATE VOTES
    # ========================================================

    votes = candidate["type_votes"]

    if predicted_type not in votes:

        votes[predicted_type] = {
            "count": 0,
            "confidence_sum": 0.0,
        }

    votes[predicted_type]["count"] += 1

    votes[predicted_type]["confidence_sum"] += confidence

    # ========================================================
    # CALCULATE TYPE SCORES
    # ========================================================

    type_scores = {}

    for entity_type, vote_data in votes.items():

        count = vote_data.get(
            "count",
            0
        )

        confidence_sum = vote_data.get(
            "confidence_sum",
            0.0
        )

        if count <= 0:
            continue

        average_confidence = (
            confidence_sum / count
        )

        type_scores[entity_type] = {
            "count": count,
            "average_confidence": average_confidence,
        }

    if not type_scores:

        candidate["updated_at"] = now

        candidates[normalized] = candidate

        _save_candidates(candidates)

        return candidate

    # ========================================================
    # FIND BEST TYPE
    # ========================================================

    best_type = max(
        type_scores,
        key=lambda entity_type: (
            type_scores[entity_type]["count"],
            type_scores[entity_type]["average_confidence"],
        )
    )

    best_count = type_scores[
        best_type
    ]["count"]

    best_average = type_scores[
        best_type
    ]["average_confidence"]

    # ========================================================
    # TYPE CONSISTENCY
    # ========================================================

    total_predictions = sum(
        value["count"]
        for value in type_scores.values()
    )

    if total_predictions > 0:

        consistency = (
            best_count /
            total_predictions
        )

    else:

        consistency = 0.0

    # ========================================================
    # UPDATE BEST TYPE
    # ========================================================

    candidate["type"] = best_type

    candidate["type_confidence"] = round(
        best_average,
        4
    )

    candidate["type_consistency"] = round(
        consistency,
        4
    )

    # ========================================================
    # CONFIRMATION CHECK
    # ========================================================

    enough_occurrences = (
        candidate.get(
            "occurrences",
            0
        )
        >= MIN_OCCURRENCES_FOR_CONFIRMATION
    )

    enough_confidence = (
        best_average
        >= MIN_AVERAGE_CONFIDENCE
    )

    consistent_type = (
        consistency
        >= MIN_TYPE_CONSISTENCY
    )

    # ========================================================
    # CONFIRM
    # ========================================================

    if (
        enough_occurrences
        and enough_confidence
        and consistent_type
    ):

        candidate["status"] = "CONFIRMED"

        candidate["confirmed_at"] = now

    # ========================================================
    # CONTINUE LEARNING
    # ========================================================

    else:

        candidate["status"] = "LEARNING"

    candidate["updated_at"] = now

    candidates[normalized] = candidate

    _save_candidates(candidates)

    return candidate


# ============================================================
# RESET TYPE LEARNING
# ============================================================

def reset_type_learning(text: str):

    normalized = _normalize(text)

    candidates = _load_candidates()

    candidate = candidates.get(normalized)

    if candidate is None:
        return None

    candidate["type"] = None

    candidate["type_confidence"] = 0.0

    candidate["type_consistency"] = 0.0

    candidate["type_history"] = []

    candidate["type_votes"] = {}

    candidate["status"] = (
        "LEARNING"
        if candidate.get("occurrences", 0) > 0
        else "NEW"
    )

    candidate["updated_at"] = (
        datetime.utcnow().isoformat()
    )

    candidates[normalized] = candidate

    _save_candidates(candidates)

    return candidate