import json
from pathlib import Path
from datetime import datetime


CANDIDATE_FILE = Path(
    "data/candidates/candidates.json"
)


def load_candidates():

    if not CANDIDATE_FILE.exists():
        return {}

    with open(
        CANDIDATE_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def save_candidates(candidates):

    CANDIDATE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        CANDIDATE_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            candidates,
            file,
            indent=2,
            ensure_ascii=False,
        )


def add_candidate(
    text: str,
    proposed_type: str,
    context: str,
):

    candidates = load_candidates()

    key = text.lower().strip()

    if key not in candidates:

        candidates[key] = {
            "text": text,
            "normalized": key,
            "proposed_type": proposed_type,
            "occurrences": 0,
            "contexts": [],
            "status": "PENDING",
            "created_at": datetime.utcnow().isoformat(),
        }

    candidates[key]["occurrences"] += 1

    if context not in candidates[key]["contexts"]:
        candidates[key]["contexts"].append(context)

    save_candidates(candidates)

    return candidates[key]