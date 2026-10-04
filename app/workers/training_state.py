import json

from pathlib import Path
from datetime import datetime


STATE_FILE = Path(
    "data/training_state.json"
)


DEFAULT_STATE = {
    "status": "idle",
    "started_at": None,
    "completed_at": None,
    "error": None,
    "model_version": None,
    "metrics": None,
}


def load_training_state():
    if not STATE_FILE.exists():
        return DEFAULT_STATE.copy()

    try:
        with open(
            STATE_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            saved_state = json.load(file)

        state = DEFAULT_STATE.copy()
        state.update(saved_state)

        return state

    except Exception:
        return DEFAULT_STATE.copy()


def save_training_state(state: dict):
    STATE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_file = STATE_FILE.with_suffix(
        ".tmp"
    )

    with open(
        temp_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            state,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temp_file.replace(STATE_FILE)


# ============================================================
# READ STATUS
# ============================================================

def get_training_state():
    return load_training_state()


def get_training_status():
    return load_training_state()


# ============================================================
# SET STATUS
# ============================================================

def set_training_status(
    status: str,
    error: str | None = None,
    model_version: str | None = None,
    metrics: dict | None = None,
):
    state = load_training_state()

    state["status"] = status

    if status == "running":
        state["started_at"] = (
            datetime.utcnow().isoformat()
        )
        state["completed_at"] = None
        state["error"] = None

    elif status in {
        "completed",
        "failed",
        "rejected",
    }:
        state["completed_at"] = (
            datetime.utcnow().isoformat()
        )
        state["error"] = error

    elif error is not None:
        state["error"] = error

    if model_version is not None:
        state["model_version"] = model_version

    if metrics is not None:
        state["metrics"] = metrics

    save_training_state(state)

    return state


# ============================================================
# UPDATE STATUS
# ============================================================

def update_training_state(
    status: str,
    error: str | None = None,
    model_version: str | None = None,
    metrics: dict | None = None,
):
    return set_training_status(
        status=status,
        error=error,
        model_version=model_version,
        metrics=metrics,
    )


def update_training_status(
    status: str,
    error: str | None = None,
    model_version: str | None = None,
    metrics: dict | None = None,
):
    return set_training_status(
        status=status,
        error=error,
        model_version=model_version,
        metrics=metrics,
    )