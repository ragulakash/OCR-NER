from datetime import datetime


training_state = {
    "status": "idle",
    "started_at": None,
    "completed_at": None,
    "error": None,
    "model_version": None,
}


def set_training_status(
    status: str,
    error=None,
    model_version=None,
):

    training_state["status"] = status
    training_state["error"] = error
    training_state["model_version"] = model_version

    if status == "training":

        training_state["started_at"] = (
            datetime.utcnow().isoformat()
        )

        training_state["completed_at"] = None

    elif status in {
        "completed",
        "failed",
    }:

        training_state["completed_at"] = (
            datetime.utcnow().isoformat()
        )


def get_training_status():

    return training_state.copy()