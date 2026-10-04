import json
import subprocess
import sys
from pathlib import Path


QUEUE_FILE = Path("data/candidates/learning_queue.json")
WORKER_LOCK = Path("data/candidates/type_learning.lock")


def _ensure():
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)


def enqueue_entity(entity_text: str):
    """
    Add an entity to the background type-learning queue.

    Duplicate entities are not added twice.
    """

    _ensure()

    try:
        queue = json.loads(
            QUEUE_FILE.read_text(encoding="utf-8")
        )

        if not isinstance(queue, list):
            queue = []

    except Exception:
        queue = []

    normalized = entity_text.strip().lower()

    if not normalized:
        return False

    if normalized not in queue:
        queue.append(normalized)

        QUEUE_FILE.write_text(
            json.dumps(
                queue,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return True

    return False


def load_queue():
    _ensure()

    if not QUEUE_FILE.exists():
        return []

    try:
        queue = json.loads(
            QUEUE_FILE.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(queue, list):
            return queue

        return []

    except Exception:
        return []


def save_queue(queue):
    _ensure()

    if not isinstance(queue, list):
        queue = []

    QUEUE_FILE.write_text(
        json.dumps(
            queue,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def start_learning_worker():
    """
    Start the type-learning worker in a separate process.

    This is intentionally NOT executed inside FastAPI's
    request-processing thread.
    """

    _ensure()

    # Another worker is already running.
    if WORKER_LOCK.exists():
        return False

    WORKER_LOCK.write_text(
        "RUNNING",
        encoding="utf-8",
    )

    try:

        creation_flags = 0

        if sys.platform == "win32":
            creation_flags = (
                subprocess.CREATE_NEW_PROCESS_GROUP
            )

        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "app.discovery.type_learning_worker",
            ],
            creationflags=creation_flags,
        )

        return True

    except Exception:

        WORKER_LOCK.unlink(
            missing_ok=True
        )

        raise