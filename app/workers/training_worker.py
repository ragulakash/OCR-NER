from threading import Lock


_training_lock = Lock()


def run_training():

    if not _training_lock.acquire(
        blocking=False
    ):

        return {
            "status": "already_running"
        }

    try:

        # Training implementation will be
        # connected here.

        return {
            "status": "completed"
        }

    finally:

        _training_lock.release()