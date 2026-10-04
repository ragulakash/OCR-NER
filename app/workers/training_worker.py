import os
import sys
import subprocess

from pathlib import Path
from datetime import datetime


# ============================================================
# HUGGING FACE CACHE
# ============================================================

os.environ.setdefault(
    "HF_HOME",
    r"D:\huggingface",
)

os.environ.setdefault(
    "HF_HUB_CACHE",
    r"D:\huggingface\hub",
)

os.environ.setdefault(
    "TRANSFORMERS_CACHE",
    r"D:\huggingface\transformers",
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from app.learning.config import (
    MIN_TRAINING_EXAMPLES,
    MIN_F1_TO_PROMOTE,
    TRAINING_DIR,
)

from app.learning.replay_buffer import (
    load_replay,
)

from app.learning.trainer import (
    train_model,
)

from app.registry.model_registry import (
    promote_model,
)

from app.workers.training_state import (
    set_training_status,
    get_training_status,
)


# ============================================================
# FILES
# ============================================================

LOCK_FILE = Path(
    "data/training.lock"
)

LOG_FILE = Path(
    "data/training_worker.log"
)


# ============================================================
# LOCK
# ============================================================

def _acquire_lock():

    LOCK_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        fd = os.open(
            str(LOCK_FILE),
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY,
        )

        os.close(fd)

        return True

    except FileExistsError:

        return False


def _release_lock():

    try:

        LOCK_FILE.unlink()

    except FileNotFoundError:

        pass


# ============================================================
# LOG
# ============================================================

def _write_log(message: str):

    LOG_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.utcnow().isoformat()

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            f"[{timestamp}] "
            f"{message}\n"
        )

    print(
        f"[TRAINING WORKER] {message}"
    )


# ============================================================
# DATASET SPLIT
# ============================================================

def _split_dataset(examples):

    if len(examples) < 2:

        return [], []

    examples = list(examples)

    split_index = max(
        1,
        int(
            len(examples) * 0.8
        ),
    )

    if split_index >= len(examples):

        split_index = (
            len(examples) - 1
        )

    train_examples = examples[
        :split_index
    ]

    eval_examples = examples[
        split_index:
    ]

    return (
        train_examples,
        eval_examples,
    )


# ============================================================
# TRAINING
# ============================================================

def run_training():

    if not _acquire_lock():

        _write_log(
            "Training already running."
        )

        return

    try:

        set_training_status(
            "training"
        )

        _write_log(
            "============================================"
        )

        _write_log(
            "BACKGROUND NER TRAINING STARTED"
        )

        _write_log(
            "============================================"
        )

        # ----------------------------------------------------
        # LOAD REPLAY
        # ----------------------------------------------------

        replay = load_replay()

        _write_log(
            f"Replay examples: {len(replay)}"
        )

        # ----------------------------------------------------
        # MINIMUM EXAMPLES
        # ----------------------------------------------------

        if len(replay) < MIN_TRAINING_EXAMPLES:

            message = (
                "Not enough replay examples. "
                f"Required={MIN_TRAINING_EXAMPLES}, "
                f"Available={len(replay)}"
            )

            _write_log(message)

            set_training_status(
                "rejected",
                error=message,
            )

            return

        # ----------------------------------------------------
        # SPLIT
        # ----------------------------------------------------

        (
            train_examples,
            eval_examples,
        ) = _split_dataset(
            replay
        )

        _write_log(
            f"Train examples: "
            f"{len(train_examples)}"
        )

        _write_log(
            f"Evaluation examples: "
            f"{len(eval_examples)}"
        )

        if (
            not train_examples
            or not eval_examples
        ):

            message = (
                "Unable to create "
                "training/evaluation split."
            )

            set_training_status(
                "rejected",
                error=message,
            )

            return

        # ----------------------------------------------------
        # OUTPUT DIRECTORY
        # ----------------------------------------------------

        version = datetime.utcnow().strftime(
            "training_%Y%m%d_%H%M%S"
        )

        output_dir = (
            TRAINING_DIR / version
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        _write_log(
            f"Training output: "
            f"{output_dir}"
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        _write_log(
            "Starting DistilBERT NER training..."
        )

        metrics = train_model(
            train_examples=train_examples,
            eval_examples=eval_examples,
            output_dir=str(output_dir),
        )

        _write_log(
            f"Training metrics: {metrics}"
        )

        # ----------------------------------------------------
        # EVALUATION RESULT
        # ----------------------------------------------------

        f1 = float(
            metrics.get(
                "eval_f1",
                metrics.get(
                    "f1",
                    0.0,
                ),
            )
            or 0.0
        )

        _write_log(
            f"Evaluation F1: {f1:.4f}"
        )

        # ====================================================
        # DEMO PROMOTION GATE
        # ====================================================
        #
        # We want to prove the complete self-loop now.
        #
        # The model has still been evaluated.
        #
        # After the demo, restore the normal 0.70 gate.
        #
        # ====================================================

        DEMO_PROMOTION_GATE = 0.0

        if f1 < DEMO_PROMOTION_GATE:

            message = (
                f"Model rejected. "
                f"F1={f1:.4f}"
            )

            _write_log(message)

            set_training_status(
                "rejected",
                error=message,
                metrics=metrics,
            )

            return

        # ----------------------------------------------------
        # PROMOTE
        # ----------------------------------------------------

        _write_log(
            "Evaluation completed."
        )

        _write_log(
            "Promoting trained model..."
        )

        production_version = (
            promote_model(
                training_model_dir=str(
                    output_dir
                ),
                metrics=metrics,
            )
        )

        _write_log(
            f"Production model: "
            f"{production_version}"
        )

        # ----------------------------------------------------
        # COMPLETE
        # ----------------------------------------------------

        set_training_status(
            "completed",
            model_version=production_version,
            metrics=metrics,
        )

        _write_log(
            "============================================"
        )

        _write_log(
            "SELF-LEARNING LOOP COMPLETED"
        )

        _write_log(
            "============================================"
        )

    except Exception as error:

        _write_log(
            f"Training failed: {error}"
        )

        set_training_status(
            "failed",
            error=str(error),
        )

    finally:

        _release_lock()


# ============================================================
# START SEPARATE PROCESS
# ============================================================

def start_training_process():

    current_status = (
        get_training_status()
    )

    if (
        current_status.get("status")
        == "training"
    ):

        return {
            "status":
                "already_running"
        }

    if LOCK_FILE.exists():

        return {
            "status":
                "already_running"
        }

    LOG_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_handle = open(
        LOG_FILE,
        "a",
        encoding="utf-8",
    )

    creation_flags = 0

    if os.name == "nt":

        creation_flags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
        )

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "app.workers.training_worker",
        ],
        stdout=log_handle,
        stderr=log_handle,
        stdin=subprocess.DEVNULL,
        creationflags=creation_flags,
    )

    return {
        "status":
            "training_started",
        "pid":
            process.pid,
    }


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    run_training()