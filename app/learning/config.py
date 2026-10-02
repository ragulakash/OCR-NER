from pathlib import Path


# ============================================================
# MODEL
# ============================================================

BASE_MODEL = "distilbert/distilbert-base-uncased"


# ============================================================
# DIRECTORIES
# ============================================================

TRAINING_DIR = Path("models/training")
PRODUCTION_DIR = Path("models/production")

DATASET_DIR = Path("data/datasets")
REPLAY_DIR = Path("data/replay")
EVALUATION_DIR = Path("data/evaluation")


# ============================================================
# TRAINING
# ============================================================

NUM_EPOCHS = 2

LEARNING_RATE = 5e-5

TRAIN_BATCH_SIZE = 8

EVAL_BATCH_SIZE = 8

WEIGHT_DECAY = 0.01


# ============================================================
# AUTOMATIC LEARNING
# ============================================================

MIN_CANDIDATE_OCCURRENCES = 3

MIN_TRAINING_EXAMPLES = 10

MIN_F1_TO_PROMOTE = 0.70


# ============================================================
# NER LABELS
# ============================================================

ENTITY_TYPES = [
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "TECHNOLOGY",
    "PRODUCT",
    "DATABASE",
    "DATE",
    "EMAIL",
    "PHONE",
]


# BIO labels

LABEL_LIST = ["O"]

for entity_type in ENTITY_TYPES:
    LABEL_LIST.append(f"B-{entity_type}")
    LABEL_LIST.append(f"I-{entity_type}")


LABEL2ID = {
    label: index
    for index, label in enumerate(LABEL_LIST)
}


ID2LABEL = {
    index: label
    for label, index in LABEL2ID.items()
}