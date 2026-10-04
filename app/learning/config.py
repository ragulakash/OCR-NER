from pathlib import Path


# ============================================================
# BASE MODEL
# ============================================================

BASE_MODEL = (
    "distilbert/distilbert-base-uncased"
)


# ============================================================
# MODEL DIRECTORIES
# ============================================================

TRAINING_DIR = Path(
    "models/training"
)

PRODUCTION_DIR = Path(
    "models/production"
)

TYPE_CLASSIFIER_DIR = Path(
    "models/type_classifier"
)


# ============================================================
# DATA DIRECTORIES
# ============================================================

DATASET_DIR = Path(
    "data/datasets"
)

REPLAY_DIR = Path(
    "data/replay"
)

EVALUATION_DIR = Path(
    "data/evaluation"
)


# ============================================================
# NER TRAINING PARAMETERS
# ============================================================

NUM_EPOCHS = 5

LEARNING_RATE = 5e-5

TRAIN_BATCH_SIZE = 4

EVAL_BATCH_SIZE = 4

WEIGHT_DECAY = 0.01

MAX_LENGTH = 256


# ============================================================
# CANDIDATE LEARNING GATES
# ============================================================

# Candidate must be observed at least this many times
# before it can become CONFIRMED.

MIN_CANDIDATE_OCCURRENCES = 3


# Candidate/type learning should not start training with
# a tiny replay dataset.

MIN_TRAINING_EXAMPLES = 30


# Minimum held-out F1 required before a trained model
# can be promoted to production.

MIN_F1_TO_PROMOTE = 0.70


# Minimum GLiNER candidate confidence.

MIN_CANDIDATE_CONFIDENCE = 0.70


# ============================================================
# ENTITY TYPES
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


# ============================================================
# TYPES THAT CAN BE LEARNED BY DISTILBERT NER
# ============================================================

LEARNABLE_ENTITY_TYPES = [
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "TECHNOLOGY",
    "PRODUCT",
    "DATABASE",
]


# ============================================================
# STRUCTURED TYPES
#
# These are handled through deterministic validation rather
# than automatic supervised NER learning.
# ============================================================

STRUCTURED_ENTITY_TYPES = [
    "DATE",
    "EMAIL",
    "PHONE",
]


# ============================================================
# BIO LABELS
# ============================================================

LABEL_LIST = [
    "O"
]

for entity_type in ENTITY_TYPES:

    LABEL_LIST.append(
        f"B-{entity_type}"
    )

    LABEL_LIST.append(
        f"I-{entity_type}"
    )


# ============================================================
# LABEL MAPPINGS
# ============================================================

LABEL2ID = {
    label: index
    for index, label in enumerate(
        LABEL_LIST
    )
}

ID2LABEL = {
    index: label
    for label, index in LABEL2ID.items()
}