import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from app.learning.config import (
    BASE_MODEL,
    TYPE_CLASSIFIER_DIR,
    LEARNABLE_ENTITY_TYPES,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_FILE = Path(
    "data/datasets/type_training.json"
)

OUTPUT_DIR = Path(TYPE_CLASSIFIER_DIR)

MAX_LENGTH = 256
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
WEIGHT_DECAY = 0.01

MIN_EXAMPLES = 10


# ============================================================
# LABELS
# ============================================================

LABELS = list(LEARNABLE_ENTITY_TYPES)

LABEL2ID = {
    label: index
    for index, label in enumerate(LABELS)
}

ID2LABEL = {
    index: label
    for label, index in LABEL2ID.items()
}


# ============================================================
# LOAD DATASET
# ============================================================

def load_training_data():

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}"
        )

    with open(
        DATASET_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "type_training.json must contain a JSON list."
        )

    examples = []

    for item in data:

        if not isinstance(item, dict):
            continue

        entity = str(
            item.get("entity", "")
        ).strip()

        context = str(
            item.get("context", "")
        ).strip()

        label = str(
            item.get("label", "")
        ).upper().strip()

        if not entity:
            continue

        if not context:
            continue

        if label not in LABEL2ID:
            print(
                f"[TYPE TRAINER] "
                f"Skipping unsupported label: {label}"
            )
            continue

        examples.append(
            {
                "entity": entity,
                "context": context,
                "label": label,
            }
        )

    if len(examples) < MIN_EXAMPLES:
        raise ValueError(
            f"Need at least {MIN_EXAMPLES} "
            f"valid examples. Found {len(examples)}."
        )

    return examples


# ============================================================
# BUILD HUGGING FACE DATASET
# ============================================================

def build_dataset(examples):

    rows = []

    for item in examples:

        text = (
            f"Entity: {item['entity']}\n"
            f"Context: {item['context']}"
        )

        rows.append(
            {
                "text": text,
                "label": LABEL2ID[
                    item["label"]
                ],
            }
        )

    return Dataset.from_list(rows)


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_dataset(dataset, tokenizer):

    def tokenize(batch):

        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )


# ============================================================
# TRAIN
# ============================================================

def train():

    print("=" * 70)
    print("TYPE CLASSIFIER TRAINING")
    print("=" * 70)

    print(
        f"Base model : {BASE_MODEL}"
    )

    print(
        f"Labels     : {LABELS}"
    )

    print(
        f"Dataset    : {DATASET_FILE}"
    )

    print(
        f"CUDA       : {torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU        : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    examples = load_training_data()

    print(
        f"Examples   : {len(examples)}"
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dataset = build_dataset(
        examples
    )

    # --------------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------

    split = dataset.train_test_split(
        test_size=0.2,
        seed=42,
    )

    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(
        f"Train      : {len(train_dataset)}"
    )

    print(
        f"Validation : {len(eval_dataset)}"
    )

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    train_dataset = tokenize_dataset(
        train_dataset,
        tokenizer,
    )

    eval_dataset = tokenize_dataset(
        eval_dataset,
        tokenizer,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            BASE_MODEL,
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # TRAINING ARGUMENTS
    # --------------------------------------------------------

    training_args = TrainingArguments(

        output_dir=str(
            OUTPUT_DIR
        ),

        num_train_epochs=NUM_EPOCHS,

        learning_rate=LEARNING_RATE,

        per_device_train_batch_size=(
            TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            EVAL_BATCH_SIZE
        ),

        weight_decay=WEIGHT_DECAY,

        eval_strategy="epoch",

        save_strategy="no",

        logging_strategy="steps",

        logging_steps=10,

        report_to="none",

        fp16=(
            torch.cuda.is_available()
        ),

        dataloader_pin_memory=(
            torch.cuda.is_available()
        ),
    )

    # --------------------------------------------------------
    # TRAINER
    # --------------------------------------------------------

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }

    # Transformers compatibility
    try:

        trainer = Trainer(
            processing_class=tokenizer,
            **trainer_kwargs,
        )

    except TypeError:

        trainer = Trainer(
            tokenizer=tokenizer,
            **trainer_kwargs,
        )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print()
    print(
        "[TYPE TRAINER] Starting training..."
    )

    trainer.train()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    print()
    print(
        "[TYPE TRAINER] Saving model..."
    )

    trainer.save_model(
        str(OUTPUT_DIR)
    )

    tokenizer.save_pretrained(
        str(OUTPUT_DIR)
    )

    # Save explicit label mapping.
    labels_file = (
        OUTPUT_DIR / "labels.json"
    )

    with open(
        labels_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "labels": LABELS,
                "label2id": LABEL2ID,
                "id2label": {
                    str(k): v
                    for k, v in ID2LABEL.items()
                },
            },
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    print()
    print(
        "[TYPE TRAINER] Evaluating..."
    )

    metrics = trainer.evaluate()

    print()
    print(
        "[TYPE TRAINER] Evaluation:"
    )

    for key, value in metrics.items():

        print(
            f"  {key}: {value}"
        )

    print()
    print("=" * 70)
    print(
        "TYPE CLASSIFIER TRAINING COMPLETE"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train()