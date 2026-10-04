import os


# ============================================================
# FORCE HUGGING FACE CACHE TO D:
# ============================================================

os.environ["HF_HOME"] = r"D:\huggingface"

os.environ["HF_HUB_CACHE"] = (
    r"D:\huggingface\hub"
)

os.environ["TRANSFORMERS_CACHE"] = (
    r"D:\huggingface\transformers"
)


from pathlib import Path

import numpy as np
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

from app.learning.config import (
    BASE_MODEL,
    NUM_EPOCHS,
    LEARNING_RATE,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    WEIGHT_DECAY,
    MAX_LENGTH,
    LABEL2ID,
    ID2LABEL,
)


# ============================================================
# TOKEN + BIO ALIGNMENT
# ============================================================

def tokenize_and_align_labels(
    examples,
    tokenizer,
):

    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
    )

    all_labels = []

    for index, offsets in enumerate(
        tokenized["offset_mapping"]
    ):

        labels = [-100] * len(offsets)

        entities = examples[
            "entities"
        ][index]

        for token_index, (
            token_start,
            token_end,
        ) in enumerate(offsets):

            # Special token.
            if token_start == token_end:
                continue

            label = "O"

            for entity in entities:

                entity_start = int(
                    entity["start"]
                )

                entity_end = int(
                    entity["end"]
                )

                entity_label = str(
                    entity["label"]
                )

                # Token overlaps entity.
                if (
                    token_start < entity_end
                    and token_end > entity_start
                ):

                    if token_start <= entity_start:
                        label = (
                            f"B-{entity_label}"
                        )
                    else:
                        label = (
                            f"I-{entity_label}"
                        )

                    break

            labels[token_index] = (
                LABEL2ID[label]
            )

        all_labels.append(labels)

    tokenized["labels"] = all_labels

    tokenized.pop(
        "offset_mapping",
        None,
    )

    return tokenized


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    evaluation,
):

    predictions = evaluation.predictions

    labels = evaluation.label_ids

    predicted_ids = np.argmax(
        predictions,
        axis=-1,
    )

    true_predictions = []

    true_labels = []

    for prediction, label in zip(
        predicted_ids,
        labels,
    ):

        current_predictions = []

        current_labels = []

        for pred, true in zip(
            prediction,
            label,
        ):

            if true == -100:
                continue

            current_predictions.append(
                ID2LABEL[int(pred)]
            )

            current_labels.append(
                ID2LABEL[int(true)]
            )

        true_predictions.append(
            current_predictions
        )

        true_labels.append(
            current_labels
        )

    precision = precision_score(
        true_labels,
        true_predictions,
        zero_division=0,
    )

    recall = recall_score(
        true_labels,
        true_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        true_labels,
        true_predictions,
        zero_division=0,
    )

    return {
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1": float(
            f1
        ),
    }


# ============================================================
# DATASET
# ============================================================

def prepare_dataset(
    examples,
    tokenizer,
):

    dataset = Dataset.from_list(
        examples
    )

    dataset = dataset.map(
        lambda batch:
            tokenize_and_align_labels(
                batch,
                tokenizer,
            ),
        batched=True,
        remove_columns=[
            "text",
            "entities",
        ],
    )

    return dataset


# ============================================================
# TRAIN
# ============================================================

def train_model(
    train_examples,
    eval_examples,
    output_dir,
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 60
    )

    print(
        "TRAINING MODEL"
    )

    print(
        "=" * 60
    )

    print(
        "HF_HOME:",
        os.environ.get(
            "HF_HOME"
        ),
    )

    print(
        "HF_HUB_CACHE:",
        os.environ.get(
            "HF_HUB_CACHE"
        ),
    )

    print(
        "Train examples:",
        len(train_examples),
    )

    print(
        "Eval examples:",
        len(eval_examples),
    )

    print(
        "Device:",
        "CUDA"
        if torch.cuda.is_available()
        else "CPU",
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    tokenizer = (
        AutoTokenizer.from_pretrained(
            BASE_MODEL
        )
    )

    # ========================================================
    # DATA
    # ========================================================

    train_dataset = (
        prepare_dataset(
            train_examples,
            tokenizer,
        )
    )

    eval_dataset = (
        prepare_dataset(
            eval_examples,
            tokenizer,
        )
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = (
        AutoModelForTokenClassification
        .from_pretrained(
            BASE_MODEL,
            num_labels=len(
                LABEL2ID
            ),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
    )

    # ========================================================
    # COLLATOR
    # ========================================================

    data_collator = (
        DataCollatorForTokenClassification(
            tokenizer=tokenizer
        )
    )

    # ========================================================
    # TRAINING ARGUMENTS
    # ========================================================

    training_args = TrainingArguments(

        output_dir=str(
            output_dir
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

        # CRITICAL:
        # Don't create huge checkpoints
        # after every epoch.
        save_strategy="no",

        logging_strategy="steps",

        logging_steps=5,

        report_to="none",

        fp16=(
            torch.cuda.is_available()
        ),

        dataloader_pin_memory=(
            torch.cuda.is_available()
        ),

        # Don't attempt to restore a
        # checkpoint because none are saved.
        load_best_model_at_end=False,
    )

    # ========================================================
    # TRAINER
    # ========================================================

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=eval_dataset,

        processing_class=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics,
    )

    # ========================================================
    # TRAIN
    # ========================================================

    trainer.train()

    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    metrics = trainer.evaluate()

    print(
        "Final evaluation:"
    )

    print(metrics)

    # ========================================================
    # SAVE FINAL MODEL ONLY
    # ========================================================

    print(
        "Saving final model..."
    )

    trainer.save_model(
        str(output_dir)
    )

    tokenizer.save_pretrained(
        str(output_dir)
    )

    print(
        "Model saved:",
        output_dir,
    )

    return metrics