from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

from app.learning.config import (
    BASE_MODEL,
    LABEL2ID,
    ID2LABEL,
    NUM_EPOCHS,
    LEARNING_RATE,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    WEIGHT_DECAY,
)


def tokenize_and_align_labels(
    examples,
    tokenizer,
):

    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )

    all_labels = []

    for batch_index, offsets in enumerate(
        tokenized["offset_mapping"]
    ):

        entities = examples["entities"][batch_index]

        labels = [0] * len(offsets)

        for entity in entities:

            start = entity["start"]
            end = entity["end"]
            entity_type = entity["label"]

            begin_label = LABEL2ID[
                f"B-{entity_type}"
            ]

            inside_label = LABEL2ID[
                f"I-{entity_type}"
            ]

            first_token = True

            for token_index, (
                token_start,
                token_end,
            ) in enumerate(offsets):

                if token_start == token_end:
                    continue

                overlaps = (
                    token_start < end
                    and token_end > start
                )

                if not overlaps:
                    continue

                if first_token:

                    labels[token_index] = (
                        begin_label
                    )

                    first_token = False

                else:

                    labels[token_index] = (
                        inside_label
                    )

        all_labels.append(labels)

    tokenized["labels"] = all_labels

    tokenized.pop(
        "offset_mapping"
    )

    return tokenized


def build_dataset(
    examples,
):

    dataset = Dataset.from_list(
        examples
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    tokenized_dataset = dataset.map(
        lambda batch:
            tokenize_and_align_labels(
                batch,
                tokenizer,
            ),
        batched=True,
        remove_columns=dataset.column_names,
    )

    return (
        tokenized_dataset,
        tokenizer,
    )


def train_model(
    train_examples: list[dict],
    eval_examples: list[dict],
    output_dir: str,
):

    train_dataset, tokenizer = build_dataset(
        train_examples
    )

    eval_dataset, _ = build_dataset(
        eval_examples
    )

    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=False,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(
        output_dir
    )

    tokenizer.save_pretrained(
        output_dir
    )

    metrics = trainer.evaluate()

    return metrics