import json
from pathlib import Path


DATASET_DIR = Path("data/datasets")

TRAIN_FILE = DATASET_DIR / "train.json"


def create_directories():

    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def create_example(
    text: str,
    entity_text: str,
    entity_type: str,
):
    """
    Create an entity example using character offsets.
    """

    start = text.find(entity_text)

    if start == -1:
        return None

    end = start + len(entity_text)

    return {
        "text": text,
        "entities": [
            {
                "start": start,
                "end": end,
                "label": entity_type,
            }
        ],
    }


def save_dataset(
    examples: list[dict],
    file_path: Path = TRAIN_FILE,
):

    create_directories()

    with open(
        file_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            examples,
            file,
            indent=2,
            ensure_ascii=False,
        )


def load_dataset(
    file_path: Path = TRAIN_FILE,
):

    if not file_path.exists():
        return []

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def add_example(
    text: str,
    entity_text: str,
    entity_type: str,
):

    example = create_example(
        text,
        entity_text,
        entity_type,
    )

    if example is None:
        return False

    examples = load_dataset()

    examples.append(example)

    save_dataset(examples)

    return True