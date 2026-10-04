import json

from pathlib import Path


DATASET_DIR = Path(
    "data/datasets"
)

TRAIN_FILE = (
    DATASET_DIR /
    "train.json"
)


# ============================================================
# DIRECTORIES
# ============================================================

def create_directories():

    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# CREATE EXAMPLE
# ============================================================

def create_example(

    text: str,

    entity_text: str,

    entity_type: str,

    start: int | None = None,

    end: int | None = None,
):

    # --------------------------------------------------------
    # Use exact span if supplied
    # --------------------------------------------------------

    if (

        start is not None

        and

        end is not None

    ):

        if (

            start < 0

            or end > len(text)

            or start >= end

        ):

            return None


        actual_text = text[
            start:end
        ]


        if not actual_text.strip():

            return None


        return {

            "text":
                text,

            "entities": [

                {

                    "start":
                        start,

                    "end":
                        end,

                    "label":
                        entity_type,
                }
            ],
        }


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    start = text.find(
        entity_text
    )


    if start == -1:

        return None


    end = (

        start

        +

        len(
            entity_text
        )
    )


    return {

        "text":
            text,

        "entities": [

            {

                "start":
                    start,

                "end":
                    end,

                "label":
                    entity_type,
            }
        ],
    }


# ============================================================
# SAVE
# ============================================================

def save_dataset(

    examples: list[dict],

    file_path:
        Path = TRAIN_FILE,
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


# ============================================================
# LOAD
# ============================================================

def load_dataset(

    file_path:
        Path = TRAIN_FILE,
):

    if not file_path.exists():

        return []


    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# ============================================================
# ADD EXAMPLE
# ============================================================

def add_example(

    text: str,

    entity_text: str,

    entity_type: str,

    start: int | None = None,

    end: int | None = None,
):

    example = create_example(

        text=text,

        entity_text=entity_text,

        entity_type=entity_type,

        start=start,

        end=end,
    )


    if example is None:

        return False


    examples = load_dataset()


    examples.append(
        example
    )


    save_dataset(
        examples
    )


    return True