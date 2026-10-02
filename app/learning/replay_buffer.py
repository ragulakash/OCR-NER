import json
from pathlib import Path


REPLAY_FILE = Path(
    "data/replay/replay.json"
)


def load_replay():

    if not REPLAY_FILE.exists():
        return []

    with open(
        REPLAY_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def save_replay(
    examples: list[dict],
):

    REPLAY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPLAY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            examples,
            file,
            indent=2,
            ensure_ascii=False,
        )


def add_examples(
    new_examples: list[dict],
):

    replay = load_replay()

    existing = {
        (
            item["text"],
            item["entities"][0]["start"],
            item["entities"][0]["end"],
            item["entities"][0]["label"],
        )
        for item in replay
        if item.get("entities")
    }

    for example in new_examples:

        if not example.get("entities"):
            continue

        entity = example["entities"][0]

        key = (
            example["text"],
            entity["start"],
            entity["end"],
            entity["label"],
        )

        if key not in existing:

            replay.append(example)

            existing.add(key)

    save_replay(replay)

    return replay