import json
from pathlib import Path


REPLAY_FILE = Path(
    "data/replay/replay.json"
)


EXAMPLES = [
    {
        "text": "Arjun Technologies hired Ragul Akash.",
        "entities": [
            {
                "start": 0,
                "end": 17,
                "label": "ORGANIZATION",
            }
        ],
    },
    {
        "text": "Ragul Akash joined Arjun Technologies.",
        "entities": [
            {
                "start": 19,
                "end": 36,
                "label": "ORGANIZATION",
            }
        ],
    },
    {
        "text": "The employee works at Arjun Technologies.",
        "entities": [
            {
                "start": 24,
                "end": 41,
                "label": "ORGANIZATION",
            }
        ],
    },
    {
        "text": "Arjun Technologies opened a new office.",
        "entities": [
            {
                "start": 0,
                "end": 17,
                "label": "ORGANIZATION",
            }
        ],
    },
    {
        "text": "The company Arjun Technologies is expanding.",
        "entities": [
            {
                "start": 12,
                "end": 29,
                "label": "ORGANIZATION",
            }
        ],
    },
    {
        "text": "Arjun Technologies is based in Chennai.",
        "entities": [
            {
                "start": 0,
                "end": 17,
                "label": "ORGANIZATION",
            }
        ],
    },
    {
        "text": "A new project was started by Arjun Technologies.",
        "entities": [
            {
                "start": 30,
                "end": 47,
                "label": "ORGANIZATION",
            }
        ],
    },
    {
        "text": "Arjun Technologies develops software.",
        "entities": [
            {
                "start": 0,
                "end": 17,
                "label": "ORGANIZATION",
            }
        ],
    },
    {
        "text": "The team at Arjun Technologies completed the project.",
        "entities": [
            {
                "start": 12,
                "end": 29,
                "label": "ORGANIZATION",
            }
        ],
    },
    {
        "text": "Ragul Akash works for Arjun Technologies.",
        "entities": [
            {
                "start": 22,
                "end": 39,
                "label": "ORGANIZATION",
            }
        ],
    },
    {
        "text": "Arjun Technologies hired new developers.",
        "entities": [
            {
                "start": 0,
                "end": 17,
                "label": "ORGANIZATION",
            }
        ],
    },
    {
        "text": "The software team belongs to Arjun Technologies.",
        "entities": [
            {
                "start": 30,
                "end": 47,
                "label": "ORGANIZATION",
            }
        ],
    },
    {
        "text": "Arjun Technologies is building an AI platform.",
        "entities": [
            {
                "start": 0,
                "end": 17,
                "label": "ORGANIZATION",
            }
        ],
    },
    {
        "text": "Developers joined Arjun Technologies this year.",
        "entities": [
            {
                "start": 18,
                "end": 35,
                "label": "ORGANIZATION",
            }
        ],
    },
    {
        "text": "The headquarters of Arjun Technologies are in Chennai.",
        "entities": [
            {
                "start": 21,
                "end": 38,
                "label": "ORGANIZATION",
            }
        ],
    },
    {
        "text": "Arjun Technologies created a document processing system.",
        "entities": [
            {
                "start": 0,
                "end": 17,
                "label": "ORGANIZATION",
            }
        ],
    },
    {
        "text": "The project was developed at Arjun Technologies.",
        "entities": [
            {
                "start": 30,
                "end": 47,
                "label": "ORGANIZATION",
            }
        ],
    },
    {
        "text": "Arjun Technologies works with AI and machine learning.",
        "entities": [
            {
                "start": 0,
                "end": 17,
                "label": "ORGANIZATION",
            }
        ],
    },
    {
        "text": "Ragul Akash is working at Arjun Technologies.",
        "entities": [
            {
                "start": 27,
                "end": 44,
                "label": "ORGANIZATION",
            }
        ],
    },
    {
        "text": "The new application was built by Arjun Technologies.",
        "entities": [
            {
                "start": 35,
                "end": 52,
                "label": "ORGANIZATION",
            }
        ],
    },
]


def load_replay():
    if not REPLAY_FILE.exists():
        return []

    with open(
        REPLAY_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_replay(data):
    REPLAY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_file = REPLAY_FILE.with_suffix(
        ".tmp"
    )

    with open(
        temp_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temp_file.replace(
        REPLAY_FILE
    )


def main():

    replay = load_replay()

    existing = {
        (
            item.get("text"),
            tuple(
                (
                    entity.get("start"),
                    entity.get("end"),
                    entity.get("label"),
                )
                for entity in item.get(
                    "entities",
                    [],
                )
            ),
        )
        for item in replay
    }

    added = 0

    for example in EXAMPLES:

        key = (
            example["text"],
            tuple(
                (
                    entity["start"],
                    entity["end"],
                    entity["label"],
                )
                for entity in example["entities"]
            ),
        )

        if key in existing:
            continue

        replay.append(example)
        existing.add(key)
        added += 1

    save_replay(replay)

    print(
        f"Added examples: {added}"
    )

    print(
        f"Total replay examples: "
        f"{len(replay)}"
    )


if __name__ == "__main__":
    main()