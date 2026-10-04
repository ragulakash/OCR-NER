import json
import re
from pathlib import Path


REPLAY_FILE = Path(
    "data/replay/replay.json"
)


# ============================================================
# STORAGE
# ============================================================

def _ensure_storage():
    REPLAY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not REPLAY_FILE.exists():
        REPLAY_FILE.write_text(
            "[]",
            encoding="utf-8",
        )


def load_replay():
    _ensure_storage()

    try:

        data = json.loads(
            REPLAY_FILE.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(data, list):
            return data

        return []

    except Exception:
        return []


def save_replay(replay):
    _ensure_storage()

    REPLAY_FILE.write_text(
        json.dumps(
            replay,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# NORMALIZATION
# ============================================================

def _normalize_text(text):
    return re.sub(
        r"\s+",
        " ",
        str(text or "").strip().lower(),
    )


def _normalize_label(label):
    return str(
        label or ""
    ).strip().upper()


# ============================================================
# VALIDATION
# ============================================================

def _validate_example(example):
    if not isinstance(
        example,
        dict,
    ):
        return False

    text = str(
        example.get(
            "text",
            "",
        )
        or ""
    )

    entities = example.get(
        "entities",
        [],
    )

    if not text.strip():
        return False

    if not isinstance(
        entities,
        list,
    ):
        return False

    for entity in entities:

        if not isinstance(
            entity,
            dict,
        ):
            return False

        start = entity.get(
            "start"
        )

        end = entity.get(
            "end"
        )

        label = entity.get(
            "label"
        )

        if not isinstance(
            start,
            int,
        ):
            return False

        if not isinstance(
            end,
            int,
        ):
            return False

        if start < 0:
            return False

        if end <= start:
            return False

        if end > len(text):
            return False

        if not str(
            label or ""
        ).strip():
            return False

    return True


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def _is_duplicate(
    replay,
    example,
):
    example_text = _normalize_text(
        example.get(
            "text",
            "",
        )
    )

    example_entities = example.get(
        "entities",
        [],
    )

    for existing in replay:

        if _normalize_text(
            existing.get(
                "text",
                "",
            )
        ) != example_text:
            continue

        existing_entities = existing.get(
            "entities",
            [],
        )

        if len(
            existing_entities
        ) != len(
            example_entities
        ):
            continue

        same = True

        for a, b in zip(
            existing_entities,
            example_entities,
        ):

            if (
                a.get("start")
                != b.get("start")
                or a.get("end")
                != b.get("end")
                or _normalize_label(
                    a.get("label")
                )
                != _normalize_label(
                    b.get("label")
                )
            ):
                same = False
                break

        if same:
            return True

    return False


# ============================================================
# BULK ADD
# ============================================================

def add_examples(examples):
    """
    Add multiple replay examples.

    IMPORTANT:
    This function returns a SUMMARY DICT.

    It does NOT return the replay list.
    """

    replay = load_replay()

    if not isinstance(
        examples,
        list,
    ):
        examples = []

    added = 0
    duplicates = 0
    invalid = 0

    for example in examples:

        if not _validate_example(
            example
        ):

            invalid += 1
            continue

        if _is_duplicate(
            replay,
            example,
        ):

            duplicates += 1
            continue

        replay.append(
            example
        )

        added += 1

    save_replay(
        replay
    )

    return {
        "added": added,
        "duplicates": duplicates,
        "invalid": invalid,
        "total": len(replay),
    }


# ============================================================
# FIND ENTITY SPAN
# ============================================================

def _find_entity_span(
    text,
    entity_text,
):
    """
    Find entity inside the actual context text.

    This is deliberately local to the context.
    We never reuse offsets from the full document.
    """

    if not text or not entity_text:
        return None

    # Exact match first.
    index = text.find(
        entity_text
    )

    if index >= 0:

        return (
            index,
            index + len(entity_text),
        )

    # Case-insensitive fallback.
    pattern = re.escape(
        entity_text
    )

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    )

    if match:

        return (
            match.start(),
            match.end(),
        )

    # Whitespace-tolerant fallback.
    parts = re.split(
        r"\s+",
        entity_text.strip(),
    )

    if parts:

        pattern = r"\s+".join(
            re.escape(part)
            for part in parts
            if part
        )

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:

            return (
                match.start(),
                match.end(),
            )

    return None


# ============================================================
# SINGLE REPLAY EXAMPLE
# ============================================================

def add_replay_example(
    text,
    label,
    contexts=None,
    entity_text=None,
    start=None,
    end=None,
):
    """
    Add one or more replay examples.

    Supported usage:

    1. Full document with explicit span:

        add_replay_example(
            text=document_text,
            label="ORGANIZATION",
            entity_text="Arjun Technologies",
            start=100,
            end=118,
        )

    2. Context text:

        add_replay_example(
            text=context_text,
            label="ORGANIZATION",
            entity_text="Arjun Technologies",
        )

    The important rule is:

        offsets always belong to `text`.

    We never apply document-global offsets to
    a shortened context string.
    """

    added_total = 0

    # --------------------------------------------------------
    # Explicit full-text example
    # --------------------------------------------------------

    if (
        entity_text
        and start is not None
        and end is not None
    ):

        try:

            start = int(start)
            end = int(end)

        except Exception:

            return 0

        example = {
            "text": str(
                text
                or ""
            ),
            "entities": [
                {
                    "start": start,
                    "end": end,
                    "label": _normalize_label(
                        label
                    ),
                }
            ],
        }

        result = add_examples(
            [example]
        )

        return int(
            result.get(
                "added",
                0,
            )
        )

    # --------------------------------------------------------
    # Context list
    # --------------------------------------------------------

    if contexts is not None:

        if not isinstance(
            contexts,
            list,
        ):
            contexts = [
                contexts
            ]

        examples = []

        for context in contexts:

            if isinstance(
                context,
                str,
            ):

                context_text = context

            elif isinstance(
                context,
                dict,
            ):

                context_text = str(
                    context.get(
                        "context",
                        context.get(
                            "context_text",
                            context.get(
                                "text",
                                "",
                            ),
                        ),
                    )
                    or ""
                )

            else:

                continue

            if not context_text.strip():
                continue

            span = _find_entity_span(
                context_text,
                entity_text,
            )

            if span is None:
                continue

            local_start, local_end = span

            examples.append(
                {
                    "text": context_text,
                    "entities": [
                        {
                            "start": local_start,
                            "end": local_end,
                            "label": _normalize_label(
                                label
                            ),
                        }
                    ],
                }
            )

        result = add_examples(
            examples
        )

        return int(
            result.get(
                "added",
                0,
            )
        )

    # --------------------------------------------------------
    # Entity text inside supplied text
    # --------------------------------------------------------

    if entity_text:

        span = _find_entity_span(
            str(text or ""),
            entity_text,
        )

        if span is None:
            return 0

        local_start, local_end = span

        example = {
            "text": str(
                text
                or ""
            ),
            "entities": [
                {
                    "start": local_start,
                    "end": local_end,
                    "label": _normalize_label(
                        label
                    ),
                }
            ],
        }

        result = add_examples(
            [example]
        )

        return int(
            result.get(
                "added",
                0,
            )
        )

    return 0


# ============================================================
# COUNT
# ============================================================

def get_replay_count():
    return len(
        load_replay()
    )