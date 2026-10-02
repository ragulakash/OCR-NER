from typing import Any


def flatten_json(data: Any) -> str:

    if isinstance(data, dict):

        parts = []

        for key, value in data.items():

            parts.append(
                str(key)
            )

            parts.append(
                flatten_json(value)
            )

        return " ".join(parts)


    if isinstance(data, list):

        return " ".join(
            flatten_json(item)
            for item in data
        )


    return str(data)


def extract_json_text(
    data: dict[str, Any]
) -> str:

    return flatten_json(data)