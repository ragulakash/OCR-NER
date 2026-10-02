from gliner2 import GLiNER2


class GLiNEREngine:

    def __init__(self):

        self.model = GLiNER2.from_pretrained(
            "fastino/gliner2-base-v1"
        )

    def extract_entities(
        self,
        text: str,
        labels: list[str],
    ):

        return self.model.extract_entities(
            text,
            labels,
            include_confidence=True,
            include_spans=True,
        )