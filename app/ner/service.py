from app.ner.gliner import GLiNEREngine
from app.ner.normalization import normalize_entity
from app.ner.mapping import map_known_entity
from app.discovery.candidates import add_candidate


class NERService:

    def __init__(self):

        self.engine = GLiNEREngine()

        self.labels = [
            "PERSON",
            "ORGANIZATION",
            "LOCATION",
            "TECHNOLOGY",
            "PRODUCT",
            "DATABASE",
            "DATE",
            "EMAIL",
            "PHONE",
        ]

    def process(self, text: str):

        response = self.engine.extract_entities(
            text,
            self.labels,
        )

        results = []

        entities_by_type = response.get(
            "entities",
            {}
        )

        for entity_type, entities in entities_by_type.items():

            if not isinstance(entities, list):
                continue

            for entity in entities:

                # GLiNER2 may return either a string
                # or a structured entity when spans
                # and confidence are requested.

                if isinstance(entity, str):

                    entity_text = entity
                    score = 0.0
                    start = text.find(entity_text)

                    if start >= 0:
                        end = start + len(entity_text)
                    else:
                        start = None
                        end = None

                elif isinstance(entity, dict):

                    entity_text = entity.get(
                        "text",
                        ""
                    )

                    score = entity.get(
                        "confidence",
                        entity.get("score", 0.0)
                    )

                    start = entity.get(
                        "start"
                    )

                    end = entity.get(
                        "end"
                    )

                else:
                    continue

                if not entity_text:
                    continue

                normalized = normalize_entity(
                    entity_text
                )

                known = map_known_entity(
                    entity_text
                )

                if known:

                    results.append({
                        "text": entity_text,
                        "normalized": normalized,
                        "type": known["type"],
                        "canonical": known["canonical"],
                        "score": score,
                        "start": start,
                        "end": end,
                        "status": "KNOWN",
                    })

                else:

                    candidate = add_candidate(
                        text=entity_text,
                        proposed_type=entity_type,
                        context=text,
                    )

                    results.append({
                        "text": entity_text,
                        "normalized": normalized,
                        "type": entity_type,
                        "score": score,
                        "start": start,
                        "end": end,
                        "status": "UNKNOWN_CANDIDATE",
                        "candidate_occurrences":
                            candidate["occurrences"],
                    })

        return results