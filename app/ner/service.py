from __future__ import annotations


from app.ner.gliner import (
    GLiNEREngine,
)

from app.ner.mapping import (
    map_known_entity,
    find_known_phrases,
)

from app.ner.normalization import (
    normalize_entity,
)

from app.ner.production import (
    ProductionNER,
)

from app.discovery.candidates import (
    add_candidate,
)

from app.discovery.learning_queue import (
    enqueue_entity,
    start_learning_worker,
)

from app.learning.replay_buffer import (
    add_examples,
)

from app.learning.gemini_validator import (
    get_gemini_validator,
)


# ============================================================
# CONFIGURATION
# ============================================================

NON_LEARNABLE_TYPES = {
    "DATE",
    "EMAIL",
    "PHONE",
}


LEARNABLE_TYPES = {
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "TECHNOLOGY",
    "PRODUCT",
    "DATABASE",
}


ALL_TYPES = (
    LEARNABLE_TYPES
    | NON_LEARNABLE_TYPES
)


MIN_GLINER_CONFIDENCE = 0.35


# ============================================================
# SERVICE
# ============================================================

class NERService:

    def __init__(self):

        print(
            "[NER SERVICE] Initializing GLiNER2..."
        )

        self.gliner = GLiNEREngine()

        print(
            "[NER SERVICE] Initializing production NER..."
        )

        self.production = ProductionNER()

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

        print(
            "[NER SERVICE] Ready."
        )

    # ========================================================
    # GLINER NORMALIZATION
    # ========================================================

    def _normalize_gliner_output(
        self,
        output,
        text: str,
    ):

        entities = []

        if not isinstance(
            output,
            dict,
        ):
            return entities

        raw_entities = output.get(
            "entities",
            {},
        )

        if not isinstance(
            raw_entities,
            dict,
        ):
            return entities

        for entity_type, values in raw_entities.items():

            if not isinstance(
                values,
                list,
            ):
                values = [
                    values
                ]

            for value in values:

                if isinstance(
                    value,
                    dict,
                ):

                    entity_text = (
                        value.get(
                            "text"
                        )
                        or value.get(
                            "entity"
                        )
                        or value.get(
                            "value"
                        )
                    )

                    confidence = float(
                        value.get(
                            "confidence",
                            value.get(
                                "score",
                                0.0,
                            ),
                        )
                        or 0.0
                    )

                    start = value.get(
                        "start"
                    )

                    end = value.get(
                        "end"
                    )

                elif isinstance(
                    value,
                    str,
                ):

                    entity_text = value

                    confidence = 0.0

                    start = text.find(
                        entity_text
                    )

                    end = (
                        start + len(
                            entity_text
                        )
                        if start >= 0
                        else None
                    )

                else:

                    continue

                if not entity_text:
                    continue

                entity_text = " ".join(
                    str(entity_text or "").strip().split()
                )

                if not entity_text:
                    continue

                # ------------------------------------------------
                # Recover span
                # ------------------------------------------------

                if (
                    start is None
                    or end is None
                ):

                    start = text.find(
                        entity_text
                    )

                    if start >= 0:

                        end = (
                            start
                            + len(
                                entity_text
                            )
                        )

                # ------------------------------------------------
                # Convert span
                # ------------------------------------------------

                if start is not None:

                    try:
                        start = int(
                            start
                        )
                    except Exception:
                        start = None

                if end is not None:

                    try:
                        end = int(
                            end
                        )
                    except Exception:
                        end = None

                # ------------------------------------------------
                # Validate span
                # ------------------------------------------------

                if (
                    start is not None
                    and end is not None
                ):

                    if (
                        start < 0
                        or end <= start
                        or end > len(text)
                    ):

                        recovered = text.find(
                            entity_text
                        )

                        if recovered >= 0:

                            start = recovered
                            end = (
                                recovered
                                + len(
                                    entity_text
                                )
                            )

                        else:

                            start = None
                            end = None

                # ------------------------------------------------
                # Label normalization
                # ------------------------------------------------

                normalized_type = str(
                    entity_type
                ).strip().upper()

                label_mapping = {
                    "PERSON": "PERSON",
                    "PERSON, INDIVIDUAL NAME": "PERSON",
                    "ORGANIZATION": "ORGANIZATION",
                    "ORGANIZATION, COMPANY, VENDOR, SUPPLIER": "ORGANIZATION",
                    "LOCATION": "LOCATION",
                    "LOCATION, ADDRESS, CITY, COUNTRY": "LOCATION",
                    "TECHNOLOGY": "TECHNOLOGY",
                    "TECHNOLOGY, SOFTWARE, ALGORITHM, MODEL, FRAMEWORK, LIBRARY, ARCHITECTURE": "TECHNOLOGY",
                    "PRODUCT": "PRODUCT",
                    "PRODUCT, COMMERCIAL ITEM, MERCHANDISE, PHYSICAL GOOD": "PRODUCT",
                    "DATABASE": "DATABASE",
                    "DATABASE, DATA STORE": "DATABASE",
                    "DATE": "DATE",
                    "DATE, TIMESTAMP": "DATE",
                    "EMAIL": "EMAIL",
                    "EMAIL ADDRESS": "EMAIL",
                    "PHONE": "PHONE",
                    "PHONE NUMBER": "PHONE",
                }

                normalized_type = label_mapping.get(
                    normalized_type,
                    normalized_type,
                )

                entities.append(
                    {
                        "text": entity_text,
                        "type": normalized_type,
                        "score": confidence,
                        "start": start,
                        "end": end,
                        "source": "gliner",
                    }
                )

        return entities

    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    def _validate_gliner_entities(
        self,
        entities,
        text: str,
    ):

        valid = []

        for entity in entities:

            entity_text = " ".join(
                str(
                    entity.get(
                        "text",
                        "",
                    )
                    or ""
                ).strip().split()
            )

            entity_type = str(
                entity.get(
                    "type",
                    "",
                )
                or ""
            ).upper()

            score = float(
                entity.get(
                    "score",
                    0.0,
                )
                or 0.0
            )

            if not entity_text:
                continue

            if len(entity_text) < 2:
                continue

            lower_text = entity_text.lower()
            noise_words = {
                "upgrade plan", "io", "rai", "station ma", "click here", "read more",
                "view invoice", "page 1", "total pages", "subtotal", "total", "amount",
                "qty", "description", "item", "price", "tax", "resnet50", "knn", "svm",
                "vgg16", "distilbert", "bert", "gpt", "gliner"
            }
            if lower_text in noise_words:
                continue

            if len(entity_text) < 3 and not entity_text.isupper():
                continue

            if entity_type not in ALL_TYPES:

                print(
                    "[GLINER] Ignoring unsupported type:",
                    entity_type,
                    entity_text,
                )

                continue

            if score < MIN_GLINER_CONFIDENCE:
                continue

            valid.append(
                entity
            )

        return valid

    # ========================================================
    # OVERLAP RESOLUTION
    # ========================================================

    def _resolve_overlaps(
        self,
        entities,
    ):

        sorted_entities = sorted(
            entities,
            key=lambda item: (
                item.get(
                    "start"
                )
                if item.get(
                    "start"
                ) is not None
                else 10**12,
                -(
                    (
                        item.get(
                            "end"
                        )
                        or 0
                    )
                    - (
                        item.get(
                            "start"
                        )
                        or 0
                    )
                ),
            ),
        )

        accepted = []

        for entity in sorted_entities:

            start = entity.get(
                "start"
            )

            end = entity.get(
                "end"
            )

            if (
                start is None
                or end is None
            ):

                accepted.append(
                    entity
                )

                continue

            overlaps = False

            for existing in accepted:

                existing_start = existing.get(
                    "start"
                )

                existing_end = existing.get(
                    "end"
                )

                if (
                    existing_start is None
                    or existing_end is None
                ):
                    continue

                if (
                    start < existing_end
                    and end > existing_start
                ):

                    overlaps = True

                    existing_score = float(
                        existing.get(
                            "score",
                            0.0,
                        )
                        or 0.0
                    )

                    current_score = float(
                        entity.get(
                            "score",
                            0.0,
                        )
                        or 0.0
                    )

                    if current_score > existing_score:

                        accepted.remove(
                            existing
                        )

                        accepted.append(
                            entity
                        )

                    break

            if not overlaps:

                accepted.append(
                    entity
                )

        return accepted

    # ========================================================
    # MAIN EXTRACTION
    # ========================================================

    def process(
        self,
        text: str,
    ):
        return self.extract_entities(text)

    def extract_entities(
        self,
        text: str,
    ):

        if not text or not text.strip():

            return {
                "total": 0,
                "known": [],
                "known_types": [],
                "production": [],
                "unknown_candidates": [],
            }

        print(
            "[NER SERVICE] Running GLiNER2..."
        )

        gliner_raw = self.gliner.extract_entities(
            text,
            [
                "person",
                "organization",
                "location",
                "technology",
                "product",
                "database",
                "date",
                "email address",
                "phone number",
            ],
        )

        print(
            "[NER SERVICE] GLiNER raw output:",
            gliner_raw,
        )

        gliner_entities = (
            self._normalize_gliner_output(
                gliner_raw,
                text,
            )
        )

        print(
            "[NER SERVICE] GLiNER normalized:",
            len(
                gliner_entities
            ),
            "entities",
        )

        gliner_entities = (
            self._validate_gliner_entities(
                gliner_entities,
                text,
            )
        )

        gliner_entities = (
            self._resolve_overlaps(
                gliner_entities
            )
        )

        # ====================================================
        # KNOWN PHRASES
        # ====================================================

        known_phrase_entities = []

        try:

            known_phrase_entities = (
                find_known_phrases(
                    text
                )
            )

        except Exception as exc:

            print(
                "[NER SERVICE] Known phrase error:",
                exc,
            )

        # ====================================================
        # PRODUCTION MODEL
        # ====================================================

        production_entities = []

        try:

            production_entities = (
                self.production.extract_entities(
                    text
                )
                or []
            )

        except Exception as exc:

            print(
                "[NER SERVICE] Production NER error:",
                exc,
            )

        # ====================================================
        # RESULT ARRAYS
        # ====================================================

        known = []
        known_types = []
        production = []
        unknown_candidates = []

        # ====================================================
        # ADD KNOWN PHRASES
        # ====================================================

        for entity in known_phrase_entities:

            if not isinstance(
                entity,
                dict,
            ):
                continue

            entity_text = str(
                entity.get(
                    "text",
                    "",
                )
                or ""
            ).strip()

            entity_type = str(
                entity.get(
                    "type",
                    entity.get(
                        "label",
                        "",
                    ),
                )
                or ""
            ).upper()

            if not entity_text:
                continue

            result = {
                "text": entity_text,
                "normalized": normalize_entity(
                    entity_text
                ),
                "type": entity_type,
                "score": float(
                    entity.get(
                        "score",
                        1.0,
                    )
                    or 1.0
                ),
                "start": entity.get(
                    "start"
                ),
                "end": entity.get(
                    "end"
                ),
                "status": "KNOWN",
                "source": "known_phrase_match",
            }

            if entity.get(
                "canonical"
            ):

                result["canonical"] = (
                    entity[
                        "canonical"
                    ]
                )

            known.append(
                result
            )

        # ====================================================
        # PRODUCTION
        # ====================================================

        for entity in production_entities:

            if not isinstance(
                entity,
                dict,
            ):
                continue

            result = dict(
                entity
            )

            result.setdefault(
                "status",
                "PRODUCTION_MODEL",
            )

            result.setdefault(
                "source",
                "distilbert",
            )

            production.append(
                result
            )

        # ====================================================
        # PROCESS GLINER DISCOVERY
        # ====================================================

        for entity in gliner_entities:

            entity_text = str(
                entity.get(
                    "text",
                    "",
                )
                or ""
            ).strip()

            entity_type = str(
                entity.get(
                    "type",
                    "",
                )
                or ""
            ).upper()

            score = float(
                entity.get(
                    "score",
                    0.0,
                )
                or 0.0
            )

            start = entity.get(
                "start"
            )

            end = entity.get(
                "end"
            )

            if not entity_text:
                continue

            # ------------------------------------------------
            # Known mapping
            # ------------------------------------------------

            canonical = None

            try:

                mapping = map_known_entity(
                    entity_text
                )

            except Exception:

                mapping = None

            if mapping:

                if isinstance(
                    mapping,
                    dict,
                ):

                    mapped_type = str(
                        mapping.get(
                            "type",
                            entity_type,
                        )
                        or entity_type
                    ).upper()

                    canonical = mapping.get(
                        "canonical"
                    )

                else:

                    mapped_type = str(
                        mapping
                    ).upper()

                result = {
                    "text": entity_text,
                    "normalized": normalize_entity(
                        entity_text
                    ),
                    "type": mapped_type,
                    "score": score,
                    "start": start,
                    "end": end,
                    "status": "KNOWN",
                    "source": "gliner_mapping",
                }

                if canonical:
                    result[
                        "canonical"
                    ] = canonical

                known.append(
                    result
                )

                continue

            # ------------------------------------------------
            # Structured types
            # ------------------------------------------------

            if entity_type in NON_LEARNABLE_TYPES:

                known_types.append(
                    {
                        "text": entity_text,
                        "normalized": normalize_entity(
                            entity_text
                        ),
                        "type": entity_type,
                        "score": score,
                        "start": start,
                        "end": end,
                        "status": "KNOWN_TYPE",
                        "source": "gliner",
                    }
                )

                continue

            # ------------------------------------------------
            # Check production overlap
            # ------------------------------------------------

            matched_production = False

            for prod in production:

                prod_start = prod.get(
                    "start"
                )

                prod_end = prod.get(
                    "end"
                )

                if (
                    start is not None
                    and end is not None
                    and prod_start is not None
                    and prod_end is not None
                ):

                    if (
                        start < prod_end
                        and end > prod_start
                    ):

                        matched_production = True
                        break

                elif normalize_entity(
                    prod.get(
                        "text",
                        "",
                    )
                ) == normalize_entity(
                    entity_text
                ):

                    matched_production = True
                    break

            if matched_production:
                continue

            # ------------------------------------------------
            # NEW UNKNOWN CANDIDATE
            # ------------------------------------------------

            context_start = max(
                0,
                (
                    start
                    if start is not None
                    else 0
                )
                - 150,
            )

            context_end = min(
                len(text),
                (
                    end
                    if end is not None
                    else len(text)
                )
                + 150,
            )

            context_text = text[
                context_start:context_end
            ]

            candidate = add_candidate(
                text=entity_text,
                proposed_type=entity_type,
                context=context_text,
                start=start,
                end=end,
                score=score,
            )

            occurrences = int(
                candidate.get(
                    "occurrences",
                    0,
                )
                or 0
            )

            # ----------------------------------------------------
            # INLINE GEMINI TEACHER REFINEMENT
            # ----------------------------------------------------
            gemini_val = get_gemini_validator()
            cand_status = candidate.get("status", "LEARNING")
            cand_type = candidate.get("type") or entity_type
            cand_conf = float(candidate.get("type_confidence", score) or score)

            if cand_status != "CONFIRMED" and gemini_val.is_available():
                g_res = gemini_val.validate_candidate(
                    entity_text=entity_text,
                    context=context_text,
                    proposed_type=entity_type,
                )
                if g_res and float(g_res.get("confidence", 0.0)) >= 0.70:
                    cand_type = g_res["type"]
                    cand_conf = float(g_res["confidence"])
                    cand_status = "CONFIRMED"

                    from app.discovery.candidates import load_candidates, save_candidates
                    from datetime import datetime
                    c_store = load_candidates()
                    norm_k = candidate.get("normalized")
                    if norm_k and norm_k in c_store:
                        c_store[norm_k]["status"] = "CONFIRMED"
                        c_store[norm_k]["type"] = cand_type
                        c_store[norm_k]["type_confidence"] = cand_conf
                        c_store[norm_k]["updated_at"] = datetime.utcnow().isoformat()
                        save_candidates(c_store)
                        candidate = c_store[norm_k]

            if cand_status == "CONFIRMED":
                known.append({
                    "text": entity_text,
                    "normalized": normalize_entity(entity_text),
                    "type": cand_type,
                    "score": cand_conf,
                    "start": start,
                    "end": end,
                    "status": "CONFIRMED",
                    "source": "teacher_validated",
                })
            else:
                unknown_candidates.append(
                    {
                        "text": entity_text,
                        "normalized": normalize_entity(
                            entity_text
                        ),
                        "proposed_type": entity_type,
                        "confidence": score,
                        "start": start,
                        "end": end,
                        "status": cand_status,
                        "occurrences": occurrences,
                        "source": "gliner",
                    }
                )

                try:
                    enqueue_entity(entity_text)
                except Exception as exc:
                    print("[NER SERVICE] Queue error:", exc)

        # ====================================================
        # START BACKGROUND TYPE LEARNING
        # ====================================================

        if unknown_candidates:

            try:

                started = start_learning_worker()

                print(
                    "[NER SERVICE] Type learning worker:",
                    "STARTED"
                    if started
                    else "ALREADY RUNNING",
                )

            except Exception as exc:

                print(
                    "[NER SERVICE] Failed to start "
                    "type-learning worker:",
                    exc,
                )

        # ====================================================
        # REPLAY KNOWN EXAMPLES
        # ====================================================

        replay_examples = []

        for entity in known:

            start = entity.get(
                "start"
            )

            end = entity.get(
                "end"
            )

            entity_type = str(
                entity.get(
                    "type",
                    "",
                )
                or ""
            ).upper()

            if (
                start is None
                or end is None
            ):
                continue

            if not entity_type:
                continue

            if entity_type in {
                "DATE",
                "EMAIL",
                "PHONE",
            }:
                continue

            replay_examples.append(
                {
                    "text": text,
                    "entities": [
                        {
                            "start": int(
                                start
                            ),
                            "end": int(
                                end
                            ),
                            "label": entity_type,
                        }
                    ],
                }
            )

        if replay_examples:

            try:

                replay_result = add_examples(
                    replay_examples
                )

                print(
                    "[NER SERVICE] Replay:",
                    replay_result,
                )

            except Exception as exc:

                print(
                    "[NER SERVICE] Replay error:",
                    exc,
                )

        # ====================================================
        # FINAL
        # ====================================================

        total = (
            len(known)
            + len(known_types)
            + len(production)
            + len(unknown_candidates)
        )

        print(
            "[NER SERVICE] Final:",
            "known=",
            len(known),
            "known_types=",
            len(known_types),
            "production=",
            len(production),
            "unknown_candidates=",
            len(unknown_candidates),
        )

        return {
            "total": total,
            "known": known,
            "known_types": known_types,
            "production": production,
            "unknown_candidates": unknown_candidates,
        }


# ============================================================
# SINGLETON
# ============================================================

_service = None


def get_ner_service():

    global _service

    if _service is None:

        _service = NERService()

    return _service


class _LazyNERServiceProxy:
    def __getattr__(self, name):
        return getattr(get_ner_service(), name)


ner_service = _LazyNERServiceProxy()


def extract_entities(
    text: str,
):

    return get_ner_service().extract_entities(
        text
    )