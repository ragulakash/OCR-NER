import sys
import gc
import torch
from typing import Any
from gliner2 import GLiNER2

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class GLiNEREngine:
    """
    GLiNER2 discovery layer.

    Purpose:
        Discover entity spans from raw document text.

    Important:
        GLiNER is a DISCOVERY model here.
        It is not the final production NER model.

    Pipeline:

        Document
            ↓
        GLiNER2
            ↓
        Candidate discovery
            ↓
        Type classifier
            ↓
        Candidate confirmation
            ↓
        Replay buffer
            ↓
        DistilBERT training
    """

    MODEL_NAME = "fastino/gliner2-base-v1"

    # GLiNER2 works with thresholds internally.
    # Keep this lower than the final candidate confirmation threshold.
    DEFAULT_THRESHOLD = 0.35

    # Use explicit long-document extraction above this size.
    LONG_TEXT_THRESHOLD = 2500

    CHUNK_SIZE = 384
    CHUNK_OVERLAP = 64

    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            print("[GLINER] Loading model:", self.MODEL_NAME)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            try:
                self._model = GLiNER2.from_pretrained(self.MODEL_NAME)
            except Exception as exc:
                print(f"[GLINER] Initial load attempt failed ({exc}), running GC and retrying...")
                gc.collect()
                self._model = GLiNER2.from_pretrained(self.MODEL_NAME)

            print("[GLINER] Model loaded successfully.")

        return self._model

    # ============================================================
    # NORMALIZE LABELS
    # ============================================================

    def _normalize_labels(
        self,
        labels: list[str],
    ) -> list[str]:

        """
        GLiNER2 works well with natural-language/lowercase
        entity descriptions.

        Internally our application uses uppercase labels.
        GLiNER receives lowercase semantic labels.
        """

        mapping = {
            "PERSON": "person, individual name",
            "ORGANIZATION": "organization, company, vendor, supplier",
            "LOCATION": "location, address, city, country",
            "TECHNOLOGY": "technology, software, algorithm, model, framework, library, architecture",
            "PRODUCT": "product, commercial item, merchandise, physical good",
            "DATABASE": "database, data store",
            "DATE": "date, timestamp",
            "EMAIL": "email address",
            "PHONE": "phone number",
        }

        normalized = []

        for label in labels:

            label_upper = str(label).upper()

            normalized.append(
                mapping.get(
                    label_upper,
                    str(label).lower(),
                )
            )

        return normalized

    # ============================================================
    # EXTRACT ENTITIES
    # ============================================================

    def extract_entities(
        self,
        text: str,
        labels: list[str],
    ) -> dict[str, Any]:

        if not text or not text.strip():

            return {
                "entities": {}
            }

        gliner_labels = self._normalize_labels(labels)

        print(
            "[GLINER] Extracting entities...",
            "text_length=",
            len(text),
            "labels=",
            gliner_labels,
        )

        try:

            # ----------------------------------------------------
            # LONG DOCUMENT
            # ----------------------------------------------------

            if len(text) > self.LONG_TEXT_THRESHOLD:

                print(
                    "[GLINER] Using long-document extraction."
                )

                result = self.model.extract_entities_long(
                    text,
                    gliner_labels,
                    threshold=self.DEFAULT_THRESHOLD,
                    chunk_size=self.CHUNK_SIZE,
                    chunk_overlap=self.CHUNK_OVERLAP,
                    include_confidence=True,
                    include_spans=True,
                )

            # ----------------------------------------------------
            # NORMAL DOCUMENT
            # ----------------------------------------------------

            else:

                result = self.model.extract_entities(
                    text,
                    gliner_labels,
                    threshold=self.DEFAULT_THRESHOLD,
                    include_confidence=True,
                    include_spans=True,
                )

            # ----------------------------------------------------
            # DEBUG
            # ----------------------------------------------------

            print(
                "[GLINER] Raw result:"
            )

            print(result)

            # ----------------------------------------------------
            # GUARANTEE DICT RESULT
            # ----------------------------------------------------

            if not isinstance(result, dict):

                print(
                    "[GLINER] WARNING: unexpected result type:",
                    type(result),
                )

                return {
                    "entities": {}
                }

            # ----------------------------------------------------
            # GUARANTEE ENTITIES KEY
            # ----------------------------------------------------

            if "entities" not in result:

                print(
                    "[GLINER] WARNING: result has no 'entities' key."
                )

                return {
                    "entities": {}
                }

            return result

        except Exception as exc:

            print(
                "[GLINER] Extraction error:",
                repr(exc),
            )

            return {
                "entities": {}
            }