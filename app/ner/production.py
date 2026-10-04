from pathlib import Path

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)

from app.registry.model_registry import (
    get_production_model,
)


class ProductionNER:

    def __init__(self):

        self.model = None
        self.tokenizer = None
        self.version = None

        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )


    def _load_current_model(self):

        production = (
            get_production_model()
        )

        if production is None:
            return False


        version = production.get(
            "version"
        )


        if version == self.version:
            return True


        model_path = Path(
            production["path"]
        )


        if not model_path.exists():
            return False


        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                str(model_path)
            )
        )


        self.model = (
            AutoModelForTokenClassification
            .from_pretrained(
                str(model_path)
            )
        )


        self.model.to(
            self.device
        )

        self.model.eval()

        self.version = version


        return True


    def is_available(self):

        return self._load_current_model()


    def extract_entities(
        self,
        text: str,
    ):

        if not self._load_current_model():

            return []


        encoding = self.tokenizer(

            text,

            return_tensors="pt",

            return_offsets_mapping=True,

            truncation=True,

            max_length=512,
        )


        offsets = encoding.pop(
            "offset_mapping"
        )


        encoding = {

            key: value.to(
                self.device
            )

            for key, value in encoding.items()
        }


        with torch.no_grad():

            outputs = self.model(
                **encoding
            )


        predictions = (
            outputs.logits.argmax(
                dim=-1
            )[0]
        )


        id2label = (
            self.model.config.id2label
        )


        entities = []


        current_entity = None


        for index, label_id in enumerate(
            predictions
        ):

            label = id2label[
                int(label_id)
            ]


            token_start = int(
                offsets[0][index][0]
            )

            token_end = int(
                offsets[0][index][1]
            )


            if (
                token_start
                ==
                token_end
            ):

                continue


            # ------------------------------------------------
            # Outside
            # ------------------------------------------------

            if label == "O":

                if current_entity:

                    entities.append(
                        current_entity
                    )

                    current_entity = None

                continue


            # ------------------------------------------------
            # BIO parsing
            # ------------------------------------------------

            if label.startswith(
                "B-"
            ):

                if current_entity:

                    entities.append(
                        current_entity
                    )


                entity_type = (
                    label[2:]
                )


                current_entity = {

                    "start":
                        token_start,

                    "end":
                        token_end,

                    "type":
                        entity_type,
                }


            elif label.startswith(
                "I-"
            ):

                entity_type = (
                    label[2:]
                )


                if (
                    current_entity
                    and
                    current_entity["type"]
                    ==
                    entity_type
                ):

                    current_entity[
                        "end"
                    ] = token_end

                else:

                    current_entity = {

                        "start":
                            token_start,

                        "end":
                            token_end,

                        "type":
                            entity_type,
                    }


        if current_entity:

            entities.append(
                current_entity
            )


        results = []


        for entity in entities:

            start = entity[
                "start"
            ]

            end = entity[
                "end"
            ]


            results.append({

                "text":
                    text[start:end],

                "normalized":
                    text[start:end].strip().lower(),

                "type":
                    entity["type"],

                "start":
                    start,

                "end":
                    end,

                "status":
                    "PRODUCTION_MODEL",

                "model_version":
                    self.version,
            })


        return results