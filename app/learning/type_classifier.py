from pathlib import Path
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.learning.config import LEARNABLE_ENTITY_TYPES


MODEL_DIR = Path("models/type_classifier")


class EntityTypeClassifier:

    def __init__(self):
        self.model = None
        self.tokenizer = None

        if not MODEL_DIR.exists():
            return

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR
        )

        self.model.eval()

    def is_available(self):
        return self.model is not None and self.tokenizer is not None

    def predict(self, entity: str, context: str):

        if not self.is_available():
            return []

        text = f"Entity: {entity}\nContext: {context}"

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        results = []

        for index, probability in enumerate(probabilities):
            label = self.model.config.id2label[index]

            if label in LEARNABLE_ENTITY_TYPES:
                results.append({
                    "type": label,
                    "confidence": round(float(probability), 4),
                })

        results.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

        return results


_classifier = None


def get_type_classifier():

    global _classifier

    if _classifier is None:
        _classifier = EntityTypeClassifier()

    return _classifier