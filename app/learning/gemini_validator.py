# ============================================================
# app/learning/gemini_validator.py
# GEMINI TEACHER / VALIDATOR FOR AMBIGUOUS NER CANDIDATES
# ============================================================

import os
import json
import logging
from typing import Optional, Dict, Any
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class GeminiValidator:
    """
    Optional Gemini teacher/validator for low-confidence or ambiguous NER candidates.
    Gemini does NOT replace DistilBERT; it acts as a teacher to confirm hard candidates
    which are then added to the replay buffer to train DistilBERT.
    """

    def __init__(self):
        self.enabled_str = os.getenv("GEMINI_ENABLED", "false").lower()
        self.enabled = self.enabled_str in ("true", "1", "t", "yes")
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def is_available(self) -> bool:
        """
        Check if Gemini validation is explicitly enabled and API key is configured.
        """
        # Re-check env vars in case they were updated at runtime
        enabled_val = os.getenv("GEMINI_ENABLED", "false").lower() in ("true", "1", "t", "yes")
        api_key_val = os.getenv("GEMINI_API_KEY", "").strip()
        return enabled_val and bool(api_key_val)

    def validate_candidate(
        self,
        entity_text: str,
        context: str,
        proposed_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Send ONLY the entity and small relevant context to Gemini to resolve low-confidence cases.

        Returns structured response:
        {
          "entity": "Arjun Technologies",
          "type": "VENDOR",
          "confidence": 0.93
        }
        """
        if not self.is_available():
            return None

        entity_clean = str(entity_text or "").strip()
        context_clean = str(context or "").strip()

        if not entity_clean or not context_clean:
            return None

        # Truncate context to small relevant window (max 300 chars)
        if len(context_clean) > 300:
            pos = context_clean.find(entity_clean)
            if pos >= 0:
                start_c = max(0, pos - 100)
                end_c = min(len(context_clean), pos + len(entity_clean) + 100)
                context_clean = context_clean[start_c:end_c]
            else:
                context_clean = context_clean[:300]

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        model_name = os.getenv("GEMINI_MODEL", self.model_name)

        prompt = f"""You are an expert AI teacher for a document intelligence NER system.
Analyze the entity text in its surrounding context with full semantic understanding.

Entity: "{entity_clean}"
Context: "{context_clean}"
GLiNER Proposed Type: "{proposed_type or 'UNKNOWN'}"

Task:
1. Determine the true entity category.
   Standard categories:
   - PERSON (Individual names)
   - ORGANIZATION (Companies, vendors, suppliers, agencies, institutions)
   - LOCATION (Cities, addresses, countries, regions)
   - TECHNOLOGY (Algorithms, ML models, software, libraries, frameworks, tools)
   - PRODUCT (Commercial items, merchandise, physical goods, services)
   - DATABASE (Database systems, storage engines)
   - DATE, EMAIL, PHONE
2. Provide a canonical normalized name if applicable.
3. Assign a confidence score from 0.0 to 1.0.

Respond ONLY with a valid JSON object matching this schema:
{{
  "entity": "{entity_clean}",
  "canonical": "Clean Canonical Name",
  "type": "CATEGORY_UPPERCASE",
  "confidence": 0.95,
  "reasoning": "Brief explanation"
}}
"""

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.1
            }
        }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=12)

            if response.status_code != 200:
                # Try fallback model if default model failed
                fallback_models = ["gemini-1.5-flash", "gemini-2.0-flash"]
                for fb in fallback_models:
                    if fb == model_name:
                        continue
                    fb_url = f"https://generativelanguage.googleapis.com/v1beta/models/{fb}:generateContent?key={api_key}"
                    response = requests.post(fb_url, json=payload, headers=headers, timeout=12)
                    if response.status_code == 200:
                        break

            if response.status_code != 200:
                print(f"[GEMINI VALIDATOR] API call failed ({response.status_code}): {response.text[:200]}")
                return None

            res_json = response.json()
            candidates = res_json.get("candidates", [])
            if not candidates:
                return None

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                return None

            text_resp = parts[0].get("text", "").strip()

            # Clean markdown code block if present
            if text_resp.startswith("```"):
                text_resp = text_resp.split("```")[1]
                if text_resp.startswith("json"):
                    text_resp = text_resp[4:].strip()

            data = json.loads(text_resp)

            result_entity = str(data.get("entity", entity_clean)).strip()
            result_type = str(data.get("type", "")).strip().upper()
            confidence = float(data.get("confidence", 0.0) or 0.0)

            # Map common aliases to project types
            type_aliases = {
                "VENDOR": "ORGANIZATION",
                "COMPANY": "ORGANIZATION",
                "SUPPLIER": "ORGANIZATION",
                "BUSINESS": "ORGANIZATION",
                "CITY": "LOCATION",
                "COUNTRY": "LOCATION",
                "STATE": "LOCATION",
            }

            final_type = type_aliases.get(result_type, result_type)

            canonical = str(data.get("canonical", result_entity)).strip()
            reasoning = str(data.get("reasoning", "")).strip()

            print(
                f"[GEMINI VALIDATOR] Gemini output for '{entity_clean}': "
                f"type={final_type} (raw={result_type}), canonical='{canonical}', confidence={confidence:.4f}"
            )

            return {
                "entity": result_entity,
                "canonical": canonical,
                "type": final_type,
                "raw_type": result_type,
                "confidence": round(confidence, 4),
                "reasoning": reasoning,
            }

        except Exception as exc:
            print(f"[GEMINI VALIDATOR] Exception during Gemini validation: {exc}")
            return None


_gemini_validator = None


def get_gemini_validator() -> GeminiValidator:
    global _gemini_validator
    if _gemini_validator is None:
        _gemini_validator = GeminiValidator()
    return _gemini_validator
