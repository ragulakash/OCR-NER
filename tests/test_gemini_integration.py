# ============================================================
# tests/test_gemini_integration.py
# INTEGRATION TESTS FOR DYNAMIC NER & GEMINI VALIDATION
# ============================================================

import os
import sys
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from app.learning.gemini_validator import GeminiValidator, get_gemini_validator
from app.ner.service import NERService, get_ner_service, ner_service


class TestDynamicNERPipeline(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        """Test GET /health"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(data.get("service"), "dynamic-ner")

    def test_gemini_disabled_by_default(self):
        """Verify Gemini is disabled when GEMINI_ENABLED=false"""
        os.environ["GEMINI_ENABLED"] = "false"
        os.environ["GEMINI_API_KEY"] = ""
        
        validator = GeminiValidator()
        self.assertFalse(validator.is_available())
        
        # Test candidate validation returns None when disabled
        result = validator.validate_candidate(
            entity_text="Test Vendor Corp",
            context="Invoice issued by Test Vendor Corp for services.",
        )
        self.assertIsNone(result)

    def test_ner_service_methods(self):
        """Verify NERService exposes process and extract_entities"""
        service = get_ner_service()
        self.assertTrue(hasattr(service, "extract_entities"))
        self.assertTrue(hasattr(service, "process"))
        
        # Verify proxy
        self.assertTrue(hasattr(ner_service, "process"))

    def test_document_upload_text(self):
        """Test POST /documents/upload with text file"""
        sample_text = """
        INVOICE #10492
        Date: 2026-10-04
        Vendor: Arjun Technologies Inc
        Billed To: Ragul Akash
        Amount: $1,450.00
        Email: contact@arjuntech.com
        Phone: +1-555-0199
        """
        
        response = self.client.post(
            "/documents/upload",
            files={"file": ("sample_invoice.txt", sample_text.encode("utf-8"), "text/plain")}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "success")
        self.assertIn("ner", data)
        ner_res = data["ner"]
        
        # Verify NER output dictionary structure
        self.assertIn("total", ner_res)
        self.assertIn("known", ner_res)
        self.assertIn("known_types", ner_res)
        self.assertIn("production", ner_res)
        self.assertIn("unknown_candidates", ner_res)

    def test_training_status_endpoint(self):
        """Test GET /training/status"""
        response = self.client.get("/training/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)


if __name__ == "__main__":
    unittest.main()
