# ============================================================
# scripts/test_all.py
# VERIFICATION SUITE FOR DYNAMIC NER & GEMINI INTEGRATION
# ============================================================

import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app.main import app
from app.learning.gemini_validator import get_gemini_validator

def run_verifications():
    print("=" * 60)
    print("RUNNING END-TO-END VERIFICATION SUITE")
    print("=" * 60)

    client = TestClient(app)

    # 1. Health check
    print("\n[1] Testing GET /health ...")
    r_health = client.get("/health")
    print("    Status:", r_health.status_code)
    print("    Body  :", r_health.json())
    assert r_health.status_code == 200, "Health check failed!"

    # 2. Gemini disabled test
    print("\n[2] Testing Gemini disabled mode (GEMINI_ENABLED=false) ...")
    os.environ["GEMINI_ENABLED"] = "false"
    os.environ["GEMINI_API_KEY"] = ""
    gemini = get_gemini_validator()
    is_avail = gemini.is_available()
    print("    Gemini available?", is_avail)
    assert not is_avail, "Gemini should be disabled!"
    
    val_res = gemini.validate_candidate("Test Entity", "Some context")
    print("    Validation result when disabled:", val_res)
    assert val_res is None, "Gemini validation should return None when disabled!"

    # 3. Text document upload
    print("\n[3] Testing POST /documents/upload (Text invoice) ...")
    sample_invoice = """
    NON-PO INVOICE #88201
    Date: 2026-10-04
    Vendor: Arjun Technologies Pvt Ltd
    Customer: Ragul Akash
    Item: Software Development Consultancy
    Amount: 45,000 INR
    Email: billing@arjuntech.com
    Phone: +91-9876543210
    """
    
    r_upload = client.post(
        "/documents/upload",
        files={"file": ("invoice_88201.txt", sample_invoice.encode("utf-8"), "text/plain")}
    )
    print("    Upload Status:", r_upload.status_code)
    data = r_upload.json()
    print("    Filename:", data.get("filename"))
    print("    NER keys:", list(data.get("ner", {}).keys()))
    print("    Known count:", len(data.get("ner", {}).get("known", [])))
    print("    Unknown candidates:", data.get("ner", {}).get("unknown_candidates"))
    print("    Learning queue:", data.get("learning"))
    assert r_upload.status_code == 200, "Upload failed!"

    # 4. Training status check
    print("\n[4] Testing GET /training/status ...")
    r_train = client.get("/training/status")
    print("    Status:", r_train.status_code)
    print("    Body  :", r_train.json())
    assert r_train.status_code == 200, "Training status failed!"

    print("\n" + "=" * 60)
    print("ALL VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_verifications()
