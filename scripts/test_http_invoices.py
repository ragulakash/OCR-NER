# ============================================================
# scripts/test_http_invoices.py
# TEST SAMPLE INVOICES AGAINST ACTIVE FASTAPI SERVER
# ============================================================

import sys
import json
import requests

# UTF-8 encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_URL = "http://127.0.0.1:8000"

def test_live_invoices():
    # Check health
    try:
        r_h = requests.get(f"{BASE_URL}/health", timeout=5)
        print("Server Health:", r_h.json())
    except Exception as e:
        print("Server health check failed:", e)
        return

    # Sample Non-PO Invoice 1
    inv1 = """
    NON-PO VENDOR INVOICE
    ----------------------------------------
    Invoice Number : INV-99012
    Date           : 2026-10-04
    Vendor Name    : Arjun Technologies Pvt Ltd
    Billed To      : Ragul Akash
    Description    : Custom Software Consulting & Neural Network Tuning
    Total Amount   : 45,000 INR
    Email          : billing@arjuntech.com
    Phone          : +91-9876543210
    ----------------------------------------
    """

    print("\n" + "=" * 60)
    print("UPLOADING NON-PO INVOICE 1: Arjun Technologies")
    print("=" * 60)
    res1 = requests.post(
        f"{BASE_URL}/documents/upload",
        files={"file": ("invoice_arjun.txt", inv1.encode("utf-8"), "text/plain")},
        timeout=30
    )
    print("HTTP Status:", res1.status_code)
    data1 = res1.json()
    print("Non-PO Invoice Output:")
    print(json.dumps(data1.get("non_po_invoice"), indent=2))

    # Sample Non-PO Invoice 2
    inv2 = """
    RECURRING SERVICES INVOICE
    ----------------------------------------
    INV-40291
    Date: 2026-10-04
    Supplier: Kryptos Infosys
    Service: PostgreSQL High-Availability Hosting
    Total Amount Due: $2,350.00 USD
    Email: support@kryptos.com
    ----------------------------------------
    """

    print("\n" + "=" * 60)
    print("UPLOADING NON-PO INVOICE 2: Kryptos Infosys")
    print("=" * 60)
    res2 = requests.post(
        f"{BASE_URL}/documents/upload",
        files={"file": ("invoice_kryptos.txt", inv2.encode("utf-8"), "text/plain")},
        timeout=30
    )
    print("HTTP Status:", res2.status_code)
    data2 = res2.json()
    print("Non-PO Invoice Output:")
    print(json.dumps(data2.get("non_po_invoice"), indent=2))

if __name__ == "__main__":
    test_live_invoices()
