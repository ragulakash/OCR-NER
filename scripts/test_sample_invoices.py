# ============================================================
# scripts/test_sample_invoices.py
# TEST SAMPLE INVOICES AGAINST DYNAMIC NER API
# ============================================================

import sys
from pathlib import Path
from fastapi.testclient import TestClient

# UTF-8 encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app

def test_invoices():
    client = TestClient(app)

    # ---------------------------------------------------------
    # Sample 1: Non-PO Vendor Invoice (Arjun Technologies)
    # ---------------------------------------------------------
    invoice_1 = """
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
    print("TESTING INVOICE 1: NON-PO VENDOR INVOICE (Arjun Technologies)")
    print("=" * 60)
    r1 = client.post(
        "/documents/upload",
        files={"file": ("invoice_arjun_tech.txt", invoice_1.encode("utf-8"), "text/plain")}
    )
    print("Status Code:", r1.status_code)
    data1 = r1.json()
    print("Extracted Non-PO Invoice Schema:")
    print("  Vendor              :", data1.get("non_po_invoice", {}).get("vendor"))
    print("  Invoice Number      :", data1.get("non_po_invoice", {}).get("invoice_number"))
    print("  Invoice Date        :", data1.get("non_po_invoice", {}).get("invoice_date"))
    print("  Total Amount        :", data1.get("non_po_invoice", {}).get("total_amount"))
    print("  Vendor Memory Status:", data1.get("non_po_invoice", {}).get("vendor_memory_status"))
    print("  Known Entities Count:", len(data1.get("ner", {}).get("known", [])))

    # ---------------------------------------------------------
    # Sample 2: Non-PO Supplier Invoice (Kryptos Infosys)
    # ---------------------------------------------------------
    invoice_2 = """
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
    print("TESTING INVOICE 2: RECURRING SUPPLIER INVOICE (Kryptos Infosys)")
    print("=" * 60)
    r2 = client.post(
        "/documents/upload",
        files={"file": ("invoice_kryptos.txt", invoice_2.encode("utf-8"), "text/plain")}
    )
    print("Status Code:", r2.status_code)
    data2 = r2.json()
    print("Extracted Non-PO Invoice Schema:")
    print("  Vendor              :", data2.get("non_po_invoice", {}).get("vendor"))
    print("  Invoice Number      :", data2.get("non_po_invoice", {}).get("invoice_number"))
    print("  Invoice Date        :", data2.get("non_po_invoice", {}).get("invoice_date"))
    print("  Total Amount        :", data2.get("non_po_invoice", {}).get("total_amount"))
    print("  Vendor Memory Status:", data2.get("non_po_invoice", {}).get("vendor_memory_status"))
    print("  Known Entities Count:", len(data2.get("ner", {}).get("known", [])))

    # ---------------------------------------------------------
    # Sample 3: PO-Based Invoice (Purchase Order #PO-88491)
    # ---------------------------------------------------------
    invoice_3 = """
    PURCHASE ORDER INVOICE
    ----------------------------------------
    Invoice Number: INV-88219
    PO Number     : PO-88491
    Date          : 2026-10-04
    Vendor        : Global Hardware Supplies Ltd
    Customer      : Enterprise Tech Corp
    Line Items    : 10x Server Racks, 20x Switches
    Total Amount  : $12,850.00
    ----------------------------------------
    """

    print("\n" + "=" * 60)
    print("TESTING INVOICE 3: PURCHASE ORDER INVOICE (PO #PO-88491)")
    print("=" * 60)
    r3 = client.post(
        "/documents/upload",
        files={"file": ("invoice_po_88491.txt", invoice_3.encode("utf-8"), "text/plain")}
    )
    print("Status Code:", r3.status_code)
    data3 = r3.json()
    print("Extracted Non-PO Invoice Schema:")
    print("  Vendor              :", data3.get("non_po_invoice", {}).get("vendor"))
    print("  Invoice Number      :", data3.get("non_po_invoice", {}).get("invoice_number"))
    print("  Invoice Date        :", data3.get("non_po_invoice", {}).get("invoice_date"))
    print("  Total Amount        :", data3.get("non_po_invoice", {}).get("total_amount"))
    print("  Vendor Memory Status:", data3.get("non_po_invoice", {}).get("vendor_memory_status"))

    print("\n" + "=" * 60)
    print("ALL SAMPLE INVOICE TESTS COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_invoices()
    except Exception as exc:
        import traceback
        traceback.print_exc()
