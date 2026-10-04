# ============================================================
# app/discovery/non_po_memory.py
# INDUSTRY-STANDARD NON-PO INVOICE MEMORY & SPATIAL EXTRACTION ENGINE
# ============================================================

import re
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

from app.database.sqlite_db import (
    db_save_invoice,
    db_get_invoices,
    db_get_invoice_by_id,
    db_remember_vendor,
    db_lookup_vendor,
    db_load_memory,
)

NON_INVOICE_NOISE = {
    "resnet50", "knn", "svm", "vgg16", "distilbert", "bert", "gpt", "gliner",
    "upgrade plan", "io", "rai", "station ma", "click here", "read more",
    "view invoice", "page 1", "total pages", "subtotal", "total", "amount",
    "qty", "description", "item", "price", "tax", "status", "unknown", "learning"
}

NON_PO_ALLOWED_TYPES = {
    "VENDOR", "ORGANIZATION", "CUSTOMER", "PERSON", "INVOICE_NUMBER",
    "DATE", "AMOUNT", "TOTAL_AMOUNT", "TAX_AMOUNT", "NET_AMOUNT",
    "ADDRESS", "EMAIL", "PHONE", "LINE_ITEM", "CURRENCY", "PRODUCT"
}


def load_non_po_memory() -> Dict[str, Any]:
    return db_load_memory()


def save_non_po_memory(data: Dict[str, Any]):
    pass  # Automatically synchronized via SQLite database


def load_processed_invoices() -> List[Dict[str, Any]]:
    return db_get_invoices()


def save_processed_invoice(invoice: Dict[str, Any]) -> Dict[str, Any]:
    return db_save_invoice(invoice)


def get_processed_invoices(vendor_filter: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    return db_get_invoices(vendor_filter=vendor_filter, limit=limit)


def get_processed_invoice_by_id(invoice_id: str) -> Optional[Dict[str, Any]]:
    return db_get_invoice_by_id(invoice_id)


def remember_vendor(vendor_name: str, canonical: Optional[str] = None, confidence: float = 1.0) -> Dict[str, Any]:
    return db_remember_vendor(vendor_name=vendor_name, canonical=canonical, confidence=confidence)


def lookup_non_po_vendor(text: str) -> Optional[Dict[str, Any]]:
    return db_lookup_vendor(text)


def filter_non_po_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clean_entities = []
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        t = str(ent.get("text", "")).strip()
        norm_t = t.lower()
        if not t or norm_t in NON_INVOICE_NOISE or len(t) < 2:
            continue
        clean_entities.append(ent)
    return clean_entities


def assemble_non_po_invoice_schema(
    text: str, 
    extracted_entities: List[Dict[str, Any]], 
    filename: str = "uploaded_document"
) -> Dict[str, Any]:
    clean_entities = filter_non_po_entities(extracted_entities)

    # 1. Regex Extractors for Document Structure & Header
    inv_num_pattern = re.compile(r"(?:INVOICE\s*(?:NUMBER|NO|#)?|INV\s*#?)\s*[:.]\s*([A-Z0-9\-_]{1,25})", re.IGNORECASE)
    po_num_pattern = re.compile(r"(?:PO\s*(?:NUMBER|NO|#)?)\s*[:.]\s*([A-Z0-9\-_]{1,25})", re.IGNORECASE)
    
    date_pattern = re.compile(r"(?:INVOICE\s*DATE|DATE)\s*[:.-]?\s*(\d{2}[-/.]\d{2}[-/.]\d{4}|\d{4}[-/.]\d{2}[-/.]\d{2})", re.IGNORECASE)
    due_date_pattern = re.compile(r"(?:DUE\s*DATE)\s*[:.-]?\s*(\d{2}[-/.]\d{2}[-/.]\d{4}|\d{4}[-/.]\d{2}[-/.]\d{2})", re.IGNORECASE)

    vendor_name_pattern = re.compile(r"(?:Company\s*Name|Vendor\s*Name|Vendor\s*Company)\s*[:.-]?\s*([^\n]+)", re.IGNORECASE)
    email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    phone_pattern = re.compile(r"(?:Phone|Tel|Mobile)\s*[:.-]?\s*(\+?\d[\d\s\-()]{7,15}\d)", re.IGNORECASE)
    address_pattern = re.compile(r"(?:Address)\s*[:.-]?\s*([^\n]+)", re.IGNORECASE)
    tax_id_pattern = re.compile(r"(?:GSTIN|VAT|TAX\s*ID|EIN)\s*[:.-]?\s*([A-Z0-9\-_]{5,20})", re.IGNORECASE)

    cust_name_pattern = re.compile(r"(?:Customer\s*Name|Attn)\s*[:.-]?\s*([^\n]+)", re.IGNORECASE)
    dept_pattern = re.compile(r"(?:Department|Dept)\s*[:.-]?\s*([^\n]+)", re.IGNORECASE)

    net_amt_pattern = re.compile(r"(?:SUBTOTAL|NET\s*AMOUNT)\s*[:.-]?\s*([$€£₹]?\s*[\d,]+\.?\d*\s*(?:USD|INR|EUR|GBP)?)", re.IGNORECASE)
    tax_amt_pattern = re.compile(r"(?:TAX\s*AMOUNT|GST|VAT|TAX)\s*[:.-]?\s*([$€£₹]?\s*[\d,]+\.?\d*\s*(?:USD|INR|EUR|GBP)?(?:\s*\([^)]+\))?)", re.IGNORECASE)
    tot_amt_pattern = re.compile(r"(?:INVOICE\s*TOTAL|TOTAL\s*AMOUNT|AMOUNT\s*DUE)\s*[:.-]?\s*([$€£₹]?\s*[\d,]+\.?\d*\s*(?:USD|INR|EUR|GBP)?)", re.IGNORECASE)
    tot_amt_fallback = re.compile(r"INVOICE\s*TOTAL[\s:]*([$€£₹]?\s*[\d,]+\.?\d*)", re.IGNORECASE)

    # Bank Payment Information Pattern
    bank_account_pattern = re.compile(r"(?:Account|Acc\s*#?|IBAN)\s*[:.-]?\s*([A-Z0-9]{8,34})", re.IGNORECASE)
    bank_name_pattern = re.compile(r"(?:Bank\s*of\s*[A-Za-z]+|[A-Za-z]+\s*Bank)", re.IGNORECASE)

    # Extract Header Values
    inv_num = inv_num_pattern.search(text).group(1).strip() if inv_num_pattern.search(text) else None
    po_num = po_num_pattern.search(text).group(1).strip() if po_num_pattern.search(text) else None

    inv_date = date_pattern.search(text).group(1).strip() if date_pattern.search(text) else None
    due_date = due_date_pattern.search(text).group(1).strip() if due_date_pattern.search(text) else None

    v_name = vendor_name_pattern.search(text).group(1).strip() if vendor_name_pattern.search(text) else None
    v_email = email_pattern.search(text).group(0).strip() if email_pattern.search(text) else None
    v_phone = phone_pattern.search(text).group(1).strip() if phone_pattern.search(text) else None
    v_addr = address_pattern.search(text).group(1).strip() if address_pattern.search(text) else None
    v_tax_id = tax_id_pattern.search(text).group(1).strip() if tax_id_pattern.search(text) else None

    cust_name = cust_name_pattern.search(text).group(1).strip() if cust_name_pattern.search(text) else None
    cust_dept = dept_pattern.search(text).group(1).strip() if dept_pattern.search(text) else None

    net_amt = net_amt_pattern.search(text).group(1).strip() if net_amt_pattern.search(text) else None
    tax_amt = tax_amt_pattern.search(text).group(1).strip() if tax_amt_pattern.search(text) else None

    tot_match = tot_amt_pattern.search(text) or tot_amt_fallback.search(text)
    tot_amt = tot_match.group(1).strip() if tot_match else None

    # Bank Payment Information
    payee_bank = bank_name_pattern.search(text).group(0).strip() if bank_name_pattern.search(text) else None
    payee_acc = bank_account_pattern.search(text).group(1).strip() if bank_account_pattern.search(text) else None

    # Currency Detection
    currency = "USD"
    if "INR" in text.upper() or "₹" in text:
        currency = "INR"
    elif "EUR" in text.upper() or "€" in text:
        currency = "EUR"
    elif "GBP" in text.upper() or "£" in text:
        currency = "GBP"
    elif "$" in text:
        currency = "USD"

    # Line Items Parsing
    line_items = []
    
    # 1. Numbered format: 1. Item Name : $100
    item_pattern_1 = re.compile(r"^\s*(\d+)[\.\)]\s*(.+?)\s*[:.-]\s*([$€£₹]?\s*[\d,]+\.?\d*\s*(?:USD|INR|EUR|GBP)?)", re.MULTILINE | re.IGNORECASE)
    for match in item_pattern_1.finditer(text):
        line_items.append({
            "item_number": int(match.group(1)),
            "description": match.group(2).strip(),
            "amount": match.group(3).strip()
        })

    # 2. Tabular format: White t-shirts  $10.00  50  10%  $500.00
    if not line_items:
        item_pattern_2 = re.compile(r"([A-Za-z0-9\s\-]+?)\s+([$€£₹]\s*[\d,]+\.?\d*)\s+(\d+)\s+(?:\d+%\s+)?([$€£₹]\s*[\d,]+\.?\d*)", re.MULTILINE)
        idx = 1
        for match in item_pattern_2.finditer(text):
            desc = match.group(1).strip()
            desc_lines = [d.strip() for d in desc.splitlines() if d.strip()]
            valid_desc_lines = [
                d for d in desc_lines 
                if not re.search(r"\b(?:product|description|unit price|quantity|tax rate|amount|qty|item)\b", d, re.IGNORECASE)
            ]
            final_desc = " ".join(valid_desc_lines) if valid_desc_lines else desc_lines[-1] if desc_lines else desc
            
            if final_desc and final_desc.lower() not in ("product", "description", "item", "unit price"):
                line_items.append({
                    "item_number": idx,
                    "description": final_desc,
                    "unit_price": match.group(2).strip(),
                    "quantity": int(match.group(3)),
                    "amount": match.group(4).strip()
                })
                idx += 1

    # Fallback & Disambiguation for Vendor & Customer Organizations
    vendor_mem_status = "UNKNOWN_VENDOR"
    org_entities = [e for e in clean_entities if e.get("type") in ("ORGANIZATION", "VENDOR")]

    # Check vendor memory matching
    for org in org_entities:
        org_t = org.get("text", "").strip()
        mem_match = db_lookup_vendor(org_t)
        if mem_match:
            v_name = mem_match.get("canonical", org_t)
            vendor_mem_status = "REMEMBERED_VENDOR"
            break

    # If vendor is still unknown or invalid, pick top organization entity
    if not v_name or "INVOICE" in str(v_name).upper() or str(v_name).startswith(":"):
        if org_entities:
            v_name = org_entities[0].get("text")
        else:
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            if lines:
                candidate_v = lines[0]
                candidate_v = re.sub(r"^[iI:\|\!\\/\-]\s*", "", candidate_v)
                candidate_v = re.sub(r"\s*INVOICE.*$", "", candidate_v, flags=re.IGNORECASE).strip()
                v_name = candidate_v or lines[0]

    # Clean vendor name string
    if v_name:
        v_name = re.sub(r"^[iI:\|\!\\/\-]\s*", "", v_name)
        v_name = re.sub(r"\s*INVOICE.*$", "", v_name, flags=re.IGNORECASE).strip()

    # Vendor Memory update
    if v_name and vendor_mem_status == "UNKNOWN_VENDOR":
        mem_rec = db_remember_vendor(v_name)
        if mem_rec:
            vendor_mem_status = "NEW_VENDOR_LEARNED"

    # Assign Customer Name from remaining distinct Organization entity
    if not cust_name:
        for org in org_entities:
            org_t = org.get("text", "").strip()
            if org_t and org_t.lower() != str(v_name).lower() and org_t.lower() not in ("bank of america", "bank"):
                cust_name = org_t
                break

    # Build Standard Structured Invoice Output
    invoice_id = f"NONPO-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    
    structured_invoice = {
        "invoice_id": invoice_id,
        "document_type": "NON_PO_INVOICE",
        "header": {
            "invoice_number": inv_num or f"INV-{uuid.uuid4().hex[:6].upper()}",
            "po_reference": po_num,
            "invoice_date": inv_date or datetime.utcnow().strftime("%Y-%m-%d"),
            "due_date": due_date,
            "currency": currency,
            "payment_terms": "NET_30"
        },
        "vendor": {
            "name": v_name or "Unknown Vendor",
            "canonical_name": v_name.title() if v_name else None,
            "address": v_addr,
            "email": v_email,
            "phone": v_phone,
            "tax_id": v_tax_id,
            "vendor_memory_status": vendor_mem_status
        },
        "customer": {
            "name": cust_name,
            "department": cust_dept
        },
        "line_items": line_items,
        "financials": {
            "net_amount": net_amt,
            "tax_amount": tax_amt,
            "total_amount": tot_amt or net_amt
        },
        "payment_details": {
            "bank_name": payee_bank,
            "account_number": payee_acc
        },
        "metadata": {
            "filename": filename,
            "processed_at": datetime.utcnow().isoformat(),
            "extracted_entities_count": len(clean_entities)
        },
        "non_po_entities": clean_entities
    }

    db_save_invoice(structured_invoice)
    return structured_invoice
