# ============================================================
# app/database/sqlite_db.py
# FAST, ZERO-COST SQLITE PERSISTENCE ENGINE
# ============================================================

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

DB_FILE = Path("data/memory/database.db")


def get_db_connection() -> sqlite3.Connection:
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes SQLite tables and indexes."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 1. Processed Non-PO Invoices Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_invoices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                invoice_id TEXT UNIQUE NOT NULL,
                invoice_number TEXT,
                po_reference TEXT,
                invoice_date TEXT,
                due_date TEXT,
                vendor_name TEXT,
                customer_name TEXT,
                currency TEXT,
                net_amount TEXT,
                tax_amount TEXT,
                total_amount TEXT,
                bank_name TEXT,
                account_number TEXT,
                raw_json TEXT NOT NULL,
                filename TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 2. Vendor Profiles & Memory Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vendor_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vendor_key TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                canonical TEXT NOT NULL,
                occurrences INTEGER DEFAULT 1,
                confidence REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Indexes for fast querying
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inv_id ON processed_invoices(invoice_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inv_vendor ON processed_invoices(vendor_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_vendor_key ON vendor_memory(vendor_key)")
        conn.commit()

    # Migrate legacy JSON data if present
    _migrate_legacy_json()


def _migrate_legacy_json():
    """Migrates existing JSON files into SQLite database if not already present."""
    memory_file = Path("data/memory/non_po_memory.json")
    invoices_file = Path("data/memory/processed_invoices.json")

    # Migrate Vendors
    if memory_file.exists():
        try:
            data = json.loads(memory_file.read_text(encoding="utf-8"))
            vendors = data.get("vendors", {})
            with get_db_connection() as conn:
                cursor = conn.cursor()
                for key, v in vendors.items():
                    cursor.execute("""
                        INSERT OR IGNORE INTO vendor_memory 
                        (vendor_key, name, canonical, occurrences, confidence, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        key,
                        v.get("name", key),
                        v.get("canonical", key.title()),
                        v.get("occurrences", 1),
                        v.get("confidence", 1.0),
                        v.get("created_at", datetime.utcnow().isoformat()),
                        v.get("updated_at", datetime.utcnow().isoformat())
                    ))
                conn.commit()
        except Exception as e:
            print("[SQLITE] Legacy vendor migration notice:", e)

    # Migrate Invoices
    if invoices_file.exists():
        try:
            invoices = json.loads(invoices_file.read_text(encoding="utf-8"))
            if isinstance(invoices, list):
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    for inv in invoices:
                        inv_id = inv.get("invoice_id")
                        if not inv_id:
                            continue
                        header = inv.get("header", {})
                        vendor = inv.get("vendor", {})
                        customer = inv.get("customer", {})
                        fin = inv.get("financials", {})
                        pay = inv.get("payment_details", {})
                        meta = inv.get("metadata", {})

                        cursor.execute("""
                            INSERT OR IGNORE INTO processed_invoices
                            (invoice_id, invoice_number, po_reference, invoice_date, due_date,
                             vendor_name, customer_name, currency, net_amount, tax_amount, total_amount,
                             bank_name, account_number, raw_json, filename, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            inv_id,
                            header.get("invoice_number"),
                            header.get("po_reference"),
                            header.get("invoice_date"),
                            header.get("due_date"),
                            vendor.get("name"),
                            customer.get("name"),
                            header.get("currency", "USD"),
                            fin.get("net_amount"),
                            fin.get("tax_amount"),
                            fin.get("total_amount"),
                            pay.get("bank_name"),
                            pay.get("account_number"),
                            json.dumps(inv, ensure_ascii=False),
                            meta.get("filename"),
                            meta.get("processed_at", datetime.utcnow().isoformat())
                        ))
                    conn.commit()
        except Exception as e:
            print("[SQLITE] Legacy invoice migration notice:", e)


# ============================================================
# DB OPERATIONS: INVOICES
# ============================================================

def db_save_invoice(invoice: Dict[str, Any]) -> Dict[str, Any]:
    init_db()
    inv_id = invoice.get("invoice_id")
    header = invoice.get("header", {})
    vendor = invoice.get("vendor", {})
    customer = invoice.get("customer", {})
    fin = invoice.get("financials", {})
    pay = invoice.get("payment_details", {})
    meta = invoice.get("metadata", {})

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO processed_invoices
            (invoice_id, invoice_number, po_reference, invoice_date, due_date,
             vendor_name, customer_name, currency, net_amount, tax_amount, total_amount,
             bank_name, account_number, raw_json, filename, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            inv_id,
            header.get("invoice_number"),
            header.get("po_reference"),
            header.get("invoice_date"),
            header.get("due_date"),
            vendor.get("name"),
            customer.get("name"),
            header.get("currency", "USD"),
            fin.get("net_amount"),
            fin.get("tax_amount"),
            fin.get("total_amount"),
            pay.get("bank_name"),
            pay.get("account_number"),
            json.dumps(invoice, ensure_ascii=False),
            meta.get("filename"),
            meta.get("processed_at", datetime.utcnow().isoformat())
        ))
        conn.commit()
    return invoice


def db_get_invoices(vendor_filter: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if vendor_filter:
            pattern = f"%{vendor_filter.strip().lower()}%"
            cursor.execute("""
                SELECT raw_json FROM processed_invoices 
                WHERE LOWER(vendor_name) LIKE ? 
                ORDER BY id DESC LIMIT ?
            """, (pattern, limit))
        else:
            cursor.execute("SELECT raw_json FROM processed_invoices ORDER BY id DESC LIMIT ?", (limit,))

        rows = cursor.fetchall()
        invoices = []
        for r in rows:
            try:
                invoices.append(json.loads(r["raw_json"]))
            except Exception:
                pass
        return invoices


def db_get_invoice_by_id(invoice_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT raw_json FROM processed_invoices 
            WHERE invoice_id = ? OR invoice_number = ?
        """, (invoice_id, invoice_id))
        row = cursor.fetchone()
        if row:
            try:
                return json.loads(row["raw_json"])
            except Exception:
                pass
        return None


# ============================================================
# DB OPERATIONS: VENDOR MEMORY
# ============================================================

def db_remember_vendor(vendor_name: str, canonical: Optional[str] = None, confidence: float = 1.0) -> Dict[str, Any]:
    init_db()
    norm = " ".join(vendor_name.strip().lower().split())
    if not norm or len(norm) < 2:
        return {}

    now = datetime.utcnow().isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM vendor_memory WHERE vendor_key = ?", (norm,))
        row = cursor.fetchone()

        if not row:
            canon = canonical or vendor_name.title()
            cursor.execute("""
                INSERT INTO vendor_memory 
                (vendor_key, name, canonical, occurrences, confidence, created_at, updated_at)
                VALUES (?, ?, ?, 1, ?, ?, ?)
            """, (norm, vendor_name, canon, confidence, now, now))
            conn.commit()
            return {
                "name": vendor_name,
                "canonical": canon,
                "type": "VENDOR",
                "occurrences": 1,
                "confidence": confidence,
                "created_at": now,
                "updated_at": now
            }
        else:
            new_occ = row["occurrences"] + 1
            new_conf = max(row["confidence"], confidence)
            cursor.execute("""
                UPDATE vendor_memory 
                SET occurrences = ?, confidence = ?, updated_at = ?
                WHERE vendor_key = ?
            """, (new_occ, new_conf, now, norm))
            conn.commit()
            return {
                "name": row["name"],
                "canonical": row["canonical"],
                "type": "VENDOR",
                "occurrences": new_occ,
                "confidence": new_conf,
                "created_at": row["created_at"],
                "updated_at": now
            }


def db_lookup_vendor(text: str) -> Optional[Dict[str, Any]]:
    init_db()
    norm = " ".join(text.strip().lower().split())
    if not norm:
        return None

    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Direct match
        cursor.execute("SELECT * FROM vendor_memory WHERE vendor_key = ?", (norm,))
        row = cursor.fetchone()
        if row:
            return dict(row)

        # Substring search
        cursor.execute("SELECT * FROM vendor_memory WHERE vendor_key LIKE ?", (f"%{norm}%",))
        row = cursor.fetchone()
        if row:
            return dict(row)

        return None


def db_load_memory() -> Dict[str, Any]:
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM vendor_memory")
        rows = cursor.fetchall()
        vendors = {}
        for r in rows:
            vendors[r["vendor_key"]] = dict(r)
        return {"vendors": vendors, "entities": {}}
