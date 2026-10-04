# ============================================================
# app/main.py
# DYNAMIC NER DOCUMENT INTELLIGENCE API
# ============================================================

from pathlib import Path
import json
import csv
import sys
import os

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.responses import HTMLResponse


# ============================================================
# NER SERVICE
# ============================================================

from app.ner.service import NERService


ner_service = NERService()


# ============================================================
# LEARNING QUEUE
# ============================================================

from app.discovery.learning_queue import (
    enqueue_entity,
    start_learning_worker,
)


# ============================================================
# TRAINING WORKER
# ============================================================

from app.workers.training_worker import (
    get_training_status,
    start_training_process,
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Dynamic NER Document Intelligence API",
    version="1.0.0",
)


# ============================================================
# DIRECTORIES
# ============================================================

UPLOAD_DIR = Path(
    "data/incoming"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_document_text(
    file_path: Path,
) -> str:

    extension = (
        file_path.suffix.lower()
    )


    # ========================================================
    # TXT
    # ========================================================

    if extension == ".txt":

        try:

            return file_path.read_text(
                encoding="utf-8"
            )

        except UnicodeDecodeError:

            return file_path.read_text(
                encoding="latin-1"
            )


    # ========================================================
    # JSON
    # ========================================================

    if extension == ".json":

        from app.extraction.json_extractor import (
            extract_json_text,
        )

        try:

            data = json.loads(
                file_path.read_text(
                    encoding="utf-8"
                )
            )

        except json.JSONDecodeError as exc:

            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON file: {exc}",
            )

        return extract_json_text(
            data
        )


    # ========================================================
    # PDF
    # ========================================================

    if extension == ".pdf":

        from app.extraction.pdf_extractor import (
            extract_pdf_text,
        )

        return extract_pdf_text(
            file_path
        )


    # ========================================================
    # IMAGE / OCR
    # ========================================================

    if extension in {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
    }:

        from app.extraction.ocr import (
            extract_text_from_image,
        )

        return extract_text_from_image(
            file_path
        )


    # ========================================================
    # CSV
    # ========================================================

    if extension == ".csv":

        rows = []

        try:

            with open(
                file_path,
                "r",
                encoding="utf-8",
                newline="",
            ) as file:

                reader = csv.reader(
                    file
                )

                for row in reader:

                    rows.append(
                        " ".join(
                            str(value)
                            for value in row
                        )
                    )

        except UnicodeDecodeError:

            with open(
                file_path,
                "r",
                encoding="latin-1",
                newline="",
            ) as file:

                reader = csv.reader(
                    file
                )

                for row in reader:

                    rows.append(
                        " ".join(
                            str(value)
                            for value in row
                        )
                    )

        return "\n".join(
            rows
        )


    # ========================================================
    # DOCX
    # ========================================================

    if extension == ".docx":

        try:

            from docx import Document

            document = Document(
                file_path
            )

            paragraphs = []

            for paragraph in (
                document.paragraphs
            ):

                value = (
                    paragraph.text.strip()
                )

                if value:

                    paragraphs.append(
                        value
                    )

            return "\n".join(
                paragraphs
            )

        except Exception as exc:

            raise HTTPException(
                status_code=500,
                detail=(
                    "DOCX extraction failed: "
                    f"{exc}"
                ),
            )


    # ========================================================
    # UNSUPPORTED
    # ========================================================

    raise HTTPException(
        status_code=400,
        detail=(
            f"Unsupported file type: "
            f"{extension}"
        ),
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "service": (
            "Dynamic NER "
            "Document Intelligence API"
        ),
        "version": "1.0.0",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "dynamic-ner",
    }


# ============================================================
# TRAINING STATUS
# ============================================================

@app.get(
    "/training/status"
)
def training_status():

    return get_training_status()


# ============================================================
# START TRAINING
# ============================================================

@app.post(
    "/training/start"
)
def start_training():

    started = (
        start_training_process()
    )

    return {
        "started": started,
        "training_status":
            get_training_status(),
    }


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

@app.post(
    "/documents/upload"
)
async def upload_document(
    file: UploadFile = File(...),
):

    # ========================================================
    # FILE NAME
    # ========================================================

    filename = (
        file.filename
        or "uploaded_document"
    )

    extension = (
        Path(filename)
        .suffix
        .lower()
    )


    # ========================================================
    # SUPPORTED FILES
    # ========================================================

    supported_extensions = {
        ".txt",
        ".json",
        ".pdf",
        ".docx",
        ".csv",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
    }


    if extension not in (
        supported_extensions
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: "
                f"{extension}"
            ),
        )


    # ========================================================
    # SAVE FILE
    # ========================================================

    safe_filename = (
        Path(filename).name
    )

    file_path = (
        UPLOAD_DIR
        / safe_filename
    )


    try:

        content = await file.read()

        file_path.write_bytes(
            content
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to save file: "
                f"{exc}"
            ),
        )


    # ========================================================
    # EXTRACT
    # ========================================================

    try:

        text = extract_document_text(
            file_path
        )

    except HTTPException:

        raise

    except Exception as exc:

        print(
            "[API] EXTRACTION ERROR:",
            exc,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Document extraction failed: "
                f"{exc}"
            ),
        )


    # ========================================================
    # NORMALIZE TEXT
    # ========================================================

    if not isinstance(
        text,
        str,
    ):

        text = str(text)


    text = text.strip()


    # ========================================================
    # EMPTY DOCUMENT
    # ========================================================

    if not text:

        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded document "
                "contains no readable text."
            ),
        )


    # ========================================================
    # LOG
    # ========================================================

    print()
    print(
        "=" * 60
    )
    print(
        "[API] DOCUMENT UPLOADED"
    )
    print(
        f"[API] File       : {filename}"
    )
    print(
        f"[API] Characters : {len(text)}"
    )
    print(
        "=" * 60
    )


    # ========================================================
    # NER
    # ========================================================

    try:

        result = (
            ner_service.process(
                text
            )
        )

    except Exception as exc:

        print(
            "[API] NER ERROR:",
            exc,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "NER processing failed: "
                f"{exc}"
            ),
        )


    # ========================================================
    # UNKNOWN CANDIDATES
    # ========================================================

    unknown_candidates = (
        result.get(
            "unknown_candidates",
            [],
        )
    )


    # ========================================================
    # LEARNING QUEUE
    # ========================================================

    queued_entities = []


    for candidate in (
        unknown_candidates
    ):

        if isinstance(
            candidate,
            dict,
        ):

            entity_text = str(
                candidate.get(
                    "text",
                    "",
                )
            ).strip()

        else:

            entity_text = str(
                candidate
            ).strip()


        if not entity_text:

            continue


        try:

            enqueue_entity(
                entity_text
            )

            queued_entities.append(
                entity_text
            )

        except Exception as exc:

            print(
                "[API] Queue error:",
                entity_text,
                exc,
            )


    # ========================================================
    # START LEARNING WORKER
    # ========================================================

    worker_started = False


    if queued_entities:

        try:

            start_learning_worker()

            worker_started = True

        except Exception as exc:

            print(
                "[API] Learning worker "
                "could not start:",
                exc,
            )


    # ========================================================
    # NON-PO INVOICE SCHEMA & MEMORY ASSEMBLY
    # ========================================================

    from app.discovery.non_po_memory import assemble_non_po_invoice_schema

    all_entities = []
    if isinstance(result, dict):
        all_entities.extend(result.get("known", []))
        all_entities.extend(result.get("known_types", []))
        all_entities.extend(result.get("production", []))
        for cand in result.get("unknown_candidates", []):
            if isinstance(cand, dict):
                c_copy = dict(cand)
                if "proposed_type" in c_copy and "type" not in c_copy:
                    c_copy["type"] = c_copy["proposed_type"]
                all_entities.append(c_copy)

    non_po_invoice = assemble_non_po_invoice_schema(
        text=text,
        extracted_entities=all_entities,
        filename=filename,
    )

    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "status": "success",

        "filename": filename,

        "text_length": len(text),

        "non_po_invoice": non_po_invoice,

        "ner": result,

        "learning": {

            "queued_entities":
                queued_entities,

            "queue_count":
                len(queued_entities),

            "worker_started":
                worker_started,

        },

    }


# ============================================================
# NON-PO MEMORY & INVOICE INSPECTION ENDPOINTS
# ============================================================

from typing import Optional
from app.discovery.non_po_memory import (
    load_non_po_memory,
    get_processed_invoices,
    get_processed_invoice_by_id,
)
from app.discovery.candidates import load_candidates


@app.get("/non-po/invoices")
def list_non_po_invoices(vendor: Optional[str] = None, limit: int = 50):
    """View stored Non-PO invoices extracted by the system."""
    invoices = get_processed_invoices(vendor_filter=vendor, limit=limit)
    return {
        "status": "success",
        "count": len(invoices),
        "invoices": invoices,
    }


@app.get("/non-po/invoices/{invoice_id}")
def get_non_po_invoice(invoice_id: str):
    """View a specific stored Non-PO invoice record by ID."""
    invoice = get_processed_invoice_by_id(invoice_id)
    if not invoice:
        raise HTTPException(
            status_code=404, 
            detail=f"Invoice '{invoice_id}' not found."
        )
    return {
        "status": "success",
        "invoice": invoice,
    }


@app.get("/non-po/memory")
def get_non_po_memory():
    """View stored vendor profiles, canonical names, and confidence scores."""
    memory = load_non_po_memory()
    vendors = memory.get("vendors", {})
    return {
        "status": "success",
        "vendors_count": len(vendors),
        "vendors": vendors,
    }


@app.get("/non-po/candidates")
def get_non_po_candidates():
    """View learned and candidates state in the learning queue."""
    candidates = load_candidates()
    return {
        "status": "success",
        "count": len(candidates),
        "candidates": candidates,
    }


@app.get("/non-po/table", response_class=HTMLResponse)
def get_non_po_invoices_table():
    """Render interactive HTML Table of all stored Non-PO invoices."""
    invoices = get_processed_invoices(limit=100)
    
    rows_html = ""
    for inv in invoices:
        inv_id = inv.get("invoice_id", "-")
        h = inv.get("header", {})
        v = inv.get("vendor", {})
        c = inv.get("customer", {})
        f = inv.get("financials", {})
        m = inv.get("metadata", {})

        inv_num = h.get("invoice_number", "-")
        po_ref = h.get("po_reference", "-") or "-"
        inv_date = h.get("invoice_date", "-")
        vendor_name = v.get("name", "-") or "-"
        customer_name = c.get("name", "-") or "-"
        tot_amt = f.get("total_amount", "-") or "-"
        status = v.get("vendor_memory_status", "PROCESSED")
        filename = m.get("filename", "-")

        rows_html += f"""
        <tr>
            <td style="font-weight: bold; color: #2563eb;">{inv_id}</td>
            <td>{inv_num}</td>
            <td><span style="background:#e0e7ff; color:#3730a3; padding:2px 8px; border-radius:12px; font-size:12px;">{po_ref}</span></td>
            <td>{inv_date}</td>
            <td style="font-weight: 600;">{vendor_name}</td>
            <td>{customer_name}</td>
            <td style="font-weight: bold; color: #059669;">{tot_amt}</td>
            <td><span style="background:#d1fae5; color:#065f46; padding:2px 8px; border-radius:12px; font-size:12px;">{status}</span></td>
            <td style="color:#6b7280; font-size:13px;">{filename}</td>
        </tr>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Non-PO Document Intelligence Dashboard</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background: #f8fafc; color: #1e293b; }}
            h1 {{ color: #0f172a; margin-bottom: 5px; }}
            p {{ color: #64748b; margin-top: 0; margin-bottom: 25px; }}
            .card {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
            table {{ width: 100%; border-collapse: collapse; text-align: left; }}
            th {{ background: #f1f5f9; padding: 14px 16px; font-size: 13px; font-weight: 700; color: #475569; text-transform: uppercase; border-bottom: 2px solid #e2e8f0; }}
            td {{ padding: 14px 16px; border-bottom: 1px solid #f1f5f9; font-size: 14px; }}
            tr:hover {{ background-color: #f8fafc; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>📄 Non-PO Invoice Intelligence Database</h1>
            <p>Live extracted Non-PO invoice records stored persistently in SQLite database (<code>data/memory/database.db</code>).</p>
            <table>
                <thead>
                    <tr>
                        <th>Invoice ID</th>
                        <th>Invoice #</th>
                        <th>PO Ref</th>
                        <th>Date</th>
                        <th>Vendor Name</th>
                        <th>Customer Name</th>
                        <th>Total Amount</th>
                        <th>Memory Status</th>
                        <th>Filename</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/non-po/table/json")
def get_non_po_invoices_table_json():
    """Returns structured table rows for Non-PO invoices as a clean JSON array."""
    invoices = get_processed_invoices(limit=100)
    table_rows = []
    for inv in invoices:
        h = inv.get("header", {})
        v = inv.get("vendor", {})
        c = inv.get("customer", {})
        f = inv.get("financials", {})
        m = inv.get("metadata", {})
        table_rows.append({
            "invoice_id": inv.get("invoice_id", "-"),
            "invoice_number": h.get("invoice_number", "-"),
            "po_reference": h.get("po_reference", "-") or "-",
            "invoice_date": h.get("invoice_date", "-"),
            "vendor_name": v.get("name", "-") or "-",
            "customer_name": c.get("name", "-") or "-",
            "total_amount": f.get("total_amount", "-") or "-",
            "vendor_memory_status": v.get("vendor_memory_status", "PROCESSED"),
            "filename": m.get("filename", "-")
        })
    return {
        "status": "success",
        "count": len(table_rows),
        "table_rows": table_rows
    }

