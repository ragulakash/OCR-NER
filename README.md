# Non-PO Invoice & Document Intelligence System

A high-accuracy, zero-cost Non-PO Invoice Document Intelligence architecture combining Dual-Pass OCR, GLiNER2 zero-shot entity discovery, DistilBERT sequence classification, SQLite persistence, and optional Gemini Teacher validation.

---

## ⚡ Quick Start: Running Commands & Endpoints

### 1. Terminal Commands to Run the Application

```powershell
# Navigate to the project root directory
cd C:\Users\ragul\OneDrive\Desktop\OCR-NER\dynamic-ner

# Start the FastAPI Server (Port 8000)
.venv\Scripts\python.exe -m uvicorn app.main:app --reload

# Or if virtual environment is activated:
python -m uvicorn app.main:app --reload
```

---

### 2. Primary Web & API Endpoints

| Endpoint | Type | Description |
| :--- | :--- | :--- |
| **`http://127.0.0.1:8000/docs`** | OpenAPI | Interactive FastAPI Swagger UI documentation. |
| **`http://127.0.0.1:8000/non-po/table`** | Web Dashboard | Visual CSS-styled HTML dashboard of all stored Non-PO invoices. |
| **`http://127.0.0.1:8000/non-po/table/json`** | API JSON | Clean JSON array of all stored invoice table rows. |
| **`POST /documents/upload`** | API Upload | Extract & process invoices (`.png`, `.jpg`, `.pdf`, `.docx`, `.txt`). |
| **`GET /non-po/invoices`** | API Query | Search stored invoices from SQLite (`?vendor=...&limit=50`). |
| **`GET /non-po/invoices/{id}`** | API Query | Retrieve a specific invoice JSON by ID. |
| **`GET /non-po/memory`** | API Query | Inspect learned vendor profiles and occurrence metrics. |

---

### 3. Essential Python Module Imports

```python
# 1. Non-PO Invoice Schema & Memory Assembly
from app.discovery.non_po_memory import (
    assemble_non_po_invoice_schema,
    get_processed_invoices,
    get_processed_invoice_by_id,
    load_non_po_memory,
)

# 2. SQLite Database Persistence Engine
from app.database.sqlite_db import (
    init_db,
    db_save_invoice,
    db_get_invoices,
    db_remember_vendor,
    db_load_memory,
)

# 3. Dual-Pass Preprocessed OCR
from app.extraction.ocr import extract_text_from_image, extract_text_from_pil

# 4. NER Engine & Discovery Service
from app.ner.service import ner_service, extract_entities

# 5. Gemini Teacher Validator (Optional Cloud Teacher)
from app.learning.gemini_validator import get_gemini_validator
```

---

### 4. How to Test Invoice Upload

#### Using PowerShell:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/documents/upload" `
  -Method Post `
  -Form @{ file = Get-Item "sample_invoices\sample_non_po_1.txt" } `
  | ConvertTo-Json -Depth 6
```

#### Using cURL:
```bash
curl -X POST "http://127.0.0.1:8000/documents/upload" \
  -F "file=@sample_invoices/sample_non_po_2.docx"
```

#### Using Python:
```python
import requests

url = "http://127.0.0.1:8000/documents/upload"
with open("sample_invoices/sample_non_po_1.txt", "rb") as f:
    response = requests.post(url, files={"file": f})

print(response.json()["non_po_invoice"])
```

---

## 🏗️ System Architecture & Workflow

```
 ┌─────────────────────────────────────────────────────────┐
 │               Document Upload / Image Input             │
 └────────────────────────────┬────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │   Dual-Pass Preprocessed OCR / DOCX / PDF Extractor     │
 │        (2x Lanczos Scaling + PSM 3 Page Segmentation)   │
 └────────────────────────────┬────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │             GLiNER2 Zero-Shot Discovery Engine          │
 └────────────────────────────┬────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │        DistilBERT Production Type Classifier            │
 └────────────────────────────┬────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
 [High Confidence (≥ 0.75)]                      [Low Confidence (< 0.75)]
        │                                           │
        ▼                                           ▼
 ┌──────────────┐                        ┌─────────────────────┐
 │ Auto-Confirm │                        │  Optional Gemini    │
 └──────┬───────┘                        │  Teacher Validator  │
        │                                └──────────┬──────────┘
        │                                           │
        └─────────────────────┬─────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │     Non-PO Invoice Spatial Assembly & Vendor Memory     │
 └────────────────────────────┬────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │    SQLite Database Storage (data/memory/database.db)    │
 └─────────────────────────────────────────────────────────┘
```

---

## 🗄️ Database Schema & Storage

Extracted invoices and vendor profiles are saved in SQLite at `data/memory/database.db`:

1. **`processed_invoices`**:
   - `invoice_id`, `invoice_number`, `po_reference`, `invoice_date`, `due_date`, `vendor_name`, `customer_name`, `total_amount`, `currency`, `bank_name`, `account_number`, `raw_json`, `filename`, `created_at`.
2. **`vendor_memory`**:
   - `vendor_key`, `name`, `canonical`, `occurrences`, `confidence`, `updated_at`.

---

## ⚙️ Environment Variables (Optional Gemini Teacher)

Create a `.env` file in the project root:

```env
# Gemini Teacher Integration (Optional)
GEMINI_ENABLED=false
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

---

## 📂 Complete Project File Structure

```
dynamic-ner/
├── app/
│   ├── database/          # SQLite persistence engine
│   │   └── sqlite_db.py
│   ├── discovery/         # Candidate discovery, Non-PO memory & schema assembly
│   │   ├── candidates.py
│   │   ├── learning_queue.py
│   │   ├── non_po_memory.py
│   │   └── type_learning_worker.py
│   ├── extraction/        # Preprocessed OCR & document extractors
│   │   ├── ocr.py
│   │   ├── pdf_extractor.py
│   │   └── json_extractor.py
│   ├── learning/          # DistilBERT classifier, replay buffer, & Gemini validator
│   │   ├── config.py
│   │   ├── dataset_builder.py
│   │   ├── evaluator.py
│   │   ├── gemini_validator.py
│   │   ├── replay_buffer.py
│   │   ├── trainer.py
│   │   └── type_classifier.py
│   ├── ner/               # GLiNER discovery, phrase mapping, & production model
│   │   ├── gliner.py
│   │   ├── mapping.py
│   │   ├── production.py
│   │   └── service.py
│   ├── registry/          # Model registry & promotion
│   │   └── model_registry.py
│   ├── workers/           # Async background training workers
│   │   ├── training_state.py
│   │   └── training_worker.py
│   └── main.py            # FastAPI main application & UI routes
├── data/
│   ├── memory/            # SQLite database (database.db)
│   ├── candidates/        # Candidate learning store
│   └── replay/            # Active learning replay buffer
├── sample_invoices/       # Test invoice files (.txt, .docx)
├── scripts/               # Test and verification scripts
└── README.md
```
