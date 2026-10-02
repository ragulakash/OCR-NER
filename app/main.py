from pathlib import Path

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)

from app.extraction.document_extractor import (
    extract_document,
)

from app.ner.service import NERService

from app.discovery.candidates import (
    load_candidates,
)

from app.workers.training_state import (
    get_training_status,
)

from app.workers.training_worker import (
    run_training,
)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Dynamic NER Learning Pipeline",
    description=(
        "Automatic document extraction, OCR, "
        "dynamic entity discovery and continuous NER learning"
    ),
    version="0.1.0",
)


# ============================================================
# DIRECTORIES
# ============================================================

UPLOAD_DIR = Path("data/incoming")

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SUPPORTED FILE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    ".json",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


# ============================================================
# NER SERVICE
# ============================================================

# GLiNER2 is loaded once when FastAPI starts.
#
# It is NOT loaded for every request.

ner_service = NERService()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "service": "dynamic-ner",
        "version": "0.1.0",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
    }


# ============================================================
# UNIFIED DOCUMENT UPLOAD
# ============================================================

@app.post("/documents/upload")
async def upload_document(
    document_id: str,
    file: UploadFile = File(...),
):

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Filename is required",
        )


    # --------------------------------------------------------
    # Detect file extension
    # --------------------------------------------------------

    extension = Path(
        file.filename
    ).suffix.lower()


    # --------------------------------------------------------
    # Validate file type
    # --------------------------------------------------------

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail={
                "error": "Unsupported file type",
                "received": extension,
                "supported": sorted(
                    ALLOWED_EXTENSIONS
                ),
            },
        )


    # --------------------------------------------------------
    # Create file path
    # --------------------------------------------------------

    file_path = (
        UPLOAD_DIR
        / f"{document_id}{extension}"
    )


    # --------------------------------------------------------
    # Read uploaded file
    # --------------------------------------------------------

    try:

        content = await file.read()

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unable to read uploaded file: "
                f"{str(error)}"
            ),
        )


    # --------------------------------------------------------
    # Save uploaded file
    # --------------------------------------------------------

    try:

        file_path.write_bytes(
            content
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to save uploaded file: "
                f"{str(error)}"
            ),
        )


    # --------------------------------------------------------
    # DOCUMENT EXTRACTION
    #
    # JSON
    # PDF text
    # Scanned PDF -> OCR
    # Image -> OCR
    # --------------------------------------------------------

    try:

        text, extraction_method = (
            extract_document(
                str(file_path)
            )
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail={
                "error": "Document extraction failed",
                "message": str(error),
            },
        )


    # --------------------------------------------------------
    # Validate extracted text
    # --------------------------------------------------------

    if not text or not text.strip():

        raise HTTPException(
            status_code=422,
            detail=(
                "No text could be extracted "
                "from the uploaded document."
            ),
        )


    text = text.strip()


    # --------------------------------------------------------
    # GLINER2 ENTITY DISCOVERY
    # --------------------------------------------------------

    try:

        entities = ner_service.process(
            text
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail={
                "error": "NER processing failed",
                "message": str(error),
            },
        )


    # --------------------------------------------------------
    # Separate known and unknown entities
    # --------------------------------------------------------

    known_entities = [
        entity
        for entity in entities
        if entity.get("status") == "KNOWN"
    ]


    unknown_candidates = [
        entity
        for entity in entities
        if entity.get("status")
        == "UNKNOWN_CANDIDATE"
    ]


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {

        "status": "processed",

        "document": {
            "document_id": document_id,
            "filename": file.filename,
            "file_type": extension,
        },

        "extraction": {
            "method": extraction_method,
            "text_length": len(text),
            "text": text,
        },

        "entities": {

            "total": len(entities),

            "known": known_entities,

            "unknown_candidates": (
                unknown_candidates
            ),
        },
    }


# ============================================================
# CANDIDATE MEMORY
# ============================================================

@app.get("/candidates")
def get_candidates():

    try:

        candidates = load_candidates()

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail={
                "error": "Unable to load candidates",
                "message": str(error),
            },
        )


    return {
        "total": len(candidates),
        "candidates": candidates,
    }


# ============================================================
# TRAINING STATUS
# ============================================================

@app.get("/training/status")
def training_status():

    return get_training_status()


# ============================================================
# START BACKGROUND TRAINING
# ============================================================

@app.post("/training/start")
def start_training(
    background_tasks: BackgroundTasks,
):

    current_status = get_training_status()

    # --------------------------------------------------------
    # Prevent duplicate training jobs
    # --------------------------------------------------------

    if current_status.get("status") == "training":

        return {
            "status": "already_running",
            "message": (
                "A training job is already running."
            ),
        }


    # --------------------------------------------------------
    # Queue training
    # --------------------------------------------------------

    background_tasks.add_task(
        run_training
    )


    return {
        "status": "training_queued",
        "message": (
            "Training has been moved to "
            "the background."
        ),
    }


# ============================================================
# PRODUCTION MODEL
# ============================================================

@app.get("/models/production")
def production_model():

    return {
        "status": "not_available",
        "message": (
            "Production model registry "
            "will be connected after training."
        ),
    }