import json
from pathlib import Path
from typing import Tuple

from app.extraction.json_extractor import extract_json_text
from app.extraction.pdf_extractor import (
    extract_pdf_text,
    extract_pdf_with_ocr,
)
from app.extraction.ocr import extract_image_text


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


def extract_document(file_path: str) -> Tuple[str, str]:
    """
    Automatically extract text from JSON, PDF, or image files.

    Returns:
        (extracted_text, extraction_method)
    """

    path = Path(file_path)
    extension = path.suffix.lower()

    # --------------------------------
    # JSON
    # --------------------------------

    if extension == ".json":

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        text = extract_json_text(data)

        return text, "json"


    # --------------------------------
    # IMAGE
    # --------------------------------

    if extension in IMAGE_EXTENSIONS:

        text = extract_image_text(file_path)

        return text, "ocr"


    # --------------------------------
    # PDF
    # --------------------------------

    if extension == ".pdf":

        # First attempt:
        # Extract embedded text.
        text = extract_pdf_text(file_path)

        if text.strip():

            return text, "pdf_text"


        # No embedded text means this
        # may be a scanned PDF.
        text = extract_pdf_with_ocr(file_path)

        return text, "pdf_ocr"


    # --------------------------------
    # Unsupported
    # --------------------------------

    raise ValueError(
        f"Unsupported file type: {extension}"
    )