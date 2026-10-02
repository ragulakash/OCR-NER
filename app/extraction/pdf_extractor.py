import pymupdf

from io import BytesIO

from PIL import Image
import pytesseract


def extract_pdf_text(file_path: str) -> str:
    """
    Extract embedded text from a normal text-based PDF.
    """

    document = pymupdf.open(file_path)

    pages = []

    try:

        for page in document:

            text = page.get_text()

            if text.strip():

                pages.append(
                    text.strip()
                )

    finally:

        document.close()

    return "\n".join(pages).strip()


def extract_pdf_with_ocr(file_path: str) -> str:
    """
    Render each PDF page as an image
    and extract text using Tesseract OCR.
    """

    document = fitz.open(file_path)

    pages = []

    try:

        for page in document:

            # 2x resolution for better OCR
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(2, 2)
            )

            image_bytes = pixmap.tobytes(
                "png"
            )

            image = Image.open(
                BytesIO(image_bytes)
            )

            try:

                text = pytesseract.image_to_string(
                    image
                )

                if text.strip():

                    pages.append(
                        text.strip()
                    )

            finally:

                image.close()

    finally:

        document.close()

    return "\n".join(pages).strip()