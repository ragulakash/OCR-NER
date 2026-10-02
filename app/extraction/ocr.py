from PIL import Image
import pytesseract


def extract_image_text(file_path: str) -> str:
    """
    Extract text from an image using Tesseract OCR.
    """

    image = Image.open(file_path)

    try:

        text = pytesseract.image_to_string(
            image
        )

    finally:

        image.close()

    return text.strip()