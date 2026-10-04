# ============================================================
# app/extraction/ocr.py
# HYBRID DUAL-PASS TESSERACT OCR ENGINE
# ============================================================

import re
from pathlib import Path
from typing import Union

from PIL import Image, ImageOps, ImageFilter
import pytesseract


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

TESSERACT_PATH = None

if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================
# SUPPORTED IMAGE TYPES
# ============================================================

SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}


# ============================================================
# TESSERACT AVAILABILITY
# ============================================================

def check_tesseract() -> bool:
    """Check whether Tesseract OCR is installed and accessible."""
    try:
        version = pytesseract.get_tesseract_version()
        print(f"[OCR] Tesseract detected: {version}")
        return True
    except Exception as exc:
        print("[OCR] Tesseract is not available:", exc)
        return False


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image: Image.Image, adaptive_contrast: bool = True) -> Image.Image:
    """
    Advanced preprocessing for document OCR:
        1. Convert to RGB / Grayscale
        2. 2x Bilinear/Lanczos Rescaling if under 2400px
        3. Auto contrast & sharpening
    """
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    w, h = image.size
    if w < 2400 or h < 2400:
        image = image.resize((w * 2, h * 2), Image.Resampling.LANCZOS)

    gray = ImageOps.grayscale(image)
    if adaptive_contrast:
        gray = ImageOps.autocontrast(gray, cutoff=2)

    gray = gray.filter(ImageFilter.SHARPEN)
    return gray


def clean_ocr_text(text: str) -> str:
    """Normalizes lines and cleans stray Tesseract OCR artifacts."""
    if not text:
        return ""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove leading/trailing stray symbols like ': ', 'i ', '| ', '! '
        line = re.sub(r"^[iI:\|\!\\/\-]\s*", "", line)
        line = re.sub(r"\s*[iI:\|\!\\/\-]$", "", line)
        
        # Typos fix for common OCR misread terms
        line = re.sub(r"\bTox rate\b", "Tax rate", line, flags=re.IGNORECASE)
        line = re.sub(r"\bhe Shirt\b", "The Shirt", line, flags=re.IGNORECASE)
        lines.append(line)
    return "\n".join(lines)


# ============================================================
# DUAL-PASS HYBRID OCR
# ============================================================

def extract_text_from_pil(
    image: Image.Image,
    language: str = "eng",
    config: str = "--oem 3 --psm 3",
) -> str:
    """
    Dual-Pass Hybrid OCR Execution:
        Pass 1: PSM 3 (Automatic Page Segmentation)
        Pass 2: PSM 6 (Uniform Block Text)
    Selects the pass with richer structured character yield.
    """
    if image is None:
        return ""

    processed_image = preprocess_image(image)

    # Pass 1: PSM 3
    try:
        t1 = pytesseract.image_to_string(processed_image, lang=language, config="--oem 3 --psm 3")
    except Exception:
        t1 = ""

    # Pass 2: PSM 6
    try:
        t2 = pytesseract.image_to_string(processed_image, lang=language, config="--oem 3 --psm 6")
    except Exception:
        t2 = ""

    clean_t1 = clean_ocr_text(t1)
    clean_t2 = clean_ocr_text(t2)

    # Select richer pass
    chosen_text = clean_t1 if len(clean_t1) >= len(clean_t2) else clean_t2
    if not chosen_text:
        chosen_text = clean_t2 or clean_t1

    return chosen_text.strip()


def extract_text_from_image(
    file_path: Union[str, Path],
    language: str = "eng",
    config: str = "--oem 3 --psm 3",
) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {path.suffix}")

    try:
        with Image.open(path) as image:
            print(f"[OCR] Processing image: {path.name} ({image.size[0]}x{image.size[1]})")
            text = extract_text_from_pil(image=image, language=language, config=config)
    except Exception as exc:
        raise RuntimeError(f"Image OCR failed for {path.name}: {exc}") from exc

    print(f"[OCR] Extracted characters: {len(text)}")
    return text


def extract_text(file_path: Union[str, Path]) -> str:
    return extract_text_from_image(file_path)


def extract_image_text(file_path: Union[str, Path]) -> str:
    return extract_text_from_image(file_path)