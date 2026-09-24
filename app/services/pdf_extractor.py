import io
import logging
import re
from typing import Any, Dict, List, Tuple
import fitz  # PyMuPDF
from PIL import Image
try:
    import pytesseract
except ImportError:
    pytesseract = None

from app.core.config import settings

logger = logging.getLogger("pdf_extractor")


class PDFLayoutParser:
    """
    Layout-aware PDF parser with multi-column block ordering, OCR fallback, 
    noise cleaning, and context windowing.
    """

    def __init__(self):
        self.noise_patterns = [
            r"Machine Translated by Google\s*",
            r"Page\s+\d+\s+of\s+\d+\s*",
            r"Seite:\s*\d+\s*von\s*\d+\s*",
            r"Confidential\s*-\s*Internal Use Only\s*",
            r"^\s*ISO\s+Copyright\s+office.*$",
            r"All rights reserved.*$",
        ]

    def extract_ordered_blocks(self, page: fitz.Page) -> str:
        """
        Extracts text blocks sorted in visual reading order (y-bucketing + x).
        Prevents column interleaved text corruption on multi-column layouts.
        """
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
        # Bucket by ~20px y-coordinates, then sort left-to-right (x0)
        text_blocks.sort(key=lambda b: (round(b[1] / 20) * 20, b[0]))
        return "\n".join(b[4].strip() for b in text_blocks)

    def is_image_heavy_page(self, page: fitz.Page, extracted_text: str) -> bool:
        """Detects image-heavy pages / moodboards with negligible text content."""
        if len(extracted_text.strip()) >= settings.MIN_PAGE_TEXT_CHARS:
            return False
        try:
            img_area = sum(b["width"] * b["height"] for b in page.get_image_info())
            page_area = page.rect.width * page.rect.height
            return page_area > 0 and (img_area / page_area) >= settings.IMAGE_AREA_RATIO
        except Exception:
            return False

    def ocr_page(self, page: fitz.Page) -> str:
        """Fallback OCR when embedded text streams are missing or sparse."""
        if pytesseract is None:
            return ""
        try:
            pix = page.get_pixmap(dpi=settings.OCR_DPI)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            return pytesseract.image_to_string(img, lang="eng")
        except Exception as exc:
            logger.warning("OCR fallback skipped or failed on page: %s", exc)
            return ""

    def clean_text(self, text: str) -> str:
        """Strips noise patterns, repeated blank lines, and watermarks."""
        for pattern in self.noise_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def parse_pdf_bytes(self, pdf_bytes: bytes) -> List[Tuple[int, str]]:
        """Parses PDF bytes into page-level text items (page_num, clean_text)."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages: List[Tuple[int, str]] = []

        for page_num, page in enumerate(doc, start=1):
            text = self.extract_ordered_blocks(page)
            # Check if page needs OCR
            if len(text.strip()) < settings.MIN_PAGE_TEXT_CHARS:
                ocr_text = self.ocr_page(page)
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text

            # Skip moodboards / image slides
            if self.is_image_heavy_page(page, text):
                logger.info("Page %d: detected as image-heavy, skipping", page_num)
                continue

            text = self.clean_text(text)
            if text:
                pages.append((page_num, text))

        doc.close()
        return pages

    def build_windows(self, pages: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
        """
        Batches consecutive pages into context-preserving windows up to MAX_WINDOW_CHARS.
        """
        windows: List[Dict[str, Any]] = []
        current_chunks: List[str] = []
        current_pages: List[int] = []
        current_len = 0

        for page_num, text in pages:
            chunk = f"[Page {page_num}]\n{text}"
            chunk_len = len(chunk)

            if current_len + chunk_len > settings.MAX_WINDOW_CHARS and current_chunks:
                windows.append({
                    "text": "\n\n".join(current_chunks),
                    "pages": list(current_pages)
                })
                current_chunks = []
                current_pages = []
                current_len = 0

            current_chunks.append(chunk)
            current_pages.append(page_num)
            current_len += chunk_len

        if current_chunks:
            windows.append({
                "text": "\n\n".join(current_chunks),
                "pages": list(current_pages)
            })

        return windows


pdf_parser = PDFLayoutParser()
