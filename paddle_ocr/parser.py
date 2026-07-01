"""
paddle_ocr/parser.py
────────────────────
Field-level text extraction from raw OCR boxes.

Dedicated functions for each target field:
  • ``extract_nin``         — 18-digit National Identification Number
  • ``extract_arabic_name`` — Arabic full name
  • ``extract_french_name`` — French (Latin) full name

Each function uses:
  • bounding-box spatial position
  • regex patterns
  • confidence filtering
  • keyword blacklists
  • text normalisation
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import List

from .config import (
    ARABIC_BLACKLIST_KEYWORDS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    FRENCH_BLACKLIST_KEYWORDS,
    NAME_CONFIDENCE_THRESHOLD,
    NIN_CONFIDENCE_THRESHOLD,
    NIN_EXACT_LENGTH,
    NIN_REGEX,
)
from .models import FieldResult, OCRBox

logger = logging.getLogger("paddle_ocr.parser")

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

_ARABIC_RANGE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
_LATIN_RANGE  = re.compile(r"[A-Za-zÀ-ÿ]")


def _is_arabic(text: str) -> bool:
    """Return True if the majority of alphabetic chars are Arabic."""
    ar = len(_ARABIC_RANGE.findall(text))
    la = len(_LATIN_RANGE.findall(text))
    return ar > la


def _is_latin(text: str) -> bool:
    """Return True if the majority of alphabetic chars are Latin."""
    la = len(_LATIN_RANGE.findall(text))
    ar = len(_ARABIC_RANGE.findall(text))
    return la > ar


def _normalise(text: str) -> str:
    """Strip extra whitespace and control characters."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _contains_blacklisted(text: str, blacklist: list[str]) -> bool:
    """Check if *text* (lowered) matches any blacklist keyword."""
    lower = text.lower()
    # Also try without diacritics
    stripped = unicodedata.normalize("NFD", lower)
    stripped = "".join(c for c in stripped if unicodedata.category(c) != "Mn")
    for kw in blacklist:
        if kw in lower or kw in stripped:
            return True
    return False


def _extract_digits(text: str) -> str:
    """Return only ASCII digit characters from *text*."""
    return "".join(c for c in text if c.isdigit())


# ──────────────────────────────────────────────────────────────────────
# Public extraction functions
# ──────────────────────────────────────────────────────────────────────

class OCRParser:
    """Stateless parser that converts raw OCRBox lists into FieldResults."""

    def __init__(self, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> None:
        self._threshold = confidence_threshold

    # ── NIN ──────────────────────────────────────────────────────────
    def extract_nin(self, boxes: List[OCRBox]) -> FieldResult:
        raw_texts: list[str] = [b.text for b in boxes]
        
        # Sort boxes top-to-bottom, left-to-right
        sorted_boxes = sorted(boxes, key=lambda b: (int(b.center_y / 10), b.center_x))
        
        candidates: list[tuple[str, float]] = []
        
        # 1. Look for direct match in individual boxes
        for box in sorted_boxes:
            digits = _extract_digits(box.text)
            matches = re.findall(NIN_REGEX, digits)
            for m in matches:
                if len(m) == NIN_EXACT_LENGTH:
                    candidates.append((m, box.confidence))

        # 2. Look for split NINs across the same line
        if not candidates:
            # Group by approximate Y (within 15 pixels)
            lines = []
            current_line = []
            current_y = -1
            for box in sorted_boxes:
                if current_y == -1 or abs(box.center_y - current_y) < 15:
                    current_line.append(box)
                    current_y = box.center_y
                else:
                    lines.append(current_line)
                    current_line = [box]
                    current_y = box.center_y
            if current_line:
                lines.append(current_line)
                
            for line in lines:
                line.sort(key=lambda b: b.center_x)
                line_text = " ".join(b.text for b in line)
                line_digits = _extract_digits(line_text)
                matches = re.findall(NIN_REGEX, line_digits)
                avg_conf = sum(b.confidence for b in line) / max(len(line), 1)
                for m in matches:
                    if len(m) == NIN_EXACT_LENGTH:
                        candidates.append((m, avg_conf))

        # 3. Fallback: Accept partial 15-17 digit NINs if exact 18-digit is not found
        if not candidates:
            for box in sorted_boxes:
                digits = _extract_digits(box.text)
                matches = re.findall(r"\d{15,17}", digits)
                for m in matches:
                    candidates.append((m, box.confidence * 0.8))
                    
            if not candidates:
                for line in lines:
                    line.sort(key=lambda b: b.center_x)
                    line_text = " ".join(b.text for b in line)
                    line_digits = _extract_digits(line_text)
                    matches = re.findall(r"\d{15,17}", line_digits)
                    avg_conf = sum(b.confidence for b in line) / max(len(line), 1)
                    for m in matches:
                        candidates.append((m, avg_conf * 0.8))

        if not candidates:
            return FieldResult(
                raw_texts=raw_texts,
                error="NIN not found in OCR output.",
            )

        candidates.sort(key=lambda x: (-len(x[0]), -x[1]))
        best_value, best_conf = candidates[0]

        if best_conf < NIN_CONFIDENCE_THRESHOLD:
            return FieldResult(
                value=best_value,
                confidence=best_conf,
                raw_texts=raw_texts,
                error=f"NIN confidence too low ({best_conf:.2f}).",
            )

        logger.info("NIN extracted: %s (conf=%.3f)", best_value, best_conf)
        return FieldResult(value=best_value, confidence=best_conf, raw_texts=raw_texts)

    # ── Arabic Name ──────────────────────────────────────────────────
    def extract_arabic_name(self, boxes: List[OCRBox]) -> FieldResult:
        raw_texts: list[str] = []
        name_parts: list[tuple[str, float, float]] = []

        for box in boxes:
            raw_texts.append(box.text)
            text = _normalise(box.text)

            if not text or box.confidence < NAME_CONFIDENCE_THRESHOLD:
                continue

            if not _is_arabic(text):
                continue

            # PaddleOCR outputs visual RTL. Reverse characters to get logical Arabic string!
            # e.g., "ةقاطب" becomes "بطاقة"
            text = text[::-1]

            # Remove blacklisted keywords instead of skipping the entire box
            for kw in ARABIC_BLACKLIST_KEYWORDS:
                if kw in text:
                    text = text.replace(kw, "")

            # Skip single characters or strings that are just numbers/punctuation
            cleaned = re.sub(r"[^\w\s]", "", text).strip()
            if len(cleaned) < 2:
                continue

            name_parts.append((cleaned, box.confidence, box.center_y))

        if not name_parts:
            return FieldResult(
                raw_texts=raw_texts,
                error="Arabic name not found in OCR output.",
            )

        # Sort top-to-bottom
        name_parts.sort(key=lambda x: x[2])

        combined = " ".join(p[0] for p in name_parts)
        avg_conf = sum(p[1] for p in name_parts) / len(name_parts)

        if avg_conf < NAME_CONFIDENCE_THRESHOLD:
            return FieldResult(
                value=combined,
                confidence=avg_conf,
                raw_texts=raw_texts,
                error=f"Arabic name confidence too low ({avg_conf:.2f}).",
            )

        logger.info("Arabic name extracted: %s (conf=%.3f)", combined, avg_conf)
        return FieldResult(value=combined, confidence=avg_conf, raw_texts=raw_texts)

    # ── French Name ──────────────────────────────────────────────────
    def extract_french_name(self, boxes: List[OCRBox]) -> FieldResult:
        raw_texts: list[str] = []
        name_parts: list[tuple[str, float, float, float]] = []

        for box in boxes:
            raw_texts.append(box.text)
            
            # Fast-path: Check for MRZ name format (e.g., MOHAMMEDI<<ILYES<<<<)
            if "<<" in box.text:
                mrz_text = box.text.strip("< ")
                # Only process if it doesn't look like the document/number line (e.g., IDDZA...)
                if not mrz_text.startswith("IDDZA") and not any(c.isdigit() for c in mrz_text):
                    parts = [p.strip() for p in mrz_text.split("<<") if p.strip()]
                    # Usually "LASTNAME<<FIRSTNAME<SECONDNAME"
                    if len(parts) >= 2:
                        mrz_name = " ".join(parts).replace("<", " ")
                        logger.info("French name extracted from MRZ: %s (conf=%.3f)", mrz_name, box.confidence)
                        return FieldResult(value=mrz_name, confidence=box.confidence, raw_texts=raw_texts)

            text = _normalise(box.text)

            if not text or box.confidence < NAME_CONFIDENCE_THRESHOLD:
                continue

            if not _is_latin(text):
                continue

            if _contains_blacklisted(text, FRENCH_BLACKLIST_KEYWORDS):
                continue

            cleaned = re.sub(r"[^A-Za-zÀ-ÿ\s\-']", "", text).strip()
            if len(cleaned) < 2:
                continue

            # Only accept mostly uppercase words (names are uppercase on the ID)
            upper_count = sum(1 for c in cleaned if c.isupper())
            if upper_count < len(cleaned) * 0.5:
                continue

            name_parts.append((cleaned, box.confidence, box.center_y, box.center_x))

        if not name_parts:
            return FieldResult(
                raw_texts=raw_texts,
                error="French name not found in OCR output.",
            )

        # Sort top-to-bottom, left-to-right
        name_parts.sort(key=lambda x: (x[2], x[3]))

        combined = " ".join(p[0] for p in name_parts)
        avg_conf = sum(p[1] for p in name_parts) / len(name_parts)

        if avg_conf < NAME_CONFIDENCE_THRESHOLD:
            return FieldResult(
                value=combined,
                confidence=avg_conf,
                raw_texts=raw_texts,
                error=f"French name confidence too low ({avg_conf:.2f}).",
            )

        logger.info("French name extracted: %s (conf=%.3f)", combined, avg_conf)
        return FieldResult(value=combined, confidence=avg_conf, raw_texts=raw_texts)
