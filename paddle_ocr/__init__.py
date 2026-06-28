"""
paddle_ocr — GPU-accelerated OCR for Algerian ID documents
══════════════════════════════════════════════════════════════

Public API
──────────
    from paddle_ocr import OCRManager

    ocr = OCRManager()                          # init once
    result = ocr.process("ID CARD RECTO", img)  # → dict
"""

from .manager import OCRManager

__all__ = ["OCRManager"]
