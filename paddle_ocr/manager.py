"""
paddle_ocr/manager.py
─────────────────────
The **single entry point** for the rest of the application.

Usage from the detector
───────────────────────
    from paddle_ocr import OCRManager

    ocr = OCRManager()                          # initialise once
    result = ocr.process("ID CARD RECTO", img)  # returns dict
    print(result)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from .config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DOC_DRIVER_LICENSE,
    DOC_ID_RECTO,
    DOC_ID_VERSO,
    DOC_OTHER,
    SUPPORTED_DOCUMENTS,
    BLUR_THRESHOLD,
)
from .engine import OCREngine
from .extractor import OCRExtractor
from .models import ExtractionResult, FieldResult
from .parser import OCRParser
from .state_machine import OCRStateMachine

logger = logging.getLogger("paddle_ocr.manager")


class OCRManager:
    """
    Orchestrates the full OCR pipeline.

    The detector calls ``process(document_type, image)`` and gets back a
    plain dict with the extracted fields.  Everything else — engine init,
    ROI cropping, parsing, state tracking — happens behind this class.
    """

    def __init__(
        self,
        use_gpu: bool = True,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._engine = OCREngine(use_gpu=use_gpu)
        self._extractor = OCRExtractor(self._engine)
        self._parser = OCRParser(confidence_threshold=confidence_threshold)
        self._state_machine = OCRStateMachine()
        logger.info(
            "OCRManager ready  (gpu=%s, threshold=%.2f)",
            self._engine.is_gpu, confidence_threshold,
        )

    # ── public API ──────────────────────────────────────────────────
    def process(
        self,
        document_type: str,
        image: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run the full OCR pipeline for a single frame.

        Parameters
        ----------
        document_type : str
            The stable classification label from the detector
            (``"ID CARD RECTO"``, ``"ID CARD VERSO"``, ``"DRIVERS LICENSE"``,
            or ``"OTHER DOCUMENT"``).
        image : np.ndarray
            The perspective-corrected, orientation-fixed card image (BGR).

        Returns
        -------
        Dict[str, Any]
            Structured extraction result (see ``ExtractionResult.to_dict``).
        """
        # ── unsupported document ────────────────────────────────────
        if document_type not in SUPPORTED_DOCUMENTS:
            logger.debug("Unsupported document type: %s", document_type)
            return {
                "document_type": document_type,
                "status": "unsupported_document",
                "completed": False,
            }

        # ── image quality gate ──────────────────────────────────────
        if image is None or image.size == 0:
            return self._error_dict(document_type, "Empty image received.")

        # ── run OCR on ROIs ─────────────────────────────────────────
        try:
            raw_boxes = self._extractor.extract(document_type, image)
        except Exception:
            logger.exception("OCR extraction failed.")
            return self._error_dict(document_type, "OCR extraction failed.")

        # ── parse fields ────────────────────────────────────────────
        result = self._parse_for_document(document_type, raw_boxes)

        # ── state machine ───────────────────────────────────────────
        result = self._apply_state(document_type, result)

        return result.to_dict()

    def reset(self) -> None:
        """Reset the state machine (start a fresh scan session)."""
        self._state_machine.reset()
        logger.info("OCRManager state reset.")

    @property
    def status(self) -> str:
        """Current state machine state name."""
        return self._state_machine.state_name

    @property
    def is_waiting_for_verso(self) -> bool:
        return self._state_machine.is_waiting_for_verso()

    @property
    def is_completed(self) -> bool:
        return self._state_machine.is_completed()

    @property
    def last_recto_result(self) -> Optional[Dict[str, Any]]:
        """Return stored recto data as dict, or None."""
        r = self._state_machine.recto_result
        return r.to_dict() if r else None

    # ── internals ───────────────────────────────────────────────────
    def _parse_for_document(
        self,
        document_type: str,
        raw_boxes: Dict[str, list],
    ) -> ExtractionResult:
        """Route to the correct parser methods for each document type."""

        result = ExtractionResult(document_type=document_type)

        if document_type == DOC_ID_RECTO:
            # NIN + Arabic Name
            nin_boxes = raw_boxes.get("nin", [])
            name_ar_boxes = raw_boxes.get("arabic_name", [])

            nin = self._parser.extract_nin(nin_boxes)
            arabic = self._parser.extract_arabic_name(name_ar_boxes)

            self._fill_nin(result, nin)
            self._fill_arabic(result, arabic)
            result._fields = {"nin": nin, "arabic_name": arabic}

        elif document_type == DOC_ID_VERSO:
            # French Name only
            name_fr_boxes = raw_boxes.get("french_name", [])
            french = self._parser.extract_french_name(name_fr_boxes)

            self._fill_french(result, french)
            result._fields = {"french_name": french}

        elif document_type == DOC_DRIVER_LICENSE:
            # All three fields
            nin_boxes = raw_boxes.get("nin", [])
            name_ar_boxes = raw_boxes.get("arabic_name", [])
            name_fr_boxes = raw_boxes.get("french_name", [])

            nin = self._parser.extract_nin(nin_boxes)
            arabic = self._parser.extract_arabic_name(name_ar_boxes)
            french = self._parser.extract_french_name(name_fr_boxes)

            self._fill_nin(result, nin)
            self._fill_arabic(result, arabic)
            self._fill_french(result, french)
            result._fields = {"nin": nin, "arabic_name": arabic, "french_name": french}

        return result

    def _apply_state(
        self, document_type: str, result: ExtractionResult,
    ) -> ExtractionResult:
        """Push the extraction result through the state machine."""

        if document_type == DOC_ID_RECTO:
            return self._state_machine.handle_recto(result)

        elif document_type == DOC_ID_VERSO:
            if self._state_machine.is_waiting_for_verso():
                return self._state_machine.handle_verso(result)
            else:
                # Verso scanned without recto — still return what we got
                result.status = "missing_recto"
                result.error = "Please scan the front of the ID card first."
                return result

        elif document_type == DOC_DRIVER_LICENSE:
            return self._state_machine.handle_driver_license(result)

        # Fallback
        result.status = "unknown"
        return result

    # ── field helpers ───────────────────────────────────────────────
    @staticmethod
    def _fill_nin(result: ExtractionResult, field: FieldResult) -> None:
        if field.is_valid:
            result.nin = field.value
            result.nin_confidence = field.confidence

    @staticmethod
    def _fill_arabic(result: ExtractionResult, field: FieldResult) -> None:
        if field.is_valid:
            result.arabic_name = field.value
            result.arabic_name_confidence = field.confidence

    @staticmethod
    def _fill_french(result: ExtractionResult, field: FieldResult) -> None:
        if field.is_valid:
            result.french_name = field.value
            result.french_name_confidence = field.confidence

    @staticmethod
    def _error_dict(document_type: str, error: str) -> Dict[str, Any]:
        return {
            "document_type": document_type,
            "status": "error",
            "error": error,
            "completed": False,
        }
