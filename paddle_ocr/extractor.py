"""
paddle_ocr/extractor.py
───────────────────────
Crops ROI regions from a rectified card image and feeds them
through the OCREngine.

This layer answers *"where to look"* — the parser answers
*"what to look for"*.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from .config import (
    DOC_ID_RECTO, DOC_ID_VERSO, DOC_DRIVER_LICENSE, MIN_IMAGE_SIZE, DOCUMENT_ROIS, ROI
)
from .engine import OCREngine
from .models import OCRBox

logger = logging.getLogger("paddle_ocr.extractor")


class OCRExtractor:
    """
    Given a document type and a rectified card image, runs OCR on the full image
    and then filters the resulting boxes using the spatial ROI definitions.
    """

    def __init__(self, engine: OCREngine) -> None:
        self._engine = engine

    def extract(
        self,
        document_type: str,
        image: np.ndarray,
    ) -> Dict[str, List[OCRBox]]:
        h, w = image.shape[:2]
        if h < MIN_IMAGE_SIZE or w < MIN_IMAGE_SIZE:
            logger.warning("Image too small (%dx%d) for reliable OCR.", w, h)
            return {}

        results: Dict[str, List[OCRBox]] = {}
        doc_rois = DOCUMENT_ROIS.get(document_type, [])

        if document_type == DOC_ID_RECTO:
            boxes_ar = self._engine.run(image, lang="ar")
            results["nin"] = self._filter_boxes(boxes_ar, "nin", doc_rois, h, w)
            results["arabic_name"] = self._filter_boxes(boxes_ar, "arabic_name", doc_rois, h, w)

        elif document_type == DOC_ID_VERSO:
            boxes_fr = self._engine.run(image, lang="fr")
            results["french_name"] = self._filter_boxes(boxes_fr, "french_name", doc_rois, h, w)

        elif document_type == DOC_DRIVER_LICENSE:
            boxes_fr = self._engine.run(image, lang="fr")
            results["nin"] = self._filter_boxes(boxes_fr, "nin", doc_rois, h, w)
            results["french_name"] = self._filter_boxes(boxes_fr, "french_name", doc_rois, h, w)

        return results

    @staticmethod
    def _filter_boxes(boxes: List[OCRBox], field_name: str, rois: List[ROI], h: int, w: int) -> List[OCRBox]:
        """Keep only boxes whose center point falls within the corresponding ROI."""
        target_roi = next((r for r in rois if r.field == field_name), None)
        if not target_roi:
            return boxes
            
        filtered = []
        for b in boxes:
            rel_x = b.center_x / w
            rel_y = b.center_y / h
            if target_roi.x1 <= rel_x <= target_roi.x2 and target_roi.y1 <= rel_y <= target_roi.y2:
                filtered.append(b)
        return filtered
