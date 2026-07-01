import cv2
import os
from Idcard import load_templates, fix_orientation, classify_document, get_perspective_transform
from paddle_ocr.manager import OCRManager
from typing import Dict, Any, Optional

class Scanner:
    def __init__(self):
        print("Initializing OCR Manager...")
        self.ocr_manager = OCRManager(use_gpu=True)
        
        print("Loading Templates...")
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        template_dir = os.path.join(BASE_DIR, "templates")
        self.templates = load_templates(template_dir)
        print("Scanner initialized.")

    def _detect_and_rectify(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                warped = get_perspective_transform(frame, pts, width=856, height=540)
                return warped
        return None

    def scan_id(self, frame) -> Optional[Dict[str, Any]]:
        """
        Takes an OpenCV frame (BGR). Returns a dictionary of extracted fields if successful.
        """
        rectified_card = self._detect_and_rectify(frame)
        if rectified_card is None:
            # Fallback if card isn't properly framed, use the whole frame
            rectified_card = cv2.resize(frame, (856, 540))
            
        oriented_card = fix_orientation(rectified_card)
        if oriented_card is None:
            oriented_card = rectified_card
            
        best_label, score = classify_document(oriented_card, self.templates)
        
        if not best_label or score < 5.0:
            # Not a recognized document
            return None
            
        print(f"Classified as: {best_label} (score: {score})")
        extracted_fields = self.ocr_manager.process(best_label, oriented_card)
        
        # Flatten the fields to a simple dict
        result = {
            "document_type": best_label,
            "nin": None,
            "french_name": None,
            "arabic_name": None
        }
        
        for k, v in extracted_fields.items():
            result[k] = v.value
            
        # Optional validation
        if not result["nin"] and not result["french_name"] and not result["arabic_name"]:
            return None
            
        return result
