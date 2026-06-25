"""
Text extraction module using PaddleOCR.
Dependency: paddlepaddle-gpu + paddleocr[doc-parser]

Responsibilities:
  1. Initialize OCR reader (Arabic + French for Algerian IDs)
  2. Extract all text from the card image
  3. Identify specific fields: NIN, name, first name, date of birth
  4. Return a structured dictionary of extracted fields
"""

import sys
import os

# Auto-switch to .venv Python when run directly
if __name__ == "__main__":
    _VENV_PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "python.exe")
    if sys.platform == "win32" and os.path.exists(_VENV_PYTHON) and sys.executable.lower() != _VENV_PYTHON.lower():
        import subprocess
        ret = subprocess.call([_VENV_PYTHON] + sys.argv)
        sys.exit(ret)

# Suppress PaddlePaddle startup noise during import
_devnull_fd = os.open(os.devnull, os.O_WRONLY)
_old_stdout = os.dup(1)
_old_stderr = os.dup(2)
os.dup2(_devnull_fd, 1)
os.dup2(_devnull_fd, 2)
os.close(_devnull_fd)
try:
    import paddle
    import paddleocr
except ImportError:
    raise ImportError("Missing dependencies. Run: pip install paddlepaddle==3.2.1 paddleocr[doc-parser]>=3.6.0")
finally:
    os.dup2(_old_stdout, 1)
    os.dup2(_old_stderr, 2)
    os.close(_old_stdout)
    os.close(_old_stderr)


os.system("chcp 65001 >nul 2>&1")
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import re
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ExtractedFields:
    """Fields extracted from the ID card."""
    raw_text: str
    raw_blocks: List[Dict] = field(default_factory=list)
    nin: Optional[str] = None
    nom: Optional[str] = None
    prenom: Optional[str] = None
    date_naissance: Optional[str] = None
    lieu_naissance: Optional[str] = None
    all_numbers: List[str] = field(default_factory=list)
    confidence_moyenne: float = 0.0


def extraire_texte_roi(paddle_engine, roi_image, is_num=False, allowlist=None):
    if allowlist is None:
        if is_num:
            allowlist = '0123456789'
        else:
            arabic = ''.join(chr(c) for c in range(0x0600, 0x0700))
            allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ' + arabic
    result = paddle_engine.predict(roi_image)
    text = ""
    for res in result:
        text = res['rec_text']
        break
    text = ''.join(c for c in text if c in allowlist)
    return text


LABELS_FR = {
    "nom": ["nom", "surname", "nom de famille", "family name"],
    "prenom": ["prénom", "prenom", "first name", "prénoms"],
    "date_naissance": ["date de naissance", "date naissance", "born", "birth date", "né(e) le"],
    "lieu_naissance": ["lieu de naissance", "lieu naissance", "born in", "birth place", "né à"],
    "nin": ["nin", "n° identification", "numéro identification", "national id"],
}

LABELS_AR = {
    "nom": ["اللقب", "الاسم العائلي"],
    "prenom": ["الاسم", "الاسم الشخصي"],
    "date_naissance": ["تاريخ الميلاد", "تاريخ الازدياد"],
    "lieu_naissance": ["مكان الميلاد", "محل الميلاد", "بلدية الميلاد"],
    "nin": ["رقم التعريف الوطني", "رقم التعريف"],
}



class CardOCR:
    """
    Extracts and structures text from an Algerian ID card.

    PaddleOCR usage:
      - PaddleOCR(lang='ar') : Arabic reader (detection + recognition)
      - PaddleOCR()          : French/English reader
      - TextRecognition()    : ROI refinement engine
    """

    def __init__(self, languages: List[str] = None, gpu: bool = False):
        self.languages = languages or ['ar', 'en']
        self.gpu = gpu
        self.ocr_ar = None
        self.ocr_fr = None
        self.rec_ar = None
        self.rec_fr = None

    def _init_reader(self):
        """Lazy initialization of the PaddleOCR engines."""
        if self.ocr_ar is None:
            print("[OCR] Loading models...")
            from paddleocr import PaddleOCR, TextRecognition

            # Suppress PaddlePaddle GLOG noise (C++ level, not Python)
            devnull = os.open(os.devnull, os.O_WRONLY)
            old_stdout = os.dup(1)
            old_stderr = os.dup(2)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            os.close(devnull)
            try:
                self.ocr_ar = PaddleOCR(
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="arabic_PP-OCRv5_mobile_rec",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
                self.rec_ar = TextRecognition(model_name="arabic_PP-OCRv5_mobile_rec")
                self.ocr_fr = PaddleOCR(
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="PP-OCRv5_mobile_rec",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
                self.rec_fr = TextRecognition(model_name="PP-OCRv5_mobile_rec")
            finally:
                os.dup2(old_stdout, 1)
                os.dup2(old_stderr, 2)
                os.close(old_stdout)
                os.close(old_stderr)
            print("[OCR] Models loaded.")

    def _run_ocr(self, ocr_engine, image):
        """Run PaddleOCR and return list of (bbox, text, confidence) compatible with EasyOCR format."""
        results = ocr_engine.predict(image)
        output = []
        for res in results:
            polys = res['dt_polys']
            texts = res['rec_texts']
            scores = res['rec_scores']
            for i in range(len(texts)):
                bbox = polys[i].tolist()
                output.append((bbox, texts[i], float(scores[i])))
        return output

    def extract(self, image: np.ndarray) -> ExtractedFields:
        """
        Extract text from an ID card image using two OCR passes
        (Arabic + French) and merge results.
        """
        self._init_reader()

        from cv2 import cvtColor, COLOR_BGR2RGB
        rgb_image = cvtColor(image, COLOR_BGR2RGB)

        print("[OCR] Extracting text (Arabic pass)...")
        results_ar = self._run_ocr(self.ocr_ar, rgb_image)
        print("[OCR] Extracting text (French pass)...")
        results_fr = self._run_ocr(self.ocr_fr, rgb_image)

        # Merge results, deduplicate by bbox proximity
        # When two blocks overlap, prefer: more digits, then higher confidence
        all_results = []
        seen_centers = []

        for (bbox, text, confidence) in results_ar + results_fr:
            cx = sum(p[0] for p in bbox) / 4
            cy = sum(p[1] for p in bbox) / 4

            is_dup = False
            for idx, (sx, sy) in enumerate(seen_centers):
                if abs(cx - sx) < 30 and abs(cy - sy) < 30:
                    is_dup = True
                    existing = all_results[idx]
                    existing_digits = sum(c.isdigit() for c in existing[1])
                    new_digits = sum(c.isdigit() for c in text)
                    if new_digits > existing_digits or (new_digits == existing_digits and confidence > existing[2]):
                        all_results[idx] = (bbox, text, confidence)
                    break

            if not is_dup:
                all_results.append((bbox, text, confidence))
                seen_centers.append((cx, cy))

        raw_text_parts = []
        blocks = []
        all_confidences = []

        for (bbox, text, confidence) in all_results:
            raw_text_parts.append(text)
            blocks.append({
                "text": text,
                "confidence": confidence,
                "bbox": bbox
            })
            all_confidences.append(confidence)

        raw_text = " ".join(raw_text_parts)
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

        extracted = ExtractedFields(
            raw_text=raw_text,
            raw_blocks=blocks,
            all_numbers=self._find_all_numbers(raw_text),
            confidence_moyenne=avg_confidence
        )

        extracted.nin = self._find_nin(raw_text, blocks)
        self._find_identity_fields(extracted, blocks)

        # --- Allowlist refinement pass ---
        # Re-read each detected field's bbox region with the correct allowlist
        # to fix OCR errors like 0/O, 1/l, 8/B
        if extracted.nin:
            for block in blocks:
                block_cleaned = re.sub(r'\s+', '', block["text"])
                if extracted.nin in block_cleaned or re.search(r'\d{10,}', block_cleaned):
                    bbox = block["bbox"]
                    x1 = int(min(p[0] for p in bbox))
                    y1 = int(min(p[1] for p in bbox))
                    x2 = int(max(p[0] for p in bbox))
                    y2 = int(max(p[1] for p in bbox))
                    roi = rgb_image[y1:y2, x1:x2]
                    if roi.size > 0:
                        cleaned = extraire_texte_roi(self.rec_fr, roi, allowlist='0123456789')
                        if cleaned and len(cleaned) >= 16:
                            extracted.nin = cleaned[:18]
                    break

        # Refine date: re-read with digits + slash allowlist
        if extracted.date_naissance:
            for block in blocks:
                if extracted.date_naissance in block["text"]:
                    bbox = block["bbox"]
                    x1 = int(min(p[0] for p in bbox))
                    y1 = int(min(p[1] for p in bbox))
                    x2 = int(max(p[0] for p in bbox))
                    y2 = int(max(p[1] for p in bbox))
                    roi = rgb_image[y1:y2, x1:x2]
                    if roi.size > 0:
                        cleaned = extraire_texte_roi(
                            self.rec_fr, roi,
                            allowlist='0123456789/.'
                        )
                        if cleaned:
                            extracted.date_naissance = cleaned
                    break

        # Refine text fields: nom, prenom, lieu_naissance
        # Known label prefixes that OCR may include in re-read
        _label_prefixes = [
            "اللقب", "للقب", "الاسم", "الإسم", "للاسم", "مكان الميلاد",
            "تاريخ الميلاد", "تاريخ الإستخراج", "تاريخ الانتهاء",
        ]
        for field_name in ["nom", "prenom", "lieu_naissance"]:
            field_value = getattr(extracted, field_name)
            if field_value:
                for block in blocks:
                    if field_value.lower() in block["text"].lower():
                        bbox = block["bbox"]
                        x1 = int(min(p[0] for p in bbox))
                        y1 = int(min(p[1] for p in bbox))
                        x2 = int(max(p[0] for p in bbox))
                        y2 = int(max(p[1] for p in bbox))
                        roi = rgb_image[y1:y2, x1:x2]
                        if roi.size > 0:
                            has_arabic = any('\u0600' <= c <= '\u06FF' for c in field_value)
                            rec_engine = self.rec_ar if has_arabic else self.rec_fr
                            cleaned = extraire_texte_roi(rec_engine, roi, is_num=False)
                            if cleaned:
                                # Strip any label prefix that OCR reintroduced
                                for prefix in _label_prefixes:
                                    if cleaned.startswith(prefix) and len(cleaned) > len(prefix):
                                        candidate = cleaned[len(prefix):].lstrip(' :')
                                        if candidate:
                                            cleaned = candidate
                                            break
                                setattr(extracted, field_name, cleaned)
                        break

        return extracted

    def _find_all_numbers(self, text: str) -> List[str]:
        """Find all digit sequences in the text."""
        return re.findall(r'\d+', text)

    def _find_nin(self, text: str, blocks: List[Dict]) -> Optional[str]:
        """
        Search for the NIN (18 digits) in OCR text.
        Handles cases where the NIN is split across multiple blocks.
        """
        # Method 1: 18 consecutive digits after removing spaces
        cleaned = re.sub(r'\s+', '', text)
        match_18 = re.search(r'\b\d{18}\b', cleaned)
        if match_18:
            return match_18.group()

        # Method 2: a single OCR block contains exactly 18 digits
        for block in blocks:
            block_cleaned = re.sub(r'\s+', '', block["text"])
            if re.match(r'^\d{18}$', block_cleaned):
                return block_cleaned

        # Method 3: concatenate all digit blocks and search
        number_blocks = []
        for block in blocks:
            nums = re.findall(r'\d+', block["text"])
            number_blocks.extend(nums)

        all_digits = ''.join(number_blocks)
        match = re.search(r'\d{18}', all_digits)
        if match:
            return match.group()

        # Method 4: tolerance (16-19 digits)
        match_long = re.search(r'\d{16,19}', all_digits)
        if match_long:
            candidate = match_long.group()
            if len(candidate) >= 18:
                return candidate[:18]

        return None

    def _find_identity_fields(self, extracted: ExtractedFields, blocks: List[Dict]):
        """
        Identify name, first name, date of birth using card labels.
        Handles both Arabic and French text with bbox positioning.
        """
        # Date of birth: find dates associated with تاريخ الميلاد label
        # First try: find date block immediately after تاريخ الميلاد label in sorted blocks
        date_naissance_labels = ["تاريخ الميلاد", "date de naissance", "born on"]
        date_pattern_dmy = re.compile(r'\b(\d{2}[/-]\d{2}[/-]\d{4})\b')
        date_pattern_dot_dmy = re.compile(r'\b(\d{2}\.\d{2}\.\d{4})\b')
        date_pattern_dot_ymd = re.compile(r'\b(\d{4}\.\d{2}\.\d{2})\b')
        all_date_patterns = [date_pattern_dmy, date_pattern_dot_dmy, date_pattern_dot_ymd]

        # Build expanded_blocks first (needed here too for label detection)
        _expanded = []
        _label_value_pats = [
            r'(اللقب)\s*[:\u061A\u061B]?\s*(.+)',
            r'(الاسم)\s*[:\u061A\u061B]?\s*(.+)',
            r'(الإسم)\s*[:\u061A\u061B]?\s*(.+)',
            r'(مكان الميلاد)\s*[:\u061A\u061B]?\s*(.+)',
            r'(تاريخ الميلاد)\s*[:\u061A\u061B]?\s*(.+)',
            r'(تاريخ الإستخراج)\s*[:\u061A\u061B]?\s*(.+)',
            r'(تاريخ الانتهاء)\s*[:\u061A\u061B]?\s*(.+)',
            r'(للقب)\s*[:\u061A\u061B]?\s*(.+)',
            r'(للاسم)\s*[:\u061A\u061B]?\s*(.+)',
        ]
        for block in blocks:
            text = block["text"]
            matched = False
            for pat in _label_value_pats:
                m = re.search(pat, text)
                if m:
                    _expanded.append({"text": m.group(1), "confidence": block["confidence"], "bbox": block["bbox"]})
                    _expanded.append({"text": m.group(2), "confidence": block["confidence"], "bbox": block["bbox"]})
                    matched = True
                    break
            if not matched:
                _expanded.append(block)

        _sorted = sorted(_expanded, key=lambda b: b["bbox"][0][1])

        date_found = False
        for i, block in enumerate(_sorted):
            text_lower = block["text"].lower().strip()
            if any(lab in text_lower for lab in date_naissance_labels):
                for j in range(i + 1, len(_sorted)):
                    next_text = _sorted[j]["text"].strip()
                    for dp in all_date_patterns:
                        dm = dp.search(next_text)
                        if dm:
                            extracted.date_naissance = dm.group(1)
                            date_found = True
                            break
                    if date_found:
                        break
                    # Stop if we hit another label (don't cross into expiry date)
                    if any(lab in next_text.lower() for lab in ["تاريخ", "date"]):
                        break
                break

        # Fallback: pick earliest YYYY.MM.DD date (birth dates are chronologically earliest)
        if not date_found:
            ymd_dates = []
            for block in blocks:
                dm = date_pattern_dot_ymd.search(block["text"])
                if dm:
                    ymd_dates.append(dm.group(1))
            if ymd_dates:
                ymd_dates.sort()
                extracted.date_naissance = ymd_dates[0]
            else:
                # Last resort: first date found in raw text
                for dp in all_date_patterns:
                    dm = dp.search(extracted.raw_text)
                    if dm:
                        extracted.date_naissance = dm.group(1)
                        break

        # Labels to search for (Arabic + French + OCR misread variants)
        nom_labels = [
            "اللقب", "للقب", "اسم العائلة", "الاسم العائلي", "nom", "surname",
            "nom de famille", "family name", "اللقب:", "اللقب :"
        ]
        prenom_labels = [
            "الاسم", "الإسم", "للاسم", "الاسم الشخصي", "prenom", "prénom", "first name",
            "prénoms", "الاسم الأول", "الاسم:", "الاسم :", "الإسم:", "الإسم :"
        ]
        lieu_labels = [
            "مكان الميلاد", "محل الميلاد", "بلدية الميلاد", " lieu de naissance",
            "lieu naissance", "born in", "birth place", "né à",
            "مكان الميلاد:", "مكان الميلاد :"
        ]

        # Passe 1: Séparer les blocs "label: valeur" fusionnés par OCR arabe
        # Ex: "اللقب: طهني" lu comme 1 bloc → 2 blocs séparés
        expanded_blocks = []
        label_value_patterns = [
            r'(اللقب)\s*[:\u061A\u061B]?\s*(.+)',
            r'(للقب)\s*[:\u061A\u061B]?\s*(.+)',
            r'(الاسم)\s*[:\u061A\u061B]?\s*(.+)',
            r'(للاسم)\s*[:\u061A\u061B]?\s*(.+)',
            r'(الإسم)\s*[:\u061A\u061B]?\s*(.+)',
            r'(مكان الميلاد)\s*[:\u061A\u061B]?\s*(.+)',
            r'(تاريخ الميلاد)\s*[:\u061A\u061B]?\s*(.+)',
            r'(تاريخ الإستخراج)\s*[:\u061A\u061B]?\s*(.+)',
            r'(تاريخ الانتهاء)\s*[:\u061A\u061B]?\s*(.+)',
        ]
        for block in blocks:
            text = block["text"]
            matched = False
            for pattern in label_value_patterns:
                m = re.search(pattern, text)
                if m:
                    expanded_blocks.append({"text": m.group(1), "confidence": block["confidence"], "bbox": block["bbox"]})
                    expanded_blocks.append({"text": m.group(2), "confidence": block["confidence"], "bbox": block["bbox"]})
                    matched = True
                    break
            if not matched:
                expanded_blocks.append(block)

        # Sort blocks by y-position (top to bottom)
        sorted_blocks = sorted(expanded_blocks, key=lambda b: b["bbox"][0][1])

        for i, block in enumerate(sorted_blocks):
            text = block["text"].strip()
            text_lower = text.lower()

            # Check if this block is a label
            is_nom_label = any(kw.lower() in text_lower for kw in nom_labels)
            is_prenom_label = any(kw.lower() in text_lower for kw in prenom_labels)
            is_lieu_label = any(kw.lower() in text_lower for kw in lieu_labels)

            if is_nom_label or is_prenom_label or is_lieu_label:
                # Find the next non-label block as value
                for j in range(i + 1, len(sorted_blocks)):
                    next_text = sorted_blocks[j]["text"].strip()
                    next_lower = next_text.lower()

                    # Skip if next block is also a label
                    is_next_label = False
                    for kw_list in [nom_labels, prenom_labels, lieu_labels]:
                        if any(kw.lower() in next_lower for kw in kw_list):
                            is_next_label = True
                            break

                    if is_next_label or len(next_text) < 2:
                        continue

                    # Assign value to the correct field
                    if is_nom_label and extracted.nom is None:
                        extracted.nom = next_text
                        break
                    elif is_prenom_label and extracted.prenom is None:
                        extracted.prenom = next_text
                        break
                    elif is_lieu_label and extracted.lieu_naissance is None:
                        extracted.lieu_naissance = next_text
                        break

        # Position-based fallback: find nom/prenom below NIN on right side
        # On CNI cards, اللقب and الاسم are always below the NIN on the right
        if extracted.nom is None and extracted.prenom is None:
            nin_bbox = None
            for block in blocks:
                cleaned = re.sub(r'\s+', '', block["text"])
                if re.search(r'\d{10,}', cleaned):
                    nin_bbox = block["bbox"]
                    break

            if nin_bbox is not None:
                nin_bottom = max(p[1] for p in nin_bbox)
                nin_left = min(p[0] for p in nin_bbox)

                below_blocks = []
                for block in expanded_blocks:
                    text = block["text"].strip()
                    bbox = block["bbox"]
                    block_top = min(p[1] for p in bbox)

                    if (block_top > nin_bottom
                        and block_top - nin_bottom < 200
                        and len(text) >= 2 and len(text) <= 25
                        and any('\u0600' <= c <= '\u06FF' for c in text)
                        and not re.search(r'\d', text)):
                        is_label = any(kw in text for kw in [
                            "اللقب", "الاسم", "رقم", "تاريخ", "مكان",
                            "التعريف", "الوطني", "الوطنية", "الرقم",
                            "بطاقة", "الميلاد", "الاستخراج", "الانتهاء",
                        ])
                        if not is_label:
                            below_blocks.append(block)

                if len(below_blocks) >= 2:
                    extracted.nom = below_blocks[0]["text"].strip()
                    extracted.prenom = below_blocks[1]["text"].strip()
                elif len(below_blocks) == 1:
                    extracted.nom = below_blocks[0]["text"].strip()

        # Fallback: if no name found, use text blocks without digits
        if extracted.nom is None and extracted.prenom is None:
            # Common non-name words to skip
            skip_words = [
                "الجمهورية", "الجزائرية", "الشعبية", "الديمقراطورية",
                "رخصة", "السباقة", "سارية", "المدى", "_personne",
                "république", "algérienne", "démocratique", "populaire",
                "permis", "conduire", "valide", "card", "identity",
                "national", "algerie", "dz", "number",
                # En-tête CNI arabe
                "بطاقة", "التعريف", "الوطني", "الوطنية", "الرقم",
                "تاريخ", "مكان", "الميلاد", "الاستخراج", "الانتهاء",
                "الإستخراج", "الأكثر", "استعمال", "المستعمل",
                "الجنس", "ال blood", "الفئة",
            ]

            def is_clean_name(text):
                """Check if text looks like a name (clean Latin or Arabic, no garbled)."""
                t = text.strip()
                if len(t) < 2 or len(t) > 30:
                    return False
                if any(c.isdigit() for c in t):
                    return False
                if any(kw.lower() in t.lower() for kw in skip_words):
                    return False
                has_garbled = any(c in t for c in '!@#$%^&*()_+=[]{}|;:"<>?/~`؟،٥٧')
                if has_garbled:
                    return False
                has_latin = any('a' <= c.lower() <= 'z' for c in t)
                has_arabic = any('\u0600' <= c <= '\u06FF' for c in t)
                # Accept pure Latin names (like BRAHIMI, ABDELNASSER)
                if has_latin and not has_arabic:
                    words = t.split()
                    for w in words:
                        if len(w) > 2:
                            is_all_upper = w == w.upper()
                            is_all_lower = w == w.lower()
                            if not is_all_upper and not is_all_lower:
                                return False
                    return True
                if has_arabic and not has_latin:
                    return True
                return False

            # First try: look for clean Latin names (usually most reliable)
            latin_blocks = [b for b in sorted_blocks
                           if is_clean_name(b["text"])
                           and any('a' <= c.lower() <= 'z' for c in b["text"])
                           and not any('\u0600' <= c <= '\u06FF' for c in b["text"])]

            if len(latin_blocks) >= 2:
                extracted.nom = latin_blocks[0]["text"].strip()
                extracted.prenom = latin_blocks[1]["text"].strip()
            else:
                # Fallback: any clean text blocks
                text_blocks = [b for b in sorted_blocks if is_clean_name(b["text"])]
                if len(text_blocks) >= 2:
                    extracted.nom = text_blocks[0]["text"].strip()
                    extracted.prenom = text_blocks[1]["text"].strip()
                elif len(text_blocks) == 1:
                    extracted.nom = text_blocks[0]["text"].strip()


# ─────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────
_default_reader: Optional[CardOCR] = None


def extract_fields(image: np.ndarray, gpu: bool = False) -> ExtractedFields:
    """
    Shortcut to extract fields from a card image.
    Reuses a single reader instance across calls.
    """
    global _default_reader
    if _default_reader is None:
        _default_reader = CardOCR(gpu=gpu)
    return _default_reader.extract(image)



if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        import cv2
        from cv2 import cvtColor, COLOR_BGR2RGB
        img = cv2.imread(sys.argv[1])
        if img is not None:
            fields = extract_fields(img)

            # Show raw OCR text for comparison
            rgb = cvtColor(img, COLOR_BGR2RGB)
            reader = _default_reader.ocr_fr
            print("\n=== AVANT allowlist (raw predict) ===")
            raw_results = _default_reader._run_ocr(reader, rgb)
            for (bbox, text, conf) in raw_results:
                print(f"  [{conf:.2f}] {text}")

            print("\n=== APRÈS allowlist (refined fields) ===")
            print(f"  NIN              : {fields.nin}")
            print(f"  Nom              : {fields.nom}")
            print(f"  Prénom           : {fields.prenom}")
            print(f"  Date naissance   : {fields.date_naissance}")
            print(f"  Lieu naissance   : {fields.lieu_naissance}")
            print(f"  Confiance        : {fields.confidence_moyenne:.1%}")
        else:
            print(f"Cannot load: {sys.argv[1]}")
    else:
        print("Usage: python ocr_extractor.py <image.jpg>")
