"""
Text extraction module using EasyOCR.
Dependency: easyocr (+ torch/torchvision)

Responsibilities:
  1. Initialize OCR reader (French + English for Algerian IDs)
  2. Extract all text from the card image
  3. Identify specific fields: NIN, name, first name, date of birth
  4. Return a structured dictionary of extracted fields
"""

import subprocess
import sys
import os

def _ensure_package(pkg, import_name=None):
    try:
        __import__(import_name or pkg)
    except ImportError:
        print(f"[SETUP] Installing {pkg}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

_ensure_package("torch")
_ensure_package("torchvision")
_ensure_package("easyocr")


os.system("chcp 65001 >nul 2>&1")
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import re
import easyocr
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

    EasyOCR usage:
      - easyocr.Reader(['fr', 'en']) : initialize reader
      - reader.readtext(image) : run OCR on image
      - Each result = (bbox, text, confidence)
    """

    def __init__(self, languages: List[str] = None, gpu: bool = False):
        self.languages = languages or ['ar', 'en']
        self.gpu = gpu
        self.reader_ar = None
        self.reader_fr = None

    def _init_reader(self):
        """Lazy initialization of the EasyOCR readers."""
        if self.reader_ar is None:
            print("[OCR] Loading EasyOCR Arabic model (ar, en)...")
            self.reader_ar = easyocr.Reader(['ar', 'en'], gpu=self.gpu)
            print("[OCR] Arabic model loaded.")
        if self.reader_fr is None:
            print("[OCR] Loading EasyOCR French model (fr, en)...")
            self.reader_fr = easyocr.Reader(['fr', 'en'], gpu=self.gpu)
            print("[OCR] French model loaded.")

    def extract(self, image: np.ndarray) -> ExtractedFields:
        """
        Extract text from an ID card image using two OCR passes
        (Arabic + French) and merge results.
        """
        self._init_reader()

        from cv2 import cvtColor, COLOR_BGR2RGB
        rgb_image = cvtColor(image, COLOR_BGR2RGB)

        print("[OCR] Extracting text (Arabic pass)...")
        results_ar = self.reader_ar.readtext(rgb_image)
        print("[OCR] Extracting text (French pass)...")
        results_fr = self.reader_fr.readtext(rgb_image)

        # Merge results, deduplicate by bbox proximity
        all_results = []
        seen_centers = []

        for (bbox, text, confidence) in results_ar + results_fr:
            # Compute center of bbox
            cx = sum(p[0] for p in bbox) / 4
            cy = sum(p[1] for p in bbox) / 4

            # Check if similar center already exists
            is_dup = False
            for (sx, sy) in seen_centers:
                if abs(cx - sx) < 30 and abs(cy - sy) < 30:
                    is_dup = True
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
        # Date of birth (common formats: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY)
        date_match = re.search(r'\b(\d{2}[/-]\d{2}[/-]\d{4})\b', extracted.raw_text)
        if date_match:
            extracted.date_naissance = date_match.group(1)
        else:
            date_match2 = re.search(r'\b(\d{2}\.\d{2}\.\d{4})\b', extracted.raw_text)
            if date_match2:
                extracted.date_naissance = date_match2.group(1)

        # Labels to search for (Arabic + French)
        nom_labels = [
            "اللقب", "اسم العائلة", "الاسم العائلي", "nom", "surname",
            "nom de famille", "family name"
        ]
        prenom_labels = [
            "الاسم", "الاسم الشخصي", "prenom", "prénom", "first name",
            "prénoms", "الاسم الأول"
        ]
        lieu_labels = [
            "مكان الميلاد", "محل الميلاد", "بلدية الميلاد", "lieu de naissance",
            "lieu naissance", "born in", "birth place", "né à"
        ]

        # Sort blocks by y-position (top to bottom)
        sorted_blocks = sorted(blocks, key=lambda b: b["bbox"][0][1])

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

        # Fallback: if no name found, use text blocks without digits
        if extracted.nom is None and extracted.prenom is None:
            # Common non-name words to skip
            skip_words = [
                "الجمهورية", "الجزائرية", "الشعبية", "الديمقراطورية",
                "رخصة", "السباقة", "سارية", "المدى", "_personne",
                "république", "algérienne", "démocratique", "populaire",
                "permis", "conduire", "valide", "card", "identity",
                "national", "algerie", "dz", "number",
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
                    # Real names are ALL UPPERCASE or all lowercase
                    # Mixed case within words = garbled OCR
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


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        import cv2
        img = cv2.imread(sys.argv[1])
        if img is not None:
            fields = extract_fields(img)
            print(f"\n=== OCR Result ===")
            print(f"Raw text  : {fields.raw_text[:200]}...")
            print(f"NIN       : {fields.nin}")
            print(f"Name      : {fields.nom}")
            print(f"First name: {fields.prenom}")
            print(f"DOB       : {fields.date_naissance}")
            print(f"Confidence: {fields.confidence_moyenne:.1%}")
        else:
            print(f"Cannot load: {sys.argv[1]}")
    else:
        print("Usage: python ocr_extractor.py <image.jpg>")
