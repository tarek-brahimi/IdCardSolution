"""
paddle_ocr/config.py
─────────────────────
Central configuration for the OCR pipeline.

All tunable constants live here so the rest of the package
never hard-codes magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# ──────────────────────────────────────────────
# Document-type labels  (must match Idcard.py)
# ──────────────────────────────────────────────
DOC_ID_RECTO: str = "ID CARD RECTO"
DOC_ID_VERSO: str = "ID CARD VERSO"
DOC_DRIVER_LICENSE: str = "DRIVERS LICENSE"
DOC_OTHER: str = "OTHER DOCUMENT"

SUPPORTED_DOCUMENTS: set[str] = {DOC_ID_RECTO, DOC_ID_VERSO, DOC_DRIVER_LICENSE}


# ──────────────────────────────────────────────
# ROI definitions  (normalised fractions 0..1)
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class ROI:
    """A rectangular region of interest on the card image."""
    field: str          # logical field name
    lang: str           # "ar" | "fr"
    y1: float
    y2: float
    x1: float
    x2: float


# --- Algerian National ID Card — Recto ---------------------------------
ID_RECTO_ROIS: List[ROI] = [
    # Arabic full name: right half, middle portion (below NIN, above Date/Place of birth)
    ROI(field="arabic_name", lang="ar", y1=0.55, y2=0.83, x1=0.30, x2=1.00),
    # NIN: middle/lower
    ROI(field="nin",         lang="ar", y1=0.40, y2=0.80, x1=0.10, x2=1.00),
]

# --- Algerian National ID Card — Verso ----------------------------------
ID_VERSO_ROIS: List[ROI] = [
    # French full name: covers entire card to capture the MRZ at the bottom
    ROI(field="french_name", lang="fr", y1=0.00, y2=1.00, x1=0.00, x2=1.00),
]

# --- Algerian Driver License --------------------------------------------
DRIVER_LICENSE_ROIS: List[ROI] = [
    # French full name: middle block
    ROI(field="french_name", lang="fr", y1=0.35, y2=0.80, x1=0.20, x2=1.00),
    # Arabic full name: tightly constrained to middle-right
    ROI(field="arabic_name", lang="ar", y1=0.30, y2=0.55, x1=0.45, x2=1.00),
    # NIN
    ROI(field="nin",         lang="ar", y1=0.60, y2=0.97, x1=0.18, x2=1.00),
]

# Map document labels → ROI lists
DOCUMENT_ROIS: Dict[str, List[ROI]] = {
    DOC_ID_RECTO:       ID_RECTO_ROIS,
    DOC_ID_VERSO:       ID_VERSO_ROIS,
    DOC_DRIVER_LICENSE: DRIVER_LICENSE_ROIS,
}


# ──────────────────────────────────────────────
# OCR confidence thresholds
# ──────────────────────────────────────────────
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.60
NIN_CONFIDENCE_THRESHOLD: float = 0.55      # NIN is digits-only → easier to validate
NAME_CONFIDENCE_THRESHOLD: float = 0.55


# ──────────────────────────────────────────────
# NIN pattern
# ──────────────────────────────────────────────
# Algerian NIN is exactly 18 digits.
NIN_REGEX: str = r"\d{15,18}"
NIN_EXACT_LENGTH: int = 18


# ──────────────────────────────────────────────
# Keyword blacklists — text to ignore
# ──────────────────────────────────────────────
# Common header / label text that appears on cards but is NOT a name.
ARABIC_BLACKLIST_KEYWORDS: list[str] = [
    "الجمهورية",
    "الجزائرية",
    "الديمقراطية",
    "الشعبية",
    "بطاقة",
    "التعريف",
    "الوطنية",
    "اللقب",
    "الإسم",
    "الاسم",
    "تاريخ",
    "الميلاد",
    "مكان",
    "الجنس",
    "ذكر",
    "أنثى",
    "رقم",
    "رخصة",
    "السياقة",
    "الإصدار",
    "الانتهاء",
    "سلطة",
    "السلطة",
    "الوطني",
    "الإمضاء",
    "إمضاء",
]

FRENCH_BLACKLIST_KEYWORDS: list[str] = [
    "republique",
    "république",
    "algerienne",
    "algérienne",
    "democratique",
    "démocratique",
    "populaire",
    "carte",
    "nationale",
    "identite",
    "identité",
    "nom",
    "prenom",
    "prénom",
    "date",
    "naissance",
    "lieu",
    "sexe",
    "masculin",
    "feminin",
    "féminin",
    "permis",
    "conduire",
    "catégorie",
    "categorie",
    "delivre",
    "délivré",
    "wilaya",
]


# ──────────────────────────────────────────────
# Image quality thresholds
# ──────────────────────────────────────────────
BLUR_THRESHOLD: float = 50.0        # Laplacian variance below this → blurry
MIN_IMAGE_SIZE: int = 100           # pixels, either axis
