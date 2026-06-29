"""
paddle_ocr/models.py
────────────────────
Data classes for structured OCR results.

No raw strings leave this module — everything is typed and validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class OCRBox:
    """A single text detection from PaddleOCR."""
    text: str
    confidence: float
    bbox: List[List[float]]         # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]

    @property
    def center_x(self) -> float:
        """Horizontal centre of the bounding box."""
        return sum(p[0] for p in self.bbox) / 4

    @property
    def center_y(self) -> float:
        """Vertical centre of the bounding box."""
        return sum(p[1] for p in self.bbox) / 4

    @property
    def top(self) -> float:
        """Top edge (minimum y)."""
        return min(p[1] for p in self.bbox)

    @property
    def bottom(self) -> float:
        """Bottom edge (maximum y)."""
        return max(p[1] for p in self.bbox)

    @property
    def left(self) -> float:
        """Left edge (minimum x)."""
        return min(p[0] for p in self.bbox)

    @property
    def right(self) -> float:
        """Right edge (maximum x)."""
        return max(p[0] for p in self.bbox)

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def width(self) -> float:
        return self.right - self.left


@dataclass
class FieldResult:
    """Result for a single extracted field (NIN, name, etc.)."""
    value: Optional[str] = None
    confidence: float = 0.0
    raw_texts: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """True when a value was extracted with acceptable confidence."""
        return self.value is not None and self.error is None


@dataclass
class ExtractionResult:
    """Full result of an OCR processing pass."""
    document_type: str = ""
    nin: Optional[str] = None
    nin_confidence: float = 0.0
    arabic_name: Optional[str] = None
    arabic_name_confidence: float = 0.0
    french_name: Optional[str] = None
    french_name_confidence: float = 0.0
    status: str = ""                    # e.g. "waiting_for_verso", "completed", …
    completed: bool = False
    error: Optional[str] = None

    # Keep full FieldResult objects for debugging
    _fields: Dict[str, FieldResult] = field(default_factory=dict, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to the dict format expected by the caller."""
        d: Dict[str, Any] = {"document_type": self.document_type, "status": self.status}

        if self.nin is not None:
            d["nin"] = self.nin
            d["nin_confidence"] = round(self.nin_confidence, 3)

        if self.arabic_name is not None:
            d["arabic_name"] = self.arabic_name
            d["arabic_name_confidence"] = round(self.arabic_name_confidence, 3)

        if self.french_name is not None:
            d["french_name"] = self.french_name
            d["french_name_confidence"] = round(self.french_name_confidence, 3)

        d["completed"] = self.completed

        if self.error:
            d["error"] = self.error

        return d
