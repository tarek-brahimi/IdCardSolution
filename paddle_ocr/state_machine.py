"""
paddle_ocr/state_machine.py
────────────────────────────
Tracks the recto / verso state for National ID Card scanning.

States
──────
  IDLE                → nothing in progress
  WAITING_FOR_RECTO   → expecting the front of an ID card
  WAITING_FOR_VERSO   → recto done, expecting the back
  COMPLETED           → both sides scanned (or single-side document done)
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Optional

from .models import ExtractionResult

logger = logging.getLogger("paddle_ocr.state_machine")


class State(Enum):
    IDLE = auto()
    WAITING_FOR_RECTO = auto()
    WAITING_FOR_VERSO = auto()
    COMPLETED = auto()


class OCRStateMachine:
    """
    Manages the multi-step scanning flow for documents that require
    two passes (ID card recto + verso).

    For single-pass documents (driver license) the machine transitions
    directly to COMPLETED.
    """

    def __init__(self) -> None:
        self._state: State = State.IDLE
        self._recto_result: Optional[ExtractionResult] = None
        logger.debug("StateMachine initialised → IDLE")

    # ── public API ──────────────────────────────────────────────────
    @property
    def state(self) -> State:
        return self._state

    @property
    def state_name(self) -> str:
        return self._state.name

    @property
    def recto_result(self) -> Optional[ExtractionResult]:
        """Return stored recto data (or None)."""
        return self._recto_result

    def handle_recto(self, result: ExtractionResult) -> ExtractionResult:
        """
        Process a recto scan.

        Stores the partial result and transitions to WAITING_FOR_VERSO.
        """
        self._recto_result = result
        self._state = State.WAITING_FOR_VERSO
        result.status = "waiting_for_verso"
        result.completed = False
        logger.info("Recto stored → WAITING_FOR_VERSO")
        return result

    def handle_verso(self, verso_result: ExtractionResult) -> ExtractionResult:
        """
        Process a verso scan.

        Combines with stored recto data and transitions to COMPLETED.
        """
        if self._recto_result is None:
            # Edge case: verso scanned without prior recto
            logger.warning("Verso scanned but no recto data stored.")
            verso_result.status = "missing_recto"
            verso_result.error = "Recto was not scanned first."
            return verso_result

        # Merge recto + verso into a final result
        combined = ExtractionResult(
            document_type="id_card",
            nin=self._recto_result.nin,
            nin_confidence=self._recto_result.nin_confidence,
            arabic_name=self._recto_result.arabic_name,
            arabic_name_confidence=self._recto_result.arabic_name_confidence,
            french_name=verso_result.french_name,
            french_name_confidence=verso_result.french_name_confidence,
            status="completed",
            completed=True,
        )

        self._state = State.COMPLETED
        logger.info("Verso processed → COMPLETED")
        return combined

    def handle_driver_license(self, result: ExtractionResult) -> ExtractionResult:
        """
        Process a driver license scan (single pass).
        """
        result.document_type = "driver_license"
        result.status = "completed"
        result.completed = True
        self._state = State.COMPLETED
        logger.info("Driver license processed → COMPLETED")
        return result

    def reset(self) -> None:
        """Reset to idle, clearing any stored partial results."""
        self._state = State.IDLE
        self._recto_result = None
        logger.debug("StateMachine reset → IDLE")

    def is_waiting_for_verso(self) -> bool:
        return self._state == State.WAITING_FOR_VERSO

    def is_completed(self) -> bool:
        return self._state == State.COMPLETED
