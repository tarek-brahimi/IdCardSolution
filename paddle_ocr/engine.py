"""
paddle_ocr/engine.py
────────────────────
Low-level wrapper around PaddleOCR.

Responsibilities
────────────────
* Auto-detect CUDA / fall back to CPU.
* Initialise PaddleOCR instances (Arabic + French) **once**.
* Provide a thin ``run()`` method that accepts a BGR numpy image
  and returns ``List[OCRBox]``.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import cv2
import numpy as np

from .models import OCRBox

logger = logging.getLogger("paddle_ocr.engine")


def _detect_gpu() -> bool:
    """Return True if CUDA is available through PaddlePaddle."""
    try:
        import paddle
        available = paddle.device.is_compiled_with_cuda()
        if available:
            logger.info("CUDA detected — PaddlePaddle will use GPU.")
        else:
            logger.info("CUDA not available — PaddlePaddle will use CPU.")
        return available
    except Exception:
        logger.warning("Could not query PaddlePaddle for CUDA support; defaulting to CPU.")
        return False


class OCREngine:
    """
    Singleton-style PaddleOCR engine.

    Creates two internal PaddleOCR readers (Arabic and French) on first use
    and reuses them for the lifetime of the process.

    Supports both PaddleOCR 2.x (`.ocr()`) and 3.x (`.predict()`) APIs
    transparently.
    """

    def __init__(self, use_gpu: bool = True) -> None:
        self._use_gpu: bool = use_gpu and _detect_gpu()
        self._reader_ar: Optional[object] = None   # lazy
        self._reader_fr: Optional[object] = None   # lazy
        self._initialised: bool = False
        self._api_version: int = 2                  # detected during init
        logger.info("OCREngine created  (gpu=%s)", self._use_gpu)

    # ── lazy initialisation ─────────────────────────────────────────────
    def _ensure_init(self) -> None:
        """Lazy-load PaddleOCR models on first call."""
        if self._initialised:
            return

        # Disable oneDNN/MKLDNN before importing paddleocr
        os.environ["FLAGS_use_mkldnn"] = "0"

        # On Windows, Python >= 3.8 requires explicit DLL directories for cudnn/cublas
        if os.name == "nt":
            import site
            import glob
            site_packages = site.getsitepackages()
            for sp in site_packages:
                nvidia_paths = glob.glob(os.path.join(sp, "nvidia", "*", "bin"))
                for path in nvidia_paths:
                    if os.path.isdir(path):
                        try:
                            os.add_dll_directory(path)
                            logger.debug("Added DLL directory: %s", path)
                        except Exception as e:
                            logger.warning("Failed to add DLL dir %s: %s", path, e)

        from paddleocr import PaddleOCR  # type: ignore[import-untyped]

        # Detect API version: v3 uses 'device', v2 uses 'use_gpu'
        import inspect
        init_params = inspect.signature(PaddleOCR.__init__).parameters
        use_v3 = "device" in init_params and "use_gpu" not in init_params

        if use_v3:
            self._api_version = 3
            device = "gpu" if self._use_gpu else "cpu"
            logger.info("PaddleOCR 3.x API detected (device=%s).", device)
            logger.info("Loading PaddleOCR Arabic model …")
            self._reader_ar = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang="ar",
                device=device,
                enable_mkldnn=False,
            )
            logger.info("Loading PaddleOCR French model …")
            self._reader_fr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang="fr",
                device=device,
                enable_mkldnn=False,
            )
        else:
            self._api_version = 2
            logger.info("PaddleOCR 2.x API detected (use_gpu=%s).", self._use_gpu)
            logger.info("Loading PaddleOCR Arabic model …")
            self._reader_ar = PaddleOCR(
                use_angle_cls=False,
                lang="ar",
                use_gpu=self._use_gpu,
                show_log=False,
            )
            logger.info("Loading PaddleOCR French model …")
            self._reader_fr = PaddleOCR(
                use_angle_cls=False,
                lang="fr",
                use_gpu=self._use_gpu,
                show_log=False,
            )

        self._initialised = True
        logger.info("PaddleOCR models loaded successfully (api_version=%d).", self._api_version)

    # ── public API ──────────────────────────────────────────────────────
    def run(self, image: np.ndarray, lang: str = "ar") -> List[OCRBox]:
        """
        Run OCR on a BGR numpy image.

        Parameters
        ----------
        image : np.ndarray
            BGR image (OpenCV format).
        lang : str
            ``"ar"`` for Arabic reader, ``"fr"`` for French reader.

        Returns
        -------
        List[OCRBox]
            Detected text boxes with text, confidence and bounding-box
            coordinates.
        """
        self._ensure_init()

        reader = self._reader_ar if lang == "ar" else self._reader_fr
        if reader is None:
            logger.error("OCR reader for lang=%s is None after init.", lang)
            return []

        if image is None or image.size == 0:
            logger.warning("run() called with empty image.")
            return []

        boxes: List[OCRBox] = []
        try:
            if self._api_version >= 3:
                results = self._run_v3(reader, image)
            else:
                results = self._run_v2(reader, image)
            boxes = results
            logger.debug("OCR lang=%s  →  %d boxes detected.", lang, len(boxes))

        except Exception:
            logger.exception("OCR inference failed (lang=%s).", lang)

        return boxes

    # ── v2 API (.ocr) ──────────────────────────────────────────────────
    @staticmethod
    def _run_v2(reader: object, image: np.ndarray) -> List[OCRBox]:
        """PaddleOCR 2.x: uses .ocr(image, cls=False)."""
        results = reader.ocr(image, cls=False)  # type: ignore[attr-defined]
        boxes: List[OCRBox] = []
        if not results:
            return boxes

        for page in results:
            if page is None:
                continue
            for line in page:
                # line = [bbox, (text, confidence)]
                if line is None or len(line) < 2:
                    continue
                bbox = line[0]              # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                text_info = line[1]         # (text, confidence)
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                    text = str(text_info[0])
                    conf = float(text_info[1])
                else:
                    continue
                if text.strip():
                    boxes.append(OCRBox(
                        text=text,
                        confidence=conf,
                        bbox=bbox if isinstance(bbox, list) else [[0, 0]] * 4,
                    ))
        return boxes

    # ── v3 API (.predict) ──────────────────────────────────────────────
    @staticmethod
    def _run_v3(reader: object, image: np.ndarray) -> List[OCRBox]:
        """PaddleOCR 3.x: uses .predict(image_path) with temp file."""
        import tempfile

        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            cv2.imwrite(tmp_path, image)
            results = reader.predict(tmp_path)  # type: ignore[attr-defined]
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        boxes: List[OCRBox] = []
        if not results:
            return boxes

        for page in results:
            if isinstance(page, dict):
                rec_texts = page.get("rec_texts", [])
                rec_scores = page.get("rec_scores", [])
                dt_polys = page.get("dt_polys", [])
                for i, text in enumerate(rec_texts):
                    if not text:
                        continue
                    conf = rec_scores[i] if i < len(rec_scores) else 0.0
                    bbox = dt_polys[i].tolist() if i < len(dt_polys) else [[0, 0]] * 4
                    boxes.append(OCRBox(text=text, confidence=float(conf), bbox=bbox))
            elif isinstance(page, list):
                for item in page:
                    if isinstance(item, dict):
                        for i, text in enumerate(item.get("rec_texts", [])):
                            conf = item.get("rec_scores", [0.0])[i] if i < len(item.get("rec_scores", [])) else 0.0
                            bbox_list = item.get("dt_polys", [])
                            bbox = bbox_list[i].tolist() if i < len(bbox_list) else [[0, 0]] * 4
                            boxes.append(OCRBox(text=text, confidence=float(conf), bbox=bbox))
        return boxes

    # ── properties ──────────────────────────────────────────────────────
    @property
    def is_gpu(self) -> bool:
        return self._use_gpu

    @property
    def is_ready(self) -> bool:
        return self._initialised
