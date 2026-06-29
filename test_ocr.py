"""Quick smoke test: load a saved card image and run OCR on it."""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paddle_ocr import OCRManager
import cv2

manager = OCRManager(use_gpu=True)

# Test with a saved driver license image
test_images = [
    ("DRIVERS LICENSE", "saved_id_cards/drivers_license_1782397123.png"),
    ("ID CARD RECTO",   "saved_id_cards/id_card_recto_1782170144.png"),
]

for doc_type, path in test_images:
    full = os.path.join(os.path.dirname(__file__), path)
    if not os.path.exists(full):
        print(f"[SKIP] {path} not found")
        continue
    img = cv2.imread(full)
    if img is None:
        print(f"[SKIP] Could not read {path}")
        continue
    print(f"\n{'='*60}")
    print(f"Testing: {doc_type} → {path}")
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")
    print(f"{'='*60}")
    result = manager.process(doc_type, img)
    for k, v in result.items():
        print(f"  {k}: {v}")
    # Reset for next test
    manager.reset()
