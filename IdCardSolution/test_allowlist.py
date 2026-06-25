"""
Test script for allowlist refinement.
Shows AVANT/APRES comparison for each field.

Usage:
  python test_allowlist.py carte.jpg
"""

import sys
import cv2
from cv2 import cvtColor, COLOR_BGR2RGB

from ocr_extractor import extract_fields, _default_reader


def test_allowlist(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot load: {image_path}")
        return

    # --- AVANT: raw predict without allowlist ---
    rgb = cvtColor(img, COLOR_BGR2RGB)
    reader = _default_reader.ocr_fr
    print("\n=== AVANT allowlist (raw predict) ===")
    raw_results = _default_reader._run_ocr(reader, rgb)
    for (bbox, text, conf) in raw_results:
        print(f"  [{conf:.2f}] {text}")

   
    fields = extract_fields(img)
    print("\n=== APRES allowlist (refined fields) ===")
    print(f"  NIN              : {fields.nin}")
    print(f"  Nom              : {fields.nom}")
    print(f"  Prenom           : {fields.prenom}")
    print(f"  Date naissance   : {fields.date_naissance}")
    print(f"  Lieu naissance   : {fields.lieu_naissance}")
    print(f"  Confiance        : {fields.confidence_moyenne:.1%}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_allowlist(sys.argv[1])
    else:
        print("Usage: python test_allowlist.py <image.jpg>")
