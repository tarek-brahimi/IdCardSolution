"""
Smart Card Scanner — detection temps reel + capture finale.

Detection identique a oh.py :
  Canny(40,120) → approxPolyDP(0.03) → perspective transform
  → classify_document avec ROI regions + templates id_card_emblem.png / driver_license_dz_badge.png
  → stabilite Counter(6/10) + auto-save + OCR

Usage :
  python smart_scanner.py                     # DroidCam
  python smart_scanner.py --camera 0          # Webcam locale
  python smart_scanner.py --gpu               # GPU pour PaddleOCR
"""

import sys
import os
import time
import argparse
from datetime import datetime
from typing import Optional, Tuple
from collections import Counter

# Auto-switch to .venv Python if available and not already using it
_VENV_PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "python.exe")
if sys.platform == "win32" and os.path.exists(_VENV_PYTHON) and sys.executable.lower() != _VENV_PYTHON.lower():
    import subprocess
    ret = subprocess.call([_VENV_PYTHON] + sys.argv)
    sys.exit(ret)

os.system("chcp 65001 >nul 2>&1")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import cv2
import numpy as np

from ocr_extractor import extract_fields

CAMERA_DEFAULT = "http://192.168.100.9:4747/video"

CAPTURE_DIR = "captures"
DIR_NATIONAL_ID = os.path.join(CAPTURE_DIR, "national_id")
DIR_DRIVERS_LICENSE = os.path.join(CAPTURE_DIR, "drivers_license")

CLASSIFY_EVERY_N = 5 
HISTORY_SIZE = 10
STABILITY_THRESHOLD = 6

AREA_RATIO_MIN = 0.15
BLUR_THRESHOLD = 60.0
SAVE_COOLDOWN = 3.0
DETECT_WIDTH = 640


# Fonctions detection

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_perspective_transform(frame, pts, width=856, height=540):
    ordered_pts = order_points(pts)
    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
    warped = cv2.warpPerspective(frame, M, (width, height))
    return warped

def is_blurry(image, threshold=BLUR_THRESHOLD):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold, variance

def load_templates(template_dir):
    templates = {}
    id_path = os.path.join(template_dir, "id_card_emblem.png")
    id_img = cv2.imread(id_path)
    if id_img is not None:
        templates["id_card"] = {
            "image": id_img,
            "gray": cv2.cvtColor(id_img, cv2.COLOR_BGR2GRAY),
            "label": "ALGERIAN ID CARD",
            "color": (0, 255, 0),
        }
        print(f"  Loaded ID card template {id_img.shape[1]}x{id_img.shape[0]}")
    else:
        print(f"  Could not load {id_path}")
    dl_path = os.path.join(template_dir, "driver_license_dz_badge.png")
    dl_img = cv2.imread(dl_path)
    if dl_img is not None:
        templates["driver_license"] = {
            "image": dl_img,
            "gray": cv2.cvtColor(dl_img, cv2.COLOR_BGR2GRAY),
            "label": "DRIVERS LICENSE",
            "color": (255, 165, 0),
        }
        print(f"  Loaded driver license template {dl_img.shape[1]}x{dl_img.shape[0]}")
    else:
        print(f"  Could not load {dl_path}")
    return templates

def classify_document(warped_card, templates, match_threshold=0.45):
    if not templates:
        return "NO TEMPLATES", (128, 128, 128), 0.0, {}
    card_h, card_w = warped_card.shape[:2]
    card_gray = cv2.cvtColor(warped_card, cv2.COLOR_BGR2GRAY)
    roi_regions = {
        "id_card": {"y1": 0.00, "y2": 0.40, "x1": 0.00, "x2": 0.25},
        "driver_license": {"y1": 0.00, "y2": 0.25, "x1": 0.00, "x2": 0.15},
    }
    all_scores = {}
    best_type = None
    best_score = -1.0
    for doc_type, tpl_data in templates.items():
        if doc_type not in roi_regions:
            continue
        roi_cfg = roi_regions[doc_type]
        template_gray = tpl_data["gray"]
        tpl_h, tpl_w = template_gray.shape[:2]
        y1 = int(card_h * roi_cfg["y1"])
        y2 = int(card_h * roi_cfg["y2"])
        x1 = int(card_w * roi_cfg["x1"])
        x2 = int(card_w * roi_cfg["x2"])
        roi_gray = card_gray[y1:y2, x1:x2]
        roi_h, roi_w = roi_gray.shape[:2]
        target_w = int(roi_w * 0.90)
        target_h = int(roi_h * 0.90)
        scale = min(target_w / tpl_w, target_h / tpl_h)
        new_w = max(int(tpl_w * scale), 1)
        new_h = max(int(tpl_h * scale), 1)
        if new_w >= roi_w or new_h >= roi_h:
            new_w = roi_w - 2
            new_h = roi_h - 2
        if new_w < 5 or new_h < 5:
            all_scores[doc_type] = 0.0
            continue
        resized_template = cv2.resize(template_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(roi_gray, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        all_scores[doc_type] = round(max_val, 3)
        if max_val > best_score:
            best_score = max_val
            best_type = doc_type
    if best_type is not None and best_score >= match_threshold:
        label = templates[best_type]["label"]
        color = templates[best_type]["color"]
        return label, color, best_score, all_scores
    else:
        return "OTHER DOCUMENT", (0, 0, 255), best_score, all_scores




def ensure_dirs():
    os.makedirs(DIR_NATIONAL_ID, exist_ok=True)
    os.makedirs(DIR_DRIVERS_LICENSE, exist_ok=True)

def get_save_folder(card_type):
    if card_type in ("CNI", "ALGERIAN ID CARD"):
        return DIR_NATIONAL_ID
    elif card_type in ("PERMIS", "DRIVERS LICENSE"):
        return DIR_DRIVERS_LICENSE
    return DIR_NATIONAL_ID

def get_folder_name(card_type):
    if card_type in ("CNI", "ALGERIAN ID CARD"):
        return "national_id"
    elif card_type in ("PERMIS", "DRIVERS LICENSE"):
        return "drivers_license"
    return "unknown"

def map_label(stable_label):
    if stable_label == "ALGERIAN ID CARD":
        return "CNI"
    elif stable_label == "DRIVERS LICENSE":
        return "PERMIS"
    return "INCONNU"

def get_display_label(stable_label):
    labels = {
        "ALGERIAN ID CARD": "CNI Algerienne",
        "DRIVERS LICENSE": "Permis de Conduire",
        "OTHER DOCUMENT": "Document Inconnu",
        "DETECTING": "Detection...",
        "NO TEMPLATES": "Pas de templates",
    }
    return labels.get(stable_label, stable_label)

def get_stable_color(stable_label):
    if stable_label == "ALGERIAN ID CARD":
        return (0, 255, 0)
    elif stable_label == "DRIVERS LICENSE":
        return (255, 165, 0)
    elif stable_label in ("OTHER DOCUMENT", "NO TEMPLATES"):
        return (0, 0, 255)
    return (200, 200, 200)

def save_capture(image, card_type):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%Hh%Mm%S")
    filename = f"card_{timestamp}.jpg"
    folder = get_save_folder(card_type)
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, image)
    return filepath

def draw_label(frame, text, position, bg_color=(0, 180, 0), text_color=(255, 255, 255), scale=0.7, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    cv2.rectangle(frame, (x - 5, y - th - 10), (x + tw + 10, y + baseline + 5), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, scale, text_color, thickness)

def print_capture_result(label, score, all_scores, fields, filepath, warped_card):
    card_type = map_label(label)
    print()
    print("=" * 50)
    print(f"  TYPE      : {card_type} — ", end="")
    if card_type == "CNI":
        print("Carte Nationale Algerienne")
    elif card_type == "PERMIS":
        print("Permis de Conduire Algerien")
    else:
        print("Document Non Reconnu")
    print(f"  Label     : {label}")
    print(f"  Confiance : {score:.0%}")
    print(f"  Scores    : {all_scores}")
    if fields.nin:
        print(f"  NIN       : {fields.nin}")
    if fields.nom:
        print(f"  Nom       : {fields.nom}")
    if fields.prenom:
        print(f"  Prenom    : {fields.prenom}")
    print(f"  Sauvegarde: {filepath}")
    print("=" * 50)

def print_inconnu():
    print()
    print("=" * 50)
    print("  AUTRE CARTE DETECTEE")
    print("  Document non reconnu.")
    print("  Presentez une CNI ou un Permis algerien.")
    print("  Aucune sauvegarde effectuee.")
    print("=" * 50)



def run_live_scan(camera_url=CAMERA_DEFAULT, gpu=False, debug=False):
    print()
    print("=" * 55)
    print("  SMART CARD SCANNER — LIVE MODE")
    print("=" * 55)
    print()
    print("[INFO] Demarrage de la webcam...")
    print("[INFO] SPACE = Capturer (OCR) | Q = Quitter")
    print("[INFO] Auto-save si carte stable et bien visible")
    print()

    ensure_dirs()

    print("Loading templates")
    templates = load_templates(os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"))

    if isinstance(camera_url, int):
        cap = cv2.VideoCapture(camera_url)
    else:
        cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        print("Error could not open camera")
        return False

    classification_history = []
    stable_label = "DETECTING"
    stable_color = (200, 200, 200)
    frame_count = 0
    last_save_time = 0.0
    current_card_image = None
    confirmation_msg = None
    confirmation_expiry = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame_count += 1
            frame_height, frame_width = frame.shape[:2]
            frame_area = frame_width * frame_height

            # --- Detection sur frame reduite pour fluidite ---
            detect_h = int(frame_height * DETECT_WIDTH / frame_width)
            small = cv2.resize(frame, (DETECT_WIDTH, detect_h), interpolation=cv2.INTER_AREA)
            small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            small_blurred = cv2.GaussianBlur(small_gray, (5, 5), 0)
            small_edges = cv2.Canny(small_blurred, 40, 120)
            contours, _ = cv2.findContours(small_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

            card_contour = None
            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.03 * peri, True)
                if len(approx) == 4:
                    card_contour = approx
                    break

            display = frame.copy()

            if card_contour is not None:
                # Remettre les points a l'echelle originale
                scale_x = frame_width / DETECT_WIDTH
                scale_y = frame_height / detect_h
                pts = card_contour.reshape(4, 2).astype(np.float32)
                pts[:, 0] *= scale_x
                pts[:, 1] *= scale_y
                pts = pts.astype(np.int32)

                cv2.drawContours(display, [pts.reshape(-1, 1, 2)], -1, (0, 255, 0), 3)
                for pt in pts:
                    cv2.circle(display, tuple(pt), 7, (255, 0, 0), -1)

                rectified_card = get_perspective_transform(frame, pts, width=856, height=540)

                # Classification toutes les N frames
                if templates and (frame_count % CLASSIFY_EVERY_N == 0):
                    label, color, score, all_scores = classify_document(
                        rectified_card, templates, match_threshold=0.45
                    )
                    classification_history.append(label)
                    if len(classification_history) > HISTORY_SIZE:
                        classification_history.pop(0)
                    if len(classification_history) >= STABILITY_THRESHOLD:
                        counts = Counter(classification_history)
                        most_common_label, most_common_count = counts.most_common(1)[0]
                        if most_common_count >= STABILITY_THRESHOLD:
                            stable_label = most_common_label
                            stable_color = get_stable_color(stable_label)
                        else:
                            stable_label = "DETECTING"
                            stable_color = (200, 200, 200)
                    scores_str = " | ".join([f"{k}: {v:.3f}" for k, v in all_scores.items()])
                    if debug:
                        print(f"[CLASSIFY] {label} (score: {score:.3f}) | {scores_str}")

                # Label sur le frame
                display_label = get_display_label(stable_label)
                x, y, w, h = cv2.boundingRect(pts.reshape(-1, 1, 2))
                label_y = max(y - 15, 25)
                draw_label(display, display_label, (x, label_y), bg_color=stable_color)

                # Stocker pour capture
                current_card_image = rectified_card

                # Auto-save: trigger when card is detected, classified or not
                card_area = cv2.contourArea(pts.reshape(-1, 1, 2))
                area_ratio = card_area / frame_area
                blurry, var_val = is_blurry(rectified_card)
                now_seconds = cv2.getTickCount() / cv2.getTickFrequency()

                if (stable_label in ("ALGERIAN ID CARD", "DRIVERS LICENSE", "DETECTING")
                    and area_ratio > AREA_RATIO_MIN
                    and not blurry
                    and (now_seconds - last_save_time) > SAVE_COOLDOWN):

                    card_type = map_label(stable_label)
                    if card_type == "INCONNU":
                        card_type = "CNI"
                    filepath = save_capture(rectified_card, card_type)
                    folder_name = get_folder_name(stable_label)
                    print(f"[AUTO-SAVE] {stable_label} -> {filepath} (area={area_ratio:.2f}, sharp={var_val:.1f})")
                    last_save_time = now_seconds
                    confirmation_msg = f"Auto-save: {folder_name}/{os.path.basename(filepath)}"
                    confirmation_expiry = now_seconds + 3.0

            else:
                current_card_image = None
                classification_history.clear()
                stable_label = "DETECTING"
                stable_color = (200, 200, 200)
                draw_label(display, "Aucune carte detectee", (10, 30), bg_color=(0, 0, 200))

            # Barre de controls
            h_frame = display.shape[0]
            cv2.rectangle(display, (0, h_frame - 40), (display.shape[1], h_frame), (50, 50, 50), -1)
            cv2.putText(display, "SPACE = Capturer (OCR)  |  Q = Quitter",
                        (10, h_frame - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Message de confirmation
            now_seconds = cv2.getTickCount() / cv2.getTickFrequency()
            if confirmation_msg and now_seconds < confirmation_expiry:
                draw_label(display, confirmation_msg, (50, 60), bg_color=(200, 150, 0))
            elif confirmation_msg:
                confirmation_msg = None

            cv2.imshow("Smart Card Scanner", display)

            key = cv2.waitKey(1) & 0xFF

            if key == 32:  # ESPACE
                if current_card_image is not None:
                    draw_label(display, "Traitement OCR en cours...", (50, 60), bg_color=(0, 120, 220))
                    cv2.imshow("Smart Card Scanner", display)
                    cv2.waitKey(1)

                    fields = extract_fields(current_card_image, gpu=gpu)
                    card_type = map_label(stable_label)

                    # Always save if card image exists, even if not classified
                    if card_type == "INCONNU":
                        card_type = "CNI"

                    filepath = save_capture(current_card_image, card_type)
                    print_capture_result(stable_label, 0.0, {}, fields, filepath, current_card_image)
                    folder_name = get_folder_name(stable_label)
                    nom_str = fields.nom or ""
                    prenom_str = fields.prenom or ""
                    name_part = f" — {nom_str} {prenom_str}" if nom_str else ""
                    confirmation_msg = f"SAUVE: {folder_name}/{os.path.basename(filepath)}{name_part}"
                    confirmation_expiry = now_seconds + 4.0
                else:
                    print("[WARN] Aucune carte detectee, impossible de capturer.")

            elif key == ord('q') or key == ord('Q'):
                print("[INFO] Arret par l'utilisateur.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Smart Card Scanner — detection et capture en temps reel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python smart_scanner.py                     # Camera WiFi DroidCam
  python smart_scanner.py --camera 0          # Webcam locale
  python smart_scanner.py --gpu               # GPU pour PaddleOCR
  python smart_scanner.py --debug             # Afficher le debug
        """
    )
    parser.add_argument("--camera", "-c", type=str, default=CAMERA_DEFAULT,
                        help="Index camera (0, 1) ou URL DroidCam")
    parser.add_argument("--gpu", action="store_true",
                        help="Utiliser le GPU pour PaddleOCR")
    parser.add_argument("--debug", action="store_true",
                        help="Afficher les infos de debug")
    args = parser.parse_args()

    try:
        camera = args.camera
        if camera.isdigit():
            camera = int(camera)
        success = run_live_scan(camera_url=camera, gpu=args.gpu, debug=args.debug)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[INFO] Interrompu par l'utilisateur.")
        sys.exit(130)


if __name__ == "__main__":
    main()
