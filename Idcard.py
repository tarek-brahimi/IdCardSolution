import cv2
import numpy as np
import os
import time
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# ── OCR Module ──────────────────────────────────────────────────────
# The paddle_ocr package handles all PaddleOCR internals.
# Set OCR_AVAILABLE = False to disable OCR entirely.
try:
    from paddle_ocr import OCRManager
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("idcard")

# Order corner points
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# Warp perspective to rectify card
def get_perspective_transform(frame, pts, width=856, height=540, margin=15):
    ordered_pts = order_points(pts)
    center = ordered_pts.mean(axis=0)
    expanded_pts = np.zeros_like(ordered_pts)
    for i in range(4):
        direction = ordered_pts[i] - center
        norm = np.linalg.norm(direction)
        if norm > 0:
            expanded_pts[i] = ordered_pts[i] + (direction / norm) * margin
        else:
            expanded_pts[i] = ordered_pts[i]
    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(expanded_pts, dst_pts)
    warped = cv2.warpPerspective(frame, M, (width, height))
    return warped

# Check if image is blurry
def is_blurry(image, threshold=60.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold, variance

# Load templates for classification
def load_templates(template_dir):
    templates = {}
    id_path = os.path.join(template_dir, "id_card_emblem.png")
    id_img = cv2.imread(id_path)
    if id_img is not None:
        templates["id_card"] = {
            "image": id_img,
            "gray": cv2.cvtColor(id_img, cv2.COLOR_BGR2GRAY),
            "label": "ID CARD RECTO",
            "color": (0, 255, 0),
        }
        print(f"  Loaded ID recto template {id_img.shape[1]}x{id_img.shape[0]}")
    else:
        print(f"  Could not load {id_path}")
    verso_seal_path = os.path.join(template_dir, "id_verso_seal.png")
    verso_orig_path = os.path.join(template_dir, "id_verso.png")
    if os.path.exists(verso_seal_path):
        verso_img = cv2.imread(verso_seal_path)
    else:
        raw = cv2.imread(verso_orig_path)
        if raw is not None:
            h, w = raw.shape[:2]
            verso_img = raw[0:int(h * 0.65), :]
        else:
            verso_img = None
    if verso_img is not None:
        templates["id_card_verso"] = {
            "image": verso_img,
            "gray": cv2.cvtColor(verso_img, cv2.COLOR_BGR2GRAY),
            "label": "ID CARD VERSO",
            "color": (255, 255, 0),
        }
        print(f"  Loaded ID verso template {verso_img.shape[1]}x{verso_img.shape[0]}")
    else:
        print(f"  Could not load verso template")
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

# ROI regions for each template
ROI_REGIONS = {
    "id_card":        {"y1": 0.00, "y2": 0.45, "x1": 0.00, "x2": 0.30},
    "id_card_verso":  {"y1": 0.00, "y2": 0.50, "x1": 0.50, "x2": 1.00},
    "driver_license": {"y1": 0.00, "y2": 0.35, "x1": 0.00, "x2": 0.25},
}

# Match one template against a region using multi-scale
def match_template_in_roi(card_gray, template_gray, roi_cfg):
    card_h, card_w = card_gray.shape[:2]
    tpl_h, tpl_w = template_gray.shape[:2]
    y1 = int(card_h * roi_cfg["y1"])
    y2 = int(card_h * roi_cfg["y2"])
    x1 = int(card_w * roi_cfg["x1"])
    x2 = int(card_w * roi_cfg["x2"])
    roi_gray = card_gray[y1:y2, x1:x2]
    roi_h, roi_w = roi_gray.shape[:2]
    best_score = 0.0
    for scale_pct in [0.5, 0.7, 0.85]:
        tw = int(roi_w * scale_pct)
        th = int(roi_h * scale_pct)
        aspect = tpl_w / tpl_h
        if tw / th > aspect:
            tw = max(int(th * aspect), 5)
        else:
            th = max(int(tw / aspect), 5)
        if tw >= roi_w or th >= roi_h or tw < 5 or th < 5:
            continue
        resized = cv2.resize(template_gray, (tw, th), interpolation=cv2.INTER_LINEAR)
        result = cv2.matchTemplate(roi_gray, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > best_score:
            best_score = max_val
    return best_score

# Fix orientation using all templates
def fix_orientation(warped_card, templates):
    if not templates:
        return warped_card, False
    card_gray = cv2.cvtColor(warped_card, cv2.COLOR_BGR2GRAY)
    rotated_card = cv2.rotate(warped_card, cv2.ROTATE_180)
    rotated_gray = cv2.cvtColor(rotated_card, cv2.COLOR_BGR2GRAY)
    normal_best = 0.0
    rotated_best = 0.0
    for doc_type, tpl_data in templates.items():
        if doc_type not in ROI_REGIONS:
            continue
        roi_cfg = ROI_REGIONS[doc_type]
        template_gray = tpl_data["gray"]
        score_n = match_template_in_roi(card_gray, template_gray, roi_cfg)
        score_r = match_template_in_roi(rotated_gray, template_gray, roi_cfg)
        normal_best = max(normal_best, score_n)
        rotated_best = max(rotated_best, score_r)
    if rotated_best > normal_best and rotated_best > 0.20:
        return rotated_card, True
    return warped_card, False

# Classify document using template matching
def classify_document(warped_card, templates, match_threshold=0.50):
    if not templates:
        return "NO TEMPLATES", (128, 128, 128), 0.0, {}
    card_gray = cv2.cvtColor(warped_card, cv2.COLOR_BGR2GRAY)
    all_scores = {}
    best_type = None
    best_score = -1.0
    for doc_type, tpl_data in templates.items():
        if doc_type not in ROI_REGIONS:
            continue
        roi_cfg = ROI_REGIONS[doc_type]
        template_gray = tpl_data["gray"]
        max_val = match_template_in_roi(card_gray, template_gray, roi_cfg)
        all_scores[doc_type] = round(max_val, 3)
        if max_val > best_score:
            best_score = max_val
            best_type = doc_type
    if best_type is not None and best_score >= match_threshold:
        if best_type == "driver_license":
            id_recto = all_scores.get("id_card", 0)
            id_verso = all_scores.get("id_card_verso", 0)
            if id_recto >= match_threshold and id_recto >= best_score * 0.8:
                best_type = "id_card"
                best_score = id_recto
            elif id_verso >= match_threshold and id_verso >= best_score * 0.8:
                best_type = "id_card_verso"
                best_score = id_verso
        label = templates[best_type]["label"]
        color = templates[best_type]["color"]
        return label, color, best_score, all_scores
    else:
        return "OTHER DOCUMENT", (0, 0, 255), best_score, all_scores

# Select camera source
def select_camera():
    print("")
    print("SELECT CAMERA SOURCE")
    print("Press L -> Laptop webcam")
    print("Press P -> Phone IP webcam")
    while True:
        choice = input("Your choice (L/P): ").strip().lower()
        if choice == "l":
            print("Using laptop webcam")
            return cv2.VideoCapture(0)
        elif choice == "p":
            url = "http://192.168.100.9:4747/video"
            print(f"Connecting to {url}")
            return cv2.VideoCapture(url)
        else:
            print("Invalid choice try again")

# ── Helper: draw OCR results on the display card ────────────────────
def put_text_arabic(img, text, org, color=(255, 255, 255), font_size=24):
    """Draw Arabic (and Latin) text using PIL, Arabic Reshaper, and python-bidi."""
    from PIL import Image, ImageDraw, ImageFont
    import arabic_reshaper
    from bidi.algorithm import get_display
    
    reshaped = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped)
    
    # Convert BGR to RGB for PIL, then back to BGR
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        # Use a standard Windows font that supports Arabic
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
        
    # Pillow takes RGB colors, so convert the given BGR to RGB
    rgb_color = (color[2], color[1], color[0])
    draw.text(org, bidi_text, font=font, fill=rgb_color)
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_ocr_overlay(display_card, ocr_result, ocr_in_progress, is_waiting_verso):
    """Render OCR status / results onto the rectified-card window."""
    y_offset = 80

    if is_waiting_verso:
        cv2.putText(display_card, "Please flip your ID card",
                    (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        y_offset += 35
        # Also show what we already have from recto
        if ocr_result:
            for key in ("nin", "arabic_name"):
                if key in ocr_result and ocr_result[key]:
                    val = ocr_result[key]
                    if "arabic" in key:
                        # Re-assign display_card with PIL rendering
                        display_card[:] = put_text_arabic(display_card, f"{key}: {val}", (15, y_offset), (0, 255, 255), 22)
                    else:
                        cv2.putText(display_card, f"{key}: {val}",
                                    (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    y_offset += 28
        return

    if ocr_in_progress:
        cv2.putText(display_card, "Extracting text...",
                    (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        return

    if not ocr_result:
        return

    status = ocr_result.get("status", "")
    if status == "unsupported_document":
        cv2.putText(display_card, "Unsupported card",
                    (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return

    # Draw extracted fields
    for key in ("nin", "arabic_name", "french_name"):
        val = ocr_result.get(key)
        if val:
            conf_key = f"{key}_confidence"
            conf = ocr_result.get(conf_key, 0)
            color = (0, 255, 0) if conf >= 0.80 else (0, 255, 255) if conf >= 0.60 else (0, 165, 255)
            
            if "arabic" in key:
                display_card[:] = put_text_arabic(display_card, f"{key}: {val} ({conf:.0%})", (15, y_offset), color, 22)
            else:
                cv2.putText(display_card, f"{key}: {val} ({conf:.0%})",
                            (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 28

    if ocr_result.get("completed"):
        cv2.putText(display_card, "SCAN COMPLETE",
                    (15, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


# Main application loop
def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(BASE_DIR, "saved_id_cards")
    template_dir = os.path.join(BASE_DIR, "templates")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Loading templates")
    templates = load_templates(template_dir)

    # ── OCR initialisation ──────────────────────────────────────────
    ocr_manager = None
    if OCR_AVAILABLE:
        print("Initialising OCR module (PaddleOCR) …")
        ocr_manager = OCRManager(use_gpu=True)
        print("OCR module ready.")
    else:
        print("PaddleOCR not installed. OCR disabled.")

    cap = select_camera()
    if not cap.isOpened():
        print("Error could not open camera")
        return
    print("")
    print("ID Card Detector Started")
    print("E -> Extract Text")
    print("S -> Save card")
    print("R -> Reset OCR state")
    print("Q -> Quit")
    WIN_W, WIN_H = 640, 480
    for name in ["1 Original", "2 Edges", "3 Contours", "4 Rectified Card"]:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, WIN_W, WIN_H)
    classification_history = []
    HISTORY_SIZE = 10
    STABILITY_THRESHOLD = 6
    stable_label = "DETECTING"
    stable_color = (200, 200, 200)
    frame_count = 0
    CLASSIFY_EVERY_N = 5

    # ── OCR async state ─────────────────────────────────────────────
    last_ocr_time = 0
    ocr_result = {}           # latest result dict from OCRManager
    ocr_in_progress = False
    ocr_executor = ThreadPoolExecutor(max_workers=1) if ocr_manager else None

    def ocr_callback(future):
        nonlocal ocr_result, ocr_in_progress
        try:
            result = future.result()
            if result:
                ocr_result = result
                print("\n" + "="*50)
                print(f"   EXTRACTED DATA: {result.get('document_type', 'UNKNOWN')}")
                print("="*50)
                for key in ['nin', 'arabic_name', 'french_name']:
                    val = result.get(key)
                    if val:
                        print(f"  {key.upper():<15}: {val}")
                print("="*50 + "\n")
        except Exception as e:
            logger.error("OCR error: %s", e)
        finally:
            ocr_in_progress = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        frame_count += 1
        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_width * frame_height
        contour_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        card_contour = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            if len(approx) == 4:
                card_contour = approx
                break
        rectified_card = None
        if card_contour is not None:
            pts = card_contour.reshape(4, 2)
            cv2.drawContours(contour_frame, [card_contour], -1, (0, 255, 0), 3)
            for pt in pts:
                cv2.circle(contour_frame, tuple(pt), 7, (255, 0, 0), -1)
            warped = get_perspective_transform(frame, pts, width=856, height=540)
            rectified_card, was_flipped = fix_orientation(warped, templates)
            if was_flipped and (frame_count % CLASSIFY_EVERY_N == 0):
                print("[ORIENT] Card was upside down -> rotated 180")
            if templates and (frame_count % CLASSIFY_EVERY_N == 0):
                label, color, score, all_scores = classify_document(
                    rectified_card, templates, match_threshold=0.50
                )
                classification_history.append(label)
                if len(classification_history) > HISTORY_SIZE:
                    classification_history.pop(0)
                if len(classification_history) >= STABILITY_THRESHOLD:
                    counts = Counter(classification_history)
                    most_common_label, most_common_count = counts.most_common(1)[0]
                    if most_common_count >= STABILITY_THRESHOLD:
                        stable_label = most_common_label
                        if stable_label == "ID CARD RECTO":
                            stable_color = (0, 255, 0)
                        elif stable_label == "ID CARD VERSO":
                            stable_color = (255, 255, 0)
                        elif stable_label == "DRIVERS LICENSE":
                            stable_color = (255, 165, 0)
                        else:
                            stable_color = (0, 0, 255)
                    else:
                        stable_label = "DETECTING"
                        stable_color = (200, 200, 200)
                scores_str = " | ".join([f"{k}: {v:.3f}" for k, v in all_scores.items()])
                # print(f"[CLASSIFY] {label} (score: {score:.3f}) | {scores_str}")
            display_card = rectified_card.copy()
            cv2.rectangle(display_card, (0, 0), (856, 50), (0, 0, 0), -1)
            cv2.putText(display_card, stable_label, (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, stable_color, 2)

            # ── Draw OCR overlay ────────────────────────────────────
            is_waiting = ocr_manager.is_waiting_for_verso if ocr_manager else False
            draw_ocr_overlay(display_card, ocr_result, ocr_in_progress, is_waiting)

            cv2.imshow("4 Rectified Card", display_card)
            card_area = cv2.contourArea(card_contour)
            area_ratio = card_area / frame_area
            blurry, var_val = is_blurry(rectified_card, threshold=60.0)
            status_text = f"Area: {area_ratio:.2f} Sharp: {var_val:.1f} | {stable_label}"
            color = (0, 255, 0) if not blurry else (0, 165, 255)
            cv2.putText(contour_frame, status_text, (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            cv2.putText(contour_frame, stable_label, (40, frame_height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, stable_color, 3)
        else:
            classification_history.clear()
            stable_label = "DETECTING"
            stable_color = (200, 200, 200)
            ocr_result = {}
            # Reset state machine when card is lost
            if ocr_manager:
                ocr_manager.reset()

            placeholder = np.zeros((540, 856, 3), dtype="uint8")
            cv2.putText(placeholder, "Searching for card", (280, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow("4 Rectified Card", placeholder)
        cv2.imshow("1 Original", frame)
        cv2.imshow("2 Edges", edges)
        cv2.imshow("3 Contours", contour_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            # Manual trigger for OCR
            if ocr_manager and stable_label != "DETECTING" and stable_label != "OTHER DOCUMENT":
                if not ocr_in_progress:
                    if rectified_card is not None:
                        print(f"\n[OCR] Extracting text from {stable_label}...")
                        ocr_in_progress = True
                        card_copy = rectified_card.copy()
                        future = ocr_executor.submit(
                            ocr_manager.process, stable_label, card_copy
                        )
                        future.add_done_callback(ocr_callback)
                else:
                    print("[OCR] Extraction already in progress, please wait...")
            else:
                print("[OCR] No valid document detected yet. Wait for a stable classification.")
        elif key == ord('r'):
            # Manual reset of OCR state
            if ocr_manager:
                ocr_manager.reset()
                ocr_result = {}
                print("[OCR] State reset manually.")
        elif key == ord('s'):
            if card_contour is not None and rectified_card is not None:
                safe_label = stable_label.lower().replace("'", "").replace(" ", "_")
                filename = os.path.join(output_dir, f"{safe_label}_{int(time.time())}.png")
                success = cv2.imwrite(filename, rectified_card)
                if success:
                    print(f"Saved -> {filename}")
                else:
                    print("Save failed check permissions")
            else:
                print("No card detected")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()