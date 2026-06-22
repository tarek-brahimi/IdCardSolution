import cv2
import numpy as np
import os
import time
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

try:
    from paddleocr import PaddleOCR
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

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

# --- OCR Integration ---
OCR_ROIS = {
    "ID CARD RECTO": {
        "Name_AR": {"y1": 0.10, "y2": 0.40, "x1": 0.50, "x2": 1.00},
        "DOB":     {"y1": 0.35, "y2": 0.60, "x1": 0.50, "x2": 1.00},
        "NIN":     {"y1": 0.65, "y2": 0.90, "x1": 0.30, "x2": 1.00},
    },
    "ID CARD VERSO": {
        "Name_FR": {"y1": 0.05, "y2": 0.45, "x1": 0.00, "x2": 0.60},
    },
    "DRIVERS LICENSE": {
        "Name": {"y1": 0.10, "y2": 0.40, "x1": 0.30, "x2": 1.00},
        "DOB":  {"y1": 0.40, "y2": 0.60, "x1": 0.30, "x2": 1.00},
        "License_Num": {"y1": 0.60, "y2": 0.90, "x1": 0.20, "x2": 1.00},
    }
}

def extraire_texte_roi(ocr_reader, roi_image, is_num=False):
    if ocr_reader is None: return ""
    resultats = ocr_reader.ocr(roi_image, cls=False)
    if resultats and resultats[0]:
        text = " ".join([res[1][0] for res in resultats[0]])
        if is_num:
            text = ''.join(filter(str.isdigit, text))
        return text.strip()
    return ""

def process_ocr(ocr_reader, rectified_card, label):
    results = {}
    if label not in OCR_ROIS or not OCR_AVAILABLE:
        return results
        
    rois = OCR_ROIS[label]
    h, w = rectified_card.shape[:2]
    
    def extract_field(field_name, roi_cfg):
        x1, x2 = int(w * roi_cfg["x1"]), int(w * roi_cfg["x2"])
        y1, y2 = int(h * roi_cfg["y1"]), int(h * roi_cfg["y2"])
        roi_img = rectified_card[y1:y2, x1:x2]
        is_num = field_name in ["DOB", "NIN"]
        return field_name, extraire_texte_roi(ocr_reader, roi_img, is_num)

    for name, cfg in rois.items():
        field_name, text = extract_field(name, cfg)
        if text:
            results[field_name] = text
            
    return results

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
            url = "http://192.168.100.2:8080/video"
            print(f"Connecting to {url}")
            return cv2.VideoCapture(url)
        else:
            print("Invalid choice try again")

# Main application loop
def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(BASE_DIR, "saved_id_cards")
    template_dir = os.path.join(BASE_DIR, "templates")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Loading templates")
    templates = load_templates(template_dir)
    
    ocr_reader = None
    if OCR_AVAILABLE:
        print("Initializing PaddleOCR...")
        # Use 'ar' lang because it supports English/French + Arabic digits and text
        ocr_reader = PaddleOCR(use_textline_orientation=False, lang='ar')
        print("PaddleOCR Initialized.")
    else:
        print("PaddleOCR not installed. OCR disabled.")
    cap = select_camera()
    if not cap.isOpened():
        print("Error could not open camera")
        return
    print("")
    print("ID Card Detector Started")
    print("S -> Save card")
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
    
    # OCR Async State
    last_ocr_time = 0
    ocr_results = {}
    ocr_in_progress = False
    ocr_executor = ThreadPoolExecutor(max_workers=1) if OCR_AVAILABLE else None

    def ocr_callback(future):
        nonlocal ocr_results, ocr_in_progress
        try:
            ocr_results = future.result()
            if ocr_results:
                print(f"[OCR] Extracted: {ocr_results}")
        except Exception as e:
            print(f"[OCR] Error: {e}")
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
            
            # Async OCR Trigger
            if stable_label in OCR_ROIS and OCR_AVAILABLE:
                current_time = time.time()
                if not ocr_in_progress and current_time - last_ocr_time > 2.0:
                    ocr_in_progress = True
                    last_ocr_time = current_time
                    future = ocr_executor.submit(process_ocr, ocr_reader, rectified_card.copy(), stable_label)
                    future.add_done_callback(ocr_callback)
            
            # Draw OCR Results on screen
            if stable_label in OCR_ROIS and ocr_results:
                y_offset = 80
                for k, v in ocr_results.items():
                    cv2.putText(display_card, f"{k}: {v}", (15, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    y_offset += 30
            elif stable_label in OCR_ROIS and ocr_in_progress:
                cv2.putText(display_card, "Extracting text...", (15, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

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
            ocr_results.clear()
            
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