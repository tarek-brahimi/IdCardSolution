import cv2
import numpy as np
import os
import time
from collections import Counter

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

# Classify document using template matching
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
            rectified_card = get_perspective_transform(frame, pts, width=856, height=540)
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
                        if stable_label == "ALGERIAN ID CARD":
                            stable_color = (0, 255, 0)
                        elif stable_label == "DRIVERS LICENSE":
                            stable_color = (255, 165, 0)
                        else:
                            stable_color = (0, 0, 255)
                    else:
                        stable_label = "DETECTING"
                        stable_color = (200, 200, 200)
                scores_str = " | ".join([f"{k}: {v:.3f}" for k, v in all_scores.items()])
                print(f"[CLASSIFY] {label} (score: {score:.3f}) | {scores_str}")
            display_card = rectified_card.copy()
            cv2.rectangle(display_card, (0, 0), (856, 50), (0, 0, 0), -1)
            cv2.putText(display_card, stable_label, (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, stable_color, 2)
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