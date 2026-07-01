import io
import cv2
import numpy as np
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import existing backend logic
from Idcard import load_templates, fix_orientation, classify_document
from paddle_ocr.manager import OCRManager

app = FastAPI(title="Biometric OCR API")

# Enable CORS so the local frontend can communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold the loaded models and templates
ocr_manager = None
templates = None

@app.on_event("startup")
async def startup_event():
    global ocr_manager, templates
    print("Loading OCR models...")
    ocr_manager = OCRManager(use_gpu=True)
    print("OCR models loaded successfully.")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(BASE_DIR, "templates")
    templates = load_templates(template_dir)

def detect_and_rectify(frame):
    from Idcard import get_perspective_transform
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            warped = get_perspective_transform(frame, pts, width=856, height=540)
            return warped
    return None

@app.post("/extract")
async def extract_data(file: UploadFile = File(...), expected_type: str = Form(None)):
    if not file:
        return JSONResponse(status_code=400, content={"error": "No image uploaded"})
        
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image format"})

    # Try automatic contour snapping first
    rectified_card = detect_and_rectify(image)
    if rectified_card is None:
        # Fallback to the manual crop provided by the frontend if contour detection fails
        rectified_card = cv2.resize(image, (856, 540))

    if not expected_type:
        # 1. Check if the frontend image is upside down using Template Matching
        rectified_card, was_rotated = fix_orientation(rectified_card, templates)

        # 2. Classify Document
        label, _, score, _ = classify_document(rectified_card, templates)

        if label in ["NO TEMPLATES", "OTHER DOCUMENT"]:
            return {"error": "Unsupported or unrecognized document format. Please align the card clearly in the frame."}
    else:
        label = expected_type

    # 3. Extract Text via PaddleOCR
    result = ocr_manager.process(label, rectified_card)

    return result

@app.post("/debug")
async def debug_ocr(file: UploadFile = File(...)):
    """Returns raw OCR boxes with positions for debugging."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return {"error": "Invalid image"}

    rectified_card = detect_and_rectify(image)
    if rectified_card is None:
        rectified_card = cv2.resize(image, (856, 540))

    rectified_card, _ = fix_orientation(rectified_card, templates)
    label, _, score, _ = classify_document(rectified_card, templates)

    h, w = rectified_card.shape[:2]

    # Run OCR for both languages
    from paddle_ocr.extractor import OCRExtractor
    extractor = ocr_manager._extractor
    raw_boxes = extractor.extract(label, rectified_card)

    debug_data = {"document_type": label, "image_size": f"{w}x{h}", "fields": {}}
    for field_name, boxes in raw_boxes.items():
        box_list = []
        for b in boxes:
            box_list.append({
                "text": b.text,
                "reversed": b.text[::-1],
                "confidence": round(b.confidence, 3),
                "center_x_rel": round(b.center_x / w, 3),
                "center_y_rel": round(b.center_y / h, 3),
                "center_x": round(b.center_x, 1),
                "center_y": round(b.center_y, 1),
            })
        debug_data["fields"][field_name] = box_list

    # Also include the parsed result
    result = ocr_manager.process(label, rectified_card)
    debug_data["parsed_result"] = result

    return debug_data

from pydantic import BaseModel
import requests

class CropRect(BaseModel):
    x: int
    y: int
    w: int
    h: int

class ExtractUrlPayload(BaseModel):
    url: str
    crop: CropRect

@app.post("/extract_url")
async def extract_from_url(payload: ExtractUrlPayload):
    try:
        # Download the image from IP Webcam
        resp = requests.get(payload.url, timeout=5)
        if resp.status_code != 200:
            return JSONResponse(status_code=400, content={"error": "Could not download image from IP Camera."})
            
        nparr = np.frombuffer(resp.content, np.uint8)
        full_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if full_image is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image received from IP Camera."})
            
        # Mathematically crop the image using coordinates provided by the frontend
        c = payload.crop
        h, w = full_image.shape[:2]
        
        # Expand crop box slightly (10%) to ensure we capture the physical edges for contour detection
        pad_x = int(c.w * 0.10)
        pad_y = int(c.h * 0.10)
        
        x1 = max(0, c.x - pad_x)
        y1 = max(0, c.y - pad_y)
        x2 = min(w, c.x + c.w + pad_x)
        y2 = min(h, c.y + c.h + pad_y)
        
        padded_crop = full_image[y1:y2, x1:x2]
        
        # Try automatic contour snapping on the padded crop
        rectified_card = detect_and_rectify(padded_crop)
        
        if rectified_card is None:
            # Fallback: strictly use the exact frontend crop and resize
            fx1 = max(0, c.x)
            fy1 = max(0, c.y)
            fx2 = min(w, c.x + c.w)
            fy2 = min(h, c.y + c.h)
            exact_crop = full_image[fy1:fy2, fx1:fx2]
            rectified_card = cv2.resize(exact_crop, (856, 540))
        
        # Fix orientation if upside down
        rectified_card, _ = fix_orientation(rectified_card, templates)
        
        # Classify and OCR
        label, _, score, _ = classify_document(rectified_card, templates)
        if label in ["NO TEMPLATES", "OTHER DOCUMENT"]:
            return {"error": "Unsupported or unrecognized document format. Please align the card clearly."}
            
        result = ocr_manager.process(label, rectified_card)
        return result
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})





# --- VISITOR MANAGEMENT ENDPOINTS ---

from pydantic import BaseModel
import database as db

class UserCreate(BaseModel):
    nin: str
    french_name: str
    arabic_name: str
    category: str

class UserUpdate(BaseModel):
    old_nin: str
    nin: str
    french_name: str
    arabic_name: str
    category: str
    arabic_name: str
    category: str

class LogCreate(BaseModel):
    action: str

@app.get('/stats/today')
def get_stats_today():
    return {'total_entries_today': db.get_total_entries_today()}

@app.get('/users/{nin}')
def check_user(nin: str):
    user = db.get_user(nin)
    if not user:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail='User not found')
    user['last_action'] = db.get_last_action(nin)
    return user

@app.post('/users')
def create_user(user: UserCreate):
    db.create_user(user.nin, user.french_name, user.arabic_name, user.category)
    return {'status': 'success'}

@app.post('/users/{nin}/log')
def log_user_action(nin: str, log: LogCreate):
    db.log_access(nin, log.action)
    return {'status': 'success'}

@app.get('/users')
def get_users():
    return db.get_all_users()

@app.get('/users/{nin}/logs')
def get_logs(nin: str):
    return db.get_user_logs(nin)

@app.put('/users/{nin}')
def update_user_info(nin: str, user: UserUpdate):
    db.update_user(user.old_nin, user.nin, user.french_name, user.arabic_name, user.category)
    return {'status': 'success'}

@app.delete('/users/{nin}')
def delete_user(nin: str):
    db.delete_user(nin)
    return {'status': 'success'}

@app.delete('/logs/{log_id}')
def delete_log(log_id: int):
    db.delete_log(log_id)
    return {'status': 'success'}

