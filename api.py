import io
import cv2
import numpy as np
import os
from fastapi import FastAPI, UploadFile, File
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

@app.post("/extract")
async def extract_data(file: UploadFile = File(...)):
    if not file:
        return JSONResponse(status_code=400, content={"error": "No image uploaded"})
        
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image format"})

    # 1. Check if the frontend image is upside down using Template Matching
    rectified_card, was_rotated = fix_orientation(image, templates)

    # 2. Classify Document
    label, _, score, _ = classify_document(rectified_card, templates)

    if label in ["NO TEMPLATES", "OTHER DOCUMENT"]:
        return {"error": "Unsupported or unrecognized document format. Please align the card clearly in the frame."}

    # 3. Extract Text via PaddleOCR
    result = ocr_manager.process(label, rectified_card)

    return result

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
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image received from IP Camera."})
            
        # Mathematically crop the image using coordinates provided by the frontend
        c = payload.crop
        h, w = image.shape[:2]
        
        # Ensure coordinates are within bounds
        x1 = max(0, c.x)
        y1 = max(0, c.y)
        x2 = min(w, c.x + c.w)
        y2 = min(h, c.y + c.h)
        
        cropped = image[y1:y2, x1:x2]
        
        # Resize to standard OCR dimensions (856x540)
        rectified_card = cv2.resize(cropped, (856, 540))
        
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

