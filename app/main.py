from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
import time
import logging
import os
import json
import io
from PIL import Image
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# -----------------------
# APP INIT
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("dermai")

app = FastAPI()

# -----------------------
# SAFE PATHS
# -----------------------
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR  = os.path.join(BASE_DIR, "templates")
STATIC_DIR    = os.path.join(BASE_DIR, "static")
MODEL_PATH    = os.path.join(BASE_DIR, "models", "skin_model.h5")
MAPPING_PATH  = os.path.join(BASE_DIR, "models", "class_mapping.json")

# -----------------------
# STATIC + TEMPLATES
# -----------------------
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=TEMPLATE_DIR)

# -----------------------
# CLASS NAMES + INFO
# -----------------------
with open(MAPPING_PATH) as f:
    _mapping = json.load(f)
CLASS_NAMES = [k for k, v in sorted(_mapping.items(), key=lambda x: x[1])]

CLASS_INFO = {
    "akiec": {"full_name": "Actinic Keratosis / Intraepithelial Carcinoma", "severity": "moderate"},
    "bcc":   {"full_name": "Basal Cell Carcinoma",                          "severity": "high"},
    "bkl":   {"full_name": "Benign Keratosis",                              "severity": "low"},
    "df":    {"full_name": "Dermatofibroma",                                "severity": "low"},
    "mel":   {"full_name": "Melanoma",                                      "severity": "high"},
    "nv":    {"full_name": "Melanocytic Nevi (Mole)",                       "severity": "low"},
    "vasc":  {"full_name": "Vascular Lesion",                               "severity": "moderate"},
}

CONFIDENCE_THRESHOLD = 0.60

# -----------------------
# GROQ CLIENT
# -----------------------
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------
# LOAD MODEL
# -----------------------
model = None

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model not found at {MODEL_PATH}")
        return
    logger.info(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    dummy = np.zeros((1, 128, 128, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)
    logger.info(f"Model ready. Classes: {CLASS_NAMES}")

# -----------------------
# HELPERS
# -----------------------
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def get_groq_explanation(label: str, confidence: float) -> dict:
    info      = CLASS_INFO.get(label, {})
    full_name = info.get("full_name", label)

    prompt = f"""You are a medical AI assistant explaining a skin condition diagnosis to a patient.
The AI model detected: {full_name} (confidence: {confidence:.1f}%).

Reply ONLY with a JSON object (no markdown, no extra text) with exactly these 4 keys:
{{
  "what_it_is": "1-2 sentence plain-English description",
  "common_symptoms": "2-3 key visual symptoms separated by semicolons",
  "risk_level": "Low or Moderate or High — one word only",
  "when_to_see_doctor": "1 sentence advice on urgency"
}}"""

    try:
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        logger.error(f"Groq error: {e}")
        return {
            "what_it_is":        f"{full_name} was detected by the model.",
            "common_symptoms":   "Consult a dermatologist for a detailed assessment.",
            "risk_level":        info.get("severity", "moderate").capitalize(),
            "when_to_see_doctor":"Please consult a qualified dermatologist.",
        }

# -----------------------
# HOME ROUTE
# -----------------------
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")

# -----------------------
# PREDICT ROUTE
# -----------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        if model is None:
            return JSONResponse(status_code=503, content={"error": "Model not loaded."})

        if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
            return JSONResponse(status_code=400, content={"error": "Only JPEG, PNG, WebP supported."})

        image_bytes = await file.read()
        logger.info(f"Image received: {file.filename} ({len(image_bytes)/1024:.1f} KB)")

        arr        = preprocess_image(image_bytes)
        preds      = model.predict(arr, verbose=0)[0]
        top_idx    = int(np.argmax(preds))
        confidence = float(preds[top_idx])
        label      = CLASS_NAMES[top_idx]
        info       = CLASS_INFO.get(label, {})
        all_probs  = {cls: round(float(p) * 100, 1) for cls, p in zip(CLASS_NAMES, preds)}
        latency_ms = int((time.time() - start_time) * 1000)

        if confidence < CONFIDENCE_THRESHOLD:
            return JSONResponse(content={
                "uncertain":   True,
                "confidence":  round(confidence * 100, 1),
                "all_probs":   all_probs,
                "latency_ms":  latency_ms,
                "message":     "Confidence too low. Please consult a dermatologist.",
                "label":       label,
                "full_name":   info.get("full_name", label),
                "severity":    info.get("severity", "unknown"),
                "explanation": None,
            })

        explanation = get_groq_explanation(label, confidence * 100)

        return JSONResponse(content={
            "uncertain":   False,
            "label":       label,
            "full_name":   info.get("full_name", label),
            "severity":    info.get("severity", "unknown"),
            "confidence":  round(confidence * 100, 1),
            "all_probs":   all_probs,
            "latency_ms":  latency_ms,
            "message":     "Prediction successful",
            "explanation": explanation,
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Analysis failed: {str(e)}"})

# -----------------------
# HEALTH CHECK
# -----------------------
@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "classes":      CLASS_NAMES,
        "threshold":    CONFIDENCE_THRESHOLD,
    }