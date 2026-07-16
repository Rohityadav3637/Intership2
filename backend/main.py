"""
main.py
-------
SteelSight FastAPI inference server.

Endpoints:
  POST /predict       — Upload an image, run YOLOv8, return detections + uncertainty flag
  GET  /health        — Liveness check with model path confirmation
  GET  /predictions   — Last 50 logged predictions, most-recent-first

All predictions are logged to a local SQLite database (inference_log.db) via SQLAlchemy.
CORS is enabled for all origins to support a Streamlit front-end on a different port.

Run:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
  or from repo root:
    python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Boolean, Column, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# ── Paths ────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# Search for the most-recently-modified best.pt under runs/
_candidates = sorted(
    BASE_DIR.glob("runs/**/weights/best.pt"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
if not _candidates:
    raise RuntimeError(
        f"No best.pt found under {BASE_DIR / 'runs'}. "
        "Run src/train/train_baseline.py first."
    )
MODEL_PATH = _candidates[0]

DB_PATH = BASE_DIR / "src" / "inference_server" / "inference_log.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.5

CLASS_NAMES = {0: "defect_1", 1: "defect_2", 2: "defect_3", 3: "defect_4"}

# ── Database setup (SQLAlchemy) ───────────────────────────────────────────────────
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String, nullable=False)
    image_filename = Column(String, nullable=False)
    detections = Column(Text, nullable=False)   # JSON text
    is_uncertain = Column(Boolean, nullable=False)


Base.metadata.create_all(bind=engine)

# ── Model loading ─────────────────────────────────────────────────────────────────
# Load once at module import; protected by a reentrant lock so concurrent requests
# that arrive before the model is warm don't double-load it.
_model_lock = threading.Lock()
_model = None


def _get_model():
    """Return the singleton YOLO model, loading it on first call."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:           # double-checked locking
                from ultralytics import YOLO
                print(f"\n[SteelSight] Loading model from: {MODEL_PATH}\n")
                _model = YOLO(str(MODEL_PATH))
    return _model


# ── FastAPI app ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SteelSight Inference API",
    description="YOLOv8 steel-defect detection service with drift logging",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup event ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Pre-warm the model so the first request isn't slow."""
    _get_model()


# ── Helper: decode uploaded bytes to BGR array ────────────────────────────────────
def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the uploaded file as a valid image. "
                   "Please upload a JPEG, PNG, or BMP file.",
        )
    return img


# ── POST /predict ─────────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Run YOLOv8 inference on an uploaded image.

    Returns:
        image_width, image_height, detections list, is_uncertain flag
    """
    raw = await file.read()

    # Validate & decode
    img_bgr = _decode_image(raw)
    h, w = img_bgr.shape[:2]

    # Run inference (thread-safe: YOLO forward pass is GIL-protected internally)
    model = _get_model()
    results = model(img_bgr, verbose=False)

    detections: list[dict] = []
    max_conf = 0.0

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            x_min, y_min, x_max, y_max = (float(v) for v in box.xyxy[0])

            if conf > max_conf:
                max_conf = conf

            detections.append(
                {
                    "class_name": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                    "confidence": round(conf, 4),
                    "bbox": {
                        "x_min": round(x_min, 1),
                        "y_min": round(y_min, 1),
                        "x_max": round(x_max, 1),
                        "y_max": round(y_max, 1),
                    },
                }
            )

    is_uncertain = max_conf < CONFIDENCE_THRESHOLD

    # Log to database
    db: Session = SessionLocal()
    try:
        record = Prediction(
            timestamp=datetime.now(timezone.utc).isoformat(),
            image_filename=file.filename or "unknown",
            detections=json.dumps(detections),
            is_uncertain=is_uncertain,
        )
        db.add(record)
        db.commit()
    finally:
        db.close()

    return {
        "image_width": w,
        "image_height": h,
        "num_detections": len(detections),
        "is_uncertain": is_uncertain,
        "detections": detections,
    }


# ── GET /health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe — confirms the model is loaded and shows its path."""
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
    }


# ── GET /predictions ──────────────────────────────────────────────────────────────
@app.get("/predictions")
async def get_predictions() -> list[dict]:
    """Return the last 50 logged predictions, most-recent first."""
    db: Session = SessionLocal()
    try:
        rows = (
            db.query(Prediction)
            .order_by(Prediction.id.desc())
            .limit(50)
            .all()
        )
    finally:
        db.close()

    return [
        {
            "id": r.id,
            "timestamp": r.timestamp,
            "image_filename": r.image_filename,
            "detections": json.loads(r.detections),
            "is_uncertain": r.is_uncertain,
        }
        for r in rows
    ]


# ── Entrypoint for direct `python main.py` ───────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
