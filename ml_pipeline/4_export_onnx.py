"""
export_onnx.py
--------------
Exports the trained YOLOv8 model to ONNX format.

Usage:
    python src/export/export_onnx.py
"""

import sys
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
WEIGHTS   = BASE_DIR / "runs" / "train_baseline" / "weights" / "best.pt"
ONNX_OUT  = BASE_DIR / "runs" / "train_baseline" / "weights" / "best.onnx"

if not WEIGHTS.exists():
    sys.exit(f"[ERROR] Weights not found: {WEIGHTS}\nRun training first.")

from ultralytics import YOLO

print(f"Loading  : {WEIGHTS}")
model = YOLO(str(WEIGHTS))

print(f"Exporting to ONNX …")
out_path = model.export(
    format    = "onnx",
    imgsz     = 640,
    opset     = 12,        # broad runtime compatibility
    simplify  = True,      # run onnx-simplifier
    dynamic   = False,     # static batch=1 for deterministic latency
    half      = False,     # FP32 (CPU doesn't support FP16 ONNX well)
)

print(f"\n✅ ONNX model saved to: {out_path}")
