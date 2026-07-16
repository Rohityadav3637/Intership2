"""
benchmark_latency.py
--------------------
Benchmarks inference latency of:
  1. Original PyTorch  (YOLOv8 best.pt  via Ultralytics)
  2. ONNX model        (best.onnx       via onnxruntime)

on 50 randomly-sampled test images.

Reports average, p50, p95 latency (ms/image) for each backend and saves
a Markdown comparison table to docs/latency_comparison.md.

Usage:
    python src/export/benchmark_latency.py
"""

import os
import sys
import time
import random
import statistics
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
WEIGHTS_PT = BASE_DIR / "runs" / "train_baseline" / "weights" / "best.pt"
WEIGHTS_OX = BASE_DIR / "runs" / "train_baseline" / "weights" / "best.onnx"
TEST_IMGS  = BASE_DIR / "data" / "images" / "test"
DOCS_DIR   = BASE_DIR / "docs"
OUT_MD     = DOCS_DIR / "latency_comparison.md"

N_IMAGES   = 50
IMG_SIZE   = 640
WARMUP     = 3          # warm-up runs before timing (per backend)
SEED       = 42

# ── Sanity checks ───────────────────────────────────────────────────────────────
if not WEIGHTS_PT.exists():
    sys.exit(f"[ERROR] PyTorch weights not found: {WEIGHTS_PT}")
if not WEIGHTS_OX.exists():
    sys.exit(
        f"[ERROR] ONNX weights not found: {WEIGHTS_OX}\n"
        "Run src/export/export_onnx.py first."
    )
if not TEST_IMGS.exists():
    sys.exit(f"[ERROR] Test image dir not found: {TEST_IMGS}")


# ── Sample images ───────────────────────────────────────────────────────────────
all_imgs = sorted(TEST_IMGS.glob("*.jpg")) + sorted(TEST_IMGS.glob("*.png"))
if len(all_imgs) < N_IMAGES:
    print(f"[WARN] Only {len(all_imgs)} test images found; using all of them.")
    N_IMAGES = len(all_imgs)

random.seed(SEED)
sample_imgs = random.sample(all_imgs, N_IMAGES)
print(f"Sampled {N_IMAGES} test images from {TEST_IMGS}\n")


# ── Preprocessing helper ────────────────────────────────────────────────────────
def preprocess(img_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (original_bgr, NCHW_float32_640x640) for ONNX input."""
    bgr = cv2.imread(str(img_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    tensor  = resized.astype(np.float32) / 255.0          # [0,1]
    tensor  = np.transpose(tensor, (2, 0, 1))              # HWC → CHW
    tensor  = np.expand_dims(tensor, 0)                    # CHW → NCHW
    return bgr, tensor


# ═══════════════════════════════════════════════════════════════════════════════
# Backend 1 — PyTorch (Ultralytics)
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading PyTorch model …")
from ultralytics import YOLO
pt_model = YOLO(str(WEIGHTS_PT))

print(f"  Warming up PyTorch ({WARMUP} runs) …")
for img_path in sample_imgs[:WARMUP]:
    pt_model(str(img_path), imgsz=IMG_SIZE, device="cpu", verbose=False)

print(f"  Timing PyTorch inference on {N_IMAGES} images …")
pt_latencies = []
for img_path in sample_imgs:
    t0 = time.perf_counter()
    pt_model(str(img_path), imgsz=IMG_SIZE, device="cpu", verbose=False)
    pt_latencies.append((time.perf_counter() - t0) * 1000)   # → ms

pt_mean = statistics.mean(pt_latencies)
pt_med  = statistics.median(pt_latencies)
pt_p95  = sorted(pt_latencies)[int(0.95 * N_IMAGES) - 1]
pt_min  = min(pt_latencies)
pt_max  = max(pt_latencies)

print(f"  PyTorch  →  mean={pt_mean:.1f} ms  p50={pt_med:.1f} ms  p95={pt_p95:.1f} ms\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Backend 2 — ONNX Runtime
# ═══════════════════════════════════════════════════════════════════════════════
try:
    import onnxruntime as ort
except ImportError:
    sys.exit(
        "[ERROR] onnxruntime not installed.\n"
        "Run:  .venv\\Scripts\\pip install onnxruntime"
    )

print("Loading ONNX model …")
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads  = os.cpu_count()
sess_opts.inter_op_num_threads  = 1
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(
    str(WEIGHTS_OX),
    sess_options=sess_opts,
    providers=["CPUExecutionProvider"],
)
input_name = sess.get_inputs()[0].name

print(f"  Warming up ONNX ({WARMUP} runs) …")
for img_path in sample_imgs[:WARMUP]:
    _, tensor = preprocess(img_path)
    sess.run(None, {input_name: tensor})

print(f"  Timing ONNX inference on {N_IMAGES} images …")
ox_latencies = []
for img_path in sample_imgs:
    _, tensor = preprocess(img_path)
    t0 = time.perf_counter()
    sess.run(None, {input_name: tensor})
    ox_latencies.append((time.perf_counter() - t0) * 1000)

ox_mean = statistics.mean(ox_latencies)
ox_med  = statistics.median(ox_latencies)
ox_p95  = sorted(ox_latencies)[int(0.95 * N_IMAGES) - 1]
ox_min  = min(ox_latencies)
ox_max  = max(ox_latencies)

print(f"  ONNX     →  mean={ox_mean:.1f} ms  p50={ox_med:.1f} ms  p95={ox_p95:.1f} ms\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Console summary
# ═══════════════════════════════════════════════════════════════════════════════
speedup = pt_mean / ox_mean if ox_mean > 0 else float("inf")
sep  = "─" * 62
print(sep)
print(f"  {'Metric':<22} {'PyTorch':>12} {'ONNX':>12} {'ONNX/PT':>10}")
print(sep)
rows = [
    ("Mean latency (ms)",   pt_mean, ox_mean),
    ("Median / p50 (ms)",   pt_med,  ox_med),
    ("p95 latency (ms)",    pt_p95,  ox_p95),
    ("Min latency (ms)",    pt_min,  ox_min),
    ("Max latency (ms)",    pt_max,  ox_max),
]
for label, pv, ov in rows:
    ratio = pv / ov if ov > 0 else float("inf")
    print(f"  {label:<22} {pv:>12.2f} {ov:>12.2f} {ratio:>9.2f}x")
print(sep)
print(f"  ONNX is {speedup:.2f}x {'faster' if speedup >= 1 else 'slower'} than PyTorch on average.")
print(sep)


# ═══════════════════════════════════════════════════════════════════════════════
# Save Markdown
# ═══════════════════════════════════════════════════════════════════════════════
DOCS_DIR.mkdir(parents=True, exist_ok=True)
ts = datetime.now().strftime("%Y-%m-%d %H:%M")

direction = "faster" if speedup >= 1 else "slower"
lines = [
    "# Latency Benchmark: PyTorch vs ONNX",
    "",
    f"**Model**: YOLOv8n fine-tuned — Severstal Steel Defect Detection  ",
    f"**Device**: CPU  ",
    f"**Images**: {N_IMAGES} randomly sampled test images  ",
    f"**Image size**: {IMG_SIZE} × {IMG_SIZE}  ",
    f"**Warm-up runs**: {WARMUP} (excluded from timing)  ",
    f"**Benchmarked**: {ts}  ",
    "",
    "## Results",
    "",
    "| Metric | PyTorch (ms) | ONNX (ms) | ONNX / PyTorch |",
    "| :----- | -----------: | --------: | -------------: |",
]
for label, pv, ov in rows:
    ratio = pv / ov if ov > 0 else float("inf")
    lines.append(f"| {label} | {pv:.2f} | {ov:.2f} | {ratio:.2f}× |")

lines += [
    "",
    "## Summary",
    "",
    f"ONNX Runtime is **{speedup:.2f}× {direction}** than the PyTorch backend on average.",
    "",
    "> Benchmarks run on CPU only (no GPU available). "
    "ONNX speedups are typically larger on GPU or with TensorRT.",
]

OUT_MD.write_text("\n".join(lines), encoding="utf-8")
print(f"\n✅ Results saved to: {OUT_MD}")
