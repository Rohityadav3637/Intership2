"""
evaluate.py
-----------
Loads the trained YOLOv8 model and evaluates it on the **test** split.
Prints a per-class and overall metrics table, then saves it to
docs/baseline_results.md
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(r"c:\Users\Rishuraj Kumar\steelsight")
WEIGHTS     = BASE_DIR / "runs" / "train_baseline" / "weights" / "best.pt"
DATASET_YAML = BASE_DIR / "dataset.yaml"
DOCS_DIR    = BASE_DIR / "docs"
RESULTS_MD  = DOCS_DIR / "baseline_results.md"

# ── Sanity check ────────────────────────────────────────────────────────────────
if not WEIGHTS.exists():
    sys.exit(
        f"[ERROR] Weights not found at {WEIGHTS}.\n"
        "Run src/train/train_baseline.py first."
    )

from ultralytics import YOLO

# ── Load model ─────────────────────────────────────────────────────────────────
print(f"\nLoading weights from: {WEIGHTS}")
model = YOLO(str(WEIGHTS))

# ── Run validation on test split ───────────────────────────────────────────────
print("Running evaluation on test split …\n")
metrics = model.val(
    data    = str(DATASET_YAML),
    split   = "test",          # use the test images/labels
    device  = "cpu",
    batch   = 8,
    workers = 0,
    verbose = True,
)

# ── Extract overall metrics ────────────────────────────────────────────────────
map50    = metrics.box.map50       # scalar
map5095  = metrics.box.map         # scalar (mAP@0.50:0.95)
prec_all = metrics.box.mp          # mean precision
rec_all  = metrics.box.mr          # mean recall

# Per-class metrics — each is a list aligned to class indices
prec_pc  = metrics.box.p           # list[float]
rec_pc   = metrics.box.r           # list[float]
ap50_pc  = metrics.box.ap50        # list[float]
ap_pc    = metrics.box.ap          # list[float]  (AP@0.50:0.95)

class_names = ["defect_1", "defect_2", "defect_3", "defect_4"]

# ── Console table ──────────────────────────────────────────────────────────────
sep  = "-" * 72
hdr  = f"{'Class':<14} {'Precision':>10} {'Recall':>10} {'mAP50':>10} {'mAP50-95':>10}"
print(sep)
print(hdr)
print(sep)
for i, name in enumerate(class_names):
    print(
        f"{name:<14} "
        f"{prec_pc[i]:>10.4f} "
        f"{rec_pc[i]:>10.4f} "
        f"{ap50_pc[i]:>10.4f} "
        f"{ap_pc[i]:>10.4f}"
    )
print(sep)
print(
    f"{'ALL':<14} "
    f"{prec_all:>10.4f} "
    f"{rec_all:>10.4f} "
    f"{map50:>10.4f} "
    f"{map5095:>10.4f}"
)
print(sep)
print(f"\nmAP50    : {map50:.4f}")
print(f"mAP50-95 : {map5095:.4f}")

# ── Save to docs/baseline_results.md ──────────────────────────────────────────
DOCS_DIR.mkdir(parents=True, exist_ok=True)

ts = datetime.now().strftime("%Y-%m-%d %H:%M")
md_lines = [
    "# Baseline Evaluation Results",
    "",
    f"**Model**: YOLOv8n fine-tuned on Severstal Steel Defect Detection  ",
    f"**Weights**: `{WEIGHTS}`  ",
    f"**Split**: test  ",
    f"**Evaluated**: {ts}  ",
    "",
    "## Per-Class Metrics",
    "",
    "| Class | Precision | Recall | mAP50 | mAP50-95 |",
    "| :---- | --------: | -----: | ----: | -------: |",
]

for i, name in enumerate(class_names):
    md_lines.append(
        f"| `{name}` "
        f"| {prec_pc[i]:.4f} "
        f"| {rec_pc[i]:.4f} "
        f"| {ap50_pc[i]:.4f} "
        f"| {ap_pc[i]:.4f} |"
    )

md_lines += [
    f"| **ALL** "
    f"| **{prec_all:.4f}** "
    f"| **{rec_all:.4f}** "
    f"| **{map50:.4f}** "
    f"| **{map5095:.4f}** |",
    "",
    "## Summary",
    "",
    f"| Metric | Value |",
    f"| :----- | ----: |",
    f"| mAP50 (all classes) | **{map50:.4f}** |",
    f"| mAP50-95 (all classes) | **{map5095:.4f}** |",
    "",
    "> **Note**: Results are from a 2-epoch baseline training run on CPU.",
    "> Further training and hyperparameter tuning is expected to improve these scores.",
]

RESULTS_MD.write_text("\n".join(md_lines), encoding="utf-8")
print(f"\n✅ Results saved to: {RESULTS_MD}")
