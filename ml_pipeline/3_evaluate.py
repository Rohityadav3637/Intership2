"""
3_evaluate.py
-------------
Evaluates a trained YOLOv8 model on the test split.
Supports comparing improved model against the baseline.

Usage:
  python ml_pipeline/3_evaluate.py                    # evaluates latest best.pt
  python ml_pipeline/3_evaluate.py --weights path.pt  # evaluates specific weights
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

BASE_DIR     = Path(__file__).resolve().parent.parent
DATASET_YAML = BASE_DIR / "dataset.yaml"
DOCS_DIR     = BASE_DIR / "docs"

# Known baseline numbers for comparison table
BASELINE = {
    "model":     "YOLOv8n (2 epochs)",
    "mAP50":     0.127,
    "mAP50-95":  0.0478,
    "precision": None,
    "recall":    None,
}

CLASS_NAMES = ["defect_1", "defect_2", "defect_3", "defect_4"]


def find_best_weights():
    """Return the most-recently-modified best.pt under runs/."""
    candidates = sorted(
        BASE_DIR.glob("runs/**/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def evaluate(weights_path: Path, output_file: str = "improved_results.md") -> dict:
    from ultralytics import YOLO

    print(f"\n[Evaluate] Loading weights: {weights_path}")
    model = YOLO(str(weights_path))

    print("[Evaluate] Running validation on test split (conf=0.001, iou=0.6)...\n")
    metrics = model.val(
        data     = str(DATASET_YAML),
        split    = "test",
        conf     = 0.001,   # standard YOLO eval threshold
        iou      = 0.6,     # standard IoU threshold
        device   = "cpu",
        batch    = 8,
        workers  = 0,
        verbose  = True,
    )

    map50    = float(metrics.box.map50)
    map5095  = float(metrics.box.map)
    prec_all = float(metrics.box.mp)
    rec_all  = float(metrics.box.mr)
    prec_pc  = list(metrics.box.p)
    rec_pc   = list(metrics.box.r)
    ap50_pc  = list(metrics.box.ap50)
    ap_pc    = list(metrics.box.ap)

    # ── Console table ────────────────────────────────────────────────────────
    SEP = "-" * 74
    print(SEP)
    print(f"{'Class':<14} {'Precision':>10} {'Recall':>10} {'mAP50':>10} {'mAP50-95':>10}")
    print(SEP)
    for i, name in enumerate(CLASS_NAMES):
        print(
            f"{name:<14} "
            f"{prec_pc[i]:>10.4f} "
            f"{rec_pc[i]:>10.4f} "
            f"{ap50_pc[i]:>10.4f} "
            f"{ap_pc[i]:>10.4f}"
        )
    print(SEP)
    print(f"{'ALL':<14} {prec_all:>10.4f} {rec_all:>10.4f} {map50:>10.4f} {map5095:>10.4f}")
    print(SEP)

    # ── Baseline comparison ──────────────────────────────────────────────────
    improvement = map50 - BASELINE["mAP50"]
    pct_change  = improvement / BASELINE["mAP50"] * 100
    print(f"\n[vs Baseline]  mAP50: {BASELINE['mAP50']:.4f}  ->  {map50:.4f}  "
          f"({improvement:+.4f} / {pct_change:+.1f}%)")

    # ── Save markdown report ─────────────────────────────────────────────────
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    run_name = weights_path.parent.parent.name   # e.g. train_improved

    lines = [
        f"# Evaluation Results — {run_name}",
        "",
        f"**Model**: {weights_path}  ",
        f"**Split**: test  ",
        f"**Evaluated**: {ts}  ",
        f"**Thresholds**: conf=0.001, iou=0.6 (standard YOLO defaults)  ",
        "",
        "## Per-Class Metrics",
        "",
        "| Class | Precision | Recall | mAP50 | mAP50-95 |",
        "| :---- | --------: | -----: | ----: | -------: |",
    ]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(
            f"| `{name}` "
            f"| {prec_pc[i]:.4f} "
            f"| {rec_pc[i]:.4f} "
            f"| {ap50_pc[i]:.4f} "
            f"| {ap_pc[i]:.4f} |"
        )
    lines += [
        f"| **ALL** | **{prec_all:.4f}** | **{rec_all:.4f}** | **{map50:.4f}** | **{map5095:.4f}** |",
        "",
        "## Comparison vs Baseline",
        "",
        "| Metric | Baseline (YOLOv8n, 2 ep) | Improved | Delta |",
        "| :----- | -----------------------: | -------: | ----: |",
        f"| mAP50 | {BASELINE['mAP50']:.4f} | {map50:.4f} | {improvement:+.4f} ({pct_change:+.1f}%) |",
        f"| mAP50-95 | {BASELINE['mAP50-95']:.4f} | {map5095:.4f} | {map5095 - BASELINE['mAP50-95']:+.4f} |",
        "",
        "## Changes Applied",
        "",
        "- **Model**: YOLOv8n -> YOLOv8s (2x capacity)",
        "- **Epochs**: 2 -> 150 with early stopping (patience=20)",
        "- **Class imbalance**: Minority-class image oversampling (up to 5x)",
        "- **Augmentation**: Mosaic=1.0, Mixup=0.15, CopyPaste=0.10, HSV jitter",
        "- **Eval thresholds**: conf=0.001, iou=0.6 (YOLO standard defaults)",
    ]

    out_path = DOCS_DIR / output_file
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[Evaluate] Results saved to: {out_path}")

    return {
        "mAP50": map50, "mAP50-95": map5095,
        "precision": prec_all, "recall": rec_all,
        "per_class": list(zip(CLASS_NAMES, prec_pc, rec_pc, ap50_pc, ap_pc)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to best.pt (auto-detected if omitted)")
    parser.add_argument("--out", type=str, default="improved_results.md")
    args = parser.parse_args()

    if args.weights:
        w = Path(args.weights)
    else:
        w = find_best_weights()
        if w is None:
            sys.exit("[ERROR] No best.pt found. Train a model first.")

    evaluate(w, output_file=args.out)
