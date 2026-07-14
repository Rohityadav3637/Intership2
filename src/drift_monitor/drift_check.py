"""
drift_check.py
--------------
Monitors covariate drift (brightness and contrast) of input images.

1. Baseline stats: Computes mean & std of pixel intensities (grayscale) for
   all images in data/images/train, saves bin edges/proportions to baseline_stats.json.
2. Target evaluation: Computes the same for a target folder, evaluates against
   baseline using Population Stability Index (PSI).
3. Thresholding:
   - PSI < 0.1: No drift (OK)
   - PSI < 0.2: Minimal drift (OK)
   - 0.2 <= PSI < 0.25: Moderate drift (WARNING)
   - PSI >= 0.25: Significant drift (DRIFT DETECTED)

Usage:
    # 1. Establish baseline & run test check against validation split:
    python src/drift_monitor/drift_check.py --target data/images/val
"""

import argparse
import json
import math
import sys
from pathlib import Path
import cv2
import numpy as np

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(r"c:\Users\Rishuraj Kumar\steelsight")
TRAIN_IMG_DIR  = BASE_DIR / "data" / "images" / "train"
BASELINE_JSON  = BASE_DIR / "src" / "drift_monitor" / "baseline_stats.json"


def compute_image_stats(image_path: Path) -> tuple[float, float]:
    """Computes brightness (mean intensity) and contrast (std of intensity) for an image."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    mean = float(np.mean(img))
    std  = float(np.std(img))
    return mean, std


def collect_folder_stats(folder_path: Path) -> tuple[list[float], list[float]]:
    """Gathers brightness and contrast lists for all images in a folder."""
    images = sorted(folder_path.glob("*.jpg")) + sorted(folder_path.glob("*.png"))
    if not images:
        raise ValueError(f"No images found in {folder_path}")

    brightnesses = []
    contrasts = []
    print(f"Processing {len(images)} images in: {folder_path.name} ...")

    # Limit printing overhead but show progress bar or status
    step = max(1, len(images) // 10)
    for idx, img_p in enumerate(images):
        try:
            mean, std = compute_image_stats(img_p)
            brightnesses.append(mean)
            contrasts.append(std)
        except Exception as e:
            print(f"[WARN] Skipping {img_p.name} due to: {e}")

        if (idx + 1) % step == 0 or (idx + 1) == len(images):
            print(f"  Processed {idx + 1}/{len(images)} images...")

    return brightnesses, contrasts


def calculate_psi(baseline: list[float], target: list[float], bin_edges: list[float]) -> float:
    """Calculates PSI between baseline and target distributions using pre-defined bin edges."""
    # Convert lists to numpy arrays
    base_arr = np.array(baseline)
    targ_arr = np.array(target)

    # Bin the data using the provided bin edges
    base_counts, _ = np.histogram(base_arr, bins=bin_edges)
    targ_counts, _ = np.histogram(targ_arr, bins=bin_edges)

    # Convert to proportions
    base_props = base_counts / len(base_arr)
    targ_props = targ_counts / len(targ_arr)

    # Add epsilon to handle zero counts smoothly
    eps = 1e-4
    base_props = np.where(base_props == 0, eps, base_props)
    targ_props = np.where(targ_props == 0, eps, targ_props)

    # Re-normalize to sum to 1
    base_props /= np.sum(base_props)
    targ_props /= np.sum(targ_props)

    # Calculate PSI
    psi_value = np.sum((targ_props - base_props) * np.log(targ_props / base_props))
    return float(psi_value)


def get_drift_status(psi: float) -> str:
    """Returns classification status based on PSI score."""
    if psi < 0.2:
        return "OK"
    elif 0.2 <= psi <= 0.25:
        return "WARNING"
    else:
        return "DRIFT DETECTED"


def main():
    parser = argparse.ArgumentParser(description="SteelSight covariate drift monitor.")
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Path to the directory containing target images to compare against baseline.",
    )
    parser.add_argument(
        "--rebuild-baseline",
        action="store_true",
        help="Force rebuild of baseline_stats.json from train images.",
    )
    args = parser.parse_args()

    # Create parent directories if they don't exist
    BASELINE_JSON.parent.mkdir(parents=True, exist_ok=True)

    # 1. Establish or load baseline
    if not BASELINE_JSON.exists() or args.rebuild_baseline:
        print("Establish baseline from training dataset...")
        if not TRAIN_IMG_DIR.exists():
            print(f"[ERROR] Training image folder not found at: {TRAIN_IMG_DIR}")
            sys.exit(1)

        b_vals, c_vals = collect_folder_stats(TRAIN_IMG_DIR)

        # We construct 10 bins (deciles) for both brightness and contrast based on baseline
        b_bins = np.percentile(b_vals, np.linspace(0, 100, 11)).tolist()
        c_bins = np.percentile(c_vals, np.linspace(0, 100, 11)).tolist()

        baseline_data = {
            "brightness_values": b_vals,
            "contrast_values": c_vals,
            "brightness_bins": b_bins,
            "contrast_bins": c_bins,
            "metadata": {
                "source": str(TRAIN_IMG_DIR),
                "num_images": len(b_vals),
                "rebuilt_at": str(np.datetime64("now")),
            }
        }

        with open(BASELINE_JSON, "w") as f:
            json.dump(baseline_data, f, indent=4)
        print(f"[SUCCESS] Baseline established and saved to: {BASELINE_JSON}\n")
    else:
        print(f"Loading baseline statistics from: {BASELINE_JSON}")
        with open(BASELINE_JSON, "r") as f:
            baseline_data = json.load(f)

    # 2. Run check against target folder if provided
    if args.target:
        target_dir = Path(args.target)
        if not target_dir.exists():
            print(f"[ERROR] Target directory not found: {target_dir}")
            sys.exit(1)

        print(f"\nEvaluating target directory: {target_dir}")
        t_b_vals, t_c_vals = collect_folder_stats(target_dir)

        # Retrieve baseline variables
        b_base = baseline_data["brightness_values"]
        c_base = baseline_data["contrast_values"]
        b_bins = baseline_data["brightness_bins"]
        c_bins = baseline_data["contrast_bins"]

        # Calculate PSI
        b_psi = calculate_psi(b_base, t_b_vals, b_bins)
        c_psi = calculate_psi(c_base, t_c_vals, c_bins)

        b_status = get_drift_status(b_psi)
        c_status = get_drift_status(c_psi)

        # Print detailed report
        print("\n" + "=" * 55)
        print("  COVARIATE DRIFT MONITORING REPORT")
        print("=" * 55)
        print(f"  Target Folder  : {target_dir.resolve()}")
        print(f"  Num Images     : {len(t_b_vals)}")
        print("-" * 55)
        print(f"  BRIGHTNESS PSI : {b_psi:.4f}  [{b_status}]")
        print(f"  CONTRAST PSI   : {c_psi:.4f}  [{c_status}]")
        print("-" * 55)
        
        # Combined flag logic
        max_psi = max(b_psi, c_psi)
        combined_status = get_drift_status(max_psi)
        print(f"  OVERALL STATUS : {combined_status}")
        print("=" * 55)


if __name__ == "__main__":
    main()
