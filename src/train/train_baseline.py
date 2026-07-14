import os
import sys

# Configure MLflow tracking BEFORE importing ultralytics
# so the built-in callback picks up the URI on module load
base_dir = r"c:\Users\Rishuraj Kumar\steelsight"
# MLflow 3.x dropped file-store support — use SQLite as local backend
mlflow_db  = os.path.join(base_dir, "mlflow.db").replace("\\", "/")
MLFLOW_URI = f"sqlite:///{mlflow_db}"
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_URI
os.environ["ULTRALYTICS_MLFLOW"]  = "True"   # Enable built-in Ultralytics MLflow callback

import mlflow
from ultralytics import YOLO

def main():
    dataset_yaml  = os.path.join(base_dir, "dataset.yaml")
    runs_dir      = os.path.join(base_dir, "runs")

    # ── MLflow setup ────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("steelsight-steel-defect-detection")
    print(f"MLflow tracking URI : {MLFLOW_URI}")
    print(f"MLflow experiment   : steelsight-steel-defect-detection")

    # ── Model ────────────────────────────────────────────────────────────────
    print("\nLoading pretrained YOLOv8n ...")
    model = YOLO("yolov8n.pt")

    # ── Training ─────────────────────────────────────────────────────────────
    print("Starting training — 2 epochs on CPU ...\n")
    results = model.train(
        data      = dataset_yaml,
        epochs    = 2,
        imgsz     = 640,
        device    = "cpu",          # no GPU detected
        batch     = 8,              # conservative for CPU RAM
        workers   = 0,              # 0 = main-process dataloader (avoids Win spawn issues)
        project   = runs_dir,
        name      = "train_baseline",
        exist_ok  = True,           # overwrite if re-running
        verbose   = True,
    )

    # ── Results ──────────────────────────────────────────────────────────────
    metrics   = results.results_dict
    mAP50     = metrics.get("metrics/mAP50(B)",    0.0)
    mAP50_95  = metrics.get("metrics/mAP50-95(B)", 0.0)

    weights   = os.path.join(runs_dir, "train_baseline", "weights", "best.pt")

    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE")
    print("=" * 55)
    print(f"  Final mAP50      : {mAP50:.4f}")
    print(f"  Final mAP50-95   : {mAP50_95:.4f}")
    print(f"  Best weights     : {weights}")
    print("=" * 55)

    # Log summary to MLflow as well
    with mlflow.start_run(run_name="train_baseline_summary", nested=True):
        mlflow.log_metric("final_mAP50",    mAP50)
        mlflow.log_metric("final_mAP50_95", mAP50_95)
        if os.path.exists(weights):
            mlflow.log_artifact(weights, artifact_path="weights")

if __name__ == "__main__":
    main()
