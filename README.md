# SteelSight - Steel Defect Detection

This project fine-tunes a YOLOv8 object detection model to identify and localize various types of defects on steel surfaces. It was originally built on the dataset from the Kaggle "Severstal: Steel Defect Detection" competition.

## Project Structure

```text
steelsight/
├── data/
│   ├── images/
│   │   ├── train/       # 70% of dataset (images)
│   │   ├── val/         # 15% of dataset (images)
│   │   └── test/        # 15% of dataset (images)
│   ├── labels/          # Corresponding YOLO format labels (.txt files)
│   ├── train_images/    # Original unzipped Kaggle images
│   └── train.csv        # Original Kaggle RLE annotations
├── docs/                # Evaluation & Latency benchmark reports
├── notebooks/
│   ├── verification_output/ # Output directory for verified bounding box images
│   └── verify_labels.py     # Script to visually check labels
├── src/
│   ├── drift_monitor/
│   │   └── drift_check.py   # Script to check for covariate drift in image brightness/contrast
│   ├── eval/
│   │   └── evaluate.py      # Script to compute mAP on the test split
│   ├── export/
│   │   ├── benchmark_latency.py # Script comparing PyTorch vs ONNX latency
│   │   └── export_onnx.py       # Script exporting YOLOv8 model to ONNX format
│   └── train/
│       ├── prepare_dataset.py   # Preprocessing script: RLE to YOLO bounding boxes & split
│       └── train_baseline.py    # Training script (YOLOv8 fine-tuning + MLflow)
├── .gitignore           # Git ignore rules
├── dataset.yaml         # YOLOv8 configuration for the dataset
├── kaggle.json          # Kaggle API credentials (ignored in git)
├── mlflow.db            # MLflow local SQLite tracking DB (ignored in git)
├── README.md            # Project documentation (this file)
└── requirements.txt     # Python dependencies
```

## Setup & Workflow

### 1. Environment & Dependencies

Make sure you have Python installed and activate your virtual environment (e.g. `.venv`). Install requirements:
```bash
pip install -r requirements.txt
```

### 2. Prepare the Dataset

Ensure `severstal-steel-defect-detection.zip` is downloaded (e.g. via Kaggle CLI) into the `data/` directory and unzipped. Then convert the RLE annotations to YOLO format and perform train/val/test splits:
```bash
python src/train/prepare_dataset.py
```

*Optional*: Visually verify the generated YOLO labels:
```bash
python notebooks/verify_labels.py
```
Check `notebooks/verification_output/` for the images with bounding boxes overlaid.

### 3. Model Training

Fine-tune the baseline YOLOv8n model on the preprocessed dataset. Logs and metrics will be tracked locally via MLflow (in `mlflow.db`):
```bash
python src/train/train_baseline.py
```

### 4. Evaluation

Evaluate the trained model on the `test` split to calculate Precision, Recall, and mAP metrics:
```bash
python src/eval/evaluate.py
```
Results are saved to `docs/baseline_results.md`.

### 5. ONNX Export & Latency Benchmarking

Export the trained model weights to the ONNX runtime format for faster CPU inference:
```bash
python src/export/export_onnx.py
```
Then, compare inference latencies between the original PyTorch model and the ONNX model:
```bash
python src/export/benchmark_latency.py
```
The comparison report is saved to `docs/latency_comparison.md`.

### 6. Drift Monitoring

Establish a baseline of image properties (brightness and contrast) from the training data, then check for covariate drift in any other folder (e.g. `val` or a new batch of data) using Population Stability Index (PSI):
```bash
python src/drift_monitor/drift_check.py --target data/images/val
```
