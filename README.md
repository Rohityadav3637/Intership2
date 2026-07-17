# SteelSight – AI-Powered Visual Quality Inspection for Sheet-Metal Manufacturing

SteelSight is a computer vision system that fine-tunes a YOLOv8 object detection model to identify and localize surface defects on steel sheet-metal parts, and wraps it in a full inspection pipeline — from raw data to a live, demo-ready dashboard.

## Business Problem

A sheet-metal manufacturer producing 50,000 parts/day currently relies on 12 human inspectors with ~92% accuracy — meaning roughly 4,000 defective parts ship every day, driving an estimated ₹8L/month in warranty returns. SteelSight automates visual inspection to catch what manual review misses, flags low-confidence cases for human review instead of guessing, and logs every inspection for traceability.

## Features

- Steel surface defect detection using YOLOv8
- Automatic dataset preprocessing (RLE mask → YOLO bounding box conversion)
- Model evaluation using Precision, Recall, and mAP
- ONNX model export with latency benchmarking (edge deployment readiness)
- FastAPI inference server with prediction logging (SQLite)
- Data drift monitoring (PSI-based, on image brightness/contrast)
- MLflow experiment tracking
- Live web dashboard: image upload, defect visualization, uncertain-prediction review queue

## Architecture

```
ml_pipeline/  --(trained model)-->  backend/  --(REST API)-->  frontend/
(data prep,                      (FastAPI +                  (HTML/CSS/JS
 training,                        SQLite                      dashboard:
 export,                          logging)                    upload, review,
 drift check)                                                 KPIs)
```

The ML pipeline produces a trained/exported model. The backend loads it and serves predictions over a REST API, logging every inspection to a database. The frontend calls that API to let a user upload parts for inspection, review low-confidence predictions, and monitor inspection stats in real time.

## Project Structure

```
steelsight/
│
├── backend/                  # API & inference server
│   ├── main.py                # FastAPI server: /predict, /health, /predictions
│   ├── check_db.py            # DB inspection utility
│   └── inference_log.db       # SQLite database of logged predictions
│
├── frontend/                  # Live dashboard (HTML/CSS/JS)
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── ml_pipeline/                # Model training & deployment prep
│   ├── 1_prepare_data.py       # RLE → YOLO label conversion, train/val/test split
│   ├── 2_train_model.py        # YOLOv8 fine-tuning, MLflow-logged
│   ├── 3_evaluate.py           # Precision/Recall/mAP evaluation
│   ├── 4_export_onnx.py        # PyTorch → ONNX export
│   ├── 5_benchmark.py          # Latency comparison (PyTorch vs ONNX)
│   └── 6_drift_monitor.py      # PSI-based input drift detection
│
├── data/                       # Raw and processed dataset (not tracked in git)
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── train.csv
│
├── notebooks/                  # Exploration & label verification notebooks
├── docs/                       # Evaluation results, latency reports
├── dataset.yaml                 # YOLO dataset config
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Preparation

```bash
python ml_pipeline/1_prepare_data.py
```

## Model Training

```bash
python ml_pipeline/2_train_model.py
```

## Model Evaluation

```bash
python ml_pipeline/3_evaluate.py
```

## ONNX Export & Latency Benchmark

```bash
python ml_pipeline/4_export_onnx.py
python ml_pipeline/5_benchmark.py
```

## Drift Monitoring

```bash
python ml_pipeline/6_drift_monitor.py --target data/images/val
```

## Running the Full Application

**1. Start the backend** (must be running first):
```bash
python -m uvicorn backend.main:app --reload --port 8000
```

**2. Start the frontend** (in a separate terminal):
```bash
cd frontend
python -m http.server 5500
```
Then open `http://localhost:5500` in your browser. The backend must remain running on port 8000 for the dashboard to work.

## Model Results

See `docs/baseline_results.md` for full per-class Precision/Recall/mAP, and `docs/latency_comparison.md` for PyTorch vs. ONNX inference latency benchmarks.

## Tech Stack

- **ML:** YOLOv8 (Ultralytics), PyTorch, ONNX, ONNX Runtime, MLflow
- **Backend:** FastAPI, SQLAlchemy, SQLite, Uvicorn
- **Frontend:** HTML, CSS, JavaScript (no framework)

## Known Limitations / Next Steps

- Active learning retraining loop is logged (uncertain predictions flagged and queryable) but the "Approve" / "Flag for Retraining" actions in the dashboard are UI-only and not yet wired to a retraining pipeline
- Drift monitor runs as a standalone script; not yet surfaced in the live dashboard
- Edge deployment currently benchmarked via ONNX Runtime; TensorRT/OpenVINO conversion not yet implemented

## Contributors

- Rohit Yadav