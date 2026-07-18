# Evaluation Results — train_baseline

**Model**: C:\Users\rohit\Downloads\rohit\steelsight\runs\train_baseline\weights\best.pt  
**Split**: test  
**Evaluated**: 2026-07-17 17:43  
**Thresholds**: conf=0.001, iou=0.6 (standard YOLO defaults)  

## Per-Class Metrics

| Class | Precision | Recall | mAP50 | mAP50-95 |
| :---- | --------: | -----: | ----: | -------: |
| `defect_1` | 0.1803 | 0.0703 | 0.0632 | 0.0211 |
| `defect_2` | 1.0000 | 0.0000 | 0.0366 | 0.0113 |
| `defect_3` | 0.2360 | 0.5866 | 0.2978 | 0.1590 |
| `defect_4` | 0.1928 | 0.2500 | 0.1383 | 0.0610 |
| **ALL** | **0.4023** | **0.2267** | **0.1340** | **0.0631** |

## Comparison vs Baseline

| Metric | Baseline (YOLOv8n, 2 ep) | Improved | Delta |
| :----- | -----------------------: | -------: | ----: |
| mAP50 | 0.1270 | 0.1340 | +0.0070 (+5.5%) |
| mAP50-95 | 0.0478 | 0.0631 | +0.0153 |

## Changes Applied

- **Model**: YOLOv8n -> YOLOv8s (2x capacity)
- **Epochs**: 2 -> 150 with early stopping (patience=20)
- **Class imbalance**: Minority-class image oversampling (up to 5x)
- **Augmentation**: Mosaic=1.0, Mixup=0.15, CopyPaste=0.10, HSV jitter
- **Eval thresholds**: conf=0.001, iou=0.6 (YOLO standard defaults)