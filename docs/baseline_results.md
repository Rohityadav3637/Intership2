# Baseline Evaluation Results

**Model**: YOLOv8n fine-tuned on Severstal Steel Defect Detection  
**Weights**: `C:\Users\rohit\Downloads\rohit\steelsight\runs\train_baseline\weights\best.pt`  
**Split**: test  
**Evaluated**: 2026-07-17 16:39  

## Per-Class Metrics

| Class | Precision | Recall | mAP50 | mAP50-95 |
| :---- | --------: | -----: | ----: | -------: |
| `defect_1` | 0.1665 | 0.0547 | 0.0567 | 0.0218 |
| `defect_2` | 1.0000 | 0.0000 | 0.0363 | 0.0120 |
| `defect_3` | 0.2308 | 0.5479 | 0.2825 | 0.1563 |
| `defect_4` | 0.2053 | 0.2333 | 0.1327 | 0.0598 |
| **ALL** | **0.4006** | **0.2090** | **0.1270** | **0.0625** |

## Summary

| Metric | Value |
| :----- | ----: |
| mAP50 (all classes) | **0.1270** |
| mAP50-95 (all classes) | **0.0625** |

> **Note**: Results are from a 2-epoch baseline training run on CPU.
> Further training and hyperparameter tuning is expected to improve these scores.