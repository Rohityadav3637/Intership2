# Latency Benchmark: PyTorch vs ONNX

**Model**: YOLOv8n fine-tuned — Severstal Steel Defect Detection  
**Device**: CPU  
**Images**: 50 randomly sampled test images  
**Image size**: 640 × 640  
**Warm-up runs**: 3 (excluded from timing)  
**Benchmarked**: 2026-07-14 09:32  

## Results

| Metric | PyTorch (ms) | ONNX (ms) | ONNX / PyTorch |
| :----- | -----------: | --------: | -------------: |
| Mean latency (ms) | 34.23 | 54.88 | 0.62× |
| Median / p50 (ms) | 33.44 | 53.41 | 0.63× |
| p95 latency (ms) | 37.20 | 66.35 | 0.56× |
| Min latency (ms) | 29.47 | 44.98 | 0.66× |
| Max latency (ms) | 53.04 | 100.82 | 0.53× |

## Summary

ONNX Runtime is **0.62× slower** than the PyTorch backend on average.

> Benchmarks run on CPU only (no GPU available). ONNX speedups are typically larger on GPU or with TensorRT.