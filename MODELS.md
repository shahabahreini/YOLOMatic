# YOLO Model Support Documentation

This document outlines the YOLO models supported in YOLOMatic and their key features based on the official Ultralytics documentation.

## Latest Release: YOLO26 🚀

**Status**: Latest (2026-01-14)  
**Focus**: Edge deployment and end-to-end NMS-free inference

### Key Features

- **End-to-End NMS-Free Inference**: Predictions generated directly without post-processing NMS
- **43% Faster CPU Inference**: Optimized for edge computing
- **DFL Removal**: Simplifies export and expands edge compatibility
- **MuSGD Optimizer**: Hybrid SGD + Muon optimizer inspired by LLM training breakthroughs
- **ProgLoss + STAL**: Improved loss functions for better accuracy, especially on small objects
- **Task Support**: Detection, Segmentation, Classification, Pose, OBB

### Performance (COCO val2017)

| Model   | mAP 50-95 | Params (M) | FLOPs (B) |
| ------- | --------- | ---------- | --------- |
| YOLO26n | 40.9      | 2.4        | 5.4       |
| YOLO26s | 48.6      | 9.5        | 20.7      |
| YOLO26m | 53.1      | 20.4       | 68.2      |
| YOLO26l | 55.0      | 24.8       | 86.4      |
| YOLO26x | 57.5      | 55.7       | 193.9     |

---

## YOLO12 - Attention-Centric Architecture

**Status**: Community Model (2025 Early)  
**Focus**: High accuracy through attention mechanisms

⚠️ **Note**: Ultralytics recommends YOLO11 or YOLO26 for production workloads due to training instability and higher memory consumption.

### Key Features

- **Area Attention Mechanism**: Efficient large receptive field processing
- **R-ELAN**: Residual Efficient Layer Aggregation Networks for better feature aggregation
- **FlashAttention Support**: Optional memory-efficient attention implementation
- **Comprehensive Task Support**: Detection, Segmentation, Classification, Pose, OBB

### Performance (COCO val2017)

| Model   | mAP 50-95 | Speed TensorRT (ms) | Params (M) | FLOPs (B) |
| ------- | --------- | ------------------- | ---------- | --------- |
| YOLO12n | 40.6      | 1.64                | 2.6        | 6.5       |
| YOLO12s | 48.0      | 2.61                | 9.3        | 21.4      |
| YOLO12m | 52.5      | 4.86                | 20.2       | 67.5      |
| YOLO12l | 53.7      | 6.77                | 26.4       | 88.9      |
| YOLO12x | 55.2      | 11.79               | 59.1       | 199.0     |

### Improvements vs Previous Versions

- **vs YOLOv10n**: +2.1% mAP, -9% speed
- **vs YOLO11m**: +1.0% mAP, -3% speed
- **vs RT-DETRv2s**: +0.1% mAP, +42% speed

---

## YOLO11 - Optimized Efficiency & Speed

**Status**: Stable (2024-09-10)  
**Focus**: Balanced accuracy and performance

### Key Features

- **Enhanced Feature Extraction**: Improved backbone and neck architecture
- **Optimized Efficiency**: Refined architectural designs and training pipelines
- **Greater Accuracy with Fewer Parameters**: 22% fewer params than YOLOv8m with higher mAP
- **Edge Deployment Ready**: Seamlessly deployable across various environments
- **Broad Task Support**: Detection, Segmentation, Classification, Pose, OBB

### Performance (COCO val2017)

| Model   | mAP 50-95 | Speed CPU (ms) | Speed TensorRT (ms) | Params (M) | FLOPs (B) |
| ------- | --------- | -------------- | ------------------- | ---------- | --------- |
| YOLO11n | 39.5      | 56.1 ± 0.8     | 1.5 ± 0.0           | 2.6        | 6.5       |
| YOLO11s | 47.0      | 90.0 ± 1.2     | 2.5 ± 0.0           | 9.4        | 21.5      |
| YOLO11m | 51.5      | 183.2 ± 2.0    | 4.7 ± 0.1           | 20.1       | 68.0      |
| YOLO11l | 53.4      | 238.6 ± 1.4    | 6.2 ± 0.1           | 25.3       | 86.9      |
| YOLO11x | 54.7      | 462.8 ± 6.7    | 11.3 ± 0.2          | 56.9       | 194.9     |

### Task Support

- ✅ Object Detection
- ✅ Instance Segmentation
- ✅ Image Classification
- ✅ Pose Estimation
- ✅ Oriented Object Detection (OBB)

All tasks support: Inference, Validation, Training, Export

---

## Other Supported Models

### YOLOv10

- 6 size variants (N, S, M, B, L, X)
- Focus: End-to-end NMS-free inference pioneer
- mAP range: 38.5 - 54.4

### YOLOv9

- 5 variants (t, s, m, c, e)
- Extended feature optimization
- mAP range: 38.3 - 55.6

### YOLOv8

- 5 variants (n, s, m, l, x)
- Production-tested stability
- mAP range: 37.3 - 53.9

### YOLOX

- 4 variants (S, M, L, X)
- Decoupled head architecture
- mAP range: 40.5 - 51.1

### YOLO-NAS

- 3 variants (S, M, L)
- Neural Architecture Search-based
- mAP range: 47.5 - 53.2

### Roboflow Upload Notes

- YOLOmatic exposes a dedicated `yolomatic-upload` CLI for Roboflow uploads.
- Upload a **full checkpoint** such as `best.pt` or `last.pt`, not generated artifacts like `state_dict.pt`.
- YOLO26 uploads require a **size-specific** Roboflow model type such as `yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, or `yolo26x`.
- Workspace defaults can be supplied with `.env` using `ROBOFLOW_API_KEY`, `ROBOFLOW_WORKSPACE`, and `ROBOFLOW_PROJECT_IDS`.
- YOLO-NAS training is supported in YOLOmatic, but upload compatibility still depends on the Roboflow SDK's support for the selected checkpoint family.

---

## Model Selection Recommendations

### Use YOLO26 When:

- ✅ Deploying on edge devices (CPU-only)
- ✅ Need maximum inference speed
- ✅ Memory is constrained
- ✅ Working with IoT, robotics, aerial imagery

### Use YOLO12 When:

- ✅ Researching attention mechanisms
- ✅ Benchmarking new architectures
- ✅ GPU resources available for training stability
- ❌ Production deployments (not recommended)

### Use YOLO11 When:

- ✅ Need production-ready stability
- ✅ Balance between speed and accuracy is needed
- ✅ Training stability is critical
- ✅ Deploying across varied environments

### Use YOLOv10 or Earlier When:

- ✅ Legacy system compatibility required
- ✅ Specific pre-trained models unavailable in newer versions
- ✅ Consistent behavior with existing pipelines needed

---

## Supported Tasks

### Detection

Identify and localize objects in images

### Segmentation

Detect objects and delineate their boundaries

### Classification

Categorize images into predefined classes

### Pose Estimation

Detect and track keypoints on human bodies

### OBB (Oriented Object Detection)

Detect rotated objects with higher precision

---

## Model Architecture Evolution

```
YOLOv8 (2023)
    ↓
YOLOv9 (Feature optimization)
    ↓
YOLOv10 (NMS-free inference)
    ↓
YOLO11 (Efficiency focus) ← Recommended for production
    ↓
YOLO12 (Attention-centric) ← Research/benchmarking
    ↓
YOLO26 (Edge optimization) ← Latest/Recommended for edge
```

---

## Performance Comparison Summary

| Metric              | YOLO26  | YOLO12 | YOLO11 | YOLOv10  | YOLOv9   |
| ------------------- | ------- | ------ | ------ | -------- | -------- |
| mAP (best variant)  | 57.5    | 55.2   | 54.7   | 54.4     | 55.6     |
| CPU Speed (fastest) | Fastest | Slower | Fast   | Moderate | Moderate |
| Params (nano)       | 2.4M    | 2.6M   | 2.6M   | -        | 2.0M     |
| Edge Optimized      | ✅✅✅  | ❌     | ✅✅   | ✅       | ✅       |
| Production Ready    | ✅      | ❌     | ✅     | ✅       | ✅       |

---

## References

- [YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26/)
- [YOLO12 Documentation](https://docs.ultralytics.com/models/yolo12/)
- [YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/)
- [Ultralytics YOLO Hub](https://docs.ultralytics.com/models/)
