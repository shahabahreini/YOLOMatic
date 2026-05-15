# Model Support Reference

> Complete reference for all model families supported by YOLOmatic — architectures, benchmarks, task support, and selection guidance.

**YOLOmatic Version:** 4.3.0  
**Last Updated:** May 14, 2026

---

## Table of Contents

1. [Quick Comparison](#quick-comparison)
2. [Benchmark Compatibility](#benchmark-compatibility)
3. [RF-DETR](#rf-detr)
4. [YOLO26](#yolo26)
5. [YOLOv12](#yolov12)
6. [YOLO11](#yolo11)
7. [YOLOv10](#yolov10)
8. [YOLOv9](#yolov9)
9. [YOLOv8](#yolov8)
10. [YOLOX](#yolox)
11. [SAM 3.1](#sam-31)
12. [Detectron2](#detectron2)
13. [Model Selection Recommendations](#model-selection-recommendations)
14. [Supported Tasks](#supported-tasks)
15. [Architecture Evolution](#architecture-evolution)
16. [Performance Comparison Summary](#performance-comparison-summary)
17. [Roboflow Upload Notes](#roboflow-upload-notes)
18. [References](#references)

---

## Quick Comparison

| Family | Best Variant mAP | Smallest (Params) | Edge Optimized | Production Ready |
|---|---|---|---|---|
| **RF-DETR** | 60.1 (2XLarge) | 30.5M (Nano) | ✅ | ✅ |
| **YOLO26** | 57.5 (x) | 2.4M (n) | ✅✅✅ | ✅ |
| **YOLOv12** | 55.2 (x) | 2.6M (n) | ❌ | ❌ |
| **YOLO11** | 54.7 (x) | 2.6M (n) | ✅✅ | ✅ |
| **YOLOv10** | 54.4 (X) | — | ✅ | ✅ |
| **YOLOv9** | 55.6 (e) | 2.0M (t) | ✅ | ✅ |
| **YOLOv8** | 53.9 (x) | 3.2M (n) | ✅ | ✅ |
| **YOLOX** | 51.1 (X) | 9.0M (S) | ✅ | ✅ |

---

## Benchmark Compatibility

The **Benchmark & Vector Analysis** feature (`Evaluate & Monitor → Benchmark Models` or `uv run yolomatic-benchmark`) supports all Ultralytics `.pt` checkpoints. Task type (detection vs. segmentation) is auto-detected at runtime.

| Checkpoint Type | Benchmark Support |
|---|---|
| YOLO detection (`.pt`) | ✅ Bounding-box IoU metrics |
| YOLO segmentation (`.pt`) | ✅ Pixel-level mask IoU metrics |
| RF-DETR (`.pth`) | ⚠️ Not yet supported — use YOLO weights |
| SAM 3.1 (HuggingFace) | ⚠️ Not yet supported — use dedicated SAM inference |
| Detectron2 (`.pth`) | ⚠️ Not yet supported |

Annotation format: COCO JSON (`_annotations.coco.json`). Roboflow export with COCO preset is directly compatible.

---

## RF-DETR

**Status:** Supported  
**Focus:** Real-time transformer detection and segmentation  
**Trainer:** Native `rfdetr` (not Ultralytics)

RF-DETR configs use the native RF-DETR trainer. Fresh training instantiates the selected model class without a local checkpoint so RF-DETR can automatically download and cache the official pretrained weights. Fine-tuning passes the selected `.pth` checkpoint as `pretrain_weights`; resume flows pass it as `resume`.

### Detection Variants

| Model | Class | mAP 50-95 | Latency T4 TensorRT (ms) | Params (M) | Resolution | License |
|---|---|---:|---:|---:|---:|---|
| RF-DETR-Nano | `RFDETRNano` | 48.4 | 2.3 | 30.5 | 384 | Apache-2.0 |
| RF-DETR-Small | `RFDETRSmall` | 53.0 | 3.5 | 32.1 | 512 | Apache-2.0 |
| RF-DETR-Medium | `RFDETRMedium` | 54.7 | 4.4 | 33.7 | 576 | Apache-2.0 |
| RF-DETR-Large | `RFDETRLarge` | 56.5 | 6.8 | 33.9 | 704 | Apache-2.0 |
| RF-DETR-XLarge | `RFDETRXLarge` | 58.6 | 11.5 | 126.4 | 700 | PML-1.0 |
| RF-DETR-2XLarge | `RFDETR2XLarge` | 60.1 | 17.2 | 126.9 | 880 | PML-1.0 |

### Segmentation Variants

YOLOmatic also exposes RF-DETR segmentation classes from Nano through 2XLarge. These use task-specific default resolutions from the RF-DETR package and output `.pth` checkpoints discoverable by the prediction and Roboflow deployment flows.

| Model | Class | Resolution | License |
|---|---|---:|---|
| RF-DETR-Seg-Nano | `RFDETRSegNano` | 312 | Apache-2.0 |
| RF-DETR-Seg-Small | `RFDETRSegSmall` | 384 | Apache-2.0 |
| RF-DETR-Seg-Medium | `RFDETRSegMedium` | 432 | Apache-2.0 |
| RF-DETR-Seg-Large | `RFDETRSegLarge` | 504 | Apache-2.0 |
| RF-DETR-Seg-XLarge | `RFDETRSegXLarge` | 624 | Apache-2.0 |
| RF-DETR-Seg-2XLarge | `RFDETRSeg2XLarge` | 768 | Apache-2.0 |

RF-DETR Plus (XLarge, 2XLarge) detection models require the `rfdetr[plus]` dependency extra and use PML-1.0 model licensing.

---

## YOLO26

**Status:** Latest (2026-01-14) 🚀  
**Focus:** Edge deployment and end-to-end NMS-free inference

### Key Features

- **End-to-End NMS-Free Inference** — predictions generated directly without post-processing NMS
- **43% Faster CPU Inference** — optimized for edge computing
- **DFL Removal** — simplifies export and expands edge compatibility
- **MuSGD Optimizer** — hybrid SGD + Muon optimizer inspired by LLM training breakthroughs
- **ProgLoss + STAL** — improved loss functions for better accuracy, especially on small objects
- **Task Support** — Detection, Segmentation, Classification, Pose, OBB

### Detection Performance (COCO val2017)

| Model | mAP 50-95 | Speed T4 TensorRT (ms) | Params (M) | FLOPs (B) |
|---|---:|---:|---:|---:|
| YOLO26n | 40.9 | 1.7 ± 0.0 | 2.4 | 5.4 |
| YOLO26s | 48.6 | 2.5 ± 0.0 | 9.5 | 20.7 |
| YOLO26m | 53.1 | 4.7 ± 0.1 | 20.4 | 68.2 |
| YOLO26l | 55.0 | 6.2 ± 0.2 | 24.8 | 86.4 |
| YOLO26x | 57.5 | 11.8 ± 0.2 | 55.7 | 193.9 |

> CPU ONNX speed benchmarks for YOLO26 have not been officially published. T4 TensorRT figures shown above.

### Segmentation Performance (COCO val2017)

| Model | mAP box 50-95 | Speed CPU ONNX (ms) | Params (M) | FLOPs (B) |
|---|---:|---:|---:|---:|
| YOLO26n-seg | 33.9 | 53.3 ± 0.5 | 2.7 | 9.1 |
| YOLO26s-seg | 40.0 | 118.4 ± 0.9 | 10.4 | 34.2 |
| YOLO26m-seg | 44.1 | 328.2 ± 2.4 | 23.6 | 121.5 |
| YOLO26l-seg | 45.5 | 387.0 ± 3.7 | 28.0 | 139.8 |
| YOLO26x-seg | 47.0 | 787.0 ± 6.8 | 62.8 | 313.5 |

> Mask mAP benchmarks for YOLO26-seg have not been officially published separately.

---

## YOLOv12

**Status:** Community Model (2025 Early)  
**Focus:** High accuracy through attention mechanisms

⚠️ **Note:** Ultralytics recommends YOLO11 or YOLO26 for production workloads due to training instability and higher memory consumption.

### Key Features

- **Area Attention Mechanism** — efficient large receptive field processing
- **R-ELAN** — Residual Efficient Layer Aggregation Networks for better feature aggregation
- **FlashAttention Support** — optional memory-efficient attention implementation
- **Task Support** — Detection, Segmentation, Classification, Pose, OBB

### Detection Performance (COCO val2017)

| Model | mAP 50-95 | Speed TensorRT (ms) | Params (M) | FLOPs (B) |
|---|---:|---:|---:|---:|
| YOLO12n | 40.6 | 1.64 | 2.6 | 6.5 |
| YOLO12s | 48.0 | 2.61 | 9.3 | 21.4 |
| YOLO12m | 52.5 | 4.86 | 20.2 | 67.5 |
| YOLO12l | 53.7 | 6.77 | 26.4 | 88.9 |
| YOLO12x | 55.2 | 11.79 | 59.1 | 199.0 |

### Improvements vs Previous Versions

- **vs YOLOv10n:** +2.1% mAP, -9% speed
- **vs YOLO11m:** +1.0% mAP, -3% speed
- **vs RT-DETRv2s:** +0.1% mAP, +42% speed

---

## YOLO11

**Status:** Stable (2024-09-10)  
**Focus:** Balanced accuracy and performance

### Key Features

- **Enhanced Feature Extraction** — improved backbone and neck architecture
- **Optimized Efficiency** — refined architectural designs and training pipelines
- **Greater Accuracy with Fewer Parameters** — 22% fewer params than YOLOv8m with higher mAP
- **Edge Deployment Ready** — seamlessly deployable across various environments
- **Task Support** — Detection, Segmentation, Classification, Pose, OBB

### Detection Performance (COCO val2017)

| Model | mAP 50-95 | Speed CPU (ms) | Speed TensorRT (ms) | Params (M) | FLOPs (B) |
|---|---:|---:|---:|---:|---:|
| YOLO11n | 39.5 | 56.1 ± 0.8 | 1.5 ± 0.0 | 2.6 | 6.5 |
| YOLO11s | 47.0 | 90.0 ± 1.2 | 2.5 ± 0.0 | 9.4 | 21.5 |
| YOLO11m | 51.5 | 183.2 ± 2.0 | 4.7 ± 0.1 | 20.1 | 68.0 |
| YOLO11l | 53.4 | 238.6 ± 1.4 | 6.2 ± 0.1 | 25.3 | 86.9 |
| YOLO11x | 54.7 | 462.8 ± 6.7 | 11.3 ± 0.2 | 56.9 | 194.9 |

### Segmentation Performance (COCO val2017)

| Model | mAP box 50-95 | mAP mask 50-95 | Params (M) | FLOPs (B) |
|---|---:|---:|---:|---:|
| YOLO11n-seg | 38.9 | 32.0 | 2.9 | 10.4 |
| YOLO11s-seg | 46.6 | 38.8 | 10.1 | 35.5 |
| YOLO11m-seg | 51.5 | 41.5 | 22.4 | 123.3 |
| YOLO11l-seg | 53.4 | 42.9 | 27.6 | 142.2 |
| YOLO11x-seg | 54.7 | 43.8 | 62.1 | 319.9 |

---

## YOLOv10

**Status:** Stable  
**Focus:** End-to-end NMS-free inference pioneer

| Model | mAP 50-95 | FLOPs (G) | Latency (ms) |
|---|---:|---:|---:|
| YOLOv10-N | 38.5 | 6.7 | 1.84 |
| YOLOv10-S | 46.3 | 21.6 | 2.49 |
| YOLOv10-M | 51.1 | 59.1 | 4.74 |
| YOLOv10-B | 52.5 | 92.0 | 5.74 |
| YOLOv10-L | 53.2 | 120.3 | 7.28 |
| YOLOv10-X | 54.4 | 160.4 | 10.70 |

**Tasks:** Detection only.

---

## YOLOv9

**Status:** Stable  
**Focus:** Extended feature optimization

| Model | mAP 50-95 | mAP 50 | Params (M) | FLOPs (B) |
|---|---:|---:|---:|---:|
| YOLOv9t | 38.3 | 53.1 | 2.0 | 7.7 |
| YOLOv9s | 46.8 | 63.4 | 7.2 | 26.7 |
| YOLOv9m | 51.4 | 68.1 | 20.1 | 76.8 |
| YOLOv9c | 53.0 | 70.2 | 25.5 | 102.8 |
| YOLOv9e | 55.6 | 72.8 | 58.1 | 192.5 |

**Tasks:** Detection + Segmentation (t/s/m/c/e + seg variants).

---

## YOLOv8

**Status:** Stable  
**Focus:** Production-tested stability

### Detection Performance

| Model | mAP 50-95 | Speed CPU ONNX (ms) | Params (M) | FLOPs (B) |
|---|---:|---:|---:|---:|
| YOLOv8n | 37.3 | 80.4 | 3.2 | 8.7 |
| YOLOv8s | 44.9 | 128.4 | 11.2 | 28.6 |
| YOLOv8m | 50.2 | 234.7 | 25.9 | 78.9 |
| YOLOv8l | 52.9 | 375.2 | 43.7 | 165.2 |
| YOLOv8x | 53.9 | 479.1 | 68.2 | 257.8 |

### Segmentation Performance

| Model | mAP box 50-95 | mAP mask 50-95 | Params (M) | FLOPs (B) |
|---|---:|---:|---:|---:|
| YOLOv8n-seg | 36.7 | 30.5 | 3.4 | 12.6 |
| YOLOv8s-seg | 44.6 | 36.8 | 11.8 | 42.6 |
| YOLOv8m-seg | 49.9 | 40.8 | 27.3 | 110.2 |
| YOLOv8l-seg | 52.3 | 42.6 | 46.0 | 220.5 |
| YOLOv8x-seg | 53.4 | 43.4 | 71.8 | 344.1 |

**Tasks:** Detection, Segmentation, Classification, Pose, OBB.

---

## YOLOX

**Status:** Stable  
**Focus:** Decoupled head architecture

| Model | mAP 50-95 | Params (M) | FLOPs (G) | FPS |
|---|---|---:|---:|---:|
| YOLOX-S | 40.5% | 9.0 | 26.8 | 102 |
| YOLOX-M | 46.9% | 25.3 | 73.8 | 81 |
| YOLOX-L | 49.7% | 54.2 | 155.6 | 69 |
| YOLOX-X | 51.1% | 99.1 | 281.9 | 58 |

**Tasks:** Detection only.

---

## SAM 3.1

**Status:** Supported  
**Focus:** Open-vocabulary segmentation (Segment Anything Model)  
**Provider:** Meta / HuggingFace (gated model)

| Model | HuggingFace ID | Params (B) | Description |
|---|---|---:|---|
| SAM 3.1 | `facebook/sam3.1` | 0.873 | Object Multiplex — 7× faster multi-object throughput |
| SAM 3 | `facebook/sam3` | 0.848 | Base model — predecessor without Object Multiplex |

**Capabilities:**
- Auto-segmentation (segment everything)
- Text-prompted open-vocabulary segmentation
- Box-prompted segmentation (from YOLO detections)
- Fine-tuning on custom COCO-format datasets

**Requirements:** HuggingFace token (`HF_TOKEN`) for downloading gated weights.

**Outputs:** PNG overlays, COCO JSON annotations, YOLO segmentation `.txt` files.

---

## Detectron2

**Status:** Supported  
**Focus:** COCO-format detection and instance segmentation  
**Provider:** Facebook Research

| Model | Task | Backbone |
|---|---|---|
| Faster R-CNN R50-FPN 3x | Detection | ResNet-50 + FPN |
| RetinaNet R50-FPN 3x | Detection | ResNet-50 + FPN |
| Mask R-CNN R50-FPN 3x | Segmentation | ResNet-50 + FPN |

Detectron2 training uses COCO-format annotations and integrates with ClearML for experiment tracking.

---

## Model Selection Recommendations

### Use RF-DETR When:

- ✅ Need highest possible mAP (60.1 with 2XLarge)
- ✅ Transformer-based real-time detection is desired
- ✅ Working with high-resolution inputs
- ✅ Server-side deployment with adequate compute

### Use YOLO26 When:

- ✅ Deploying on edge devices (CPU-only)
- ✅ Need maximum inference speed
- ✅ Memory is constrained
- ✅ Working with IoT, robotics, aerial imagery

### Use YOLOv12 When:

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

### Use SAM 3.1 When:

- ✅ Need open-vocabulary segmentation (no predefined classes)
- ✅ Upgrading detection datasets to segmentation masks
- ✅ Exploratory annotation and pseudo-label generation
- ✅ Text-prompted concept segmentation

### Use Detectron2 When:

- ✅ Working with Faster R-CNN, RetinaNet, or Mask R-CNN architectures
- ✅ Need COCO-format training with established baselines
- ✅ Integration with the Detectron2 ecosystem

---

## Supported Tasks

| Task | Description | Supported Families |
|---|---|---|
| **Detection** | Identify and localize objects in images | RF-DETR, YOLO26, YOLOv12, YOLO11, YOLOv10, YOLOv9, YOLOv8, YOLOX, Detectron2 |
| **Segmentation** | Detect objects and delineate their boundaries | RF-DETR, YOLO26, YOLOv12, YOLO11, YOLOv9, YOLOv8, SAM 3.1, Detectron2 |
| **Classification** | Categorize images into predefined classes | YOLO26, YOLO11, YOLOv8 |
| **Pose Estimation** | Detect and track keypoints on human bodies | YOLO26, YOLO11, YOLOv8 |
| **OBB** | Detect rotated objects with higher precision | YOLO26, YOLO11, YOLOv8 |
| **Open-Vocab Seg** | Segment objects by text concept (no predefined classes) | SAM 3.1 |

---

## Architecture Evolution

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
YOLO26 (Edge optimization) ← Latest / Recommended for edge
```

Parallel families:
- **RF-DETR** — transformer-based, highest mAP
- **SAM 3.1** — open-vocabulary segmentation
- **Detectron2** — Faster R-CNN / Mask R-CNN baselines

---

## Performance Comparison Summary

| Metric | RF-DETR | YOLO26 | YOLO12 | YOLO11 | YOLOv10 | YOLOv9 |
|---|---|---|---|---|---|---|
| mAP (best variant) | 60.1 | 57.5 | 55.2 | 54.7 | 54.4 | 55.6 |
| CPU Speed (fastest) | — | Fastest | Slower | Fast | Moderate | Moderate |
| Params (smallest) | 30.5M | 2.4M | 2.6M | 2.6M | — | 2.0M |
| Edge Optimized | ❌ | ✅✅✅ | ❌ | ✅✅ | ✅ | ✅ |
| Production Ready | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |

---

## Roboflow Upload Notes

- YOLOmatic exposes a dedicated `yolomatic-upload` CLI for Roboflow uploads.
- Upload a **full checkpoint** such as `best.pt` or `last.pt`, not generated artifacts like `state_dict.pt`.
- RF-DETR `.pth` checkpoints deploy through RF-DETR's `deploy_to_roboflow(...)` flow and require a workspace, project ID, and project version.
- YOLO26 uploads require a **size-specific** Roboflow model type such as `yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, or `yolo26x`.
- Workspace defaults can be supplied with `.env` using `ROBOFLOW_API_KEY`, `ROBOFLOW_WORKSPACE`, and `ROBOFLOW_PROJECT_IDS`.
- YOLO-NAS is deprecated in this build because SuperGradients conflicts with RF-DETR training dependencies.

---

## References

- [YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26/)
- [YOLO12 Documentation](https://docs.ultralytics.com/models/yolo12/)
- [YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/)
- [YOLOv10 Documentation](https://docs.ultralytics.com/models/yolov10/)
- [YOLOv9 Documentation](https://docs.ultralytics.com/models/yolov9/)
- [YOLOv8 Documentation](https://docs.ultralytics.com/models/yolov8/)
- [Ultralytics Model Hub](https://docs.ultralytics.com/models/)
- [RF-DETR GitHub](https://github.com/roboflow/rf-detr)
- [SAM 3.1 on HuggingFace](https://huggingface.co/facebook/sam3.1)
- [Detectron2 Model Zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md)
