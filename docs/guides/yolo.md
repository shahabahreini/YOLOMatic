---
description: Practical YOLOmatic workflow guide for model selection, training configuration, exports, deployment, SAM, augmentation, and NDJSON conversion.
---

# YOLOmatic Integration Guide

> Practical workflow guide — model selection, deployment scenarios, training configuration, export options, and operational reference.

**YOLOmatic Version:** 5.0.0
**Last Updated:** May 28, 2026
**Status:** Fully Supported and Integrated

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [CLI Commands](#cli-commands)
3. [Model Selection Guide](#model-selection-guide)
4. [Feature Comparison](#feature-comparison)
5. [Training Configuration](#training-configuration)
6. [Performance Benchmarks](#performance-benchmarks)
7. [Deployment Scenarios](#deployment-scenarios)
8. [Export Options](#export-options)
9. [SAM 3.1 Workflows](#sam-31-workflows)
10. [Offline Augmentation](#offline-augmentation)
11. [NDJSON Conversion](#ndjson-conversion)
12. [Troubleshooting](#troubleshooting)
13. [Summary Table](#summary-table)
14. [Additional Resources](#additional-resources)

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/shahabahreini/YOLOMatic.git
cd YOLOMatic

# Sync the project environment with uv
uv sync
```

After installation you can run the CLI via the `yolomatic` entrypoints managed by `uv`.

**Platform notes:**

- **Windows:** prefer `.venv\Scripts\python.exe -m pip ...` for manual PyTorch/CUDA repairs instead of `uv run pip ...`.
- **Linux:** YOLOmatic prepares CUDA/cuDNN runtime library paths from the active `.venv` when needed.
- **macOS:** use CPU or MPS-capable PyTorch builds; NVIDIA CUDA is not expected.

### Optional Roboflow Setup

```bash
cp .env.example .env       # Linux / macOS
# Copy-Item .env.example .env   # Windows PowerShell
```

Then fill in:

```env
ROBOFLOW_API_KEY=
ROBOFLOW_WORKSPACE=
ROBOFLOW_PROJECT_IDS=
```

Use your Roboflow **workspace slug** for `ROBOFLOW_WORKSPACE`, not a project name or display label.

---

## CLI Commands

| Command | Purpose |
|---|---|
| `uv run yolomatic` | Launch the main interactive TUI |
| `uv run yolomatic-train` | Train from a saved config (YOLO, RF-DETR, SAM, or Detectron2) |
| `uv run yolomatic-predict` | Run YOLO/RF-DETR prediction workflows |
| `uv run yolomatic-sam` | Run SAM 3.1 segmentation inference |
| `uv run yolomatic-convert` | Convert Labelbox NDJSON to YOLO/COCO |
| `uv run yolomatic-benchmark` | Benchmark trained checkpoints with interactive HTML report |
| `uv run yolomatic-upload` | Upload or deploy trained checkpoints to Roboflow |
| `uv run yolomatic-tensorboard` | Launch TensorBoard for discovered runs |
| `uv run bump patch\|minor\|major\|VERSION` | Update the package version |

### First Run

```bash
# Start the interactive model selection
uv run yolomatic

# Follow the prompts to:
# 1. Select a model (recommend YOLO26)
# 2. Choose variant (n/s/m/l/x)
# 3. Select dataset
# 4. Review configuration

# Start training
uv run yolomatic-train

# Run predictions with the TUI
uv run yolomatic-predict

# Or run single-image prediction directly
uv run yolomatic-predict --mode single --weight runs/segment/train/weights/best.pt --source /path/to/image.jpg

# Or run batch folder prediction using multiprocessing
uv run yolomatic-predict --mode folder --weight runs/segment/train/weights/best.pt --source datasets/my_dataset/test/images --workers 4

# Upload a trained checkpoint to Roboflow
uv run yolomatic-upload

# Or upload directly with explicit arguments
uv run yolomatic-upload --weight runs/segment/train2/weights/best.pt --workspace your-workspace-slug --project-ids vegmask --model-type yolo26l --model-name train2-best

# Monitor training logs
uv run yolomatic-tensorboard

# Benchmark trained models
uv run yolomatic-benchmark

# Run SAM 3.1 segmentation
uv run yolomatic-sam

# Convert Labelbox NDJSON to YOLO/COCO
uv run yolomatic-convert
```

### Training Runtime Notes

- If multiple YAML configs exist in `configs/`, `yolomatic-train` opens a TUI selector.
- If ClearML is not configured, training prompts whether to continue without ClearML or cancel.
- If CUDA is requested but PyTorch cannot use it, training prompts whether to repair the environment, continue on CPU, or cancel.
- The smart training router automatically dispatches configs to the correct trainer (YOLO, RF-DETR, SAM 3.1, or Detectron2).

### Upload Tips

- Choose a full checkpoint such as `best.pt` or `last.pt`.
- Generated artifacts like `state_dict.pt` are not uploadable checkpoints.
- YOLO26 uploads require a size-specific Roboflow type such as `yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, or `yolo26x`.
- YOLO-NAS configs no longer train because SuperGradients conflicts with RF-DETR's modern training dependency stack.

### Automatic Post-Training Upload

Add this block to your training YAML:

```yaml
roboflow:
  upload: true
  weight: "best.pt"
```

---

## Model Selection Guide

### YOLO26 — Latest Edge Optimizer

**Best For:** Edge devices, CPU inference, real-time mobile apps

**Advantages:**

- 43% faster CPU inference
- Higher accuracy on small objects
- Smaller model sizes
- NMS-free end-to-end inference
- Perfect for IoT and robotics

**Performance:**

| Size | mAP | Params | Speed (GPU T4) |
|---|---|---|---|
| Nano | 40.9 | 2.4M | 1.7ms |
| Small | 48.6 | 9.5M | 2.5ms |
| Medium | 53.1 | 20.4M | 4.7ms |
| Large | 55.0 | 24.8M | 6.2ms |
| XLarge | 57.5 | 55.7M | 11.8ms |

**When to Choose:**

- Deploying on edge devices
- Need fast CPU inference
- Memory constrained
- Mobile/IoT applications

---

### YOLOv12 — Attention-Centric Research Model

**Best For:** Research, benchmarking, attention mechanism studies

**Key Features:**

- Area Attention Mechanism
- R-ELAN architecture
- FlashAttention support (optional)
- Highest YOLO accuracy potential

**Performance:**

| Size | mAP | Params | Speed (GPU T4) |
|---|---|---|---|
| Nano | 40.6 | 2.6M | 1.64ms |
| Small | 48.0 | 9.3M | 2.61ms |
| Medium | 52.5 | 20.2M | 4.86ms |
| Large | 53.7 | 26.4M | 6.77ms |
| XLarge | 55.2 | 59.1M | 11.79ms |

**Warnings:**

- Training instability
- Higher memory consumption
- Slower CPU throughput
- Not recommended for production

---

### YOLO11 — Production-Ready Standard

**Best For:** Production systems, enterprise deployments, balanced performance

**Advantages:**

- Proven stability
- 22% fewer params than YOLOv8m (with higher accuracy)
- Fast CPU inference
- All task types supported
- Excellent documentation

**Performance:**

| Size | mAP | Params | Speed (CPU) | Speed (GPU T4) |
|---|---|---|---|---|
| Nano | 39.5 | 2.6M | 56ms | 1.5ms |
| Small | 47.0 | 9.4M | 90ms | 2.5ms |
| Medium | 51.5 | 20.1M | 183ms | 4.7ms |
| Large | 53.4 | 25.3M | 239ms | 6.2ms |
| XLarge | 54.7 | 56.9M | 463ms | 11.3ms |

---

### RF-DETR — Transformer Detection Leader

**Best For:** Maximum accuracy, high-resolution inputs, server-side deployment

**Advantages:**

- Highest mAP (60.1 with 2XLarge)
- Real-time transformer detection
- Variable resolution support (384880px)
- Both detection and segmentation variants

**Performance:**

| Size | mAP | Params | Latency (T4) |
|---|---|---|---|
| Nano | 48.4 | 30.5M | 2.3ms |
| Small | 53.0 | 32.1M | 3.5ms |
| Large | 56.5 | 33.9M | 6.8ms |
| 2XLarge | 60.1 | 126.9M | 17.2ms |

---

### SAM 3.1 — Open-Vocabulary Segmentation

**Best For:** Annotation, pseudo-label generation, concept-based segmentation

**Key Features:**

- Text-prompted segmentation (no predefined classes)
- Auto-segment everything mode
- Box-prompted from YOLO detections
- Fine-tunable on custom datasets

---

## Feature Comparison

### Computer Vision Tasks

| Task | YOLO26 | YOLO12 | YOLO11 | RF-DETR | SAM 3.1 | Detectron2 |
|---|---|---|---|---|---|---|
| Detection | Yes | Yes | Yes | Yes | — | Yes |
| Segmentation | Yes | Yes | Yes | Yes | Yes | Yes |
| Classification | Yes | Yes | Yes | — | — | — |
| Pose Estimation | Yes | Yes | Yes | — | — | — |
| OBB | Yes | Yes | Yes | — | — | — |
| Open-Vocab Seg | — | — | — | — | Yes | — |

### Architecture Comparison

| Aspect | YOLO26 | YOLO12 | YOLO11 | RF-DETR |
|---|---|---|---|---|
| **Base Architecture** | End-to-end | Attention-centric | CNN-based | Transformer |
| **NMS** | None (end-to-end) | Traditional | Traditional | None |
| **CPU Performance** | 5/5 | 2/5 | 3/5 | 2/5 |
| **GPU Performance** | 4/5 | 4/5 | 4/5 | 5/5 |
| **Training Stability** | Excellent | Variable | Excellent | Excellent |
| **Memory Usage** | Low | High | Moderate | High |

### Feature Set

| Feature | YOLO26 | YOLO12 | YOLO11 | RF-DETR |
|---|---|---|---|---|
| **DFL** | Removed | Present | Present | — |
| **NMS-Free Inference** | Yes | No | No | Yes |
| **MuSGD Optimizer** | Yes | No | No | No |
| **Area Attention** | No | Yes | No | No |
| **R-ELAN** | No | Yes | No | No |
| **FlashAttention** | No | Yes (optional) | No | No |
| **Multi-scale Proto** | Yes | No | No | No |
| **RLE for Pose** | Yes | No | No | No |

---

## Training Configuration

### Default Training Parameters (applies to all YOLO models)

```yaml
training:
  epochs: 150
  imgsz: 640
  batch: 16
  cache: false
  workers: 8
  label_smoothing: 0.1
  close_mosaic: 50

  # Augmentation
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  fliplr: 0.5
  device: auto
```

Use `cache: ram` only when the decoded dataset comfortably fits in available
memory. YOLOmatic disables `cache: disk` because Ultralytics writes an
uncompressed `.npy` beside each image; these files can consume several times the
space used by the compressed source dataset. Verified `.npy` image caches from
older runs are removed automatically when a dataset is selected for training.

### Model-Specific Recommendations

**YOLO26** — Edge deployment:

```yaml
batch: 8-16        # Reduce for edge
epochs: 100-150
imgsz: 480-640      # Smaller for edge
device: "cpu"       # Test on CPU
```

**YOLOv12** — Research configuration:

```yaml
batch: 32-64        # Larger for attention
epochs: 150-200
imgsz: 640
device: "cuda"      # Requires GPU
# Note: May have training instability
```

**YOLO11** — Production configuration:

```yaml
batch: 16-32
epochs: 100-150
imgsz: 640
device: "auto"      # Use best available
```

**RF-DETR** — High-accuracy configuration:

```yaml
# RF-DETR uses its own trainer with different parameters
epochs: 50-100
batch_size: 4-8     # Transformer models need more memory
lr: 1e-4
grad_accum_steps: 4
```

---

## Performance Benchmarks

### Accuracy (mAP on COCO val2017)

**Best Overall:**

- RF-DETR-2XLarge: 60.1 mAP
- YOLO26x: 57.5 mAP
- YOLOv9e: 55.6 mAP

**Best Nano Model:**

- YOLO26n: 40.9 mAP
- YOLO12n: 40.6 mAP
- YOLO11n: 39.5 mAP

### Speed (T4 TensorRT)

**Fastest:**

1. YOLO11n: 1.5ms
2. YOLO12n: 1.64ms
3. YOLO26n: 1.7ms
4. RF-DETR-Nano: 2.3ms

### Parameter Efficiency

**Smallest Model:**

- YOLOv9t: 2.0M params
- YOLO26n: 2.4M params
- YOLO11n / YOLO12n: 2.6M params

**Largest Model:**

- RF-DETR-2XLarge: 126.9M params
- YOLOX-X: 99.1M params
- YOLOv8x: 68.2M params

---

## Deployment Scenarios

### Scenario 1: Mobile App (Phone/Tablet)

**Recommended:** YOLO26n or YOLO26s

```python
from ultralytics import YOLO
model = YOLO("yolo26n.pt")
# Expected: 40-50ms per frame on modern phones
```

### Scenario 2: Edge Device (RPi, Jetson)

**Recommended:** YOLO26s or YOLO26m

```python
model = YOLO("yolo26s.pt")
# CPU inference optimized
```

### Scenario 3: Enterprise Server

**Recommended:** YOLO11l or YOLO11x

```python
model = YOLO("yolo11l.pt")
# GPU-accelerated inference
```

### Scenario 4: Maximum Accuracy Server

**Recommended:** RF-DETR-Large or RF-DETR-2XLarge

```python
from rfdetr import RFDETRLarge
model = RFDETRLarge()
# Transformer-based, highest mAP
```

### Scenario 5: Research Lab

**Recommended:** YOLO12m or YOLO12l

```python
model = YOLO("yolo12m.pt")
# Attention-based for study
```

### Scenario 6: Annotation Pipeline

**Recommended:** SAM 3.1

```bash
uv run yolomatic-sam
# Auto-segment or text-prompted segmentation
```

---

## Export Options

All YOLO models support export to multiple formats:

```python
from ultralytics import YOLO
model = YOLO("yolo26n.pt")

# ONNX (universal, recommended)
model.export(format="onnx")

# TensorRT (NVIDIA GPUs)
model.export(format="engine")

# CoreML (Apple devices)
model.export(format="coreml")

# TensorFlow Lite (Mobile)
model.export(format="tflite")

# OpenVINO (Intel)
model.export(format="openvino")
```

---

## SAM 3.1 Workflows

### Inference Modes

```bash
uv run yolomatic-sam
```

| Mode | Use Case |
|---|---|
| **Auto** | Exploratory annotation, pseudo-label generation, quality inspection |
| **Text-prompted** | Targeted extraction of specific object classes across image batches |
| **Box-prompted** | Upgrading existing YOLO detection datasets to segmentation masks |

### Fine-Tuning

Configure SAM 3.1 fine-tuning through the TUI:

1. Select **Configure Model** → **SAM 3.1**
2. Choose base model (SAM 3.1 or SAM 3)
3. Select COCO-format dataset
4. Configure fine-tuning strategy (decoder-only recommended)

The trainer freezes the ViT image encoder and trains only the mask decoder and prompt encoder using the HuggingFace Trainer API.

---

## Offline Augmentation

Access from the TUI under **Augment Dataset**:

| Feature | Description |
|---|---|
| **Reusable profiles** | Create, edit, clone, delete — stored as YAML in `configs/augmentation_profiles/` |
| **20+ transforms** | Geometric, color, blur, noise, weather — each with per-transform probability and parameters |
| **Multiplier** | Generate N augmented copies per source image |
| **Split redistribution** | Pool all images, then redistribute into train/val/test with configurable ratios |
| **Output formats** | YOLO Detection, YOLO Segmentation, COCO JSON |

---

## NDJSON Conversion

YOLOmatic can convert **Labelbox or Ultralytics-platform NDJSON** exports into
YOLO or COCO formats, automatically downloading images from the provided URLs.
Ultralytics pose exports support explicit YOLO Pose and COCO Pose targets.

### Features
- **Concurrent downloads** — uses a thread pool for fast image retrieval.
- **Task detection** — extracts bounding boxes, polygons, and Ultralytics pose keypoints.
- **YOLO/COCO support** — generates valid detection, segmentation, or pose datasets.
- **Public/Presigned URLs** — supports direct image downloads from Labelbox-hosted or presigned asset URLs.

### Usage
1. Export your project from Labelbox or Ultralytics Platform in **NDJSON** format.
2. Select **Convert Dataset Format** from the YOLOmatic main menu.
3. Provide the path to the `.ndjson` file.
4. Choose the target format, including YOLO Pose or COCO Pose when applicable, and output directory.

---

## Troubleshooting

### CUDA Requested but PyTorch Cannot Use the GPU

YOLOmatic offers an automatic CUDA repair flow during training. Manual Windows repair:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio
.\.venv\Scripts\python.exe -m pip install --no-cache-dir --force-reinstall torch torchvision torchaudio "numpy>=1.24.4" --index-url https://download.pytorch.org/whl/cu128
```

### YOLO-NAS Deprecation

YOLO-NAS is deprecated because SuperGradients conflicts with RF-DETR's modern training dependency stack. Use YOLO26, YOLO11, RF-DETR, or Detectron2 for new training runs.

### YOLO26 Issues

| Issue | Solution |
|---|---|
| "Model not found" | Ensure internet connection — Ultralytics auto-downloads from hub |
| "Slow on CPU" | Try smaller variant (n or s), increase batch size if memory allows |
| "Memory exceeded" | Reduce `imgsz`, reduce batch size, use smaller model variant |

### YOLOv12 Issues

| Issue | Solution |
|---|---|
| "Training instability" | Normal — reduce learning rate, use gradient clipping, or switch to YOLO11/YOLO26 |
| "High memory usage" | Expected with attention layers — reduce batch size significantly |
| "FlashAttention fails" | Not required — fallback is automatic |

### YOLO11 Issues

| Issue | Solution |
|---|---|
| "Inference slow" | Use TensorRT export for GPU, reduce image size |
| "Training convergence" | Adjust learning rate, verify dataset format, check class balance |

### SAM 3.1 Issues

| Issue | Solution |
|---|---|
| "HF token required" | Set `HF_TOKEN` env var or run `huggingface-cli login` |
| "Model download fails" | Accept Meta's terms at [huggingface.co/facebook/sam3.1](https://huggingface.co/facebook/sam3.1) |
| "Out of memory" | SAM 3.1 requires ~4GB VRAM; use CPU fallback if needed |

### Windows TensorBoard Issues

If `uv run` fails with an access-denied error on a locked `torch` file:

```powershell
.\.venv\Scripts\python.exe -m src.cli.tensorboard_launcher
```

---

## Summary Table

| Use Case | Recommended | Why |
|---|---|---|
| Mobile App | YOLO26n/s | Fastest, smallest |
| Edge Device | YOLO26m | Balanced speed/accuracy |
| IoT/Robotics | YOLO26 (any) | CPU optimized, 43% faster |
| Production Web | YOLO11l | Stable, proven, accurate |
| Enterprise | YOLO11x | Best YOLO accuracy, robust |
| Maximum Accuracy | RF-DETR-2XLarge | Highest mAP (60.1) |
| Research | YOLO12 | Cutting-edge attention architecture |
| Annotation | SAM 3.1 | Open-vocabulary segmentation |
| Legacy Compat | YOLOv8/v9 | Proven, well-documented |
| Benchmarking | YOLO26 vs YOLO11 | Compare edge vs traditional |

---

## Additional Resources

### Documentation

- [YOLO26 Official Docs](https://docs.ultralytics.com/models/yolo26/)
- [YOLO12 Official Docs](https://docs.ultralytics.com/models/yolo12/)
- [YOLO11 Official Docs](https://docs.ultralytics.com/models/yolo11/)
- [YOLOv8 Official Docs](https://docs.ultralytics.com/models/yolov8/)
- [RF-DETR GitHub](https://github.com/roboflow/rf-detr)
- [SAM 3.1 HuggingFace](https://huggingface.co/facebook/sam3.1)
- [Ultralytics Hub](https://hub.ultralytics.com/)

### Community

- [Ultralytics GitHub Issues](https://github.com/ultralytics/ultralytics/issues)
- [Ultralytics Discussions](https://github.com/ultralytics/ultralytics/discussions)
- [Discord Community](https://discord.com/invite/ultralytics)

### Related Files

- See the [README](https://github.com/shahabahreini/YOLOMatic#readme) for installation and quick start
- See the [models guide](models.md) for detailed model benchmarks and architecture reference
- See [`llms.txt`](https://github.com/shahabahreini/YOLOMatic/blob/main/llms.txt) for AI-readable repository map
