---
description: Reference for YOLOmatic-generated training configuration files and common settings.
---

# Configuration

YOLOmatic writes YAML configs under `configs/`. The generated file captures:

- Model family, variant, and task
- Dataset path and annotation format
- Image size, batch size, epochs, workers, and device
- Trainer-specific options for Ultralytics, RF-DETR, SAM 3.1, or Detectron2
- Optional ClearML, TensorBoard, Roboflow, export, and resume settings

## Device Selection

CUDA is preferred when available. Apple Silicon MPS and CPU fallbacks are
supported. If CUDA is requested but PyTorch cannot use it, the preflight flow
offers repair guidance before training starts.

## Fine-Tuning

Fine-tuning configs bind a discovered checkpoint to a new dataset and generate a
fresh training YAML. YOLO checkpoints use `.pt`, RF-DETR uses `.pth`, and SAM
uses HuggingFace model identifiers or local artifacts.

Related pages: [First training run](../getting-started/first-training-run.md), [RF-DETR](../guides/rf-detr.md), [Cloud upload](../guides/cloud-upload.md).
