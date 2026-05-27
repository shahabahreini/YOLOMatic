---
description: Frequently asked questions about YOLOmatic model support, datasets, GPUs, RF-DETR, SAM 3.1, and Roboflow upload.
---

# FAQ

### What is YOLOmatic?

YOLOmatic is a Python CLI/TUI for configuring, training, fine-tuning,
predicting, benchmarking, augmenting, converting, monitoring, and uploading
computer-vision models.

### Which model families does YOLOmatic support?

YOLO26, YOLOv12, YOLO11, YOLOv10, YOLOv9, YOLOv8, YOLOX, RF-DETR, SAM 3.1, and
Detectron2.

### Is YOLOmatic only for YOLO?

No. The name comes from YOLO workflows, but the trainer router also supports
native RF-DETR, SAM 3.1, and Detectron2 workflows.

### Does YOLOmatic require a GPU?

No. A CUDA GPU is strongly recommended for training, but CPU and Apple Silicon
MPS fallbacks are supported. The TUI detects common CUDA/PyTorch mismatches and
offers repair guidance.

### Can YOLOmatic convert Labelbox NDJSON?

Yes. It converts Labelbox and Ultralytics-platform NDJSON exports into YOLO or
COCO datasets with concurrent image downloads.

### Can I fine-tune RF-DETR?

Yes. RF-DETR `.pth` checkpoints are discoverable for fine-tuning and route to
the native RF-DETR trainer.

### Can I upload trained models to Roboflow?

Yes. YOLOmatic uploads YOLO checkpoints and deploys RF-DETR checkpoints through
the upload TUI or optional post-training upload config.

### How should I cite YOLOmatic?

Use [CITATION.cff](https://github.com/shahabahreini/YOLOMatic/blob/main/CITATION.cff)
for GitHub citation metadata or
[CITATION.bib](https://github.com/shahabahreini/YOLOMatic/blob/main/CITATION.bib)
for BibTeX.
