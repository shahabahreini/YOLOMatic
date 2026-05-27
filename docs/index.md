---
description: YOLOmatic is an interactive Python CLI/TUI for training YOLO, RF-DETR, SAM 3.1, and Detectron2 computer-vision models.
---

# YOLOmatic

Automated computer-vision training for YOLO26, YOLOv12, YOLO11, YOLOv10,
YOLOv9, YOLOv8, YOLOX, RF-DETR, SAM 3.1, and Detectron2.

![YOLOmatic terminal wizard](assets/screenshots/wizard-configure-model.png){ .hero-shot }

## Why YOLOmatic

- Interactive terminal wizards generate training, fine-tuning, prediction,
  benchmark, augmentation, conversion, TensorBoard, and upload workflows.
- Hardware-aware configuration helps pick batch sizes, workers, devices, and
  runtime fallbacks for CUDA, Apple Silicon MPS, and CPU environments.
- One CLI covers Ultralytics YOLO, native RF-DETR, SAM 3.1, Detectron2, Labelbox
  NDJSON conversion, Roboflow upload, ClearML tracking, and benchmark reports.

## Quickstart

```sh
uv tool install yolomatic
yolomatic
```

For repository development:

```sh
git clone https://github.com/shahabahreini/YOLOMatic.git
cd YOLOMatic
uv sync
uv run yolomatic
```

## Core Workflows

| Workflow | Command | Output |
| --- | --- | --- |
| Configure training | `uv run yolomatic` | YAML config for the selected model and dataset |
| Train | `uv run yolomatic-train` | Checkpoints, logs, exports, optional Roboflow upload |
| Predict | `uv run yolomatic-predict` | Annotated images or batch inference results |
| Segment with SAM | `uv run yolomatic-sam` | Auto, text-prompted, or box-prompted masks |
| Benchmark | `uv run yolomatic-benchmark` | HTML report with mAP, F1, rankings, UMAP |
| Convert data | `uv run yolomatic-convert` | YOLO or COCO dataset from NDJSON |

## Model Families

| Family | Tasks | Trainer |
| --- | --- | --- |
| YOLO26 / YOLOv12 / YOLO11 / YOLOv10 / YOLOv9 / YOLOv8 / YOLOX | Detect, segment, classify, pose, OBB where supported | Ultralytics |
| RF-DETR | Detect, segment | Native RF-DETR |
| SAM 3.1 | Open-vocabulary segmentation and mask fine-tuning | HuggingFace |
| Detectron2 | Detect, segment | Detectron2 |

## Learn More

- [Install YOLOmatic](getting-started/install.md)
- [Run the 30-second quickstart](getting-started/quickstart.md)
- [Choose a model family](guides/models.md)
- [Prepare datasets](guides/datasets.md)
- [Compare YOLOmatic with other tools](comparison.md)
- [Cite YOLOmatic](citation.md)
