---
description: Reference for YOLOmatic command-line entrypoints and their primary use cases.
---

# CLI Commands

| Command | Purpose |
| --- | --- |
| `yolomatic` / `uv run yolomatic` | Main interactive TUI |
| `uv run yolomatic-train` | Train from a saved YOLO, RF-DETR, SAM, or Detectron2 config |
| `uv run yolomatic-predict` | Run YOLO/RF-DETR prediction workflows |
| `uv run yolomatic-sam` | Run SAM 3.1 segmentation inference |
| `uv run yolomatic-convert` | Convert Labelbox or Ultralytics NDJSON to YOLO/COCO |
| `uv run yolomatic-prepare` | Prepare and split datasets |
| `uv run yolomatic-benchmark` | Benchmark checkpoints with an HTML report |
| `uv run yolomatic-upload` | Upload or deploy checkpoints to Roboflow |
| `uv run yolomatic-tensorboard` | Launch TensorBoard for discovered runs |
| `uv run yolomatic-ultralytics` | Ultralytics Platform helper workflows |
| `uv run bump patch|minor|major|VERSION` | Update the package version |

Related pages: [Quickstart](../getting-started/quickstart.md), [Configuration](configuration.md), [Cloud upload](../guides/cloud-upload.md).
