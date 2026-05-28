---
description: Reference for all YOLOmatic command-line entrypoints, flags, and usage examples.
---

# CLI Commands

## Command Overview

| Command | Purpose |
|---|---|
| `yolomatic` | Main interactive TUI |
| `yolomatic-train` | Train from a saved YAML config |
| `yolomatic-predict` | Run YOLO/RF-DETR prediction workflows |
| `yolomatic-sam` | Run SAM 3.1 segmentation inference |
| `yolomatic-convert` | Convert Labelbox or Ultralytics NDJSON to YOLO/COCO |
| `yolomatic-prepare` | Prepare and split datasets |
| `yolomatic-benchmark` | Benchmark checkpoints with an HTML report |
| `yolomatic-upload` | Upload or deploy checkpoints to Roboflow |
| `yolomatic-tensorboard` | Launch TensorBoard for discovered runs |
| `yolomatic-ultralytics` | Ultralytics Platform helper workflows |
| `bump patch\|minor\|major\|VERSION` | Update the package version |

All commands are available as `uv run <command>` from the repository root, or directly after `uv tool install yolomatic`.

---

## `yolomatic`

Launch the main interactive TUI. All primary workflows are accessible from this menu.

```sh
uv run yolomatic
# or
yolomatic
```

**TUI menu options:**

- Configure Model
- Configure Fine-Tune
- Train Model
- Predict
- Benchmark Models
- Augment Dataset
- Convert Dataset Format
- Prepare Dataset
- Upload to Roboflow
- Launch TensorBoard
- SAM Segmentation
- Ultralytics Platform

---

## `yolomatic-train`

Train from a previously saved YAML config file. If multiple configs exist in `configs/`, a selector is shown.

```sh
uv run yolomatic-train
```

The smart training router reads the config's `family` field and dispatches to the correct trainer:

- `YOLO*`, `YOLOX` → Ultralytics YOLO trainer
- `RF-DETR*` → Native RF-DETR trainer
- `SAM*` → HuggingFace SAM trainer
- `Detectron2` → Detectron2 trainer

**Runtime prompts:**

- If ClearML is not configured: continue without ClearML or cancel
- If CUDA is requested but unavailable: repair environment, fall back to CPU, or cancel

---

## `yolomatic-predict`

Run prediction on a single image or a folder of images using a trained YOLO or RF-DETR checkpoint.

```sh
uv run yolomatic-predict [--mode MODE] [--weight PATH] [--source PATH] [--conf FLOAT] [--workers INT]
```

| Flag | Default | Description |
|---|---|---|
| `--mode` | interactive | Prediction mode: `single` for one image, `folder` for batch directory |
| `--weight` | interactive | Path to a `.pt` weight file; omit to use the interactive selector |
| `--source` | interactive | Image file or folder path; omit to be prompted |
| `--input-dir` | — | Alias for `--source` in folder mode |
| `--conf` | `0.25` | Confidence threshold for predictions |
| `--workers` | `1` | Number of worker processes for folder prediction; set > 1 to enable multiprocessing |

**Examples:**

```sh
# Interactive wizard
uv run yolomatic-predict

# Single image
uv run yolomatic-predict --mode single --weight runs/detect/train/weights/best.pt --source image.jpg

# Batch folder with multiprocessing
uv run yolomatic-predict --mode folder --weight runs/detect/train/weights/best.pt --source datasets/test/images --workers 4

# Lower confidence threshold
uv run yolomatic-predict --mode single --weight best.pt --source image.jpg --conf 0.5
```

---

## `yolomatic-sam`

Run SAM 3.1 segmentation inference. Supports auto, text-prompted, and box-prompted modes.

```sh
uv run yolomatic-sam
```

This command is wizard-only — all options are presented interactively. See [SAM 3.1 guide](../guides/sam.md) for authentication and mode details.

---

## `yolomatic-convert`

Convert Labelbox or Ultralytics-platform NDJSON exports to YOLO or COCO format. Includes concurrent image downloading.

```sh
uv run yolomatic-convert
```

Wizard-only. The wizard auto-detects whether the source is a Labelbox or Ultralytics-platform NDJSON. See [NDJSON Conversion](ndjson-conversion.md) for details.

---

## `yolomatic-prepare`

Prepare and split a dataset into train/val/test subsets using random, class-balanced, or smart-balanced strategies.

```sh
uv run yolomatic-prepare
```

Wizard-only. See [Smart Split](smart-split.md) for the splitting algorithm details.

---

## `yolomatic-benchmark`

Benchmark one or more YOLO `.pt` checkpoints against COCO-format annotations and generate an interactive HTML report.

```sh
uv run yolomatic-benchmark
```

Wizard-only. Requires:

1. A trained Ultralytics `.pt` checkpoint
2. A validation set with `_annotations.coco.json`

See [Benchmarking guide](../guides/benchmarking.md) for the full workflow.

---

## `yolomatic-upload`

Upload YOLO checkpoints to Roboflow or deploy RF-DETR checkpoints via the RF-DETR deployment API.

```sh
uv run yolomatic-upload [--weight PATH] [--workspace SLUG] [--project-ids IDS] [--model-name NAME] [--model-type TYPE] [--version INT]
```

| Flag | Default | Description |
|---|---|---|
| `--weight` | interactive | Path to checkpoint (`best.pt`, `last.pt`). Omit for interactive selector. |
| `--workspace` | from `.env` | Roboflow workspace slug. Falls back to `ROBOFLOW_WORKSPACE` in `.env`. |
| `--project-ids` | from `.env` | Comma-separated project IDs. Falls back to `ROBOFLOW_PROJECT_IDS`. |
| `--model-name` | auto | Versionless model name to register in Roboflow. |
| `--model-type` | auto-detected | Override the Roboflow model type (e.g., `yolo26n`, `yolo26l`, `rf-detr-l`). |
| `--version` | `1` | Roboflow project version for RF-DETR deployment. |

**Examples:**

```sh
# Interactive wizard
uv run yolomatic-upload

# Direct upload with explicit args
uv run yolomatic-upload \
  --weight runs/segment/train/weights/best.pt \
  --workspace my-workspace \
  --project-ids my-project \
  --model-type yolo26l \
  --model-name train2-best
```

!!! warning "YOLO26 model type"
    YOLO26 uploads require a **size-specific** model type: `yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, or `yolo26x`. The generic `yolo26` type is not valid.

See [Cloud Upload guide](../guides/cloud-upload.md) for the full Roboflow workflow.

---

## `yolomatic-tensorboard`

Launch TensorBoard for discovered training run directories.

```sh
uv run yolomatic-tensorboard
```

Scans the project tree for TensorBoard event files and starts TensorBoard on port 6006. See [TensorBoard guide](../guides/tensorboard.md).

---

## `yolomatic-ultralytics`

Helpers for Ultralytics Platform workflows (dataset download, NDJSON export conversion).

```sh
uv run yolomatic-ultralytics
```

Wizard-only.

---

## `bump`

Update the project version in `src/__version__.py` and `pyproject.toml`.

```sh
uv run bump patch        # 5.0.0 → 5.0.1
uv run bump minor        # 5.0.0 → 5.1.0
uv run bump major        # 5.0.0 → 6.0.0
uv run bump 5.2.0        # explicit version
```

Related pages: [Quickstart](../getting-started/quickstart.md), [Configuration](configuration.md), [Cloud upload](../guides/cloud-upload.md).
