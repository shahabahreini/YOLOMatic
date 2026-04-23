# YOLOmatic

[ Automated Yolo Training ]

**YOLOmatic** is a CLI toolkit for configuring, training, monitoring, predicting with, and uploading modern YOLO-family models. It currently supports **YOLO26**, **YOLOv12**, **YOLOv11**, **YOLOv10**, **YOLOv9**, **YOLOv8**, **YOLOX**, and **YOLO-NAS** workflows available in this repository.

It provides an interactive terminal experience for model selection, dataset binding, prediction, TensorBoard launch, and Roboflow upload, while keeping the project runnable through the existing `uv` environment.

---

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation & Setup](#installation--setup)
4. [Dataset Preparation](#dataset-preparation)
5. [Environment Variables](#environment-variables)
6. [Usage Guide](#usage-guide)
   - [1. Configure Training](#1-configure-training)
   - [2. Start Training](#2-start-training)
   - [3. Run Predictions](#3-run-predictions)
   - [4. Upload to Roboflow](#4-upload-to-roboflow)
   - [5. Monitor Training](#5-monitor-training)
7. [ClearML Integration](#clearml-integration)
8. [Versioning](#versioning)
9. [Related Docs](#related-docs)
10. [License](#license)

---

## Features

- **Broad Model Support**: Configure and train YOLO26, YOLOv12, YOLOv11, YOLOv10, YOLOv9, YOLOv8, YOLOX, and YOLO-NAS variants exposed by the CLI.
- **Interactive Training Setup**: Generate configuration files from the TUI instead of hand-editing YAML from scratch.
- **Prediction TUI**: Discover available `.pt` weights from the project root and `runs/**/weights/`, then run single-image or folder inference.
- **Roboflow Upload Command**: Upload trained checkpoints with `yolomatic-upload`, optional `.env` defaults, workspace resolution, and Roboflow model-type prompting.
- **TensorBoard Launcher**: Discover training runs and start TensorBoard without manually hunting for log directories.
- **ClearML Integration**: Track training metadata and results through ClearML-enabled training flows.

---

## Prerequisites

- **Python**: `>=3.10,<3.11`.
- **`uv` Package Manager**: This repository relies heavily on `uv` for blazing-fast dependency management and reliable project execution. If you don't have it installed:
  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Hardware**: A CUDA-compatible GPU is highly recommended for training.

---

## Installation & Setup

1. **Clone the repository**:

   ```sh
   git clone https://github.com/shahabahreini/YOLOMatic.git
   cd YOLOMatic
   ```

2. **Sync the project environment**:
   Using `uv`, quickly sync all dependencies and set up the virtual environment:

   ```sh
   uv sync
   ```

3. **Optional: create a Roboflow environment file**:
   ```sh
   cp .env.example .env
   ```

---

## Dataset Preparation

Place your custom datasets inside the `datasets/` folder at the root of the project. Your dataset must follow standard YOLO format hierarchy:

```
YOLOMatic/
├── datasets/
│   └── your_dataset_name/
│       ├── data.yaml        # YOLO dataset config
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── valid/
│       │   ├── images/
│       │   └── labels/
│       └── test/
│           ├── images/
│           └── labels/
```

---

## Environment Variables

If you plan to use `yolomatic-upload`, configure a local `.env` file based on `.env.example`.

```env
ROBOFLOW_API_KEY=
ROBOFLOW_WORKSPACE=
ROBOFLOW_PROJECT_IDS=
```

- `ROBOFLOW_API_KEY`: required for uploads.
- `ROBOFLOW_WORKSPACE`: optional default workspace slug. Use the **workspace slug**, not a project name or display label.
- `ROBOFLOW_PROJECT_IDS`: optional comma-separated default project IDs.

The uploader can also prompt for missing values interactively.

---

## Usage Guide

YOLOmatic uses five primary commands executed via `uv`.

### 1. Configure Training

Generate an optimized configurations file by interactively selecting your YOLO model and target dataset.

```sh
uv run yolomatic
```

_Follow the on-screen terminal menus to finalize your setup. A highly optimized YAML configuration file will be uniquely generated._

### 2. Start Training

Once configured, start training with the generated configuration. Standard YOLO configs are handled by the regular trainer, while YOLO-NAS configs are routed to the NAS trainer automatically.

```sh
uv run yolomatic-train
```

### 3. Run Predictions

Launch the interactive prediction TUI to select from available `.pt` weights discovered in the project root and `runs/**/weights/`, then choose either single-image or folder inference.

```sh
uv run yolomatic-predict
```

You can also provide values directly when you do not want to step through the prompts:

```sh
uv run yolomatic-predict --mode single --weight runs/segment/train/weights/best.pt --source /path/to/image.jpg
```

### 4. Upload to Roboflow

Upload a trained checkpoint to Roboflow using the interactive upload TUI:

```sh
uv run yolomatic-upload
```

You can also override prompted values directly:

```sh
uv run yolomatic-upload --weight runs/segment/train2/weights/best.pt --workspace your-workspace-slug --project-ids vegmask --model-type yolo26l --model-name train2-best
```

Upload notes:

- Choose a **full checkpoint** such as `best.pt` or `last.pt`.
- Generated artifacts like `state_dict.pt` are not uploadable checkpoints.
- For YOLO26 uploads, Roboflow expects a **size-specific model type** such as `yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, or `yolo26x`.
- If the configured workspace cannot be loaded, the uploader can fall back to the API key's default workspace or prompt for manual entry.

### 5. Monitor Training

While training is running (or after it finishes), easily launch TensorBoard to inspect your metrics. The tool will scan all training runs and present a selection menu.

```sh
uv run yolomatic-tensorboard
```

---

## ClearML Integration

To track your experiments remotely:

1. Configure your ClearML credentials:

   ```sh
   uv run clearml-init
   ```

   _(Follow the prompts to paste your API credentials from your ClearML account)_

2. Your generated training configurations handle the rest. YOLOmatic will seamlessly sync hyper-parameters and metrics!

---

## Versioning

We manage project versions with the `bump` CLI, which updates both `src/__version__.py` and `pyproject.toml`.

```sh
# Bump versions automatically
uv run bump patch   # 0.1.2 → 0.1.3
uv run bump minor   # 0.1.2 → 0.2.0
uv run bump major   # 0.1.2 → 1.0.0

# Set an exact value
uv run bump 1.5.0
```

After bumping, ensure your environment is locked with the new version:

```sh
uv sync
```

---

## Related Docs

- `YOLO_GUIDE.md`: workflow-oriented guide for supported YOLO families and command usage.
- `MODELS.md`: architecture and model-family reference notes.

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE.md](LICENSE.md) file for details.

---

_Acknowledgments: Built proudly on top of [Ultralytics](https://github.com/ultralytics/ultralytics) and [ClearML](https://github.com/allegroai/clearml)._
