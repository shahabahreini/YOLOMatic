<div align="center">

# YOLOmatic

**Automated YOLO Training — Configure, Train, Predict, Upload**

[![macOS](https://img.shields.io/badge/macOS-supported-black?logo=apple&logoColor=white&style=for-the-badge)](https://www.apple.com/macos/)
[![Linux](https://img.shields.io/badge/Linux-supported-black?logo=linux&logoColor=white&style=for-the-badge)](https://kernel.org/)
[![Windows](https://img.shields.io/badge/Windows-supported-black?logo=windows&logoColor=white&style=for-the-badge)](https://www.microsoft.com/windows/)

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white&style=flat-square)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet?style=flat-square)](https://docs.astral.sh/uv/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-00BFFF?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square)](LICENSE.md)

</div>

---

**YOLOmatic** is a production-grade CLI toolkit for the full YOLO training lifecycle — model selection, configuration generation, training, prediction, TensorBoard monitoring, and Roboflow upload — all driven through a rich interactive terminal interface.

Supported model families: **YOLO26**, **YOLOv12**, **YOLOv11**, **YOLOv10**, **YOLOv9**, **YOLOv8**, **YOLOX**, and **YOLO-NAS**.

---

## About

YOLOmatic automates the end-to-end workflow for training YOLO-based computer vision models using [Ultralytics](https://github.com/ultralytics/ultralytics) and [SuperGradients](https://github.com/Deci-AI/super-gradients). It targets practitioners who need a reproducible, hardware-aware training pipeline without writing boilerplate configuration or shell scripts.

**Supported tasks:** object detection, instance segmentation, image classification, pose estimation, oriented object detection (OBB).

**Supported model families and variants:**

| Family | Variants | Tasks |
|---|---|---|
| YOLO26 | n / s / m / l / x + seg / cls / pose / obb | All |
| YOLOv12 | n / s / m / l / x + seg | Detect, Segment |
| YOLOv11 | n / s / m / l / x + seg / cls / pose / obb | All |
| YOLOv10 | N / S / M / B / L / X | Detect |
| YOLOv9 | t / s / m / c / e + seg | Detect, Segment |
| YOLOv8 | n / s / m / l / x + seg / cls / pose / obb | All |
| YOLOX | S / M / L / X | Detect |
| YOLO-NAS | S / M / L | Detect |

**Toolchain integrations:** PyTorch, ONNX, TensorBoard, ClearML, Roboflow, `uv`.

**Deployment targets:** CUDA GPU (NVIDIA), Apple Silicon MPS, CPU (all platforms), ONNX Runtime, TensorRT.

**Keywords:** yolo training cli, automated yolo training, yolo26 training, yolov12 training, yolov11 training, ultralytics training pipeline, yolo configuration generator, instance segmentation training, object detection cli, pytorch yolo, onnx export, clearml yolo, roboflow yolo upload, tensorboard yolo, edge deployment yolo, yolo tui, yolo interactive terminal.

---

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation & Setup](#installation--setup)
4. [Platform Notes](#platform-notes)
5. [Dataset Preparation](#dataset-preparation)
6. [Environment Variables](#environment-variables)
7. [Usage Guide](#usage-guide)
8. [ClearML Integration](#clearml-integration)
9. [Versioning](#versioning)
10. [Troubleshooting](#troubleshooting)
11. [Related Docs](#related-docs)
12. [License](#license)

---

## Features

| Feature | Description |
|---|---|
| **Broad Model Support** | YOLO26, YOLOv12, YOLOv11, YOLOv10, YOLOv9, YOLOv8, YOLOX, and YOLO-NAS — detection, segmentation, classification, pose, and OBB |
| **Interactive TUI** | Rich terminal menus for model selection, dataset binding, and config generation — no manual YAML editing required |
| **Auto-Optimized Configs** | Augmentation, compute, and worker profiles are recommended based on your dataset and hardware automatically |
| **Smart Training Router** | Standard YOLO and YOLO-NAS configs are routed to the correct trainer automatically |
| **CUDA Auto-Repair** | Detects CPU-only PyTorch when a GPU is present and offers an in-app repair flow |
| **Prediction TUI** | Discovers `.pt` weights across the project tree, then runs single-image or batch folder inference |
| **Roboflow Upload** | Uploads trained checkpoints with workspace resolution, model-type prompting, and `.env` credential support |
| **TensorBoard Launcher** | Scans all training runs and starts TensorBoard without manually locating log directories |
| **ClearML Integration** | Tracks hyper-parameters, metrics, and artifacts through ClearML for remote experiment management |
| **Version Management** | Single-command version bumping via `uv run bump` — `pyproject.toml` is the sole source of truth |

---

## Prerequisites

- **Python** `>=3.10, <3.11`
- **`uv` package manager** — fast, reliable dependency management:

  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- **Hardware** — a CUDA-compatible GPU is strongly recommended for training; CPU and Apple Silicon MPS are supported as fallbacks.

---

## Installation & Setup

**1. Clone the repository**

```sh
git clone https://github.com/shahabahreini/YOLOMatic.git
cd YOLOMatic
```

**2. Sync the environment**

```sh
uv sync
```

This installs all dependencies and creates the `.venv` in one step.

**3. Optional — configure Roboflow credentials**

```sh
# Linux / macOS
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

Edit `.env` and fill in your Roboflow API key and workspace slug.

---

## Platform Notes

### macOS

- NVIDIA CUDA is not available on macOS.
- Apple Silicon (M-series) can use `mps` acceleration when the installed PyTorch build supports it.
- If GPU acceleration is unavailable, the training flow offers a CPU fallback instead of failing hard.

### Linux

- If PyTorch fails due to missing cuDNN or CUDA runtime libraries, YOLOmatic prepares the relevant library search path automatically.
- NVIDIA runtime libraries inside `.venv` are added to `LD_LIBRARY_PATH` when needed.

### Windows

- Use `uv sync` to create `.venv`, but use `.venv\Scripts\python.exe -m pip ...` for manual PyTorch CUDA repairs.
- `uv run pip ...` may re-sync the environment and restore a CPU-only Torch build.
- `nvidia-smi` may report a GPU even when `torch.cuda.is_available()` is `False`; YOLOmatic detects this mismatch and offers an interactive repair flow.

---

## Dataset Preparation

Place datasets inside the `datasets/` folder at the project root, following standard YOLO format:

```
YOLOMatic/
└── datasets/
    └── your_dataset_name/
        ├── data.yaml
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── valid/
        │   ├── images/
        │   └── labels/
        └── test/
            ├── images/
            └── labels/
```

`data.yaml` must declare `train`, `val` (or `valid`), and `test` paths, along with `nc` (class count) and `names`.

---

## Environment Variables

Create a `.env` file from `.env.example` to enable Roboflow upload defaults:

```env
ROBOFLOW_API_KEY=
ROBOFLOW_WORKSPACE=
ROBOFLOW_PROJECT_IDS=
```

| Variable | Required | Description |
|---|---|---|
| `ROBOFLOW_API_KEY` | Yes (for uploads) | Your Roboflow API key |
| `ROBOFLOW_WORKSPACE` | No | Default workspace **slug** (not display name) |
| `ROBOFLOW_PROJECT_IDS` | No | Comma-separated default project IDs |

All values can also be entered interactively when the uploader prompts for them.

---

## Usage Guide

YOLOmatic exposes five `uv run` commands covering the full training workflow.

### 1. Configure Training

```sh
uv run yolomatic
```

Launches the interactive TUI. Navigate model families, select a dataset, and choose augmentation and compute profiles. A uniquely generated, hardware-optimized YAML config is written to `configs/`.

### 2. Start Training

```sh
uv run yolomatic-train
```

| Situation | Behavior |
|---|---|
| One config in `configs/` | Auto-selected |
| Multiple configs in `configs/` | TUI selector opens |
| ClearML not configured | Prompts to continue without ClearML or cancel |
| CUDA requested but unavailable | Prompts to repair, continue on CPU, or cancel |

Training writes a full TensorBoard event log covering losses, precision, recall, mAP, learning rate, and runtime metrics. A completeness report is printed at the end of each run.

### 3. Run Predictions

```sh
uv run yolomatic-predict
```

Discovers `.pt` weights across the project root and `runs/**/weights/`, then prompts for single-image or folder inference mode. Values can also be passed directly:

```sh
uv run yolomatic-predict --mode single --weight runs/segment/train/weights/best.pt --source /path/to/image.jpg
```

### 4. Upload to Roboflow

```sh
uv run yolomatic-upload
```

Interactive upload TUI with workspace resolution and model-type prompting. Direct flag override:

```sh
uv run yolomatic-upload \
  --weight runs/segment/train2/weights/best.pt \
  --workspace your-workspace-slug \
  --project-ids vegmask \
  --model-type yolo26l \
  --model-name train2-best
```

**Upload requirements:**

- Upload a full checkpoint: `best.pt` or `last.pt`.
- Do not upload generated artifacts such as `state_dict.pt`.
- YOLO26 uploads require a size-specific model type: `yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, or `yolo26x`.

### 5. Monitor Training

```sh
uv run yolomatic-tensorboard
```

Scans all runs for TensorBoard event files and Ultralytics `args.yaml` markers (including YOLO-NAS runs), then presents a selection menu before launching TensorBoard.

On Windows, if `uv run` fails with an access-denied error on a locked `torch` file, use the venv directly:

```powershell
.\.venv\Scripts\python.exe -m src.cli.tensorboard_launcher
```

---

## ClearML Integration

```sh
uv run clearml-init
```

Follow the prompts to paste your API credentials from your ClearML account. All subsequent training runs will automatically sync hyper-parameters, metrics, and artifacts to your ClearML project.

If ClearML is not configured when training starts, YOLOmatic prompts you to continue without it or cancel.

---

## Versioning

Version is managed in `pyproject.toml` as the single source of truth. The `bump` command updates it in one step:

```sh
uv run bump patch   # 3.0.0 -> 3.0.1
uv run bump minor   # 3.0.0 -> 3.1.0
uv run bump major   # 3.0.0 -> 4.0.0
uv run bump 3.2.0   # set exact version
```

The TUI footer and About screen read the version live from `pyproject.toml` — no other files need updating.

After bumping, lock the updated version:

```sh
uv lock
```

---

## Troubleshooting

### PyTorch cannot detect the GPU

If `nvidia-smi` works but `torch.cuda.is_available()` returns `False`, use the in-app CUDA repair flow:

1. Start training with `uv run yolomatic-train`.
2. Choose **Fix CUDA-enabled PyTorch now** when prompted.

The repair targets the active `.venv` interpreter directly and preserves `numpy==1.23.0` for YOLO-NAS compatibility.

Manual verification:

```sh
uv run python -c "import torch, numpy; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), numpy.__version__)"
```

Manual repair on Windows:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio
.\.venv\Scripts\python.exe -m pip install --no-cache-dir --force-reinstall torch torchvision torchaudio numpy==1.23.0 --index-url https://download.pytorch.org/whl/cu128
```

### YOLO-NAS fails after a CUDA repair

`super-gradients 3.7.1` requires exactly `numpy==1.23.0`. Restore it with:

```sh
uv run python -m pip install --force-reinstall numpy==1.23.0
```

### ClearML is not configured

```sh
uv run clearml-init
```

Or choose **Continue Without ClearML** when the training TUI prompts.

### macOS — no GPU acceleration

- NVIDIA CUDA is not supported on macOS.
- Apple Silicon: use an MPS-capable PyTorch build for GPU acceleration.
- Intel Mac or no MPS: training runs on CPU automatically.

---

## Related Docs

| Document | Purpose |
|---|---|
| [`YOLO_GUIDE.md`](YOLO_GUIDE.md) | Workflow guide: model selection, deployment scenarios, export options |
| [`MODELS.md`](MODELS.md) | Architecture reference, benchmark tables, model family comparison |

---

## License

Licensed under the [Apache License 2.0](LICENSE.md).

---

<div align="center">

Built on top of [Ultralytics](https://github.com/ultralytics/ultralytics) and [ClearML](https://github.com/allegroai/clearml).

</div>
