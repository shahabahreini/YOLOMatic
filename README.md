# YOLOmatic

[ Automated Yolo Training ]

**YOLOmatic** is a CLI toolkit for configuring, training, monitoring, predicting with, and uploading modern YOLO-family models. It currently supports **YOLO26**, **YOLOv12**, **YOLOv11**, **YOLOv10**, **YOLOv9**, **YOLOv8**, **YOLOX**, and **YOLO-NAS** workflows available in this repository.

It provides an interactive terminal experience for model selection, dataset binding, prediction, TensorBoard launch, and Roboflow upload, while keeping the project runnable through the existing `uv` environment.

---

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation & Setup](#installation--setup)
4. [Platform Notes](#platform-notes)
5. [Dataset Preparation](#dataset-preparation)
6. [Environment Variables](#environment-variables)
7. [Usage Guide](#usage-guide)
   - [1. Configure Training](#1-configure-training)
   - [2. Start Training](#2-start-training)
   - [3. Run Predictions](#3-run-predictions)
   - [4. Upload to Roboflow](#4-upload-to-roboflow)
   - [5. Monitor Training](#5-monitor-training)
8. [ClearML Integration](#clearml-integration)
9. [Versioning](#versioning)
10. [Troubleshooting](#troubleshooting)
11. [Related Docs](#related-docs)
12. [License](#license)

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
   - Linux/macOS:

     ```sh
     cp .env.example .env
     ```

   - Windows PowerShell:

     ```powershell
     Copy-Item .env.example .env
     ```

---

## Platform Notes

- **Windows**
  - Use `uv sync` to create `.venv`, but use `\.venv\Scripts\python.exe -m pip ...` for manual PyTorch CUDA repairs.
  - `uv run pip ...` may re-sync the environment first and can restore CPU-only Torch.
  - `nvidia-smi` may work even when `torch.cuda.is_available()` is `False`; YOLOmatic now detects this and offers an interactive repair flow.

- **Linux**
  - If PyTorch fails due to missing cuDNN or CUDA runtime libraries, YOLOmatic prepares the relevant library search path automatically.
  - NVIDIA runtime libraries inside `.venv` are added to `LD_LIBRARY_PATH` when needed.

- **macOS**
  - NVIDIA CUDA training is not expected.
  - YOLOmatic can still run on CPU, and Apple Silicon environments can use `mps` when supported by the installed PyTorch build.
  - If GPU acceleration is unavailable, the training flow offers CPU fallback instead of hard-failing.

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

Training behavior:

- If there is exactly one YAML config in `configs/`, it is auto-selected.
- If there are multiple YAML configs, YOLOmatic opens a TUI selector.
- If ClearML is not configured, YOLOmatic prompts you to continue without ClearML or cancel.
- If CUDA is requested but PyTorch cannot use it, YOLOmatic prompts you to repair the environment, continue on CPU, or cancel.

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

If ClearML is not configured when training starts, YOLOmatic will ask whether to continue without ClearML or cancel the run.

---

## Troubleshooting

### PyTorch cannot detect the GPU on Windows or Linux

If `nvidia-smi` works but training still reports a CPU-only Torch build or `torch.cuda.is_available() == False`, YOLOmatic now offers an in-app CUDA repair flow.

- Choose `Fix CUDA-enabled PyTorch now` in the TUI prompt.
- The repair uses the active `.venv` interpreter directly.
- The repair preserves `numpy==1.23.0` so `super-gradients` remains compatible.

Manual verification command:

```sh
uv run python -c "import torch, numpy; print('torch', torch.__version__); print('cuda build', torch.version.cuda); print('cuda available', torch.cuda.is_available()); print('device count', torch.cuda.device_count()); print('numpy', numpy.__version__)"
```

On Windows, if you need to repair manually, prefer:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio
.\.venv\Scripts\python.exe -m pip install --no-cache-dir --force-reinstall torch torchvision torchaudio numpy==1.23.0 --index-url https://download.pytorch.org/whl/cu128
```

### YOLO-NAS fails after CUDA repair

`super-gradients 3.7.1` requires `numpy==1.23.0` in this project.

If NumPy drifts higher, restore it with:

```sh
uv run python -m pip install --force-reinstall numpy==1.23.0
```

### ClearML is not configured

Run:

```sh
uv run clearml-init
```

Or choose `Continue Without ClearML` when prompted.

### macOS GPU notes

- NVIDIA CUDA is not expected on macOS.
- Apple Silicon users should use an MPS-capable PyTorch build if GPU acceleration is desired.
- If MPS is unavailable, run training on CPU.

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
