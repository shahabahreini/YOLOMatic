<div align="center">

# YOLOmatic

**Automated YOLO, RF-DETR & SAM Training — Configure, Train, Predict, Benchmark, Upload**

[![macOS](https://img.shields.io/badge/macOS-supported-black?logo=apple&logoColor=white&style=for-the-badge)](https://www.apple.com/macos/)
[![Linux](https://img.shields.io/badge/Linux-supported-black?logo=linux&logoColor=white&style=for-the-badge)](https://kernel.org/)
[![Windows](https://img.shields.io/badge/Windows-supported-black?logo=windows&logoColor=white&style=for-the-badge)](https://www.microsoft.com/windows/)

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white&style=flat-square)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet?style=flat-square)](https://docs.astral.sh/uv/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-00BFFF?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square)](LICENSE.md)
[![Version](https://img.shields.io/badge/Version-4.3.0-orange?style=flat-square)](pyproject.toml)

</div>

---

## What Is YOLOmatic?

YOLOmatic is a **production-ready command-line toolkit** that automates the full lifecycle of computer-vision model training — from dataset preparation and augmentation, through hardware-aware configuration and training, to evaluation, prediction, and cloud deployment. It supports **YOLO**, **RF-DETR**, **SAM 3.1**, and **Detectron2** model families through an interactive terminal interface (TUI).

Whether you are fine-tuning a nano model for a Raspberry Pi or training an XLarge transformer detector on a multi-GPU server, YOLOmatic generates optimized configs, manages checkpoints, monitors training with TensorBoard and ClearML, and uploads results to Roboflow — all without writing boilerplate code.

**Supported tasks:** Object Detection · Instance Segmentation · Image Classification · Pose Estimation · Oriented Object Detection (OBB) · Open-Vocabulary Segmentation (SAM 3.1)

---

## Supported Models

| Family | Variants | Tasks | Checkpoint |
|---|---|---|---|
| **RF-DETR** | Nano / Small / Medium / Large / XLarge / 2XLarge + Seg | Detect, Segment | `.pth` |
| **YOLO26** | n / s / m / l / x + seg / cls / pose / obb | All | `.pt` |
| **YOLOv12** | n / s / m / l / x + seg | Detect, Segment | `.pt` |
| **YOLO11** | n / s / m / l / x + seg / cls / pose / obb | All | `.pt` |
| **YOLOv10** | N / S / M / B / L / X | Detect | `.pt` |
| **YOLOv9** | t / s / m / c / e + seg | Detect, Segment | `.pt` |
| **YOLOv8** | n / s / m / l / x + seg / cls / pose / obb | All | `.pt` |
| **YOLOX** | S / M / L / X | Detect | `.pt` |
| **SAM 3.1** | SAM 3 / SAM 3.1 (Object Multiplex) | Segment (open-vocab) | HuggingFace |
| **Detectron2** | Faster R-CNN / RetinaNet / Mask R-CNN | Detect, Segment | `.pth` |

> See [`MODELS.md`](MODELS.md) for benchmark tables, architecture details, and selection recommendations.

---

## Table of Contents

1. [Features](#features)
2. [Quick Start](#quick-start)
3. [Platform Notes](#platform-notes)
4. [Dataset Preparation](#dataset-preparation)
5. [Usage Guide](#usage-guide)
6. [Benchmark & Vector Analysis](#benchmark--vector-analysis)
7. [Offline Dataset Augmentation](#offline-dataset-augmentation)
8. [SAM 3.1 Segmentation](#sam-31-segmentation)
9. [Environment Variables](#environment-variables)
10. [Global Integration Settings](#global-integration-settings)
11. [ClearML Integration](#clearml-integration)
12. [Versioning](#versioning)
13. [Troubleshooting](#troubleshooting)
14. [Related Documentation](#related-documentation)
15. [AI & Search Discovery](#ai--search-discovery)
16. [License](#license)

---

## Features

| Feature | Description |
|---|---|
| **10 Model Families** | RF-DETR, YOLO26, YOLOv12, YOLO11, YOLOv10, YOLOv9, YOLOv8, YOLOX, SAM 3.1, and Detectron2 |
| **Interactive TUI** | Rich terminal menus for every workflow — configure, fine-tune, train, predict, benchmark, augment, combine, convert, upload, and maintain |
| **NDJSON Conversion** | Convert Labelbox `.ndjson` exports to YOLO or COCO formats with concurrent image downloads and polygon/bbox support |
| **Auto-Optimized Configs** | Augmentation, compute, and worker profiles recommended based on your dataset and hardware automatically |
| **Checkpoint Fine-Tuning** | Discovers `.pt`, `.pth`, and HuggingFace checkpoints, binds them to a dataset, and generates a fresh fine-tuning YAML |
| **Automatic Pretrained Downloads** | Fresh configs load official pretrained weights automatically; local checkpoints are used only for fine-tuning or resume |
| **Smart Training Router** | YOLO, RF-DETR, SAM 3.1, and Detectron2 configs are routed to the correct trainer automatically |
| **CUDA Auto-Repair** | Detects CPU-only PyTorch when a GPU is present and offers an in-app repair flow |
| **Prediction TUI** | Discovers weights across the project tree, then runs single-image or batch inference with rich progress display |
| **SAM 3.1 Inference** | Auto-segment, text-prompted, or box-prompted (from YOLO detections) segmentation using Meta's SAM 3.1 |
| **SAM 3.1 Fine-Tuning** | Fine-tune SAM 3.1 mask decoder on custom COCO-format datasets using HuggingFace Trainer |
| **Offline Augmentation** | Albumentations-powered dataset augmentation with reusable profiles, 20+ transforms, split redistribution, and YOLO/COCO output |
| **Dataset Combiner** | Merges multiple YOLO datasets, deduplicates class names, remaps labels, and hard-links images |
| **Benchmark & Vectoring** | Evaluates checkpoints on COCO validation sets — mAP, F1, per-image rankings, UMAP vector scatter — in an interactive HTML report |
| **Roboflow Upload** | Uploads YOLO checkpoints and deploys RF-DETR checkpoints with workspace/project prompts and `.env` credential support |
| **TensorBoard Launcher** | Scans all training runs and starts TensorBoard without manually locating log directories |
| **ClearML Integration** | Tracks hyper-parameters, metrics, and artifacts for remote experiment management |
| **Detectron2 Training** | Configure and train Faster R-CNN, RetinaNet, and Mask R-CNN with COCO-format datasets |
| **Dependency Health Checks** | Checks core ML packages from the TUI and offers guided upgrades for missing or outdated dependencies |
| **Version Management** | Single-command version bumping via `uv run bump` — `pyproject.toml` is the sole source of truth |

---

## Quick Start

### Prerequisites

- **Python** `>=3.12, <3.13`
- **`uv` package manager** — fast, reliable dependency management:

  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- **Hardware** — a CUDA-compatible GPU is strongly recommended for training; CPU and Apple Silicon MPS are supported as fallbacks.

### Installation

```sh
# 1. Clone the repository
git clone https://github.com/shahabahreini/YOLOMatic.git
cd YOLOMatic

# 2. Sync the environment (installs all dependencies + creates .venv)
uv sync

# 3. Optional — configure Roboflow credentials
cp .env.example .env   # Linux / macOS
# Copy-Item .env.example .env   # Windows PowerShell
```

### First Run

```sh
# Launch the interactive TUI
uv run yolomatic

# Train from a saved config
uv run yolomatic-train

# Run predictions
uv run yolomatic-predict

# Monitor training with TensorBoard
uv run yolomatic-tensorboard
```

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

For **Detectron2** and **SAM 3.1** workflows, use COCO JSON annotations instead of YOLO `.txt` labels.

---

## Usage Guide

YOLOmatic exposes a primary TUI plus focused helper commands:

| Command | Purpose |
|---|---|
| `uv run yolomatic` | Launch the main interactive TUI |
| `uv run yolomatic-train` | Train from a saved config |
| `uv run yolomatic-predict` | Run YOLO/RF-DETR predictions |
| `uv run yolomatic-sam` | Run SAM 3.1 segmentation inference |
| `uv run yolomatic-convert` | Convert Labelbox NDJSON to YOLO/COCO |
| `uv run yolomatic-benchmark` | Benchmark trained checkpoints |
| `uv run yolomatic-upload` | Upload/deploy to Roboflow |
| `uv run yolomatic-tensorboard` | Launch TensorBoard for training runs |
| `uv run bump patch\|minor\|major` | Bump version |

### Configure Training

```sh
uv run yolomatic
```

The main menu includes:

| Menu Item | Purpose |
|---|---|
| **Configure Model** | Select a model family/variant, bind a dataset, and generate a hardware-aware training YAML |
| **Configure Fine-Tune** | Select an existing checkpoint, choose a target dataset, and generate a fresh fine-tuning YAML |
| **Train Model** | Train, validate, export, and log from a saved config |
| **Run Prediction** | Run single-image or folder inference from discovered weights |
| **SAM Segment** | Run SAM 3.1 auto/text/box-prompted segmentation |
| **Launch TensorBoard** | Open TensorBoard for a selected run or the full `runs/` tree |
| **Benchmark Models** | Evaluate trained checkpoints and generate an interactive HTML report |
| **Convert Dataset Format** | Convert Labelbox NDJSON exports to YOLO or COCO formats |
| **Augment Dataset** | Offline Albumentations augmentation with reusable profiles |
| **Combine Datasets** | Merge YOLO-format datasets with class remapping |
| **Upload to Roboflow** | Publish trained checkpoints to Roboflow |
| **Check for Updates** | Review package health for the local ML stack |
| **Maintenance → Settings** | Configure global ClearML, Roboflow, and narrative defaults |
| **About YOLOmatic** | View technical details, creator info, and version history |

### Configure Fine-Tuning

Choose **Configure Fine-Tune** from the TUI. YOLOmatic searches the project root and `runs/**/weights/` for Ultralytics `.pt` checkpoints, RF-DETR `.pth` checkpoints, and SAM 3.1 HuggingFace models.

| Strategy | Behavior |
|---|---|
| **Recommended** | Fresh fine-tune from checkpoint, no frozen layers |
| **Freeze Backbone** | Adds `freeze: 10` for small or similar-domain datasets |
| **Fully Customized** | Opens the expert parameter editor with 50+ training parameters |

### Start Training

```sh
uv run yolomatic-train
```

| Situation | Behavior |
|---|---|
| One config in `configs/` | Auto-selected |
| Multiple configs | TUI selector opens |
| ClearML not configured | Prompts to continue without ClearML or cancel |
| CUDA requested but unavailable | Prompts to repair, continue on CPU, or cancel |

### Run Predictions

```sh
# Interactive TUI
uv run yolomatic-predict

# Direct CLI
uv run yolomatic-predict --mode single --weight runs/segment/train/weights/best.pt --source /path/to/image.jpg

# Batch with multiprocessing
uv run yolomatic-predict --mode folder --weight runs/detect/train/weights/best.pt --source datasets/my_dataset/test/images --workers 4
```

### Combine Datasets

Choose **Combine Datasets** from the TUI. YOLOmatic merges selected YOLO-format datasets, deduplicates class names, remaps labels, and hard-links images where the filesystem supports it.

### Upload to Roboflow

```sh
# Interactive upload TUI
uv run yolomatic-upload

# Direct CLI with flags
uv run yolomatic-upload \
  --weight runs/segment/train2/weights/best.pt \
  --workspace your-workspace-slug \
  --project-ids vegmask \
  --model-type yolo26l \
  --model-name train2-best
```

**Automatic post-training upload** — add this block to your training YAML:

```yaml
roboflow:
  upload: true
  weight: "best.pt"
```

**Upload requirements:**

- Upload a full checkpoint: `best.pt` or `last.pt`.
- Do not upload generated artifacts such as `state_dict.pt`.
- YOLO26 uploads require a size-specific model type: `yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, or `yolo26x`.

### Monitor Training

```sh
uv run yolomatic-tensorboard
```

On Windows, if `uv run` fails with an access-denied error on a locked `torch` file:

```powershell
.\.venv\Scripts\python.exe -m src.cli.tensorboard_launcher
```

---

## Benchmark & Vector Analysis

Evaluate one or more trained checkpoints on a COCO-format validation dataset and produce a self-contained interactive HTML report.

### Running the Benchmark

```sh
# From the TUI
uv run yolomatic
# Navigate: Evaluate & Monitor → Benchmark Models

# Or directly
uv run yolomatic-benchmark
```

### What the Wizard Collects

| Step | Default |
|---|---|
| Weight files | Auto-discovered from `runs/` and project root; multi-select |
| Validation directory | `output/nir/valid` |
| Annotation file | `<validation>/_annotations.coco.json` (auto-detected) |
| Confidence threshold | `0.25` |
| Output directory | `output/benchmark_reports/<timestamp>/` |

### Report Sections

| Section | Description |
|---|---|
| **Summary Cards** | Best model name, mAP@50, mAP@50:95, F1, precision, recall |
| **Model Comparison Table** | All models side-by-side with colour-coded cells |
| **mAP by Object Size** | Grouped bar chart — small (<32²), medium (32²–96²), large (>96²) |
| **TP/FP/FN Summary** | Horizontal bar chart for the best model |
| **Per-Image Ranking** | Worst 20 and best 20 images by F1 |
| **Vector Analysis Scatter** | UMAP projection coloured by F1; click a point to see the image thumbnail |

### Task Auto-Detection

The engine auto-detects whether a model outputs bounding boxes or segmentation masks. Detection-only models use bounding-box IoU; segmentation models use pixel-level mask IoU. No manual `--task` flag is required.

### Python API

```python
from src.benchmark import BenchmarkConfig, run_benchmark, write_benchmark_report
from pathlib import Path

config = BenchmarkConfig(
    weights=[Path("runs/train/weights/best.pt")],
    validation_dir=Path("output/nir/valid"),
    conf_threshold=0.25,
)
result = run_benchmark(config)
report_path = write_benchmark_report(result, Path("output/benchmark_reports"))
print(f"Report: {report_path}")
```

---

## Offline Dataset Augmentation

YOLOmatic includes a full **Albumentations-powered offline augmentation engine** accessible from the TUI under **Augment Dataset**.

### Key Capabilities

- **Reusable profiles** — create, edit, clone, and delete augmentation profiles stored as YAML files in `configs/augmentation_profiles/`.
- **20+ transforms** — organized into groups (geometric, color, blur, noise, weather, etc.) with per-transform probability and parameter tuning.
- **Configurable multiplier** — generate N augmented copies per source image.
- **Split redistribution** — pool all images then redistribute into train/val/test splits with configurable ratios.
- **Output formats** — YOLO Detection, YOLO Segmentation, or COCO JSON.
- **Built-in profiles** — pre-configured profiles for common scenarios, ready to use or customize.

### Workflow

1. Select **Augment Dataset** from the TUI.
2. Choose or create an augmentation profile.
3. Select the source dataset and output format.
4. Configure split ratios (70/20/10, 80/15/5, 50/30/20, or custom).
5. Run augmentation with live progress display.

---

## SAM 3.1 Segmentation

YOLOmatic integrates **Meta's SAM 3.1** (Segment Anything Model) for open-vocabulary segmentation inference and fine-tuning.

### Inference

```sh
uv run yolomatic-sam
```

Three modes are available:

| Mode | Description |
|---|---|
| **Auto** | Segment everything — SAM's built-in detector finds and segments all objects automatically |
| **Text-prompted** | Provide concept labels (e.g., "vegetation, tree") — SAM 3.1's open-vocabulary detector finds and segments matching instances |
| **Box-prompted** | Supply YOLO detection `.txt` files — each bounding box becomes a SAM prompt for precise instance masks |

**Requirements:** A HuggingFace token is needed to download gated SAM 3.1 weights. Set `HF_TOKEN` in your environment or enter it interactively.

### Fine-Tuning

SAM 3.1 fine-tuning is available through the **Configure Model** flow when selecting SAM 3.1. The trainer:

- Freezes the image encoder (ViT) and trains only the mask decoder + prompt encoder.
- Uses COCO-format datasets with bounding-box and polygon annotations.
- Integrates with the HuggingFace Trainer API for checkpointing and evaluation.

### Outputs

All modes produce:
- **PNG overlays** with mask visualization
- **COCO JSON** annotations
- **YOLO segmentation `.txt`** files with normalized polygons

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
| `HF_TOKEN` | Yes (for SAM 3.1) | HuggingFace access token for gated models |

All values can also be entered interactively when prompted.

---

## Global Integration Settings

Global ClearML, Roboflow, and narrative defaults live in `configs/yolomatic_settings.yaml` and can be edited from the TUI under **Maintenance → Settings**. New YOLO, RF-DETR, SAM 3.1, and Detectron2 configs snapshot these defaults so each run remains reproducible.

Roboflow secrets stay in `.env` only. YOLOmatic reports whether credentials are configured but never writes API keys to YAML or displays their values.

ClearML setup:

```sh
uv run clearml-init
```

Set `clearml.enabled: false` to skip ClearML task creation, or `clearml.require_configured: true` to cancel training when ClearML cannot initialize. Set `roboflow.auto_upload_after_training: true` to make newly generated configs opt into post-training upload/deploy by default.

---

## ClearML Integration

```sh
uv run clearml-init
```

Follow the prompts to paste your API credentials. All subsequent training runs automatically sync hyper-parameters, metrics, and artifacts.

If ClearML is not configured when training starts, YOLOmatic prompts you to continue without it or cancel.

---

## Versioning

Version is managed in `pyproject.toml` as the single source of truth:

```sh
uv run bump patch   # 4.3.0 -> 4.3.1
uv run bump minor   # 4.3.0 -> 4.4.0
uv run bump major   # 4.3.0 -> 5.0.0
uv run bump 5.0.0   # set exact version
```

The TUI footer and About screen read the version live from `pyproject.toml`. After bumping, lock the updated version:

```sh
uv lock
```

---

## Troubleshooting

### PyTorch Cannot Detect the GPU

If `nvidia-smi` works but `torch.cuda.is_available()` returns `False`:

1. Start training with `uv run yolomatic-train`.
2. Choose **Fix CUDA-enabled PyTorch now** when prompted.

Manual verification:

```sh
uv run python -c "import torch, numpy; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), numpy.__version__)"
```

Manual repair on Windows:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio
.\.venv\Scripts\python.exe -m pip install --no-cache-dir --force-reinstall torch torchvision torchaudio "numpy>=1.24.4" --index-url https://download.pytorch.org/whl/cu128
```

### YOLO-NAS Configs No Longer Train

YOLO-NAS is deprecated because SuperGradients conflicts with RF-DETR's modern training dependency stack. Use a YOLO, RF-DETR, or Detectron2 config instead.

### ClearML Is Not Configured

```sh
uv run clearml-init
```

Or choose **Continue Without ClearML** when the training TUI prompts.

### macOS — No GPU Acceleration

- NVIDIA CUDA is not supported on macOS.
- Apple Silicon: use an MPS-capable PyTorch build.
- Intel Mac or no MPS: training runs on CPU automatically.

### SAM 3.1 — HuggingFace Authentication

SAM 3.1 is a gated model. You must:

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Accept Meta's terms at [huggingface.co/facebook/sam3.1](https://huggingface.co/facebook/sam3.1).
3. Set `HF_TOKEN` in your environment or enter it interactively when prompted.

---

## Related Documentation

| Document | Purpose |
|---|---|
| [`YOLO_GUIDE.md`](YOLO_GUIDE.md) | Workflow guide — model selection, deployment scenarios, export options |
| [`MODELS.md`](MODELS.md) | Architecture reference — benchmark tables, model family comparison |
| [`llms.txt`](llms.txt) | AI-readable repository map for LLM assistants and documentation retrieval |
| [`CITATION.cff`](CITATION.cff) | Citation metadata for academic use |

---

## AI & Search Discovery

YOLOmatic keeps a curated [`llms.txt`](llms.txt) file at the repository root. It summarizes the project, supported model families, entrypoints, important docs, and non-goals for LLM search, code assistants, and retrieval-augmented documentation tools.

Traditional search and package discovery are supported through this README, GitHub topics, PyPI metadata in `pyproject.toml`, and descriptive documentation pages.

**Search keywords:** yolo training cli, automated yolo training, yolo26 training, yolov12 training, yolov11 training, ultralytics training pipeline, RF-DETR training, rfdetr fine tuning, yolo configuration generator, instance segmentation training, object detection cli, pytorch yolo, onnx export, clearml yolo, roboflow yolo upload, tensorboard yolo, edge deployment yolo, yolo tui, yolo interactive terminal, SAM 3.1 segmentation, segment anything model, detectron2 training, dataset augmentation, model benchmarking, UMAP vector analysis, labelbox to yolo, ndjson to yolo, labelbox to coco, ndjson to coco, labelbox converter, computer vision dataset tools.


---

## License

Licensed under the [Apache License 2.0](LICENSE.md).

---

<div align="center">

Built on top of [Ultralytics](https://github.com/ultralytics/ultralytics), [RF-DETR](https://github.com/roboflow/rf-detr), [Meta SAM](https://github.com/facebookresearch/sam2), and [ClearML](https://github.com/allegroai/clearml).

</div>
