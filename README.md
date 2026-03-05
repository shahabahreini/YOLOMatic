# YOLOmatic

[ Automated Yolo Training ]

**YOLOmatic** is a comprehensive, user-friendly CLI tool for training modern YOLO (You Only Look Once) architectures. It supports **YOLO26**, YOLOv12, YOLOv11, YOLOv10, YOLOv9, YOLOv8, and YOLOX.

It streamlines model selection, robust dataset configuration with top-notch Computer Vision best practices (AutoBatch, Early Stopping, Cosine LR), training execution, TensorBoard monitoring, and effortless ClearML integration.

---

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation & Setup](#installation--setup)
4. [Dataset Preparation](#dataset-preparation)
5. [Usage Guide](#usage-guide)
   - [1. Configure Training](#1-configure-training)
   - [2. Start Training](#2-start-training)
   - [3. Monitor Training](#3-monitor-training)
6. [ClearML Integration](#clearml-integration)
7. [Versioning](#versioning)
8. [License](#license)

---

## Features

- **Broad Model Support**: Train YOLO26 (latest), YOLOv12, YOLOv11, YOLOv10, YOLOv9, YOLOv8, and YOLOX effortlessly.
- **Top-Notch Training Defaults**: Automatically utilizes CV best practices like AutoBatch, Early Stopping, and optimized augmentations for maximum performance and GPU safety.
- **Sleek CLI**: Intuitive terminal headers and tables guide you through model and dataset selection.
- **Interactive Monitoring**: Built-in TensorBoard launcher to dynamically find and monitor training runs.
- **MLOps Integrations**: Seamlessly track experiments using ClearML.

---

## Prerequisites

- **Python**: 3.10+ recommended.
- **`uv` Package Manager**: This repository relies heavily on `uv` for blazing-fast dependency management and reliable project execution. If you don't have it installed:
  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Hardware**: A CUDA-compatible GPU is highly recommended for training.

---

## Installation & Setup

1. **Clone the repository**:

   ```sh
   git clone https://github.com/shahabahreini/Automated-Yolo-Training.git
   cd Automated-Yolo-Training
   ```

2. **Sync the project environment**:
   Using `uv`, quickly sync all dependencies and set up the virtual environment:
   ```sh
   uv sync
   ```

---

## Dataset Preparation

Place your custom datasets inside the `datasets/` folder at the root of the project. Your dataset must follow standard YOLO format hierarchy:

```
Automated-Yolo-Training/
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

## Usage Guide

YOLOmatic uses three primary commands executed seamlessly via `uv`.

### 1. Configure Training

Generate an optimized configurations file by interactively selecting your YOLO model and target dataset.

```sh
uv run yolomatic
```

_Follow the on-screen terminal menus to finalize your setup. A highly optimized YAML configuration file will be uniquely generated._

### 2. Start Training

Once configured, initiate the training sequence. The trainer will automatically read the latest config and apply robust augmentations and early stopping criteria.

```sh
uv run yolomatic-train
```

### 3. Monitor Training

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

We manage project versions directly using a custom CLI script that synchronizes `src/__version__.py` and `pyproject.toml`.

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

## License

This project is licensed under the Apache License 2.0. See the [LICENSE.md](LICENSE.md) file for details.

---

_Acknowledgments: Built proudly on top of [Ultralytics](https://github.com/ultralytics/ultralytics) and [ClearML](https://github.com/allegroai/clearml)._
