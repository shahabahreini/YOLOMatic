# YOLOmatic

[ Automated Yolo Training ]

## Overview

Automated Yolo Training is a comprehensive solution for training YOLO (You Only Look Once) models, with support for YOLO26, YOLOv12, YOLOv11, YOLOv10, YOLOv9, YOLOv8, and YOLOX. This project streamlines the process of selecting models, configuring parameters, and training on custom datasets, all while integrating with ClearML for efficient experiment tracking.

<https://github.com/user-attachments/assets/d7c1e7b6-cf1b-43ca-90c6-8357e37b4638>

## Features

- Support for multiple YOLO versions: **YOLO26** (latest), YOLOv12, YOLOv11, YOLOv10, YOLOv9, YOLOv8, and YOLOX
- **YOLO26 Highlights**:
  - End-to-end NMS-free inference for faster deployment
  - Up to 43% faster CPU inference
  - Superior performance on edge devices
  - Enhanced support for detection, segmentation, pose estimation, OBB, and classification
- User-friendly command-line interface for model and dataset selection
- Enhanced UI with professional-looking headers and improved table styling
- Integration with ClearML for experiment tracking and management
- Configurable training parameters via YAML configuration
- ONNX export with optimization options
- Automatic config backup and version control

## Requirements

- Python 3.7+
- CUDA-compatible GPU (recommended)
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:

```sh
git clone https://github.com/shahabahreini/Automated-Yolo-Training.git
cd automated-yolo-training
```

2. Install the required packages:

```sh
pip install -r requirements.txt
```

## ClearML Setup

To use ClearML for experiment tracking, follow these steps:

1. **Install ClearML**:

   ```sh
   pip install clearml
   ```

2. **Configure ClearML**:
   Run the following command to configure ClearML with your credentials:

   ```sh
   clearml-init
   ```

   This will prompt you to enter your ClearML credentials (API key, secret, and server). You can find these credentials in your ClearML account settings.

3. **Update `config.yaml`**:
   Ensure that the `config.yaml` file includes the correct ClearML integration settings. The relevant section should look like this:

   ```yaml
   clearml:
     sdk:
       api:
         api_server: "https://api.clear.ml"
         web_server: "https://app.clear.ml"
         files_server: "https://files.clear.ml"
       credentials:
         access_key: "YOUR_ACCESS_KEY"
         secret_key: "YOUR_SECRET_KEY"
   ```

## Usage

You can also install the project as a package for easier invocation:

```sh
pip install -e .  # install in editable/develop mode
# then use the CLI command
yolomatic  # same as `python -m src.cli.run`
```

### Versioning

Version numbers are managed with the `uv` tool (already listed in `requirements.txt`).
After installation, you can inspect or bump the project version from the project root:

```sh
# show current version
uv version

# bump patch/minor/major
uv version --bump patch   # 0.1.2 → 0.1.3
uv version --bump minor   # 0.1.2 → 0.2.0
uv version --bump major   # 0.1.2 → 1.0.0

# set an exact value
uv version 1.5.0

# you can also invoke via `uv run` to ensure the environment is activated:
uv run uv version --bump patch

# dry-run before writing
uv version --bump patch --dry-run
```

All commands update the `version` field in `pyproject.toml` and (unless
`--no-sync` is given) refresh the lockfile.

1. Prepare your dataset in the following structure (leave the `datasets` folder at project root):

```
datasets/
└── your_dataset_name/
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

2. Run the configuration script. The code now lives under `src` and can be executed as a module, but for backwards compatibility simple wrappers are provided at the project root:

```sh
# preferred (module invocation)
python3 -m src.cli.run
# or after installing package
yolomatic
# or with uv runner
uv run -m src.cli.run

# legacy wrappers (exact same behavior)
python3 run.py
```

This interactive script will guide you through selecting a YOLO model and dataset, with an improved user interface featuring stylized headers and enhanced table presentations.

3. Start the training process:

```sh
# module execution:
python3 -m src.trainers.yolo_trainer
# or with uv:
uv run yolomatic-train
# or (legacy runner):
uv run -m src.trainers.yolo_trainer

# legacy wrapper (same as above):
python3 Yolov_trainer.py
```

## Configuration

The `config.yaml` file contains all the necessary settings for training. Key sections include:

- `settings`: General settings like model type and dataset
- `clearml`: ClearML integration settings
- `training`: Training hyperparameters
- `model`: Dataset and model-specific parameters
- `export`: Model export settings

Modify this file to customize your training process.

## Files

- `src/cli/run.py` (or `python -m src.cli.run`): Interactive script for model and dataset selection with enhanced UI
- `src/trainers/yolo_trainer.py` (or `python -m src.trainers.yolo_trainer`): Main training script

Legacy wrappers `run.py`, `Yolov_trainer.py`, `YoloNAS_trainier.py`, `NAS_datasetAnalyzer.py` and others remain at project root for backward compatibility.

- `requirements.txt`: List of Python dependencies
- `LICENSE.md`: Apache License 2.0

## License

This project is licensed under the Apache License 2.0. See the [LICENSE.md](LICENSE.md) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO implementations
- [ClearML](https://github.com/allegroai/clearml) for experiment tracking
