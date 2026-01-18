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

1. Prepare your dataset in the following structure:

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

2. Run the configuration script:

```sh
python3 run.py
```

This interactive script will guide you through selecting a YOLO model and dataset, with an improved user interface featuring stylized headers and enhanced table presentations.

3. Start the training process:

```sh
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

- `run.py`: Interactive script for model and dataset selection with enhanced UI
- `Yolov_trainer.py`: Main training script
- `requirements.txt`: List of Python dependencies
- `LICENSE.md`: Apache License 2.0

## License

This project is licensed under the Apache License 2.0. See the [LICENSE.md](LICENSE.md) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO implementations
- [ClearML](https://github.com/allegroai/clearml) for experiment tracking
