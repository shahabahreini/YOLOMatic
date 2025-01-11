import os
import yaml
from pathlib import Path
import logging
from typing import Dict, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseConfigGenerator:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        logger.info(f"Initialized with dataset path: {self.dataset_path}")
        self.data_yaml = None
        self.dataset_info = {
            "classes": [],
            "num_classes": 0,
            "train_path": "",
            "valid_path": "",
            "test_path": "",
        }

    def read_yaml(self, file_path: str) -> Optional[Dict]:
        """Read and parse YAML file."""
        try:
            with open(file_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error reading YAML file {file_path}: {e}")
            return None

    def extract_dataset_info(self):
        """Extract dataset information from data.yaml"""
        logger.info(f"Looking for data.yaml in: {self.dataset_path}")

        # Try multiple possible locations for data.yaml
        possible_paths = [
            Path(self.dataset_path) / "data.yaml",
            Path(self.dataset_path) / "dataset.yaml",
            Path(self.dataset_path).parent / "data.yaml",
        ]

        data_yaml_path = None
        for path in possible_paths:
            if path.exists():
                data_yaml_path = path
                logger.info(f"Found data.yaml at: {path}")
                break

        if data_yaml_path:
            try:
                with open(data_yaml_path, "r") as f:
                    self.data_yaml = yaml.safe_load(f)
                    logger.info(f"Loaded YAML content: {self.data_yaml}")

                if self.data_yaml:
                    # Extract class information
                    classes = self.data_yaml.get("names", [])
                    num_classes = self.data_yaml.get(
                        "nc", len(classes) if classes else 0
                    )

                    logger.info(f"Found classes: {classes}")
                    logger.info(f"Number of classes: {num_classes}")

                    # Resolve relative paths
                    yaml_dir = data_yaml_path.parent
                    train_path = self._resolve_path(
                        yaml_dir, self.data_yaml.get("train", "")
                    )
                    valid_path = self._resolve_path(
                        yaml_dir, self.data_yaml.get("val", "")
                    )
                    test_path = self._resolve_path(
                        yaml_dir, self.data_yaml.get("test", "")
                    )

                    logger.info(f"Resolved paths:")
                    logger.info(f"Train: {train_path}")
                    logger.info(f"Valid: {valid_path}")
                    logger.info(f"Test: {test_path}")

                    # Update dataset info
                    self.dataset_info.update(
                        {
                            "classes": classes,
                            "num_classes": num_classes,
                            "train_path": str(train_path),
                            "valid_path": str(valid_path),
                            "test_path": str(test_path),
                        }
                    )

                    logger.info(f"Updated dataset info: {self.dataset_info}")
                    return True
            except Exception as e:
                logger.error(f"Error reading YAML file: {e}")
                raise  # Re-raise the exception for debugging

        logger.warning("No valid data.yaml found")
        return False

    def _resolve_path(self, base_path: Path, relative_path: str) -> Path:
        """
        Resolve a potentially relative path against a base path.

        Args:
            base_path (Path): The base path to resolve against
            relative_path (str): The relative path from the YAML file

        Returns:
            Path: The resolved absolute path
        """
        if not relative_path:
            return Path("")

        # Convert the path to Path object
        path = Path(relative_path)

        # If it's an absolute path, return it as is
        if path.is_absolute():
            return path

        # If it starts with ../, resolve it relative to the base path
        resolved_path = (base_path / path).resolve()
        logger.info(
            f"Resolving {relative_path} relative to {base_path} -> {resolved_path}"
        )

        return resolved_path

    def _get_optimal_workers(self) -> int:
        """Get optimal number of workers based on CPU cores."""
        import multiprocessing

        return min(8, multiprocessing.cpu_count())

    def _detect_device(self) -> str:
        """Detect available device including MPS for Mac Silicon."""
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


class YOLONASConfigGenerator(BaseConfigGenerator):
    def generate_config(self, model_choice: str) -> Dict:
        """Generate YOLO NAS specific configuration."""
        self.extract_dataset_info()

        config = {
            "experiment": {
                "name_prefix": f"YoloNAS-{model_choice}",
                "checkpoint_dir": "./results/",
                "console_log_file": "./logs/console.log",
                "description": "Yolo NAS Training",
            },
            "clearml": {
                "project_name": "YOLO NAS Training",
                "tags": [f"YOLO NAS {model_choice}"],
            },
            "dataset": {
                "name": self.dataset_path.name,
                "base_dir": str(self.dataset_path),
                "structure": {
                    "train": {"images": "train/images", "labels": "train/labels"},
                    "valid": {"images": "valid/images", "labels": "valid/labels"},
                    "test": {"images": "test/images", "labels": "test/labels"},
                },
                "classes": self.dataset_info["classes"],
            },
            "training": {
                "batch_size": 35,
                "max_epochs": 20,
                "num_workers": self._get_optimal_workers(),
                "optimizer": {
                    "weight_decay": 0.0001,
                    "zero_weight_decay_on_bias_and_bn": True,
                },
                "learning_rate": {
                    "warmup_initial_lr": 1.0e-6,
                    "initial_lr": 5.0e-3,
                    "warmup_epochs": 3,
                    "warmup_mode": "LinearEpochLRWarmup",
                },
                "mixed_precision": {
                    "enabled": True,
                    "dtype": "qint8",
                    "loss_scale_value": 1024,
                    "loss_scale_method": "dynamic",
                },
                "checkpoint": {
                    "save_last": True,
                    "save_best_only": True,
                    "monitor": "mAP@0.50",
                    "mode": "max",
                },
                "ema": {"enabled": True, "decay": 0.9, "decay_type": "threshold"},
            },
            "model": {
                "name": model_choice.lower(),
                "pretrained_weights": "coco",
                "loss": {
                    "type": "PPYoloELoss",
                    "use_static_assigner": False,
                    "reg_max": 16,
                },
                "metrics": {
                    "score_threshold": 0.1,
                    "top_k_predictions": 300,
                    "post_prediction": {
                        "score_threshold": 0.01,
                        "nms_top_k": 1000,
                        "max_predictions": 300,
                        "nms_threshold": 0.7,
                    },
                },
            },
            "export": {
                "output_name": f"{model_choice.lower()}_int8.onnx",
                "calibration": {"batch_size": 8, "num_samples": 32, "num_workers": 0},
            },
        }
        return config


class YOLOConfigGenerator(BaseConfigGenerator):
    def generate_config(self, model_choice: str) -> Dict:
        """Generate YOLO specific configuration."""
        self.extract_dataset_info()

        # Default configuration for standard YOLO models
        config = {
            "settings": {
                "model_type": model_choice.lower(),
                "dataset": self.dataset_path.name,
            },
            "clearml": {
                "project_name": "YOLOv10 Training",
                "task_name_format": "%Y-%m-%d-%H-%M",
            },
            "training": {
                "epochs": 150,
                "imgsz": 640,
                "batch": 16,  # Will be adjusted based on GPU memory
                "cache": True,
                "workers": self._get_optimal_workers(),
                "label_smoothing": 0.1,
                "close_mosaic": 50,
                # Augmentation parameters
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "degrees": 0.0,
                "translate": 0.1,
                "scale": 0.5,
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "fliplr": 0.5,
                "mixup": 0.0,
                "device": self._detect_device(),
            },
            "model": {
                "data_dir": f"./datasets/{self.dataset_path.name}",
                "train_images_dir": "train/images",
                "train_labels_dir": "train/labels",
                "val_images_dir": "valid/images",
                "val_labels_dir": "valid/labels",
                "test_images_dir": "test/images",
                "test_labels_dir": "test/labels",
                "classes": self.dataset_info["classes"],
            },
            "export": {
                "format": "onnx",
                "optimize": True,
                "half": False,
                "nms": True,
                "int8": True,
            },
        }
        return config

    def _detect_device(self) -> str:
        """Detect available device including MPS for Mac Silicon."""
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _get_optimal_workers(self) -> int:
        """Get optimal number of workers based on CPU cores."""
        import multiprocessing

        return min(8, multiprocessing.cpu_count())
