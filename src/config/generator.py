import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import psutil
import yaml

try:
    from src.models.data import model_data_dict
    from src.utils.ml_dependencies import MLDependencyError, import_torch
except ImportError:
    try:
        from models.data import model_data_dict
        from utils.ml_dependencies import MLDependencyError, import_torch
    except ImportError:
        model_data_dict = {}
        MLDependencyError = RuntimeError

        def import_torch() -> object:
            raise RuntimeError("torch is not available.")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

AUGMENTATION_PROFILES: dict[str, dict[str, float]] = {
    "minimum": {},
    "low": {
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.1,
    },
    "medium": {
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.1,
        "degrees": 5.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "translate": 0.1,
        "scale": 0.5,
    },
}

COMPUTE_BATCH_TARGETS: dict[str, float | int] = {
    "conservative": 0.70,
    "balanced": 0.85,
    "aggressive": 0.95,
}

WORKER_PROFILE_RATIOS: dict[str, float] = {
    "light": 0.33,
    "medium": 0.60,
    "heavy": 0.85,
}

MODEL_VARIANT_RANKS = {
    "n": 0,
    "t": 0,
    "s": 1,
    "m": 2,
    "b": 2,
    "c": 3,
    "l": 3,
    "x": 4,
    "e": 4,
}


class BaseConfigGenerator:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        logger.info(f"Initialized with dataset path: {self.dataset_path}")
        self.data_yaml = None
        self.dataset_metrics: dict[str, Any] = {}
        self.system_metrics: dict[str, Any] = {}
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

                    logger.info("Resolved paths:")
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
                            "task_type": self._detect_dataset_type(train_path),
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

    def _detect_dataset_type(self, train_path: Path) -> str:
        """Analyze label files to detect if dataset is detection or segmentation."""
        if not train_path or not str(train_path).strip():
            return "unknown"

        labels_dir = Path(train_path).parent / "labels"
        if not labels_dir.exists():
            # Sometimes it's directly in labels
            labels_dir = Path(self.dataset_path) / "train" / "labels"
            if not labels_dir.exists():
                return "unknown"

        try:
            # Check up to 5 label files
            checked = 0
            for label_file in labels_dir.glob("*.txt"):
                if checked >= 5:
                    break
                with open(label_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        if len(parts) == 5:
                            return "detection"
                        elif len(parts) > 5:
                            return "segmentation"
                checked += 1
        except Exception as e:
            logger.error(f"Error reading label files to detect type: {e}")

        return "unknown"

    def _get_optimal_workers(self) -> int:
        """Get optimal number of workers based on CPU cores."""
        import multiprocessing

        return min(8, multiprocessing.cpu_count())

    def _detect_device(self) -> str:
        """Detect available device including MPS for Mac Silicon."""
        try:
            torch = import_torch()
        except MLDependencyError:
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _collect_dataset_metrics(self) -> dict[str, Any]:
        total_size_bytes = 0
        image_count = 0
        label_count = 0
        other_file_count = 0

        for file_path in self.dataset_path.rglob("*"):
            if not file_path.is_file():
                continue

            try:
                total_size_bytes += file_path.stat().st_size
            except OSError:
                continue

            suffix = file_path.suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                image_count += 1
            elif suffix == ".txt" and "labels" in file_path.parts:
                label_count += 1
            else:
                other_file_count += 1

        metrics = {
            "total_size_bytes": total_size_bytes,
            "image_count": image_count,
            "label_count": label_count,
            "other_file_count": other_file_count,
            "total_file_count": image_count + label_count + other_file_count,
        }
        self.dataset_metrics = metrics
        return metrics

    def _collect_system_metrics(self) -> dict[str, Any]:
        available_ram_bytes = int(psutil.virtual_memory().available)
        cpu_count = max(1, os.cpu_count() or 1)
        device = self._detect_device()
        available_gpu_memory_bytes: int | None = None
        total_gpu_memory_bytes: int | None = None

        if device == "cuda":
            try:
                torch = import_torch()
                if torch.cuda.is_available():
                    available_values: list[int] = []
                    total_values: list[int] = []
                    for device_index in range(torch.cuda.device_count()):
                        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
                        available_values.append(int(free_bytes))
                        total_values.append(int(total_bytes))
                    if available_values:
                        available_gpu_memory_bytes = max(available_values)
                        total_gpu_memory_bytes = max(total_values)
            except Exception:
                available_gpu_memory_bytes = None
                total_gpu_memory_bytes = None

        metrics = {
            "available_ram_bytes": available_ram_bytes,
            "cpu_count": cpu_count,
            "device": device,
            "available_gpu_memory_bytes": available_gpu_memory_bytes,
            "total_gpu_memory_bytes": total_gpu_memory_bytes,
        }
        self.system_metrics = metrics
        return metrics

    def _to_gib(self, value: int | None) -> float:
        if value is None:
            return 0.0
        return value / (1024**3)

    def _ratio(self, numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    def _parse_metric_number(self, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)

        text = str(value).strip()
        if not text or text == "-":
            return None

        cleaned = []
        decimal_found = False
        for char in text:
            if char.isdigit():
                cleaned.append(char)
            elif char == "." and not decimal_found:
                cleaned.append(char)
                decimal_found = True
            elif cleaned:
                break

        if not cleaned:
            return None

        try:
            return float("".join(cleaned))
        except ValueError:
            return None

    def _infer_model_variant_rank(self, model_choice: str) -> int:
        normalized = model_choice.lower()
        normalized = normalized.replace("-seg", "")
        if normalized.startswith("yolov10-"):
            suffix = normalized.split("-")[-1]
        else:
            suffix = normalized[-1:]
        return MODEL_VARIANT_RANKS.get(suffix, 2)

    def _classify_model_heaviness(
        self,
        params_millions: float | None,
        flops_billions: float | None,
        variant_rank: int,
    ) -> str:
        score = variant_rank
        if params_millions is not None:
            if params_millions >= 50:
                score += 2
            elif params_millions >= 20:
                score += 1
        if flops_billions is not None:
            if flops_billions >= 180:
                score += 2
            elif flops_billions >= 70:
                score += 1

        if score <= 1:
            return "light"
        if score <= 4:
            return "medium"
        return "heavy"

    def _get_model_metrics(self, model_choice: str) -> dict[str, Any]:
        normalized = model_choice.lower()
        family_key = normalized
        if normalized.startswith("yolo26"):
            family_key = "yolo26-seg" if normalized.endswith("-seg") else "yolo26"
        elif normalized.startswith("yolov12"):
            family_key = "yolov12-seg" if normalized.endswith("-seg") else "yolov12"
        elif normalized.startswith("yolov11"):
            family_key = "yolov11-seg" if normalized.endswith("-seg") else "yolov11"
        elif normalized.startswith("yolov10"):
            family_key = "yolov10"
        elif normalized.startswith("yolov9"):
            family_key = "yolov9-seg" if normalized.endswith("-seg") else "yolov9"
        elif normalized.startswith("yolov8"):
            family_key = "yolov8-seg" if normalized.endswith("-seg") else "yolov8"
        elif normalized.startswith("yolox"):
            family_key = "yolox"

        variant_rank = self._infer_model_variant_rank(model_choice)
        params_millions: float | None = None
        flops_billions: float | None = None
        input_size: float | None = None

        for entry in model_data_dict.get(family_key, []):
            entry_name = str(entry.get("Model", "")).lower()
            if entry_name == normalized:
                params_millions = self._parse_metric_number(entry.get("params (M)"))
                flops_billions = self._parse_metric_number(
                    entry.get("FLOPs (B)", entry.get("FLOPs (G)"))
                )
                input_size = self._parse_metric_number(entry.get("Input Size"))
                break

        heaviness = self._classify_model_heaviness(
            params_millions,
            flops_billions,
            variant_rank,
        )

        return {
            "model_choice": model_choice,
            "family_key": family_key,
            "variant_rank": variant_rank,
            "params_millions": params_millions,
            "flops_billions": flops_billions,
            "input_size": input_size,
            "heaviness": heaviness,
        }

    def _recommend_augmentation_profile(
        self,
        model_metrics: dict[str, Any],
        dataset_metrics: dict[str, Any],
        system_metrics: dict[str, Any],
    ) -> str:
        available_ram_gib = self._to_gib(system_metrics["available_ram_bytes"])
        dataset_size_gib = self._to_gib(dataset_metrics["total_size_bytes"])
        image_count = int(dataset_metrics["image_count"])
        total_file_count = int(dataset_metrics["total_file_count"])
        score = 0

        if image_count <= 1500:
            score += 1
        if dataset_size_gib <= 5 and total_file_count <= 20000:
            score += 1
        if available_ram_gib < 12:
            score -= 1
        if system_metrics["device"] == "cpu":
            score -= 1
        if dataset_size_gib >= 20 or total_file_count >= 100000:
            score -= 1
        if model_metrics["heaviness"] == "heavy":
            score -= 1
        elif model_metrics["heaviness"] == "light" and available_ram_gib >= 16:
            score += 1

        if score >= 1:
            return "medium"
        if score <= -2:
            return "minimum"
        return "low"

    def _recommend_compute_profile(
        self,
        model_metrics: dict[str, Any],
        dataset_metrics: dict[str, Any],
        system_metrics: dict[str, Any],
    ) -> str:
        available_ram_gib = self._to_gib(system_metrics["available_ram_bytes"])
        available_gpu_gib = self._to_gib(system_metrics["available_gpu_memory_bytes"])
        dataset_to_ram_ratio = self._ratio(
            int(dataset_metrics["total_size_bytes"]),
            int(system_metrics["available_ram_bytes"]),
        )
        total_file_count = int(dataset_metrics["total_file_count"])
        score = 0

        if available_ram_gib >= 32:
            score += 1
        elif available_ram_gib < 12:
            score -= 1

        if dataset_to_ram_ratio > 0.7:
            score -= 1
        elif dataset_to_ram_ratio < 0.2:
            score += 1

        if total_file_count > 50000:
            score -= 1

        if system_metrics["device"] == "cuda":
            if available_gpu_gib >= 12:
                score += 1
            elif available_gpu_gib and available_gpu_gib < 6:
                score -= 1
        elif system_metrics["device"] == "cpu":
            score -= 1

        if model_metrics["heaviness"] == "heavy":
            score -= 2
            if available_gpu_gib >= 20:
                score += 1
        elif model_metrics["heaviness"] == "medium":
            score -= 1
        elif model_metrics["heaviness"] == "light":
            score += 1

        if score >= 2:
            return "aggressive"
        if score <= -1:
            return "conservative"
        return "balanced"

    def _calculate_worker_profiles(
        self,
        model_metrics: dict[str, Any],
        dataset_metrics: dict[str, Any],
        system_metrics: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        cpu_count = int(system_metrics["cpu_count"])
        available_ram_gib = self._to_gib(system_metrics["available_ram_bytes"])
        available_gpu_gib = self._to_gib(system_metrics["available_gpu_memory_bytes"])
        dataset_to_ram_ratio = self._ratio(
            int(dataset_metrics["total_size_bytes"]),
            int(system_metrics["available_ram_bytes"]),
        )
        total_file_count = int(dataset_metrics["total_file_count"])

        safe_capacity = min(16, max(2, cpu_count - 1))

        if available_ram_gib < 8:
            safe_capacity = min(safe_capacity, 4)
        elif available_ram_gib < 16:
            safe_capacity = min(safe_capacity, 6)
        elif available_ram_gib < 32:
            safe_capacity = min(safe_capacity, 8)
        elif available_ram_gib < 64:
            safe_capacity = min(safe_capacity, 12)

        if dataset_to_ram_ratio > 1.0:
            safe_capacity = max(2, safe_capacity - 4)
        elif dataset_to_ram_ratio > 0.5:
            safe_capacity = max(2, safe_capacity - 2)

        if total_file_count > 100000:
            safe_capacity = max(2, safe_capacity - 2)
        elif total_file_count < 10000 and available_ram_gib >= 16:
            safe_capacity = min(cpu_count, safe_capacity + 1)

        if (
            system_metrics["device"] == "cuda"
            and available_gpu_gib >= 12
            and available_ram_gib >= 32
        ):
            safe_capacity = min(cpu_count, safe_capacity + 2)

        if model_metrics["heaviness"] == "light":
            safe_capacity = min(cpu_count, safe_capacity + 1)
        elif model_metrics["heaviness"] == "heavy":
            safe_capacity = max(2, safe_capacity - 1)

        safe_capacity = max(2, min(cpu_count, safe_capacity))

        worker_counts: dict[str, int] = {}
        for profile_name, ratio in WORKER_PROFILE_RATIOS.items():
            worker_counts[profile_name] = max(2, math.ceil(safe_capacity * ratio))

        worker_counts["medium"] = max(
            worker_counts["light"] + 1,
            min(cpu_count, worker_counts["medium"]),
        )
        worker_counts["heavy"] = max(
            worker_counts["medium"] + 1,
            min(cpu_count, worker_counts["heavy"]),
        )

        worker_counts["light"] = min(worker_counts["light"], cpu_count)
        worker_counts["medium"] = min(worker_counts["medium"], cpu_count)
        worker_counts["heavy"] = min(worker_counts["heavy"], cpu_count)

        return {
            "light": {
                "workers": worker_counts["light"],
                "description": "Lower RAM pressure, safer for large datasets or limited memory",
            },
            "medium": {
                "workers": worker_counts["medium"],
                "description": "Balanced throughput for most systems",
            },
            "heavy": {
                "workers": worker_counts["heavy"],
                "description": "Highest data-loading throughput if RAM and storage can keep up",
            },
        }

    def get_regular_yolo_profile_context(self, model_choice: str) -> dict[str, Any]:
        self.extract_dataset_info()
        model_metrics = self._get_model_metrics(model_choice)
        dataset_metrics = self._collect_dataset_metrics()
        system_metrics = self._collect_system_metrics()
        worker_profiles = self._calculate_worker_profiles(
            model_metrics,
            dataset_metrics,
            system_metrics,
        )
        compute_profile = self._recommend_compute_profile(
            model_metrics,
            dataset_metrics,
            system_metrics,
        )
        augmentation_profile = self._recommend_augmentation_profile(
            model_metrics,
            dataset_metrics,
            system_metrics,
        )
        worker_profile = {
            "conservative": "light",
            "balanced": "medium",
            "aggressive": "heavy",
        }[compute_profile]

        return {
            "model_metrics": model_metrics,
            "dataset_metrics": dataset_metrics,
            "system_metrics": system_metrics,
            "worker_profiles": worker_profiles,
            "recommended_profiles": {
                "augmentation": augmentation_profile,
                "compute": compute_profile,
                "worker": worker_profile,
            },
        }

    def _can_enable_cache(
        self,
        compute_profile: str,
        dataset_metrics: dict[str, Any],
        system_metrics: dict[str, Any],
    ) -> bool:
        available_ram_gib = self._to_gib(system_metrics["available_ram_bytes"])
        dataset_to_ram_ratio = self._ratio(
            int(dataset_metrics["total_size_bytes"]),
            int(system_metrics["available_ram_bytes"]),
        )

        if compute_profile == "conservative":
            return False
        if compute_profile == "balanced":
            return available_ram_gib >= 16 and dataset_to_ram_ratio <= 0.33
        return available_ram_gib >= 24 and dataset_to_ram_ratio <= 0.5

    def _apply_regular_yolo_profiles(
        self,
        config: dict[str, Any],
        profile_selection: dict[str, str],
        profile_context: dict[str, Any],
    ) -> dict[str, Any]:
        training = config["training"]
        dataset_metrics = profile_context["dataset_metrics"]
        system_metrics = profile_context["system_metrics"]
        worker_profile = profile_context["worker_profiles"][profile_selection["worker"]]
        compute_profile = profile_selection["compute"]
        device = str(training["device"])

        if device == "cuda":
            training["batch"] = COMPUTE_BATCH_TARGETS[compute_profile]
        else:
            training["batch"] = -1

        training["cache"] = self._can_enable_cache(
            compute_profile,
            dataset_metrics,
            system_metrics,
        )
        training["workers"] = int(worker_profile["workers"])

        for key in (
            "degrees",
            "flipud",
            "fliplr",
            "hsv_h",
            "hsv_s",
            "hsv_v",
            "mixup",
            "mosaic",
            "scale",
            "translate",
        ):
            training.pop(key, None)

        training.update(AUGMENTATION_PROFILES[profile_selection["augmentation"]])
        return config


class YOLONASConfigGenerator(BaseConfigGenerator):
    def generate_config(
        self,
        model_choice: str,
        profile_selection: dict[str, str] | None = None,
        profile_context: dict[str, Any] | None = None,
    ) -> Dict:
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
    def generate_config(
        self,
        model_choice: str,
        profile_selection: dict[str, str] | None = None,
        profile_context: dict[str, Any] | None = None,
    ) -> Dict:
        """Generate YOLO specific configuration."""
        if profile_context is None:
            profile_context = self.get_regular_yolo_profile_context(model_choice)
        elif not self.dataset_info.get("task_type"):
            self.extract_dataset_info()

        if profile_selection is None:
            profile_selection = dict(profile_context["recommended_profiles"])

        config = {
            "settings": {
                "model_type": model_choice.lower(),
                "dataset": self.dataset_path.name,
            },
            "clearml": {
                "project_name": f"YOLO Training - {model_choice.upper()}",
                "task_name_format": "%Y-%m-%d-%H-%M",
            },
            "training": {
                "epochs": 300,
                "patience": 50,
                "imgsz": 640,
                "batch": -1,
                "cache": False,
                "workers": self._get_optimal_workers(),
                "optimizer": "auto",
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
        return self._apply_regular_yolo_profiles(
            config,
            profile_selection,
            profile_context,
        )
