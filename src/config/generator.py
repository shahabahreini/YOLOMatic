import logging
import math
import os
from pathlib import Path
from typing import Any, Dict

import psutil
import yaml

try:
    from src.config.settings import load_settings, snapshot_clearml_settings, snapshot_roboflow_settings
    from src.datasets import prepare_dataset_for_family, prepared_format_for_family, summarize_dataset
    from src.datasets.cache import is_dataset_runtime_cache
    from src.models.detectron2 import get_detectron2_variant
    from src.models.rfdetr import get_rfdetr_variant
    from src.models.data import model_data_dict
    from src.utils.ml_dependencies import MLDependencyError, import_torch
except ImportError:
    try:
        from datasets import prepare_dataset_for_family, prepared_format_for_family, summarize_dataset
        from datasets.cache import is_dataset_runtime_cache
        from models.detectron2 import get_detectron2_variant
        from models.rfdetr import get_rfdetr_variant
        from models.data import model_data_dict
        from utils.ml_dependencies import MLDependencyError, import_torch
    except ImportError:
        model_data_dict = {}
        MLDependencyError = RuntimeError

        def get_rfdetr_variant(model_choice: str) -> object:
            raise RuntimeError(f"RF-DETR metadata is not available for {model_choice}.")

        def get_detectron2_variant(model_choice: str) -> object:
            raise RuntimeError(f"Detectron2 metadata is not available for {model_choice}.")

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
    "light": 0.25,
    "medium": 0.45,
    "heavy": 0.65,
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

TASK_SUFFIXES = ("-seg", "-cls", "-pose", "-obb")

# Explicit batch sizes for non-CUDA devices (MPS, CPU) where Ultralytics
# AutoBatch is a no-op and falls back to a hard-coded 16. Indexed by
# heaviness and compute profile so the generated config scales with both
# the model footprint and the user's chosen aggressiveness.
NON_CUDA_BATCH_SIZES: dict[str, dict[str, int]] = {
    "light": {"conservative": 8, "balanced": 16, "aggressive": 24},
    "medium": {"conservative": 4, "balanced": 8, "aggressive": 12},
    "heavy": {"conservative": 2, "balanced": 4, "aggressive": 6},
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

        try:
            summary = summarize_dataset(self.dataset_path)
            if summary.format in {"coco", "mixed"}:
                self.dataset_info.update(
                    {
                        "classes": summary.classes,
                        "num_classes": len(summary.classes),
                        "train_path": "",
                        "valid_path": "",
                        "test_path": "",
                        "task_type": summary.task,
                        "source_format": summary.format,
                    }
                )
                return True
        except Exception as error:
            logger.warning(f"Dataset summary fallback failed: {error}")

        logger.warning("No valid data.yaml found")
        return False

    def _prepared_dataset_config(self, family: str, task: str | None = None) -> dict[str, Any]:
        """Return public dataset config metadata with lazy format conversion."""
        try:
            prepared = prepare_dataset_for_family(self.dataset_path, family, task=task)
        except Exception as error:
            logger.warning(f"Dataset preparation metadata failed: {error}")
            summary = summarize_dataset(self.dataset_path)
            prepared_format = prepared_format_for_family(family, task)
            return {
                "name": self.dataset_path.name,
                "source_format": summary.format,
                "prepared_format": prepared_format,
                "source_dir": str(self.dataset_path),
                "prepared_dir": str(self.dataset_path),
                "base_dir": str(self.dataset_path),
                "classes": summary.classes or self.dataset_info.get("classes", []),
                "splits": {key: value.__dict__ for key, value in summary.splits.items()},
            }
        prepared["base_dir"] = prepared["prepared_dir"]
        return prepared

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

        normalized_path = str(path).replace("\\", "/")
        if normalized_path.startswith("../"):
            resolved_path = (self.dataset_path / normalized_path[3:]).resolve()
        else:
            resolved_path = (base_path / path).resolve()
        logger.info(
            f"Resolving {relative_path} relative to {base_path} -> {resolved_path}"
        )

        return resolved_path

    def _candidate_label_dirs(self, train_path: Path) -> list[Path]:
        """Return plausible locations for the training label files."""
        candidates: list[Path] = []
        seen: set[Path] = set()

        def _add(path: Path) -> None:
            try:
                resolved = path.resolve()
            except OSError:
                return
            if resolved in seen:
                return
            seen.add(resolved)
            candidates.append(path)

        if train_path and str(train_path).strip():
            train = Path(train_path)
            # Standard YOLO layout: sibling "labels" next to "images".
            _add(train.parent / "labels")
            # Some datasets nest labels directly under the split folder.
            _add(train / "labels")
            _add(train)
        # Fallback to the conventional dataset root layout.
        _add(Path(self.dataset_path) / "train" / "labels")
        _add(Path(self.dataset_path) / "labels" / "train")
        _add(Path(self.dataset_path) / "labels")
        return [path for path in candidates if path.exists() and path.is_dir()]

    def _detect_dataset_type(self, train_path: Path) -> str:
        """Detect whether labels encode bounding boxes or segmentation polygons.

        YOLO bbox rows have 5 whitespace-separated values; segmentation rows have
        an odd count >= 7 (class id + at least 3 polygon points). Counting across
        multiple files and lines avoids a single malformed entry flipping the
        classification.

        Pose datasets are identified authoritatively by the ``kpt_shape`` key in
        ``data.yaml`` (Ultralytics' own marker), since pose label rows
        (``class + bbox + K*ndim`` values) are otherwise easy to confuse with
        segmentation polygons.
        """
        if isinstance(self.data_yaml, dict) and self.data_yaml.get("kpt_shape"):
            return "pose"

        detection_hits = 0
        segmentation_hits = 0
        files_with_labels_scanned = 0
        MAX_FILES = 20
        MAX_LINES_PER_FILE = 50

        for labels_dir in self._candidate_label_dirs(Path(train_path)):
            try:
                label_files = sorted(labels_dir.glob("*.txt"))
            except OSError as error:
                logger.warning(f"Cannot list labels in {labels_dir}: {error}")
                continue

            for label_file in label_files:
                if files_with_labels_scanned >= MAX_FILES:
                    break
                file_had_label = False
                try:
                    with open(label_file, "r", encoding="utf-8") as handle:
                        for line_index, line in enumerate(handle):
                            if line_index >= MAX_LINES_PER_FILE:
                                break
                            parts = line.strip().split()
                            if not parts:
                                continue
                            file_had_label = True
                            count = len(parts)
                            if count == 5:
                                detection_hits += 1
                            elif count >= 7 and (count - 1) % 2 == 0:
                                segmentation_hits += 1
                except OSError as error:
                    logger.warning(
                        f"Skipping unreadable label file {label_file}: {error}"
                    )
                    continue
                if file_had_label:
                    files_with_labels_scanned += 1

            if detection_hits or segmentation_hits:
                break

        if detection_hits == 0 and segmentation_hits == 0:
            return "unknown"
        if segmentation_hits > detection_hits:
            return "segmentation"
        return "detection"

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
        runtime_cache_file_count = 0
        runtime_cache_size_bytes = 0

        for file_path in self.dataset_path.rglob("*"):
            if not file_path.is_file():
                continue

            try:
                file_size = file_path.stat().st_size
            except OSError:
                continue

            if is_dataset_runtime_cache(file_path):
                runtime_cache_file_count += 1
                runtime_cache_size_bytes += file_size
                continue

            total_size_bytes += file_size

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
            "runtime_cache_file_count": runtime_cache_file_count,
            "runtime_cache_size_bytes": runtime_cache_size_bytes,
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
        """Return a coarse size rank (0-4) for a YOLO variant name.

        Handles all supported shapes: ``YOLO11n``, ``YOLOv10-N``,
        ``YOLO26n-seg``, ``YOLOv9e``, ``YOLOX-M``. Strips task suffixes and the
        family-vs-variant separator, then looks up the trailing size letter.
        """
        normalized = model_choice.lower().strip()
        for suffix in TASK_SUFFIXES:
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                break
        if "-" in normalized:
            # For YOLOv10-B or YOLOX-M the variant letter is after the last dash.
            normalized = normalized.rsplit("-", 1)[-1]
        if not normalized:
            return 2
        return MODEL_VARIANT_RANKS.get(normalized[-1], 2)

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

    def _resolve_family_key(self, normalized: str) -> str:
        """Map a lowercased variant name to its key in ``model_data_dict``.

        Variant names in ``model_data_dict`` are inconsistent across families
        (e.g. ``YOLO11n`` for v11, ``YOLOv10-N`` for v10, ``YOLOX-M`` for YOLOX).
        This mapping table covers each observed shape explicitly so a new
        family name doesn't silently fall through to a missing key.
        """
        is_seg = normalized.endswith("-seg")
        is_pose = normalized.endswith("-pose")
        # Ordered list of (prefix, base_key, seg_key, pose_key). The first prefix
        # to match wins, so longer/more specific prefixes come first. ``pose_key``
        # is None for families without official pose weights (they fall back to
        # the raw name so a missing key surfaces instead of mis-mapping).
        prefixes = (
            ("yolo26", "yolo26", "yolo26-seg", "yolo26-pose"),
            ("yolov12", "yolov12", "yolov12-seg", None),
            ("yolo12", "yolov12", "yolov12-seg", None),
            ("yolov11", "yolov11", "yolov11-seg", "yolov11-pose"),
            ("yolo11", "yolov11", "yolov11-seg", "yolov11-pose"),
            ("yolov10", "yolov10", "yolov10", None),
            ("yolov9", "yolov9", "yolov9-seg", None),
            ("yolov8", "yolov8", "yolov8-seg", "yolov8-pose"),
            ("yolox", "yolox", "yolox", None),
        )
        for prefix, base_key, seg_key, pose_key in prefixes:
            if normalized.startswith(prefix):
                if is_pose:
                    return pose_key or normalized
                if is_seg:
                    return seg_key
                return base_key
        return normalized

    def _get_model_metrics(self, model_choice: str) -> dict[str, Any]:
        normalized = model_choice.lower()
        family_key = self._resolve_family_key(normalized)

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

        safe_capacity = min(12, max(2, cpu_count - 2))

        if available_ram_gib < 8:
            safe_capacity = min(safe_capacity, 4)
        elif available_ram_gib < 16:
            safe_capacity = min(safe_capacity, 5)
        elif available_ram_gib < 32:
            safe_capacity = min(safe_capacity, 7)
        elif available_ram_gib < 64:
            safe_capacity = min(safe_capacity, 10)

        if dataset_to_ram_ratio > 1.0:
            safe_capacity = max(2, safe_capacity - 4)
        elif dataset_to_ram_ratio > 0.5:
            safe_capacity = max(2, safe_capacity - 2)

        if total_file_count > 100000:
            safe_capacity = max(2, safe_capacity - 2)
        elif total_file_count < 10000:
            safe_capacity = max(2, safe_capacity - 1)

        if (
            system_metrics["device"] == "cuda"
            and available_gpu_gib >= 12
            and available_ram_gib >= 32
        ):
            safe_capacity = min(cpu_count, safe_capacity + 1)

        if system_metrics["device"] == "cpu":
            safe_capacity = max(2, safe_capacity - 1)

        if model_metrics["heaviness"] == "light":
            safe_capacity = min(cpu_count, safe_capacity + 1)
        elif model_metrics["heaviness"] == "heavy":
            safe_capacity = max(2, safe_capacity - 2)

        safe_capacity = max(2, min(cpu_count, safe_capacity))

        worker_counts: dict[str, int] = {}
        for profile_name, ratio in WORKER_PROFILE_RATIOS.items():
            worker_counts[profile_name] = max(2, math.ceil(safe_capacity * ratio))

        # Apply RAM/CPU caps to the heavy profile BEFORE enforcing the +1 gaps so
        # the final clamp to cpu_count doesn't silently erase them.
        heavy_cap = cpu_count
        if cpu_count <= 8:
            heavy_cap = min(heavy_cap, max(3, cpu_count - 1))
        if available_ram_gib < 16:
            heavy_cap = min(heavy_cap, 4)
        elif available_ram_gib < 32:
            heavy_cap = min(heavy_cap, 6)

        # Clamp every profile to cpu_count once, then restore monotonicity as
        # best we can. When cpu_count is so small that the three profiles
        # collapse onto the same value, accept that — there is no honest way
        # to distinguish them on a 2-core machine.
        worker_counts["light"] = max(2, min(worker_counts["light"], cpu_count))
        worker_counts["medium"] = max(2, min(worker_counts["medium"], cpu_count))
        worker_counts["heavy"] = max(2, min(worker_counts["heavy"], heavy_cap))

        worker_counts["medium"] = min(
            cpu_count, max(worker_counts["medium"], worker_counts["light"])
        )
        worker_counts["heavy"] = min(
            heavy_cap, max(worker_counts["heavy"], worker_counts["medium"])
        )

        return {
            "light": {
                "workers": worker_counts["light"],
                "description": "Safer baseline with lower RAM pressure and lower risk of dataloader contention",
            },
            "medium": {
                "workers": worker_counts["medium"],
                "description": "Balanced throughput for most systems without pushing loader concurrency too hard",
            },
            "heavy": {
                "workers": worker_counts["heavy"],
                "description": "Throughput-oriented option only for strong RAM, CPU, and storage headroom",
            },
        }

    def _recommend_worker_profile(
        self,
        model_metrics: dict[str, Any],
        dataset_metrics: dict[str, Any],
        system_metrics: dict[str, Any],
    ) -> tuple[str, str]:
        cpu_count = int(system_metrics["cpu_count"])
        available_ram_gib = self._to_gib(system_metrics["available_ram_bytes"])
        available_gpu_gib = self._to_gib(system_metrics["available_gpu_memory_bytes"])
        dataset_to_ram_ratio = self._ratio(
            int(dataset_metrics["total_size_bytes"]),
            int(system_metrics["available_ram_bytes"]),
        )
        total_file_count = int(dataset_metrics["total_file_count"])
        image_count = int(dataset_metrics["image_count"])
        score = 0
        reasons: list[str] = []

        if cpu_count >= 16:
            score += 1
            reasons.append("strong CPU core count")
        elif cpu_count <= 8:
            score -= 1
            reasons.append("limited CPU core count")

        if available_ram_gib >= 32:
            score += 1
            reasons.append("healthy RAM headroom")
        elif available_ram_gib < 16:
            score -= 2
            reasons.append("limited RAM headroom")

        if dataset_to_ram_ratio > 1.0:
            score -= 2
            reasons.append("dataset is larger than free RAM")
        elif dataset_to_ram_ratio > 0.5:
            score -= 1
            reasons.append("dataset is large relative to free RAM")
        elif dataset_to_ram_ratio < 0.2:
            score += 1
            reasons.append("dataset is small relative to free RAM")

        if total_file_count > 100000:
            score -= 1
            reasons.append("very high file-count dataset")
        elif image_count < 5000 and total_file_count < 20000:
            score -= 1
            reasons.append("smaller dataset usually does not benefit from many workers")

        if system_metrics["device"] == "cuda" and available_gpu_gib >= 12:
            score += 1
            reasons.append("GPU can benefit from steady batch feeding")
        elif system_metrics["device"] == "cpu":
            score -= 1
            reasons.append("CPU training gains less from high worker counts")

        if model_metrics["heaviness"] == "heavy":
            score -= 1
            reasons.append("heavier model already puts more pressure on the system")

        if score >= 3:
            return "heavy", ", ".join(reasons)
        if score <= 0:
            return "light", ", ".join(reasons)
        return "medium", ", ".join(reasons)

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
        worker_profile, worker_reason = self._recommend_worker_profile(
            model_metrics,
            dataset_metrics,
            system_metrics,
        )

        return {
            "model_metrics": model_metrics,
            "dataset_metrics": dataset_metrics,
            "system_metrics": system_metrics,
            "worker_profiles": worker_profiles,
            "worker_recommendation_reason": worker_reason,
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
        model_metrics = profile_context["model_metrics"]
        worker_profile = profile_context["worker_profiles"][profile_selection["worker"]]
        compute_profile = profile_selection["compute"]
        device = str(training["device"])

        if device == "cuda":
            # Ultralytics interprets 0 < batch < 1 as a GPU memory fraction for
            # AutoBatch. Floats here are intentional.
            training["batch"] = COMPUTE_BATCH_TARGETS[compute_profile]
        else:
            # Ultralytics AutoBatch is a no-op on MPS/CPU (returns default 16),
            # so pick an explicit size that respects model heaviness and the
            # user's compute profile instead of silently getting 16.
            training["batch"] = NON_CUDA_BATCH_SIZES[model_metrics["heaviness"]][
                compute_profile
            ]

        training["cache"] = self._can_enable_cache(
            compute_profile,
            dataset_metrics,
            system_metrics,
        )

        workers = int(worker_profile["workers"])
        # macOS + MPS with Ultralytics' dataloader has a history of stalling
        # under heavy multi-worker load; cap to a safer number there.
        if device == "mps":
            workers = min(workers, 4)
        training["workers"] = workers

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


class RFDETRConfigGenerator(BaseConfigGenerator):
    def _recommend_batch_settings(
        self,
        variant_resolution: int,
        system_metrics: dict[str, Any],
    ) -> dict[str, int]:
        available_gpu_gib = self._to_gib(system_metrics.get("available_gpu_memory_bytes"))
        device = str(system_metrics.get("device", "cpu"))

        if device != "cuda":
            return {"batch_size": 2, "grad_accum_steps": 4}
        if available_gpu_gib >= 24:
            return {"batch_size": 8 if variant_resolution <= 704 else 4, "grad_accum_steps": 1}
        if available_gpu_gib >= 12:
            return {"batch_size": 4 if variant_resolution <= 704 else 2, "grad_accum_steps": 2}
        return {"batch_size": 2, "grad_accum_steps": 4}

    def generate_config(
        self,
        model_choice: str,
        finetune_source: str | None = None,
        finetune_strategy: str | None = None,
    ) -> Dict:
        self.extract_dataset_info()
        variant = get_rfdetr_variant(model_choice)
        system_metrics = self._collect_system_metrics()
        batch_settings = self._recommend_batch_settings(
            variant.resolution,
            system_metrics,
        )
        task_name = {
            "segmentation": "RF-DETR Segmentation",
            "pose": "RF-DETR Keypoint",
        }.get(variant.task, "RF-DETR Detection")

        training: dict[str, Any] = {
            "epochs": 100,
            "resolution": variant.resolution,
            "batch_size": batch_settings["batch_size"],
            "grad_accum_steps": batch_settings["grad_accum_steps"],
            "lr": 1.0e-4,
            "lr_encoder": 1.5e-4,
            "weight_decay": 1.0e-4,
            "num_workers": self._get_optimal_workers(),
            "device": system_metrics["device"],
            "tensorboard": True,
            "early_stopping": True,
            "checkpoint_interval": 10,
            "output_dir": f"runs/rfdetr/{model_choice.lower().replace(' ', '-').replace('_', '-')}",
        }
        if variant.task == "pose":
            # RFDETRKeypointPreview defaults: COCO person-style 17 keypoints.
            training.update(
                {
                    "num_keypoints_per_class": [0, 17],
                    "keypoint_flip_pairs": [],
                    "keypoint_l1_loss_coef": 1.0,
                    "keypoint_findable_loss_coef": 1.0,
                    "keypoint_visible_loss_coef": 1.0,
                    "keypoint_nll_loss_coef": 1.0,
                }
            )
        if finetune_source:
            if finetune_strategy == "resume":
                training["resume"] = finetune_source
            else:
                training["pretrain_weights"] = finetune_source
                if finetune_strategy == "short_adaptation":
                    training["epochs"] = 50

        dataset_config = self._prepared_dataset_config("rfdetr", variant.task)
        integration_settings = load_settings()
        return {
            "settings": {
                "model_family": "rfdetr",
                "model_type": model_choice,
                "dataset": self.dataset_path.name,
                "task": variant.task,
                "class_name": variant.class_name,
                "auto_download_pretrained": finetune_source is None,
                "plus_model": variant.plus,
                "license": variant.license,
            },
            "clearml": snapshot_clearml_settings(integration_settings, task_name, model_choice),
            "dataset": {
                "name": self.dataset_path.name,
                "base_dir": dataset_config["prepared_dir"],
                "classes": dataset_config["classes"],
                "format": dataset_config["prepared_format"],
                **dataset_config,
            },
            "training": training,
            "export": {
                "enabled": True,
                "format": "onnx",
                "output_dir": "exports",
                "shape": variant.resolution,
                "batch_size": 1,
                "opset_version": 17,
                "quantization": None,
                "calibration_data": None,
            },
            "roboflow": snapshot_roboflow_settings(integration_settings),
        }


class Detectron2ConfigGenerator(BaseConfigGenerator):
    def generate_config(
        self,
        model_choice: str,
        finetune_source: str | None = None,
        finetune_strategy: str | None = None,
    ) -> Dict:
        self.extract_dataset_info()
        variant = get_detectron2_variant(model_choice)
        system_metrics = self._collect_system_metrics()
        dataset_config = self._prepared_dataset_config("detectron2", variant.task)
        output_slug = model_choice.lower().replace(" ", "-").replace("/", "-")
        training: dict[str, Any] = {
            "max_iter": 3000,
            "ims_per_batch": 2 if system_metrics["device"] != "cuda" else 4,
            "base_lr": 0.00025,
            "num_workers": self._get_optimal_workers(),
            "device": "cuda" if system_metrics["device"] == "cuda" else "cpu",
            "eval_period": 500,
            "checkpoint_period": 1000,
            "output_dir": f"runs/detectron2/{output_slug}",
        }
        if finetune_source:
            training["weights"] = finetune_source
            if finetune_strategy == "short_adaptation":
                training["max_iter"] = 1000

        integration_settings = load_settings()
        return {
            "settings": {
                "model_family": "detectron2",
                "model_type": model_choice,
                "dataset": self.dataset_path.name,
                "task": variant.task,
                "auto_download_pretrained": finetune_source is None,
            },
            "clearml": snapshot_clearml_settings(integration_settings, "Detectron2", model_choice),
            "dataset": dataset_config,
            "detectron2": {
                "config_path": variant.config_path,
                "weights_url": variant.weights_url,
            },
            "training": training,
            "export": {"enabled": False},
            "roboflow": snapshot_roboflow_settings(integration_settings),
        }


class YOLOConfigGenerator(BaseConfigGenerator):
    def generate_config(
        self,
        model_choice: str,
        profile_selection: dict[str, str] | None = None,
        profile_context: dict[str, Any] | None = None,
        finetune_source: str | None = None,
        finetune_strategy: str | None = None,
    ) -> Dict:
        """Generate YOLO specific configuration."""
        if profile_context is None:
            profile_context = self.get_regular_yolo_profile_context(model_choice)
        elif not self.dataset_info.get("task_type"):
            self.extract_dataset_info()

        if profile_selection is None:
            profile_selection = dict(profile_context["recommended_profiles"])

        model_lower = str(model_choice).lower()
        if model_lower.endswith("-pose"):
            fallback_task = "pose"
        elif model_lower.endswith("-seg"):
            fallback_task = "segmentation"
        else:
            fallback_task = "detection"
        task = self.dataset_info.get("task_type") or fallback_task
        dataset_config = self._prepared_dataset_config("yolo", task)
        integration_settings = load_settings()
        config = {
            "settings": {
                "model_family": "yolo",
                "model_type": finetune_source or model_choice.lower(),
                "dataset": self.dataset_path.name,
                "task": task,
            },
            "clearml": snapshot_clearml_settings(integration_settings, "YOLO", model_choice.upper()),
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
                "data_dir": dataset_config["prepared_dir"],
                "train_images_dir": "train/images",
                "train_labels_dir": "train/labels",
                "val_images_dir": "valid/images",
                "val_labels_dir": "valid/labels",
                "test_images_dir": "test/images",
                "test_labels_dir": "test/labels",
                "classes": dataset_config["classes"],
            },
            "dataset": dataset_config,
            "export": {
                "format": "onnx",
                "optimize": True,
                "half": False,
                "nms": True,
                "int8": False,
                "dynamic": False,
                "simplify": True,
                "opset": None,
            },
            "roboflow": snapshot_roboflow_settings(integration_settings),
        }
        if finetune_source:
            config["settings"]["base_model_type"] = model_choice.lower()
            config["settings"]["finetune_from"] = finetune_source
            config["training"]["resume"] = False
            if finetune_strategy == "freeze_backbone":
                config["training"]["freeze"] = 10
            elif finetune_strategy == "short_adaptation":
                config["training"]["epochs"] = 100
                config["training"]["patience"] = 30
        # Validate export parameters early to catch issues before training
        config = self._validate_export_params(config)

        return self._apply_regular_yolo_profiles(
            config,
            profile_selection,
            profile_context,
        )

    def _validate_export_params(self, config: dict) -> dict:
        """Validate export parameters and fix common issues."""
        export_config = config.get("export", {})
        format_type = export_config.get("format", "onnx")
        int8_enabled = export_config.get("int8", False)

        # INT8 requires calibration data - disable if not provided
        if int8_enabled and format_type == "onnx":
            has_calib_data = (
                export_config.get("data") is not None
                or config.get("model", {}).get("data_dir") is not None
            )
            if not has_calib_data:
                logger.warning(
                    "INT8 quantization requires calibration data. "
                    "Disabling int8 for ONNX export. "
                    "To enable int8, provide 'data' parameter with calibration dataset path."
                )
                export_config["int8"] = False

        # INT8 is not well-supported for ONNX without proper setup
        # Only enable for TensorRT or when data is explicitly provided
        if (
            int8_enabled
            and format_type in ("onnx", "torchscript")
            and not export_config.get("data")
        ):
            logger.warning(
                f"INT8 quantization is not recommended for {format_type} format without calibration data. "
                "Consider using TensorRT for INT8 or provide calibration data."
            )
            export_config["int8"] = False

        config["export"] = export_config
        return config


class SAMConfigGenerator(BaseConfigGenerator):
    """Generates YAML training config for SAM 3.1 fine-tuning."""

    _DEFAULTS: dict[str, Any] = {
        "base_model": "facebook/sam3.1",
        "freeze_image_encoder": True,
        "fine_tune_strategy": "decoder_only",
        "epochs": 10,
        "batch_size": 2,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "save_steps": 500,
        "max_grad_norm": 1.0,
        "fp16": True,
    }

    def generate_config(
        self,
        model_choice: str = "SAM 3.1",
        finetune_source: str | None = None,
    ) -> dict[str, Any]:
        from datetime import datetime
        try:
            from src.models.sam import get_sam_variant
        except ImportError:
            from models.sam import get_sam_variant

        variant = get_sam_variant(model_choice)
        self.extract_dataset_info()

        base_model = finetune_source or variant.hf_model_id
        run_name = f"sam3.1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        training_keys = (
            "epochs", "batch_size", "learning_rate", "weight_decay",
            "warmup_steps", "save_steps", "max_grad_norm", "fp16",
        )

        try:
            from src.config.settings import load_settings, snapshot_clearml_settings, snapshot_roboflow_settings
            integration_settings = load_settings()
            clearml_cfg = snapshot_clearml_settings(integration_settings, "SAM", model_choice)
            roboflow_cfg = snapshot_roboflow_settings(integration_settings)
        except Exception:
            clearml_cfg = {}
            roboflow_cfg = {}

        return {
            "settings": {
                "model_family": "sam3.1",
                "model_type": model_choice,
                "dataset": self.dataset_path.name,
                "task": "segmentation",
                "auto_download_pretrained": finetune_source is None,
            },
            "clearml": clearml_cfg,
            "model": {
                "base_model": base_model,
                "freeze_image_encoder": self._DEFAULTS["freeze_image_encoder"],
                "fine_tune_strategy": self._DEFAULTS["fine_tune_strategy"],
            },
            "dataset": {
                "base_dir": str(self.dataset_path),
                "format": "coco",
                "input_size": 1008,
            },
            "training": {k: self._DEFAULTS[k] for k in training_keys},
            "output": {
                "run_name": run_name,
                "output_dir": "runs/sam3.1",
            },
            "hardware": {
                "device": "auto",
            },
            "export": {"enabled": False},
            "roboflow": roboflow_cfg,
        }
