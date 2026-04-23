from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.utils.ml_dependencies import import_module_or_raise, import_torch


ARTIFACT_IMAGE_NAMES: dict[str, tuple[str, ...]] = {
    "confusion": ("confusion_matrix.png", "confusion_matrix_normalized.png"),
    "curves": ("PR_curve.png", "F1_curve.png", "P_curve.png", "R_curve.png"),
    "samples": (
        "results.png",
        "labels.jpg",
        "labels_correlogram.jpg",
        "train_batch0.jpg",
        "train_batch1.jpg",
        "train_batch2.jpg",
        "val_batch0_labels.jpg",
        "val_batch0_pred.jpg",
        "val_batch1_labels.jpg",
        "val_batch1_pred.jpg",
    ),
}

OPTIONAL_SCALAR_TAGS = ("metrics/map50_95",)


@dataclass
class TensorBoardValidationReport:
    run_dir: Path
    event_files: list[Path]
    scalar_tags: set[str]
    image_tags: set[str]
    text_tags: set[str]
    missing_required: list[str]
    missing_optional: list[str]

    @property
    def is_complete(self) -> bool:
        return not self.missing_required


def _get_summary_writer():
    tensorboard_module = import_module_or_raise("torch.utils.tensorboard")
    return tensorboard_module.SummaryWriter


def _get_event_accumulator():
    module = import_module_or_raise("tensorboard.backend.event_processing.event_accumulator")
    return module.EventAccumulator


def _normalize_tag_component(value: Any) -> str:
    text = "".join(char.lower() if char.isalnum() else "_" for char in str(value))
    return text.strip("_")


def _safe_scalar(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            return None
    return None


def _flatten_metrics(payload: Any, prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_prefix = f"{prefix}/{_normalize_tag_component(key)}" if prefix else _normalize_tag_component(key)
            flattened.update(_flatten_metrics(value, next_prefix))
        return flattened

    if isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            next_prefix = f"{prefix}/{index}" if prefix else str(index)
            flattened.update(_flatten_metrics(value, next_prefix))
        return flattened

    scalar = _safe_scalar(payload)
    if scalar is not None and prefix:
        flattened[prefix] = scalar
    return flattened


def _collect_event_files(run_dir: str | Path) -> list[Path]:
    base_dir = Path(run_dir)
    if not base_dir.exists():
        return []
    return sorted(
        path for path in base_dir.rglob("events.out.tfevents.*") if path.is_file()
    )


def _collect_event_directories(run_dir: str | Path) -> list[Path]:
    return sorted({event_file.parent for event_file in _collect_event_files(run_dir)})


def _read_image_array(image_path: Path) -> Any | None:
    cv2 = import_module_or_raise("cv2")
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _extract_runtime_stats(device: str | None) -> dict[str, float]:
    stats: dict[str, float] = {}
    if not device or not str(device).startswith("cuda"):
        return stats

    try:
        torch = import_torch()
        if not torch.cuda.is_available():
            return stats
        stats["runtime/gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
        stats["runtime/gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)
        stats["runtime/gpu_memory_max_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
    except Exception:
        return {}
    return stats


def prepare_tensorboard_log_dir(run_dir: str | Path) -> Path:
    log_dir = Path(run_dir) / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


class TensorBoardLogger:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.log_dir = prepare_tensorboard_log_dir(self.run_dir)
        self._writer = _get_summary_writer()(log_dir=str(self.log_dir))

    def log_metadata(self, metadata: dict[str, Any]) -> None:
        for key, value in sorted(metadata.items()):
            self._writer.add_text(f"metadata/{_normalize_tag_component(key)}", str(value), 0)
        self._writer.add_text("metadata/config", yaml.safe_dump(metadata, sort_keys=True), 0)
        self._writer.flush()

    def add_scalar(self, tag: str, value: Any, step: int) -> None:
        scalar = _safe_scalar(value)
        if scalar is None:
            return
        self._writer.add_scalar(tag, scalar, step)

    def add_scalars(self, metrics: dict[str, Any], step: int) -> None:
        for tag, value in metrics.items():
            self.add_scalar(tag, value, step)

    def add_text(self, tag: str, text: Any, step: int = 0) -> None:
        self._writer.add_text(tag, str(text), step)

    def add_image_from_path(self, tag: str, image_path: str | Path, step: int = 0) -> bool:
        path = Path(image_path)
        if not path.is_file():
            return False
        image = _read_image_array(path)
        if image is None:
            return False
        self._writer.add_image(tag, image, step, dataformats="HWC")
        self._writer.add_text(f"{tag}_path", str(path), step)
        return True

    def add_image_tensor(self, tag: str, image_tensor: Any, step: int) -> bool:
        torch = import_torch()
        try:
            tensor = image_tensor.detach().cpu()
        except AttributeError:
            return False

        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim != 3:
            return False

        if tensor.shape[0] in (1, 3):
            chw_tensor = tensor
        elif tensor.shape[-1] in (1, 3):
            chw_tensor = tensor.permute(2, 0, 1)
        else:
            return False

        chw_tensor = chw_tensor.float()
        if float(chw_tensor.max()) > 1.0:
            chw_tensor = chw_tensor / 255.0
        chw_tensor = torch.clamp(chw_tensor, 0.0, 1.0)
        self._writer.add_image(tag, chw_tensor, step)
        return True

    def add_runtime_stats(self, device: str | None, step: int) -> None:
        self.add_scalars(_extract_runtime_stats(device), step)

    def log_known_artifacts(self, artifact_root: str | Path, step: int = 0) -> None:
        artifact_dir = Path(artifact_root)
        for category, filenames in ARTIFACT_IMAGE_NAMES.items():
            matched_paths = [artifact_dir / name for name in filenames if (artifact_dir / name).is_file()]
            if not matched_paths:
                continue
            self.add_text(
                f"artifacts/{category}_paths",
                "\n".join(str(path) for path in matched_paths),
                step,
            )
            for image_path in matched_paths:
                self.add_image_from_path(
                    f"artifacts/{category}/{_normalize_tag_component(image_path.stem)}",
                    image_path,
                    step,
                )
        self._writer.flush()

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()


def map_ultralytics_metric_tag(raw_key: str) -> str:
    normalized = raw_key.strip()
    lowered = normalized.lower()

    if normalized.startswith("train/") and "loss" in lowered:
        suffix = normalized.split("/", 1)[1]
        return f"loss/train/{_normalize_tag_component(suffix)}"
    if normalized.startswith("val/") and "loss" in lowered:
        suffix = normalized.split("/", 1)[1]
        return f"loss/val/{_normalize_tag_component(suffix)}"
    if lowered.startswith("metrics/precision"):
        return "metrics/precision"
    if lowered.startswith("metrics/recall"):
        return "metrics/recall"
    if "map50-95" in lowered:
        return "metrics/map50_95"
    if "map50" in lowered:
        return "metrics/map50"
    if lowered.startswith("lr/"):
        suffix = normalized.split("/", 1)[1]
        return f"optimization/lr/{_normalize_tag_component(suffix)}"
    return f"metrics/raw/{_normalize_tag_component(normalized)}"


def backfill_ultralytics_tensorboard(
    run_dir: str | Path,
    metadata: dict[str, Any],
    device: str | None,
) -> None:
    run_path = Path(run_dir)
    logger = TensorBoardLogger(run_path)
    try:
        logger.log_metadata(metadata)
        results_csv = run_path / "results.csv"
        if results_csv.is_file():
            with open(results_csv, "r", encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                for index, row in enumerate(reader):
                    epoch_value = row.get("epoch", index)
                    epoch = int(float(epoch_value))
                    logger.add_scalar("runtime/epoch", epoch, epoch)
                    for key, value in row.items():
                        if key == "epoch":
                            continue
                        logger.add_scalar(map_ultralytics_metric_tag(key), value, epoch)
                    logger.add_runtime_stats(device, epoch)
        logger.log_known_artifacts(run_path)
    finally:
        logger.close()


def build_tensorboard_metadata(
    *,
    model_name: str,
    dataset_name: str,
    config_path: str,
    run_name: str,
    device: str | None,
    training_params: dict[str, Any],
) -> dict[str, Any]:
    metadata = {
        "model": model_name,
        "dataset": dataset_name,
        "config_path": config_path,
        "run_name": run_name,
        "device": device or "",
    }
    interesting_keys = (
        "epochs",
        "max_epochs",
        "batch",
        "batch_size",
        "imgsz",
        "optimizer",
        "lr0",
        "device",
    )
    for key in interesting_keys:
        if key in training_params:
            metadata[key] = training_params[key]
    return metadata


def normalize_supergradients_metrics(metrics_dict: dict[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for tag, value in _flatten_metrics(metrics_dict).items():
        lowered = tag.lower()
        if "map" in lowered and "50_95" in lowered:
            normalized["metrics/map50_95"] = value
        elif "map" in lowered and "50" in lowered:
            normalized["metrics/map50"] = value
        elif "precision" in lowered:
            normalized["metrics/precision"] = value
        elif "recall" in lowered:
            normalized["metrics/recall"] = value
        elif "loss" in lowered and "valid" in lowered:
            normalized[f"loss/val/{_normalize_tag_component(tag)}"] = value
        elif "loss" in lowered and "train" in lowered:
            normalized[f"loss/train/{_normalize_tag_component(tag)}"] = value
        else:
            normalized[f"metrics/raw/{_normalize_tag_component(tag)}"] = value
    return normalized


def validate_tensorboard_run(run_dir: str | Path) -> TensorBoardValidationReport:
    run_path = Path(run_dir)
    event_files = _collect_event_files(run_path)
    scalar_tags: set[str] = set()
    image_tags: set[str] = set()
    text_tags: set[str] = set()

    event_accumulator = _get_event_accumulator()
    for event_dir in _collect_event_directories(run_path):
        accumulator = event_accumulator(str(event_dir))
        accumulator.Reload()
        tags = accumulator.Tags()
        scalar_tags.update(tags.get("scalars", []))
        image_tags.update(tags.get("images", []))
        text_tags.update(tags.get("tensors", []))

    def has_scalar(prefix: str) -> bool:
        return any(tag.startswith(prefix) for tag in scalar_tags)

    def has_any_tag(substring: str) -> bool:
        candidates = scalar_tags | image_tags | text_tags
        return any(substring in tag for tag in candidates)

    missing_required: list[str] = []
    if not event_files:
        missing_required.append("event_files")
    if not has_any_tag("metadata/"):
        missing_required.append("metadata")
    if not has_scalar("loss/train/"):
        missing_required.append("train_loss")
    if not has_scalar("loss/val/"):
        missing_required.append("val_loss")
    if "metrics/precision" not in scalar_tags:
        missing_required.append("precision")
    if "metrics/recall" not in scalar_tags:
        missing_required.append("recall")
    if "metrics/map50" not in scalar_tags:
        missing_required.append("map50")
    if not has_scalar("optimization/lr/"):
        missing_required.append("learning_rate")
    if not has_scalar("runtime/"):
        missing_required.append("runtime")
    if not has_any_tag("artifacts/confusion"):
        missing_required.append("confusion_artifact")
    if not has_any_tag("artifacts/curves"):
        missing_required.append("curve_artifact")
    if not has_any_tag("artifacts/samples"):
        missing_required.append("sample_artifact")

    missing_optional = [tag for tag in OPTIONAL_SCALAR_TAGS if tag not in scalar_tags]
    return TensorBoardValidationReport(
        run_dir=run_path,
        event_files=event_files,
        scalar_tags=scalar_tags,
        image_tags=image_tags,
        text_tags=text_tags,
        missing_required=missing_required,
        missing_optional=missing_optional,
    )


def emit_tensorboard_report(console: Any, report: TensorBoardValidationReport) -> None:
    if report.is_complete:
        console.print(
            f"[bold green]TensorBoard completeness check passed for {report.run_dir}[/bold green]"
        )
    else:
        missing = ", ".join(report.missing_required)
        console.print(
            f"[bold yellow]TensorBoard completeness warning for {report.run_dir}: missing {missing}[/bold yellow]"
        )

    if report.missing_optional:
        console.print(
            "[bold yellow]Optional TensorBoard metrics not found:[/bold yellow] "
            + ", ".join(report.missing_optional)
        )


class SuperGradientsTensorBoardCallback:
    def __init__(self, run_dir: str | Path, metadata: dict[str, Any]):
        self.run_dir = Path(run_dir)
        self.metadata = metadata
        self.logger = TensorBoardLogger(self.run_dir)
        self._epoch_train_loss_totals: dict[str, float] = {}
        self._epoch_train_loss_counts: dict[str, int] = {}
        self._train_epoch_start_time: float | None = None
        self._epoch_sample_logged = False

    def on_training_start(self, context: Any) -> None:
        self.logger.log_metadata(self.metadata)

    def on_train_loader_start(self, context: Any) -> None:
        self._epoch_train_loss_totals = {}
        self._epoch_train_loss_counts = {}
        self._train_epoch_start_time = time.perf_counter()
        self._epoch_sample_logged = False

    def on_train_batch_loss_end(self, context: Any) -> None:
        names = getattr(context, "loss_logging_items_names", None) or []
        values = getattr(context, "loss_log_items", None)
        if values is None:
            values = []
        for name, value in zip(names, values):
            scalar = _safe_scalar(value)
            if scalar is None:
                continue
            tag = f"loss/train/{_normalize_tag_component(name)}"
            self._epoch_train_loss_totals[tag] = self._epoch_train_loss_totals.get(tag, 0.0) + scalar
            self._epoch_train_loss_counts[tag] = self._epoch_train_loss_counts.get(tag, 0) + 1

    def on_train_loader_end(self, context: Any) -> None:
        epoch = int(getattr(context, "epoch", 0))
        for tag, total in self._epoch_train_loss_totals.items():
            count = max(1, self._epoch_train_loss_counts.get(tag, 1))
            self.logger.add_scalar(tag, total / count, epoch)

        optimizer = getattr(context, "optimizer", None)
        if optimizer is not None:
            for index, param_group in enumerate(getattr(optimizer, "param_groups", [])):
                self.logger.add_scalar(
                    f"optimization/lr/pg{index}",
                    param_group.get("lr"),
                    epoch,
                )

        if self._train_epoch_start_time is not None:
            self.logger.add_scalar(
                "runtime/epoch_total_time_sec",
                time.perf_counter() - self._train_epoch_start_time,
                epoch,
            )
        self.logger.add_runtime_stats(getattr(context, "device", None), epoch)

    def on_validation_batch_end(self, context: Any) -> None:
        if self._epoch_sample_logged:
            return
        epoch = int(getattr(context, "epoch", 0))
        if self.logger.add_image_tensor("artifacts/samples/validation_input", getattr(context, "inputs", None), epoch):
            self._epoch_sample_logged = True

    def on_validation_loader_end(self, context: Any) -> None:
        epoch = int(getattr(context, "epoch", 0))
        metrics_dict = getattr(context, "metrics_dict", None) or {}
        self.logger.add_scalars(normalize_supergradients_metrics(metrics_dict), epoch)
        loss_avg_meter = getattr(context, "loss_avg_meter", None)
        avg_value = getattr(loss_avg_meter, "average", None)
        self.logger.add_scalar("loss/val/total", avg_value, epoch)
        self.logger.add_runtime_stats(getattr(context, "device", None), epoch)

    def on_training_end(self, context: Any) -> None:
        self.logger.log_known_artifacts(self.run_dir, getattr(context, "epoch", 0) or 0)
        self.logger.close()
