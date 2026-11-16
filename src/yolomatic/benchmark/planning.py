"""Task and split validation shared by the benchmark CLI and engine."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.datasets.core import SplitSummary, summarize_dataset


SUPPORTED_TASKS = frozenset({"detection", "segmentation", "pose", "semantic"})
_TASK_ALIASES = {
    "detect": "detection",
    "detection": "detection",
    "bbox": "detection",
    "box": "detection",
    "segment": "segmentation",
    "segmentation": "segmentation",
    "instance_segmentation": "segmentation",
    "pose": "pose",
    "keypoint": "pose",
    "keypoints": "pose",
    "semantic": "semantic",
    "semantic_segmentation": "semantic",
    "semseg": "semantic",
}


class BenchmarkCompatibilityError(ValueError):
    """Raised when a model, dataset, or requested split cannot be benchmarked."""


@dataclass(frozen=True)
class BenchmarkSplit:
    """One resolved dataset split available to the benchmark engine."""

    name: str
    directory: Path
    image_count: int
    annotation_count: int


def normalize_task(value: object) -> str | None:
    """Return a supported canonical task name, if *value* names one."""
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _TASK_ALIASES.get(normalized)


def _model_task_from_metadata(model: Any) -> str | None:
    """Read task metadata without relying on a checkpoint filename."""
    for candidate in (
        getattr(model, "task", None),
        getattr(getattr(model, "model", None), "task", None),
        getattr(getattr(model, "overrides", {}), "get", lambda _key: None)("task"),
    ):
        task = normalize_task(candidate)
        if task:
            return task
    return None


def resolve_ultralytics_model_task(weights: Path) -> str:
    """Load an artifact and return its verified Ultralytics task metadata.

    Filename and run-directory hints deliberately do not participate here: a
    misleading name must not produce a mathematically invalid benchmark.
    """
    try:
        from ultralytics import YOLO

        model = YOLO(str(weights))
    except Exception as exc:
        raise BenchmarkCompatibilityError(
            f"Could not load Ultralytics artifact '{weights}': {exc}"
        ) from exc
    task = _model_task_from_metadata(model)
    if task is None:
        raise BenchmarkCompatibilityError(
            f"Could not verify the task for '{weights}'. Export the model with task "
            "metadata or use a native Ultralytics checkpoint."
        )
    return task


def resolve_selected_model_task(weights: list[Path]) -> str:
    """Verify that every selected artifact has the same supported task."""
    if not weights:
        raise BenchmarkCompatibilityError("Select at least one model to benchmark.")
    tasks = {weight: resolve_ultralytics_model_task(weight) for weight in weights}
    unique = set(tasks.values())
    if len(unique) != 1:
        details = ", ".join(f"{path.name}: {task}" for path, task in tasks.items())
        raise BenchmarkCompatibilityError(
            "Selected models have different tasks (" + details + "). "
            "Benchmark one task at a time."
        )
    return unique.pop()


def _split_directory(summary: SplitSummary) -> Path | None:
    return Path(summary.images_path) if summary.images_path else None


def resolve_dataset_splits(
    dataset_dir: Path, task: str, selection: str = "valid"
) -> list[BenchmarkSplit]:
    """Return non-empty, task-compatible split directories for a benchmark run."""
    requested_task = normalize_task(task)
    if requested_task is None:
        raise BenchmarkCompatibilityError(f"Unsupported benchmark task: {task!r}.")
    summary = summarize_dataset(dataset_dir)
    dataset_task = normalize_task(summary.task)
    if dataset_task != requested_task:
        actual = summary.task or "unknown"
        raise BenchmarkCompatibilityError(
            f"Dataset '{dataset_dir}' is {actual}, but the selected model is "
            f"{requested_task}. Choose a matching dataset."
        )

    normalized_selection = selection.strip().lower()
    if normalized_selection == "val":
        normalized_selection = "valid"
    allowed = ("train", "valid", "test") if normalized_selection == "all" else (normalized_selection,)
    if any(name not in {"train", "valid", "test"} for name in allowed):
        raise BenchmarkCompatibilityError("Split selection must be train, valid, test, or all.")

    resolved: list[BenchmarkSplit] = []
    for name in allowed:
        # Dataset summaries preserve the historic ``val`` key; benchmark UX
        # standardizes it to the user-facing ``valid`` group.
        split = summary.splits.get("val") if name == "valid" else summary.splits.get(name)
        directory = _split_directory(split) if split else None
        if directory is None or not directory.exists() or split.image_count == 0:
            if normalized_selection != "all":
                raise BenchmarkCompatibilityError(
                    f"The {name} split is missing or contains no images in '{dataset_dir}'."
                )
            continue
        if split.annotation_count == 0 and split.empty_label_count == 0:
            raise BenchmarkCompatibilityError(
                f"The {name} split has no annotations. Benchmarking unlabeled data is not supported."
            )
        resolved.append(BenchmarkSplit(name, directory, split.image_count, split.annotation_count))
    if not resolved:
        raise BenchmarkCompatibilityError("No non-empty compatible splits were found.")
    return resolved
