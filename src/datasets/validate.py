"""Structural validation of a YOLO dataset against the task it declares.

The failure mode this exists to catch is silent: a dataset whose ``data.yaml`` says
one thing while its label files say another still loads, then either trains on
garbage or dies deep inside a CUDA kernel with an assert that names no file. Every
check here answers "would Ultralytics accept this row for this task?" and reports
the offending path instead.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

from src.utils.semantic import IGNORE_INDEX, semantic_max_pixel_value

IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})

# Ultralytics vocabulary; YOLOmatic writes these into data.yaml's "task" key.
TASK_DETECT = "detect"
TASK_SEGMENT = "segment"
TASK_SEMANTIC = "semantic"
TASK_POSE = "pose"
KNOWN_TASKS = (TASK_DETECT, TASK_SEGMENT, TASK_SEMANTIC, TASK_POSE)

_MAX_REPORTED_PER_RULE = 5


@dataclass
class SplitReport:
    name: str
    images: int = 0
    labels: int = 0
    masks: int = 0
    empty_labels: int = 0
    annotations: int = 0


@dataclass
class ValidationReport:
    dataset_path: Path
    task: str
    num_classes: int
    label_style: str  # "polygon" | "mask" | "bbox" | "pose"
    splits: list[SplitReport] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        counts = ", ".join(
            f"{s.name}={s.images:,} imgs/{s.labels + s.masks:,} labels" for s in self.splits
        )
        state = "OK" if self.ok else f"{len(self.errors)} error(s)"
        return f"[{state}] task={self.task} nc={self.num_classes} style={self.label_style} :: {counts}"


class _Collector:
    """Accumulates messages but only keeps the first few of each rule."""

    def __init__(self, sink: list[str]) -> None:
        self._sink = sink
        self._counts: dict[str, int] = {}

    def add(self, rule: str, message: str) -> None:
        seen = self._counts.get(rule, 0)
        self._counts[rule] = seen + 1
        if seen < _MAX_REPORTED_PER_RULE:
            self._sink.append(message)
        elif seen == _MAX_REPORTED_PER_RULE:
            self._sink.append(f"{rule}: … additional occurrences suppressed")

    def finalize(self) -> None:
        for rule, count in self._counts.items():
            if count > _MAX_REPORTED_PER_RULE:
                for index, existing in enumerate(self._sink):
                    if existing == f"{rule}: … additional occurrences suppressed":
                        self._sink[index] = (
                            f"{rule}: {count - _MAX_REPORTED_PER_RULE} further occurrence(s) suppressed"
                        )
                        break


def _sibling_path(image_path: Path, directory_name: str, suffix: str) -> Path:
    """Map an image path to its label/mask path.

    Mirrors ``ultralytics.data.utils.img2label_paths``: swap the last ``/images/``
    segment for ``/<directory_name>/`` and change the extension. Matching that exactly
    matters — validating a different path than the loader reads would defeat the point.
    """
    marker = f"{os.sep}images{os.sep}"
    replacement = f"{os.sep}{directory_name}{os.sep}"
    text = str(image_path)
    if marker in text:
        head, _, tail = text.rpartition(marker)
        text = f"{head}{replacement}{tail}"
    return Path(text).with_suffix(suffix)


def _iter_images(directory: Path) -> Iterable[Path]:
    if not directory.is_dir():
        return []
    return (p for p in sorted(directory.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES)


def _normalized(values: Iterable[float]) -> bool:
    # Ultralytics tolerates a hair past the edge from resampling; anything further
    # means the row was written in pixel space.
    return all(-0.001 <= v <= 1.001 for v in values)


def _check_row(
    task: str,
    fields: list[str],
    num_classes: int,
    keypoints: int,
    keypoint_dim: int,
) -> str | None:
    """Return an error description for one label row, or None when it is well-formed."""
    try:
        class_id = int(float(fields[0]))
        coords = [float(v) for v in fields[1:]]
    except ValueError:
        return "non-numeric field"
    if class_id < 0 or class_id >= num_classes:
        return f"class id {class_id} outside [0, {num_classes})"

    # Arity first: a row with the wrong number of fields for the task is the most
    # useful thing to report, and knowing the layout is what lets the range checks
    # below tell coordinates apart from visibility flags.
    n = len(coords)
    if task == TASK_DETECT:
        if n != 4:
            return f"expected 4 box values, found {n}"
    elif task in (TASK_SEGMENT, TASK_SEMANTIC):
        if n < 6:
            return f"polygon needs at least 3 points (6 values), found {n}"
        if n % 2 != 0:
            return f"polygon has an odd number of coordinates ({n})"
    elif task == TASK_POSE:
        expected = 4 + keypoints * keypoint_dim
        if n != expected:
            return f"expected {expected} values for {keypoints} keypoints, found {n}"

    spatial = coords
    if task == TASK_POSE and keypoint_dim == 3:
        # Every third keypoint field is a visibility flag (0/1/2), not a coordinate.
        spatial = coords[:4] + [
            value for index, value in enumerate(coords[4:]) if index % 3 != 2
        ]
        if any(v not in (0.0, 1.0, 2.0) for v in coords[6::3]):
            return "keypoint visibility flags must be 0, 1 or 2"
    if not _normalized(spatial):
        return "coordinates outside [0, 1] (expected normalized values)"

    if task == TASK_DETECT:
        _, _, w, h = coords
        if w <= 0 or h <= 0:
            return "box has non-positive width or height"
    return None


def validate_dataset(dataset_path: Path, expected_task: str | None = None) -> ValidationReport:
    """Check that a dataset's labels match the task its data.yaml declares."""
    dataset_path = Path(dataset_path)
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.is_file():
        report = ValidationReport(dataset_path, "unknown", 0, "unknown")
        report.errors.append(f"No data.yaml found in {dataset_path}.")
        return report

    meta = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    task = str(meta.get("task") or "").lower() or TASK_DETECT
    names = meta.get("names") or []
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names)]
    declared_nc = meta.get("nc")
    num_classes = int(declared_nc) if declared_nc is not None else len(names)
    masks_dir = meta.get("masks_dir")
    label_style = (
        "mask" if (task == TASK_SEMANTIC and masks_dir)
        else "polygon" if task in (TASK_SEGMENT, TASK_SEMANTIC)
        else "pose" if task == TASK_POSE
        else "bbox"
    )
    report = ValidationReport(dataset_path, task, num_classes, label_style)
    errors = _Collector(report.errors)
    warnings = _Collector(report.warnings)

    if task not in KNOWN_TASKS:
        report.errors.append(
            f"data.yaml declares task '{task}', which is not one of {list(KNOWN_TASKS)}."
        )
    if expected_task and task != expected_task:
        report.errors.append(
            f"data.yaml declares task '{task}' but '{expected_task}' was expected."
        )
    if names and declared_nc is not None and len(names) != int(declared_nc):
        report.errors.append(
            f"data.yaml has {len(names)} names but nc={declared_nc}; they must match."
        )
    if num_classes <= 0:
        report.errors.append("data.yaml declares no classes (nc must be >= 1).")
        return report

    keypoints = keypoint_dim = 0
    if task == TASK_POSE:
        kpt_shape = meta.get("kpt_shape")
        if not kpt_shape or len(kpt_shape) != 2:
            report.errors.append("A pose dataset must declare kpt_shape: [num_keypoints, dims].")
            return report
        keypoints, keypoint_dim = int(kpt_shape[0]), int(kpt_shape[1])

    if task == TASK_SEMANTIC and masks_dir:
        max_value = semantic_max_pixel_value(
            num_classes - 1 if num_classes > 1 else num_classes
        )
    else:
        max_value = 0

    found_any = False
    for split, key in (("train", "train"), ("valid", "val"), ("test", "test")):
        rel = meta.get(key)
        if not rel:
            continue
        img_dir = (dataset_path / str(rel)).resolve()
        if not img_dir.is_dir():
            report.errors.append(f"data.yaml '{key}: {rel}' points at a missing directory.")
            continue
        split_report = SplitReport(name=split)
        images = list(_iter_images(img_dir))
        split_report.images = len(images)
        found_any = found_any or bool(images)

        for image_path in images:
            if label_style == "mask":
                mask_path = _sibling_path(image_path, str(masks_dir), ".png")
                if not mask_path.is_file():
                    errors.add("missing-mask", f"{image_path.name}: no mask at {mask_path}")
                    continue
                split_report.masks += 1
                _check_mask(mask_path, max_value, errors)
                continue

            label_path = _sibling_path(image_path, "labels", ".txt")
            if not label_path.is_file():
                errors.add("missing-label", f"{image_path.name}: no label file at {label_path}")
                continue
            split_report.labels += 1
            rows = [r for r in label_path.read_text(encoding="utf-8").splitlines() if r.strip()]
            if not rows:
                split_report.empty_labels += 1
                continue
            for line_no, row in enumerate(rows, start=1):
                fields = row.split()
                if len(fields) < 5:
                    errors.add(
                        "short-row",
                        f"{label_path.name}:{line_no}: only {len(fields)} field(s) on the row",
                    )
                    continue
                problem = _check_row(task, fields, num_classes, keypoints, keypoint_dim)
                if problem:
                    errors.add(problem, f"{label_path.name}:{line_no}: {problem}")
                else:
                    split_report.annotations += 1
        report.splits.append(split_report)

    if not found_any:
        report.errors.append("No images were found in any split declared by data.yaml.")

    for split_report in report.splits:
        if split_report.images and not (split_report.labels or split_report.masks):
            report.errors.append(f"Split '{split_report.name}' has images but no labels.")
        elif split_report.images and split_report.empty_labels == split_report.labels:
            warnings.add(
                "all-empty",
                f"Split '{split_report.name}': every label file is empty (background only).",
            )

    errors.finalize()
    warnings.finalize()
    return report


def _check_mask(mask_path: Path, max_value: int, errors: _Collector) -> None:
    """Confirm a dense mask's pixel values are class indices Ultralytics can consume."""
    try:
        from src.utils.ml_dependencies import import_cv2

        cv2 = import_cv2()
        import numpy as np

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            errors.add("unreadable-mask", f"{mask_path.name}: could not be decoded")
            return
        if mask.ndim != 2:
            errors.add(
                "multichannel-mask",
                f"{mask_path.name}: expected a single-channel mask, found shape {mask.shape}",
            )
            return
        values = np.unique(mask)
        illegal = [int(v) for v in values if v > max_value and int(v) != IGNORE_INDEX]
        if illegal:
            errors.add(
                "mask-value-range",
                f"{mask_path.name}: pixel value(s) {illegal[:4]} exceed the maximum class "
                f"index {max_value}; CrossEntropyLoss would abort on these",
            )
    except Exception as exc:  # decoding problems must not abort the whole validation
        errors.add("mask-check-failed", f"{mask_path.name}: {exc}")
