"""Structural validation of datasets intended for Roboflow platform ingestion.

Validates:
1. Format & task conformity (instance-segmentation, object-detection, semantic-segmentation, pose, classification).
2. data.yaml / metadata completeness (names, nc, split paths).
3. Label row integrity (normalized coordinates [0, 1], polygon vertex count >= 3, positive box dimensions).
4. Negative sample vs missing label detection (0-byte text files are valid clean negatives).
5. Image header readability (magic byte checks).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import yaml

IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ANNOTATION_EXTENSIONS: Set[str] = {".txt", ".json", ".xml", ".csv", ".jsonl"}

# Roboflow / Ultralytics tasks
TASK_INSTANCE_SEG = "instance-segmentation"
TASK_OBJECT_DETECT = "object-detection"
TASK_SEMANTIC_SEG = "semantic-segmentation"
TASK_POSE = "pose"
TASK_CLASSIFY = "classification"

VALID_ROBOFLOW_TASKS = (
    TASK_INSTANCE_SEG,
    TASK_OBJECT_DETECT,
    TASK_SEMANTIC_SEG,
    TASK_POSE,
    TASK_CLASSIFY,
)

_MAX_REPORTED_ISSUES_PER_RULE = 5


@dataclass
class SplitStats:
    name: str
    images: int = 0
    annotated: int = 0
    negatives: int = 0  # Clean 0-byte background label files
    missing_labels: int = 0  # Images with no corresponding label file
    corrupt_images: int = 0

    @property
    def total_labeled(self) -> int:
        return self.annotated + self.negatives

    @property
    def coverage_pct(self) -> float:
        if self.images == 0:
            return 0.0
        return (self.annotated / self.images) * 100.0


@dataclass
class RoboflowValidationResult:
    dataset_path: Path
    detected_task: str
    num_classes: int
    class_names: List[str] = field(default_factory=list)
    splits: Dict[str, SplitStats] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    @property
    def total_images(self) -> int:
        return sum(s.images for s in self.splits.values())

    @property
    def total_annotated(self) -> int:
        return sum(s.annotated for s in self.splits.values())

    @property
    def total_negatives(self) -> int:
        return sum(s.negatives for s in self.splits.values())

    @property
    def overall_coverage_pct(self) -> float:
        if self.total_images == 0:
            return 0.0
        return (self.total_annotated / self.total_images) * 100.0


class _IssueCollector:
    """Collects validation messages with duplicate suppression."""

    def __init__(self, sink: List[str]) -> None:
        self._sink = sink
        self._counts: Dict[str, int] = {}

    def add(self, rule: str, message: str) -> None:
        seen = self._counts.get(rule, 0)
        self._counts[rule] = seen + 1
        if seen < _MAX_REPORTED_ISSUES_PER_RULE:
            self._sink.append(message)
        elif seen == _MAX_REPORTED_ISSUES_PER_RULE:
            self._sink.append(f"{rule}: ... additional occurrences suppressed")

    def finalize(self) -> None:
        for rule, count in self._counts.items():
            if count > _MAX_REPORTED_ISSUES_PER_RULE:
                suppressed_msg = f"{rule}: ... additional occurrences suppressed"
                for idx, existing in enumerate(self._sink):
                    if existing == suppressed_msg:
                        self._sink[idx] = f"{rule}: {count - _MAX_REPORTED_ISSUES_PER_RULE} further occurrence(s) suppressed"
                        break


def _is_valid_image_header(path: Path) -> bool:
    """Validate image file header magic bytes without heavy full-decode overhead."""
    try:
        if not path.is_file() or path.stat().st_size < 12:
            return False
        with open(path, "rb") as f:
            header = f.read(12)

        ext = path.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            return header.startswith(b"\xff\xd8\xff")
        if ext == ".png":
            return header.startswith(b"\x89PNG\r\n\x1a\n")
        if ext == ".bmp":
            return header.startswith(b"BM")
        if ext == ".webp":
            return header.startswith(b"RIFF") and header[8:12] == b"WEBP"
        if ext in (".tif", ".tiff"):
            return header.startswith(b"II*\x00") or header.startswith(b"MM\x00*")

        # Generic header check
        return (
            header.startswith(b"\xff\xd8\xff")
            or header.startswith(b"\x89PNG\r\n\x1a\n")
            or header.startswith(b"BM")
            or (header.startswith(b"RIFF") and header[8:12] == b"WEBP")
            or header.startswith(b"II*\x00")
            or header.startswith(b"MM\x00*")
        )
    except (OSError, PermissionError):
        return False


def _check_label_row(
    task: str,
    fields: Sequence[str],
    num_classes: int,
    line_no: int,
    filename: str,
) -> Optional[str]:
    """Validate a single YOLO annotation row."""
    if not fields:
        return None

    try:
        class_id = int(float(fields[0]))
    except ValueError:
        return f"{filename}:{line_no}: non-numeric class index '{fields[0]}'"

    if class_id < 0 or class_id >= num_classes:
        return f"{filename}:{line_no}: class index {class_id} outside valid range [0, {num_classes})"

    coords = []
    for idx, v in enumerate(fields[1:], start=1):
        try:
            coords.append(float(v))
        except ValueError:
            return f"{filename}:{line_no}: non-numeric coordinate '{v}' at field {idx}"

    num_coords = len(coords)

    if task == TASK_OBJECT_DETECT:
        if num_coords != 4:
            return f"{filename}:{line_no}: expected 4 box values (x, y, w, h), found {num_coords}"
        x, y, w, h = coords
        if w <= 0 or h <= 0:
            return f"{filename}:{line_no}: non-positive box dimension (w={w}, h={h})"
        if not all(-0.001 <= val <= 1.001 for val in coords):
            return f"{filename}:{line_no}: box coordinates outside normalized [0, 1] range"

    elif task in (TASK_INSTANCE_SEG, TASK_SEMANTIC_SEG):
        if num_coords < 6:
            return f"{filename}:{line_no}: polygon requires at least 3 points (6 values), found {num_coords}"
        if num_coords % 2 != 0:
            return f"{filename}:{line_no}: polygon has odd coordinate count ({num_coords})"
        if not all(-0.001 <= val <= 1.001 for val in coords):
            return f"{filename}:{line_no}: polygon coordinates outside normalized [0, 1] range"

    elif task == TASK_POSE:
        if num_coords < 4:
            return f"{filename}:{line_no}: pose requires at least 4 box coordinates, found {num_coords}"

    return None


def _check_json_annotation(
    task: str,
    content: str,
    num_classes: int,
    filename: str,
) -> Tuple[bool, Optional[str]]:
    """Validate JSON format annotations (e.g. LabelMe / Labelbox / COCO)."""
    try:
        data = json.loads(content)
    except Exception as exc:
        return False, f"{filename}: invalid JSON syntax: {exc}"

    # LabelMe format: shapes with points
    if isinstance(data, dict) and "shapes" in data:
        shapes = data.get("shapes", [])
        if not shapes:
            return True, None
        for idx, s in enumerate(shapes, start=1):
            pts = s.get("points", [])
            if len(pts) < 3 and task in (TASK_INSTANCE_SEG, TASK_SEMANTIC_SEG):
                return False, f"{filename}: shape #{idx} has fewer than 3 points"
        return True, None

    return True, None


def _infer_num_classes(dataset_path: Path) -> int:
    """Derive the class count from label files when no data.yaml declares it.

    Returns ``max(class_id) + 1`` across every readable YOLO row, or 1 when nothing
    usable is found. This exists so an undeclared dataset is not range-checked
    against a fabricated ``nc=1``, which would reject every class index >= 1.
    """
    highest = -1
    for root, _, files in os.walk(dataset_path):
        for name in files:
            if not name.lower().endswith(".txt"):
                continue
            fp = Path(root) / name
            try:
                content = fp.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for line in content.splitlines():
                fields = line.split()
                if not fields:
                    continue
                try:
                    highest = max(highest, int(float(fields[0])))
                except ValueError:
                    continue
    return highest + 1 if highest >= 0 else 1


def map_yolo_task_to_roboflow(task_name: str) -> str:
    """Map internal/Ultralytics task names to Roboflow API project types."""
    norm = task_name.lower().strip()
    if norm in ("segment", "instance-segmentation", "seg"):
        return TASK_INSTANCE_SEG
    if norm in ("detect", "object-detection", "detection"):
        return TASK_OBJECT_DETECT
    if norm in ("semantic", "semantic-segmentation", "sem"):
        return TASK_SEMANTIC_SEG
    if norm in ("pose", "keypoints"):
        return TASK_POSE
    if norm in ("classify", "classification", "cls"):
        return TASK_CLASSIFY
    return TASK_INSTANCE_SEG


def validate_roboflow_dataset(
    dataset_path: Path | str,
    target_task: Optional[str] = None,
) -> RoboflowValidationResult:
    """Pre-flight check of a dataset before uploading to Roboflow."""
    dataset_path = Path(dataset_path).resolve()

    if not dataset_path.exists():
        res = RoboflowValidationResult(
            dataset_path=dataset_path,
            detected_task=target_task or TASK_INSTANCE_SEG,
            num_classes=0,
        )
        res.errors.append(f"Dataset path does not exist: {dataset_path}")
        return res

    # 1. Parse data.yaml if present
    yaml_candidates = [
        dataset_path / "data.yaml",
        dataset_path / "dataset.yaml",
        dataset_path / "data.yml",
    ]
    meta: Dict[str, Any] = {}
    for ypath in yaml_candidates:
        if ypath.is_file():
            try:
                meta = yaml.safe_load(ypath.read_text(encoding="utf-8")) or {}
                break
            except Exception as exc:
                res = RoboflowValidationResult(
                    dataset_path=dataset_path,
                    detected_task=target_task or TASK_INSTANCE_SEG,
                    num_classes=0,
                )
                res.errors.append(f"Failed to parse {ypath.name}: {exc}")
                return res

    # Class mappings
    raw_names = meta.get("names")
    class_names: List[str] = []
    if isinstance(raw_names, list):
        class_names = [str(n) for n in raw_names]
    elif isinstance(raw_names, dict):
        class_names = [str(raw_names[k]) for k in sorted(raw_names)]

    declared_nc = meta.get("nc")
    try:
        parsed_nc = int(declared_nc) if declared_nc is not None else None
    except (TypeError, ValueError):
        parsed_nc = None

    # `nc` is authoritative when declared; otherwise fall back to the listed names,
    # and failing that infer it from the labels themselves further below. Defaulting
    # to 1 here would reject every class index >= 1 in an undeclared dataset.
    num_classes = parsed_nc if parsed_nc is not None else len(class_names)
    class_count_declared = num_classes > 0

    if not class_count_declared:
        num_classes = _infer_num_classes(dataset_path)

    if not class_names:
        class_names = [f"class_{i}" for i in range(num_classes)]

    # Determine task
    inferred_task = str(meta.get("task") or "").lower()
    resolved_task = map_yolo_task_to_roboflow(target_task or inferred_task or "instance-segmentation")

    result = RoboflowValidationResult(
        dataset_path=dataset_path,
        detected_task=resolved_task,
        num_classes=num_classes,
        class_names=class_names,
    )

    err_collector = _IssueCollector(result.errors)
    warn_collector = _IssueCollector(result.warnings)

    if not class_count_declared:
        warn_collector.add(
            "undeclared-classes",
            f"No data.yaml class declaration found; inferred {num_classes} class(es) from label files. "
            "Verify the class list on the Roboflow project after upload.",
        )

    if parsed_nc is not None and class_names and len(class_names) != parsed_nc:
        err_collector.add(
            "nc-mismatch",
            f"data.yaml declares nc={declared_nc} but lists {len(class_names)} class names.",
        )

    # 2. Fast single-pass file inventory
    all_images: List[Path] = []
    annot_by_dir_stem: Dict[Tuple[str, str], Path] = {}
    annot_by_split_stem: Dict[Tuple[str, str], Path] = {}

    for root, _, files in os.walk(dataset_path):
        root_p = Path(root)
        for f in files:
            fp = root_p / f
            ext = fp.suffix.lower()
            stem = fp.stem
            if ext in IMAGE_EXTENSIONS:
                all_images.append(fp)
            elif ext in ANNOTATION_EXTENSIONS:
                annot_by_dir_stem[(str(root_p.resolve()), stem)] = fp
                root_lower = root.lower()
                for raw_split, canonical in (
                    ("training", "train"),
                    ("train", "train"),
                    ("validation", "val"),
                    ("valid", "val"),
                    ("val", "val"),
                    ("testing", "test"),
                    ("test", "test"),
                ):
                    if f"/{raw_split}" in root_lower or f"\\{raw_split}" in root_lower:
                        # Prioritize .txt in labels/
                        existing = annot_by_split_stem.get((canonical, stem))
                        if not existing or ext == ".txt":
                            annot_by_split_stem[(canonical, stem)] = fp
                        break

    if not all_images:
        err_collector.add("no-images", f"No image files found under {dataset_path}")
        err_collector.finalize()
        warn_collector.finalize()
        return result

    # 3. Process each image & match annotation
    for img_path in all_images:
        rel = img_path.relative_to(dataset_path)
        parts = rel.parts

        split = "train"
        for p in parts:
            p_lower = p.lower()
            if p_lower in ("train", "training"):
                split = "train"
                break
            if p_lower in ("val", "valid", "validation"):
                split = "val"
                break
            if p_lower in ("test", "testing"):
                split = "test"
                break

        if split not in result.splits:
            result.splits[split] = SplitStats(name=split)
        split_stats = result.splits[split]
        split_stats.images += 1

        # Check image header integrity
        if not _is_valid_image_header(img_path):
            split_stats.corrupt_images += 1
            err_collector.add("corrupt-image", f"{img_path.name}: corrupted or invalid image header")
            continue

        stem = img_path.stem
        img_dir_str = str(img_path.parent.resolve())

        # Locate annotation (Prioritize labels/{split} over images/{split})
        annot_file = annot_by_split_stem.get((split, stem))
        if not annot_file:
            annot_file = annot_by_dir_stem.get((img_dir_str, stem))

        if not annot_file or not annot_file.exists():
            split_stats.missing_labels += 1
            continue

        try:
            content = annot_file.read_text(encoding="utf-8").strip()
        except (UnicodeDecodeError, OSError) as exc:
            err_collector.add("unreadable-label", f"{annot_file.name}: failed to read label file: {exc}")
            continue

        if not content:
            # 0-byte file: Clean negative sample (background)
            split_stats.negatives += 1
            continue

        split_stats.annotated += 1

        if annot_file.suffix.lower() == ".json":
            is_valid_json, json_err = _check_json_annotation(
                task=resolved_task,
                content=content,
                num_classes=num_classes,
                filename=annot_file.name,
            )
            if not is_valid_json and json_err:
                err_collector.add("json-format-error", json_err)
        else:
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            for line_no, line in enumerate(lines, start=1):
                fields = line.split()
                problem = _check_label_row(
                    task=resolved_task,
                    fields=fields,
                    num_classes=num_classes,
                    line_no=line_no,
                    filename=annot_file.name,
                )
                if problem:
                    err_collector.add("label-format-error", problem)

    # 4. Check dataset balance and split completeness
    if "train" not in result.splits or result.splits["train"].images == 0:
        err_collector.add("missing-train-split", "Dataset has no images assigned to the 'train' split.")

    if "val" not in result.splits or result.splits["val"].images == 0:
        warn_collector.add(
            "missing-val-split",
            "Dataset does not contain a 'val' split. Roboflow validation generation requires a validation set.",
        )

    for s_name, s_stats in result.splits.items():
        if s_stats.images > 0 and s_stats.annotated == 0 and s_stats.negatives == 0:
            warn_collector.add(
                f"unannotated-split-{s_name}",
                f"Split '{s_name}' has {s_stats.images} images but 0 annotations.",
            )

    err_collector.finalize()
    warn_collector.finalize()
    return result
