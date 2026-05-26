from __future__ import annotations

import json
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import requests
import yaml

from src.datasets.core import (
    IMAGE_EXTENSIONS,
    DatasetValidationError,
    _image_size,
    _iter_images,
    _normalize_names,
    _read_json,
    _read_yaml,
    _resolve_dataset_path,
    detect_dataset_format,
    find_coco_annotation_files,
    summarize_dataset,
)

OUTPUT_FORMATS = {"YOLO Detection", "YOLO Segmentation", "COCO"}
SPLIT_NAMES = ("train", "valid", "test")
SPLIT_STRATEGIES = {"class_balanced", "smart_balanced"}
OBJECT_SIZE_BUCKETS = ("small", "medium", "large")


@dataclass(frozen=True)
class PrepareSplitConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.20
    test_ratio: float = 0.10

    def normalized(self) -> tuple[float, float, float]:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if total <= 0:
            raise ValueError("At least one split ratio must be greater than zero.")
        if abs(total - 1.0) > 0.02:
            raise ValueError(f"Split ratios must sum to 1.0; got {total:.3f}.")
        return self.train_ratio / total, self.val_ratio / total, self.test_ratio / total


@dataclass
class PrepareDatasetConfig:
    source_path: Path
    output_format: str = "YOLO Detection"
    output_root: Path = Path("datasets")
    output_slug: str | None = None
    split_config: PrepareSplitConfig = field(default_factory=PrepareSplitConfig)
    split_strategy: str = "class_balanced"
    seed: int = 42
    overwrite: bool = False
    max_workers: int = 10


@dataclass
class PrepareDatasetStats:
    source_path: str
    source_format: str
    output_path: str
    output_format: str
    version: int
    classes: list[str]
    total_images: int
    total_annotations: int
    split_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)
    skipped_files: int = 0
    elapsed_seconds: float = 0.0
    split_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Annotation:
    class_id: int
    bbox: list[float] | None = None
    segmentation: list[float] | None = None


@dataclass
class ImageRecord:
    image_path: Path
    file_name: str
    width: int
    height: int
    annotations: list[Annotation] = field(default_factory=list)

    @property
    def class_ids(self) -> set[int]:
        return {ann.class_id for ann in self.annotations}

    @property
    def object_size_buckets(self) -> list[str]:
        return [object_size_bucket(ann, self.width, self.height) for ann in self.annotations]


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "prepared_dataset"


def resolve_versioned_output(output_root: Path, slug: str, *, overwrite: bool = False) -> tuple[Path, int]:
    output_root.mkdir(parents=True, exist_ok=True)
    match = re.search(r"_v(\d+)$", slug)
    base_slug = slug[: match.start()] if match else slug
    requested_version = int(match.group(1)) if match else None

    if requested_version is not None:
        path = output_root / slug
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output dataset already exists: {path}")
        return path, requested_version

    version = 1
    while True:
        path = output_root / f"{base_slug}_v{version:03d}"
        if overwrite or not path.exists():
            return path, version
        version += 1


def _copy_or_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _dedupe_filename(name: str, used: set[str]) -> str:
    candidate = Path(name).name
    if candidate not in used:
        used.add(candidate)
        return candidate
    stem = Path(candidate).stem
    suffix = Path(candidate).suffix
    idx = 2
    while True:
        renamed = f"{stem}_{idx:04d}{suffix}"
        if renamed not in used:
            used.add(renamed)
            return renamed
        idx += 1


def _bbox_to_segmentation(bbox: list[float], width: int, height: int) -> list[float]:
    x, y, w, h = bbox
    x1 = max(0.0, x / width)
    y1 = max(0.0, y / height)
    x2 = min(1.0, (x + w) / width)
    y2 = min(1.0, (y + h) / height)
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def _segmentation_to_bbox(segmentation: list[float], width: int, height: int) -> list[float]:
    xs = segmentation[0::2]
    ys = segmentation[1::2]
    x1 = min(xs) * width
    y1 = min(ys) * height
    x2 = max(xs) * width
    y2 = max(ys) * height
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _yolo_bbox_from_coco(bbox: list[float], width: int, height: int) -> list[float]:
    x, y, w, h = bbox
    return [
        (x + w / 2) / width,
        (y + h / 2) / height,
        w / width,
        h / height,
    ]


def _normalized_annotation_area(annotation: Annotation, width: int, height: int) -> float:
    if annotation.bbox is not None and len(annotation.bbox) == 4:
        return max(0.0, annotation.bbox[2]) * max(0.0, annotation.bbox[3])
    if annotation.segmentation is not None and len(annotation.segmentation) >= 6:
        bbox = _segmentation_to_bbox(annotation.segmentation, width, height)
        image_area = max(1.0, float(width * height))
        return max(0.0, bbox[2] * bbox[3]) / image_area
    return 0.0


def object_size_bucket(annotation: Annotation, width: int, height: int) -> str:
    area = _normalized_annotation_area(annotation, width, height)
    if area < 0.01:
        return "small"
    if area < 0.05:
        return "medium"
    return "large"


def _coco_bbox_from_yolo(bbox: list[float], width: int, height: int) -> list[float]:
    cx, cy, w, h = bbox
    return [
        (cx - w / 2) * width,
        (cy - h / 2) * height,
        w * width,
        h * height,
    ]


def _read_yolo_records(source: Path, warnings: list[str]) -> tuple[list[ImageRecord], list[str], str]:
    yaml_path = source / "data.yaml"
    if not yaml_path.exists():
        yaml_path = source / "dataset.yaml"
    data = _read_yaml(yaml_path)
    classes = _normalize_names(data.get("names"))
    task = str(data.get("task", "")).lower()
    used_names: set[str] = set()
    records: list[ImageRecord] = []
    seg_hits = bbox_hits = 0

    for split_key, aliases in {
        "train": ("train", "training"),
        "valid": ("val", "valid", "validation"),
        "test": ("test", "testing"),
    }.items():
        raw_path = next((data.get(alias) for alias in aliases if data.get(alias)), None)
        image_dir = _resolve_dataset_path(yaml_path.parent, raw_path)
        if image_dir is None or not image_dir.exists():
            continue
        label_dir = Path(str(image_dir).replace("/images", "/labels"))
        if not label_dir.exists():
            label_dir = image_dir.parent / "labels"

        for image_path in _iter_images(image_dir):
            width, height = _image_size(image_path)
            file_name = _dedupe_filename(image_path.name, used_names)
            annotations: list[Annotation] = []
            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cls = int(float(parts[0]))
                        values = [float(value) for value in parts[1:]]
                    except ValueError:
                        warnings.append(f"Skipped invalid YOLO row {label_path}:{line_no}")
                        continue
                    if len(values) == 4:
                        bbox_hits += 1
                        annotations.append(Annotation(cls, bbox=values))
                    elif len(values) >= 6 and len(values) % 2 == 0:
                        seg_hits += 1
                        annotations.append(Annotation(cls, segmentation=values))
                    else:
                        warnings.append(f"Skipped invalid YOLO row {label_path}:{line_no}")
            records.append(ImageRecord(image_path, file_name, int(width), int(height), annotations))

    if not task:
        task = "segment" if seg_hits and seg_hits >= bbox_hits else "detect"
    return records, classes, task


def _find_coco_image(source: Path, ann_path: Path, file_name: str) -> Path | None:
    candidates = [
        source / file_name,
        ann_path.parent.parent / file_name,
        ann_path.parent.parent / Path(file_name).name,
        ann_path.parent.parent / "images" / Path(file_name).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(source.rglob(Path(file_name).name))
    return matches[0] if matches else None


def _read_coco_records(source: Path, warnings: list[str]) -> tuple[list[ImageRecord], list[str], str]:
    annotation_files = find_coco_annotation_files(source)
    classes: list[str] = []
    records: list[ImageRecord] = []
    used_names: set[str] = set()
    has_seg = False

    for _, ann_path in annotation_files.items():
        data = _read_json(ann_path)
        categories = sorted(data.get("categories", []), key=lambda c: c.get("id", 0))
        if not classes:
            classes = [str(cat.get("name", cat.get("id"))) for cat in categories]
        cat_to_index = {cat.get("id"): idx for idx, cat in enumerate(categories)}
        anns_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for ann in data.get("annotations", []):
            anns_by_image[ann.get("image_id")].append(ann)

        for image in data.get("images", []):
            file_name_raw = str(image.get("file_name") or "")
            if not file_name_raw:
                warnings.append(f"Skipped COCO image without file_name in {ann_path}")
                continue
            image_path = _find_coco_image(source, ann_path, file_name_raw)
            if image_path is None:
                warnings.append(f"Missing image referenced by COCO JSON: {file_name_raw}")
                continue
            width = int(float(image.get("width") or _image_size(image_path)[0] or 1))
            height = int(float(image.get("height") or _image_size(image_path)[1] or 1))
            file_name = _dedupe_filename(Path(file_name_raw).name, used_names)
            record_annotations: list[Annotation] = []
            for ann in anns_by_image.get(image.get("id"), []):
                if ann.get("category_id") not in cat_to_index:
                    continue
                cls = cat_to_index[ann.get("category_id")]
                bbox = ann.get("bbox")
                segmentation = ann.get("segmentation")
                normalized_seg: list[float] | None = None
                if (
                    isinstance(segmentation, list)
                    and segmentation
                    and isinstance(segmentation[0], list)
                    and len(segmentation[0]) >= 6
                ):
                    coords = [float(v) for v in segmentation[0]]
                    normalized_seg = [
                        max(0.0, min(1.0, coords[i] / (width if i % 2 == 0 else height)))
                        for i in range(len(coords))
                    ]
                    has_seg = True
                coco_bbox = [float(v) for v in bbox] if isinstance(bbox, list) and len(bbox) == 4 else None
                yolo_bbox = _yolo_bbox_from_coco(coco_bbox, width, height) if coco_bbox else None
                record_annotations.append(Annotation(cls, bbox=yolo_bbox, segmentation=normalized_seg))
            records.append(ImageRecord(image_path, file_name, width, height, record_annotations))

    return records, classes, "segment" if has_seg else "detect"


def _extract_labelbox_objects(row: dict[str, Any]) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for project in row.get("projects", {}).values():
        for label in project.get("labels", []):
            objects.extend(label.get("annotations", {}).get("objects", []))
    return objects


def _download_ndjson_image(url: str, output_path: Path) -> bool:
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        output_path.write_bytes(response.content)
        return True
    except Exception:
        return False


def _read_ndjson_records(source: Path, warnings: list[str], *, max_workers: int) -> tuple[list[ImageRecord], list[str], str]:
    from PIL import Image

    rows = [json.loads(line) for line in source.read_text("utf-8").splitlines() if line.strip()]
    temp_dir = source.parent / f".{source.stem}_yolomatic_downloads"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    classes: dict[str, int] = {}
    records: list[ImageRecord] = []
    used_names: set[str] = set()
    has_seg = False

    def class_id(name: str) -> int:
        if name not in classes:
            classes[name] = len(classes)
        return classes[name]

    futures = {}
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        for idx, row in enumerate(rows):
            data_row = row.get("data_row", {})
            url = data_row.get("row_data")
            if not url:
                warnings.append(f"Skipped NDJSON row {idx + 1}: missing data_row.row_data")
                continue
            parsed_name = Path(urlparse(str(url)).path).name or f"row_{idx + 1:06d}.jpg"
            filename = Path(str(data_row.get("global_key") or parsed_name)).name
            target = temp_dir / _dedupe_filename(filename, used_names)
            futures[executor.submit(_download_ndjson_image, str(url), target)] = (idx, row, target)

        for future in as_completed(futures):
            idx, row, image_path = futures[future]
            if not future.result():
                warnings.append(f"Skipped NDJSON row {idx + 1}: image download failed")
                continue
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
            except Exception:
                warnings.append(f"Skipped NDJSON row {idx + 1}: downloaded image could not be opened")
                continue
            annotations: list[Annotation] = []
            for obj in _extract_labelbox_objects(row):
                name = str(obj.get("name") or "unnamed")
                cls = class_id(name)
                if "polygon" in obj:
                    points = obj.get("polygon") or []
                    coords: list[float] = []
                    for point in points:
                        coords.extend([
                            float(point["x"]) / width,
                            float(point["y"]) / height,
                        ])
                    if len(coords) >= 6:
                        has_seg = True
                        annotations.append(Annotation(cls, segmentation=coords))
                elif "bounding_box" in obj:
                    bbox = obj["bounding_box"]
                    top = float(bbox["top"])
                    left = float(bbox["left"])
                    h = float(bbox["height"])
                    w = float(bbox["width"])
                    annotations.append(
                        Annotation(cls, bbox=[(left + w / 2) / width, (top + h / 2) / height, w / width, h / height])
                    )
            records.append(ImageRecord(image_path, image_path.name, int(width), int(height), annotations))

    class_names = [name for name, _ in sorted(classes.items(), key=lambda item: item[1])]
    return records, class_names, "segment" if has_seg else "detect"


def _load_records(config: PrepareDatasetConfig, warnings: list[str]) -> tuple[list[ImageRecord], list[str], str, str]:
    source = Path(config.source_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")
    if source.is_file() and source.suffix.lower() == ".ndjson":
        records, classes, task = _read_ndjson_records(source, warnings, max_workers=config.max_workers)
        return records, classes, task, "ndjson"
    source_format = detect_dataset_format(source)
    if source_format == "unknown":
        summary = summarize_dataset(source)
        raise DatasetValidationError(
            "Dataset format could not be detected.",
            summary=summary,
            path=source,
            format=source_format,
            suggested_fix="Use a YOLO data.yaml, COCO annotations JSON, or Labelbox .ndjson export.",
        )
    if source_format in {"yolo", "mixed"}:
        records, classes, task = _read_yolo_records(source, warnings)
        return records, classes, task, source_format
    records, classes, task = _read_coco_records(source, warnings)
    return records, classes, task, source_format


def _target_counts(total: int, split_config: PrepareSplitConfig) -> dict[str, int]:
    train_ratio, val_ratio, test_ratio = split_config.normalized()
    train = int(total * train_ratio)
    val = int(total * val_ratio)
    test = total - train - val
    if test_ratio == 0:
        train += test
        test = 0
    return {"train": train, "valid": val, "test": test}


def split_records(
    records: list[ImageRecord],
    split_config: PrepareSplitConfig,
    seed: int,
    strategy: str = "class_balanced",
) -> dict[str, list[ImageRecord]]:
    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"Unsupported split strategy: {strategy}")
    if strategy == "smart_balanced":
        return _split_records_smart_balanced(records, split_config, seed)

    rng = random.Random(seed)
    target = _target_counts(len(records), split_config)
    result: dict[str, list[ImageRecord]] = {name: [] for name in SPLIT_NAMES}
    class_totals = Counter(cls for record in records for cls in record.class_ids)
    split_class_counts: dict[str, Counter[int]] = {name: Counter() for name in SPLIT_NAMES}

    ordered = list(records)
    rng.shuffle(ordered)
    ordered.sort(
        key=lambda record: (
            1 if record.class_ids else 2,
            min((class_totals[cls] for cls in record.class_ids), default=len(records) + 1),
            record.file_name,
        )
    )

    for record in ordered:
        candidates = [name for name in SPLIT_NAMES if len(result[name]) < target[name]]
        if not candidates:
            candidates = list(SPLIT_NAMES)

        def score(split_name: str) -> tuple[float, int, str]:
            if not record.class_ids:
                return (len(result[split_name]) / max(1, target[split_name]), len(result[split_name]), split_name)
            balance = 0.0
            for cls in record.class_ids:
                desired = max(1.0, class_totals[cls] * (target[split_name] / max(1, len(records))))
                balance += split_class_counts[split_name][cls] / desired
            return (balance / len(record.class_ids), len(result[split_name]), split_name)

        chosen = min(candidates, key=score)
        result[chosen].append(record)
        for cls in record.class_ids:
            split_class_counts[chosen][cls] += 1

    return result


def _counter_cost(counter: Counter[Any], totals: Counter[Any], ratio: float, weight: float = 1.0) -> float:
    if not totals:
        return 0.0
    cost = 0.0
    for key, total in totals.items():
        desired = max(0.0001, total * ratio)
        cost += abs(counter[key] - desired) / desired
    return (cost / max(1, len(totals))) * weight


def _split_records_smart_balanced(
    records: list[ImageRecord],
    split_config: PrepareSplitConfig,
    seed: int,
) -> dict[str, list[ImageRecord]]:
    rng = random.Random(seed)
    target = _target_counts(len(records), split_config)
    result: dict[str, list[ImageRecord]] = {name: [] for name in SPLIT_NAMES}
    split_class_counts: dict[str, Counter[int]] = {name: Counter() for name in SPLIT_NAMES}
    split_size_counts: dict[str, Counter[str]] = {name: Counter() for name in SPLIT_NAMES}
    split_object_counts: dict[str, int] = {name: 0 for name in SPLIT_NAMES}
    split_unlabeled_counts: dict[str, int] = {name: 0 for name in SPLIT_NAMES}

    class_totals = Counter(ann.class_id for record in records for ann in record.annotations)
    size_totals = Counter(bucket for record in records for bucket in record.object_size_buckets)
    total_objects = sum(len(record.annotations) for record in records)
    total_unlabeled = sum(1 for record in records if not record.annotations)
    total_images = max(1, len(records))

    ordered = list(records)
    rng.shuffle(ordered)
    ordered.sort(
        key=lambda record: (
            0 if record.annotations else 1,
            -len(record.annotations),
            min((class_totals[ann.class_id] for ann in record.annotations), default=len(records) + 1),
            record.file_name,
        )
    )

    for record in ordered:
        candidates = [name for name in SPLIT_NAMES if len(result[name]) < target[name]]
        if not candidates:
            candidates = list(SPLIT_NAMES)
        record_class_counts = Counter(ann.class_id for ann in record.annotations)
        record_size_counts = Counter(record.object_size_buckets)
        record_unlabeled = 1 if not record.annotations else 0

        def score(split_name: str) -> tuple[float, int, str]:
            ratio = target[split_name] / total_images
            class_counts = split_class_counts[split_name] + record_class_counts
            size_counts = split_size_counts[split_name] + record_size_counts
            object_count = split_object_counts[split_name] + len(record.annotations)
            unlabeled_count = split_unlabeled_counts[split_name] + record_unlabeled
            image_fill = abs((len(result[split_name]) + 1) - target[split_name]) / max(1, target[split_name])
            object_desired = max(0.0001, total_objects * ratio)
            unlabeled_desired = max(0.0001, total_unlabeled * ratio)
            object_cost = abs(object_count - object_desired) / object_desired if total_objects else 0.0
            unlabeled_cost = abs(unlabeled_count - unlabeled_desired) / unlabeled_desired if total_unlabeled else 0.0
            cost = (
                _counter_cost(class_counts, class_totals, ratio, 3.0)
                + _counter_cost(size_counts, size_totals, ratio, 2.0)
                + object_cost
                + unlabeled_cost
                + image_fill
            )
            return (cost, len(result[split_name]), split_name)

        chosen = min(candidates, key=score)
        result[chosen].append(record)
        split_class_counts[chosen].update(record_class_counts)
        split_size_counts[chosen].update(record_size_counts)
        split_object_counts[chosen] += len(record.annotations)
        split_unlabeled_counts[chosen] += record_unlabeled

    return result


def split_diagnostics(
    split_records_by_name: dict[str, list[ImageRecord]],
    classes: list[str],
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for split_name in SPLIT_NAMES:
        records = split_records_by_name.get(split_name, [])
        class_counter = Counter(ann.class_id for record in records for ann in record.annotations)
        size_counter = Counter(bucket for record in records for bucket in record.object_size_buckets)
        object_counts = [len(record.annotations) for record in records]
        diagnostics[split_name] = {
            "image_count": len(records),
            "annotation_count": sum(object_counts),
            "class_counts": {
                classes[class_id] if 0 <= class_id < len(classes) else f"class_{class_id}": count
                for class_id, count in sorted(class_counter.items())
            },
            "object_size_counts": {bucket: size_counter.get(bucket, 0) for bucket in OBJECT_SIZE_BUCKETS},
            "unlabeled_images": sum(1 for record in records if not record.annotations),
            "objects_per_image": {
                "min": min(object_counts, default=0),
                "max": max(object_counts, default=0),
                "avg": round(sum(object_counts) / len(object_counts), 3) if object_counts else 0.0,
            },
        }
    return diagnostics


def _split_warnings(
    split_records_by_name: dict[str, list[ImageRecord]],
    split_config: PrepareSplitConfig,
    strategy: str,
) -> list[str]:
    warnings: list[str] = []
    enabled_splits = [
        name
        for name, ratio in zip(SPLIT_NAMES, split_config.normalized(), strict=True)
        if ratio > 0
    ]
    for split_name in enabled_splits:
        if not split_records_by_name.get(split_name):
            warnings.append(f"{split_name} split is empty; the dataset is too small for the requested ratios.")
    if strategy == "smart_balanced":
        total_records = sum(len(records) for records in split_records_by_name.values())
        total_annotations = sum(len(record.annotations) for records in split_records_by_name.values() for record in records)
        if total_records < len(enabled_splits) * 2:
            warnings.append("Smart split degraded to deterministic best effort because the dataset is very small.")
        if total_annotations == 0:
            warnings.append("Smart split found no annotations; only image and background balance could be optimized.")
    return warnings


def _write_yolo_label(path: Path, record: ImageRecord, output_format: str) -> int:
    lines: list[str] = []
    for ann in record.annotations:
        if output_format == "YOLO Detection":
            bbox = ann.bbox
            if bbox is None and ann.segmentation is not None:
                coco_bbox = _segmentation_to_bbox(ann.segmentation, record.width, record.height)
                bbox = _yolo_bbox_from_coco(coco_bbox, record.width, record.height)
            if bbox is not None:
                lines.append(f"{ann.class_id} " + " ".join(f"{value:.6f}" for value in bbox))
        else:
            segmentation = ann.segmentation
            if segmentation is None and ann.bbox is not None:
                segmentation = _bbox_to_segmentation(_coco_bbox_from_yolo(ann.bbox, record.width, record.height), record.width, record.height)
            if segmentation is not None:
                lines.append(f"{ann.class_id} " + " ".join(f"{value:.6f}" for value in segmentation))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(dict.fromkeys(lines)), encoding="utf-8")
    return len(lines)


def _write_yolo_dataset(
    output: Path,
    split_records_by_name: dict[str, list[ImageRecord]],
    classes: list[str],
    output_format: str,
) -> int:
    total_annotations = 0
    for split_name, records in split_records_by_name.items():
        img_dir = output / split_name / "images"
        label_dir = output / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            target_image = img_dir / record.file_name
            _copy_or_link(record.image_path, target_image)
            total_annotations += _write_yolo_label(label_dir / f"{Path(record.file_name).stem}.txt", record, output_format)
    data_yaml = {
        "path": str(output.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(classes),
        "names": classes,
        "task": "segment" if output_format == "YOLO Segmentation" else "detect",
    }
    (output / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return total_annotations


def _write_coco_dataset(output: Path, split_records_by_name: dict[str, list[ImageRecord]], classes: list[str]) -> int:
    total_annotations = 0
    ann_dir = output / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    categories = [{"id": idx + 1, "name": name} for idx, name in enumerate(classes)]
    for split_name, records in split_records_by_name.items():
        img_dir = output / split_name / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        coco = {"images": [], "annotations": [], "categories": categories}
        ann_id = 1
        for image_id, record in enumerate(records, start=1):
            target_image = img_dir / record.file_name
            _copy_or_link(record.image_path, target_image)
            coco["images"].append(
                {
                    "id": image_id,
                    "file_name": str(Path(split_name) / "images" / record.file_name),
                    "width": record.width,
                    "height": record.height,
                }
            )
            for ann in record.annotations:
                if ann.bbox is not None:
                    bbox = _coco_bbox_from_yolo(ann.bbox, record.width, record.height)
                elif ann.segmentation is not None:
                    bbox = _segmentation_to_bbox(ann.segmentation, record.width, record.height)
                else:
                    continue
                segmentation: list[list[float]] = []
                if ann.segmentation is not None:
                    segmentation = [[
                        ann.segmentation[i] * (record.width if i % 2 == 0 else record.height)
                        for i in range(len(ann.segmentation))
                    ]]
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": ann.class_id + 1,
                        "bbox": bbox,
                        "area": max(0.0, bbox[2] * bbox[3]),
                        "iscrowd": 0,
                        "segmentation": segmentation,
                    }
                )
                ann_id += 1
                total_annotations += 1
        (ann_dir / f"instances_{split_name}.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")
    return total_annotations


def _write_readme(output: Path, stats: PrepareDatasetStats, split_config: PrepareSplitConfig, seed: int) -> None:
    classes = ", ".join(stats.classes[:12])
    if len(stats.classes) > 12:
        classes += ", ..."
    readme = f"""# Prepared Dataset: {Path(stats.output_path).name}

Generated by YOLOmatic Prepare / Split Dataset.

## Summary
- Source: `{stats.source_path}`
- Source format: `{stats.source_format}`
- Output format: `{stats.output_format}`
- Version: `v{stats.version:03d}`
- Seed: `{seed}`
- Split ratios: `{split_config.train_ratio:.0%} / {split_config.val_ratio:.0%} / {split_config.test_ratio:.0%}`
- Split strategy: `{stats.split_diagnostics.get("strategy", "class_balanced")}`

## Counts
| Split | Images |
|-------|--------|
| Train | {stats.split_counts.get("train", 0)} |
| Valid | {stats.split_counts.get("valid", 0)} |
| Test | {stats.split_counts.get("test", 0)} |

## Classes
- Total: {len(stats.classes)}
- Names: `{classes}`

## Next Step
Use this dataset from YOLOmatic's Configure Model or Train Model workflows.
"""
    if stats.warnings:
        readme += "\n## Warnings\n" + "\n".join(f"- {warning}" for warning in stats.warnings[:50]) + "\n"
    (output / "README.md").write_text(readme, encoding="utf-8")


def prepare_dataset(
    config: PrepareDatasetConfig,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> PrepareDatasetStats:
    started = time.time()
    if config.output_format not in OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output format: {config.output_format}")
    if config.split_strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"Unsupported split strategy: {config.split_strategy}")
    config.split_config.normalized()

    warnings: list[str] = []
    records, classes, task, source_format = _load_records(config, warnings)
    if not records:
        raise DatasetValidationError("No images were found in the source dataset.", path=config.source_path, format=source_format)
    if not classes:
        max_cls = max((ann.class_id for record in records for ann in record.annotations), default=-1)
        classes = [f"class_{idx}" for idx in range(max_cls + 1)]
    if config.output_format == "YOLO Segmentation" and task == "detect":
        warnings.append("Source is detection-only; YOLO segmentation output uses rectangle polygons from bounding boxes.")

    slug = slugify(config.output_slug or Path(config.source_path).stem)
    output, version = resolve_versioned_output(config.output_root, slug, overwrite=config.overwrite)
    if output.exists():
        if not config.overwrite:
            raise FileExistsError(f"Output dataset already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(0, len(records), "Splitting dataset...")
    split_records_by_name = split_records(records, config.split_config, config.seed, config.split_strategy)
    warnings.extend(_split_warnings(split_records_by_name, config.split_config, config.split_strategy))
    if progress_callback:
        progress_callback(len(records), len(records), "Writing output dataset...")

    if config.output_format == "COCO":
        total_annotations = _write_coco_dataset(output, split_records_by_name, classes)
    else:
        total_annotations = _write_yolo_dataset(output, split_records_by_name, classes, config.output_format)
    if source_format == "ndjson":
        shutil.rmtree(Path(config.source_path).parent / f".{Path(config.source_path).stem}_yolomatic_downloads", ignore_errors=True)

    split_counts = {name: len(items) for name, items in split_records_by_name.items()}
    diagnostics = split_diagnostics(split_records_by_name, classes)
    diagnostics["strategy"] = config.split_strategy
    stats = PrepareDatasetStats(
        source_path=str(Path(config.source_path).resolve()),
        source_format=source_format,
        output_path=str(output.resolve()),
        output_format=config.output_format,
        version=version,
        classes=classes,
        total_images=len(records),
        total_annotations=total_annotations,
        split_counts=split_counts,
        warnings=warnings,
        skipped_files=len(warnings),
        elapsed_seconds=time.time() - started,
        split_diagnostics=diagnostics,
    )
    manifest = stats.to_dict()
    manifest["split_ratios"] = asdict(config.split_config)
    manifest["split_strategy"] = config.split_strategy
    manifest["seed"] = config.seed
    manifest["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_readme(output, stats, config.split_config, config.seed)
    return stats
