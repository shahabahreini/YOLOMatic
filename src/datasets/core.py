from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation"),
    "test": ("test", "testing"),
}


@dataclass
class SplitSummary:
    name: str
    images_path: str | None = None
    labels_path: str | None = None
    annotations_path: str | None = None
    image_count: int = 0
    annotation_count: int = 0
    labeled_image_count: int = 0
    unlabeled_image_count: int = 0
    empty_label_count: int = 0
    missing_file_count: int = 0
    status: str = "missing"


@dataclass
class DatasetSummary:
    path: str
    name: str
    format: str = "unknown"
    task: str = "unknown"
    classes: list[str] = field(default_factory=list)
    image_count: int = 0
    annotation_count: int = 0
    labeled_image_count: int = 0
    unlabeled_image_count: int = 0
    empty_label_count: int = 0
    missing_file_count: int = 0
    total_size_bytes: int = 0
    splits: dict[str, SplitSummary] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    compatibility: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["splits"] = {key: asdict(value) for key, value in self.splits.items()}
        return data


class DatasetValidationError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        summary: DatasetSummary | None = None,
        path: str | Path | None = None,
        format: str | None = None,
        task: str | None = None,
        split: str | None = None,
        file: str | Path | None = None,
        suggested_fix: str | None = None,
    ) -> None:
        super().__init__(message)
        self.summary = summary
        self.path = None if path is None else str(path)
        self.format = format
        self.task = task
        self.split = split
        self.file = None if file is None else str(file)
        self.suggested_fix = suggested_fix


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return loaded if isinstance(loaded, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def _iter_images(path: Path | None) -> list[Path]:
    if path is None or not path.exists():
        return []
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _image_size(path: Path) -> tuple[float, float]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
            return float(width or 1), float(height or 1)
    except Exception:
        return 1.0, 1.0


def _resolve_dataset_path(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    normalized = str(path).replace("\\", "/")
    if normalized.startswith("../"):
        return (base / normalized[3:]).resolve()
    return (base / path).resolve()


def find_coco_annotation_files(dataset_path: str | Path) -> dict[str, Path]:
    root = Path(dataset_path)
    found: dict[str, Path] = {}
    candidates = list(root.glob("*_annotations.coco.json"))
    candidates.extend((root / "annotations").glob("instances_*.json"))
    for path in candidates:
        text = path.name.lower()
        split = "val" if "valid" in text or "val" in text else "test" if "test" in text else "train"
        found.setdefault(split, path)
    return found


def detect_dataset_format(dataset_path: str | Path) -> str:
    root = Path(dataset_path)
    has_yolo = (root / "data.yaml").exists() or (root / "dataset.yaml").exists()
    has_coco = bool(find_coco_annotation_files(root))
    if has_yolo and has_coco:
        return "mixed"
    if has_yolo:
        return "yolo"
    if has_coco:
        return "coco"
    return "unknown"


def _normalize_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        def _sort_key(value: Any) -> tuple[int, Any]:
            text = str(value)
            return (0, int(text)) if text.isdigit() else (1, text)

        return [str(names[key]) for key in sorted(names, key=_sort_key)]
    if isinstance(names, list):
        return [str(name) for name in names]
    return []


def summarize_dataset(dataset_path: str | Path, *, sample_limit: int = 5000) -> DatasetSummary:
    root = Path(dataset_path).resolve()
    summary = DatasetSummary(path=str(root), name=root.name, format=detect_dataset_format(root))
    if not root.exists():
        summary.errors.append("Dataset path does not exist.")
        return summary
    try:
        summary.total_size_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    except OSError:
        summary.warnings.append("Some files could not be inspected.")

    if summary.format in {"yolo", "mixed"}:
        _summarize_yolo(root, summary, sample_limit)
    if summary.format in {"coco", "mixed"}:
        _summarize_coco(root, summary, sample_limit)
    if summary.format == "unknown":
        summary.image_count = len(_iter_images(root))
        summary.errors.append("No data.yaml or COCO annotations were found.")

    tasks = set()
    for split in summary.splits.values():
        if split.annotation_count:
            tasks.add("segmentation" if summary.task == "segmentation" else "detection")
    if not tasks and summary.annotation_count == 0:
        summary.task = "empty" if summary.image_count else "unknown"
    summary.compatibility = {
        "yolo": "native" if summary.format in {"yolo", "mixed"} else "conversion required",
        "rfdetr": "native" if summary.format in {"yolo", "mixed"} else "conversion required",
        "detectron2": "native" if summary.format in {"coco", "mixed"} else "conversion required",
    }
    return summary


def _summarize_yolo(root: Path, summary: DatasetSummary, sample_limit: int) -> None:
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        yaml_path = root / "dataset.yaml"
    if not yaml_path.exists():
        return
    data = _read_yaml(yaml_path)
    summary.classes = summary.classes or _normalize_names(data.get("names"))
    detection_hits = segmentation_hits = 0
    for canonical, aliases in SPLIT_ALIASES.items():
        value = next((data.get(alias) for alias in aliases if data.get(alias)), None)
        images_path = _resolve_dataset_path(yaml_path.parent, value)
        labels_path = None
        if images_path is not None:
            labels_path = Path(str(images_path).replace("/images", "/labels"))
            if not labels_path.exists():
                labels_path = images_path.parent / "labels"
        split = SplitSummary(canonical, str(images_path) if images_path else None, str(labels_path) if labels_path else None)
        images = _iter_images(images_path)[:sample_limit]
        split.image_count = len(images)
        label_files = sorted(labels_path.glob("*.txt"))[:sample_limit] if labels_path and labels_path.exists() else []
        labeled_stems: set[str] = set()
        for label_file in label_files:
            try:
                lines = [line.strip() for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            except OSError:
                split.missing_file_count += 1
                continue
            if not lines:
                split.empty_label_count += 1
            else:
                labeled_stems.add(label_file.stem)
            for line in lines:
                parts = line.split()
                if len(parts) == 5:
                    detection_hits += 1
                elif len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                    segmentation_hits += 1
                split.annotation_count += 1
        image_stems = {p.stem for p in images}
        split.labeled_image_count = len(image_stems & labeled_stems)
        split.unlabeled_image_count = max(0, split.image_count - split.labeled_image_count)
        split.status = "valid" if split.image_count and (split.annotation_count or split.empty_label_count) else "warning"
        summary.splits[canonical] = split
    _rollup(summary)
    if segmentation_hits and detection_hits:
        summary.task = "mixed"
    elif segmentation_hits:
        summary.task = "segmentation"
    elif detection_hits:
        summary.task = "detection"


def _summarize_coco(root: Path, summary: DatasetSummary, sample_limit: int) -> None:
    for split_name, ann_path in find_coco_annotation_files(root).items():
        data = _read_json(ann_path)
        categories = sorted(data.get("categories", []), key=lambda c: c.get("id", 0))
        if categories and not summary.classes:
            summary.classes = [str(c.get("name", c.get("id"))) for c in categories]
        images = data.get("images", [])[:sample_limit]
        annotations = data.get("annotations", [])[:sample_limit]
        image_ids = {image.get("id") for image in images}
        labeled = {ann.get("image_id") for ann in annotations}
        split = summary.splits.get(split_name, SplitSummary(split_name))
        split.annotations_path = str(ann_path)
        split.image_count = max(split.image_count, len(images))
        split.annotation_count += len(annotations)
        split.labeled_image_count = len(image_ids & labeled)
        split.unlabeled_image_count = max(0, split.image_count - split.labeled_image_count)
        split.status = "valid" if split.image_count else "warning"
        summary.splits[split_name] = split
        if any(ann.get("segmentation") for ann in annotations):
            summary.task = "segmentation" if summary.task in {"unknown", "empty"} else "mixed"
        elif annotations and summary.task == "unknown":
            summary.task = "detection"
    _rollup(summary)


def _rollup(summary: DatasetSummary) -> None:
    summary.image_count = sum(split.image_count for split in summary.splits.values())
    summary.annotation_count = sum(split.annotation_count for split in summary.splits.values())
    summary.labeled_image_count = sum(split.labeled_image_count for split in summary.splits.values())
    summary.unlabeled_image_count = sum(split.unlabeled_image_count for split in summary.splits.values())
    summary.empty_label_count = sum(split.empty_label_count for split in summary.splits.values())
    summary.missing_file_count = sum(split.missing_file_count for split in summary.splits.values())


def _cache_dir(source: Path, family: str, prepared_format: str) -> Path:
    digest = hashlib.sha1(f"{source.resolve()}:{source.stat().st_mtime_ns}:{prepared_format}".encode()).hexdigest()[:12]
    return Path("datasets") / ".yolomatic_cache" / family / source.name / digest


def prepare_dataset_for_family(dataset_path: str | Path, family: str, *, task: str | None = None) -> dict[str, Any]:
    source = Path(dataset_path).resolve()
    summary = summarize_dataset(source)
    family_key = "detectron2" if family == "detectron2" else "yolo"
    prepared_format = "coco" if family_key == "detectron2" else "yolo"
    if summary.format == "unknown":
        raise DatasetValidationError(
            "Dataset format could not be detected.",
            summary=summary,
            path=source,
            format=summary.format,
            task=summary.task,
            suggested_fix="Add a YOLO data.yaml or COCO annotations JSON file.",
        )
    if summary.format == prepared_format or summary.format == "mixed":
        prepared_dir = source
    else:
        prepared_dir = _cache_dir(source, family_key, prepared_format)
        if prepared_format == "yolo":
            convert_coco_to_yolo(source, prepared_dir, summary=summary)
        else:
            convert_yolo_to_coco(source, prepared_dir, summary=summary)
    prepared_summary = summarize_dataset(prepared_dir)
    return {
        "name": source.name,
        "source_format": summary.format,
        "prepared_format": prepared_format,
        "source_dir": str(source),
        "prepared_dir": str(prepared_dir),
        "classes": prepared_summary.classes or summary.classes,
        "task": task or summary.task,
        "summary": summary.to_dict(),
        "splits": {key: asdict(value) for key, value in prepared_summary.splits.items()},
    }


def convert_yolo_to_coco(source_dir: str | Path, output_dir: str | Path, *, summary: DatasetSummary | None = None) -> Path:
    source = Path(source_dir).resolve()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = _read_yaml(source / "data.yaml")
    classes = _normalize_names(data.get("names"))
    manifest = {"source": str(source), "format": "coco", "warnings": [], "classes": classes}
    for split_name, aliases in SPLIT_ALIASES.items():
        value = next((data.get(alias) for alias in aliases if data.get(alias)), None)
        images_path = _resolve_dataset_path(source, value)
        if images_path is None:
            continue
        target_images = output / split_name / "images"
        target_images.mkdir(parents=True, exist_ok=True)
        images = _iter_images(images_path)
        coco = {
            "images": [],
            "annotations": [],
            "categories": [{"id": i + 1, "name": name} for i, name in enumerate(classes)],
        }
        ann_id = 1
        for image_id, image_path in enumerate(images, start=1):
            width, height = _image_size(image_path)
            target = target_images / image_path.name
            if not target.exists():
                shutil.copy2(image_path, target)
            coco["images"].append(
                {
                    "id": image_id,
                    "file_name": str(Path(split_name) / "images" / image_path.name),
                    "width": int(width),
                    "height": int(height),
                }
            )
            label_path = Path(str(images_path).replace("/images", "/labels")) / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                parts = line.strip().split()
                if len(parts) not in {5} and not (len(parts) >= 7 and (len(parts) - 1) % 2 == 0):
                    manifest["warnings"].append(f"Skipped invalid YOLO row {label_path}:{line_no}")
                    continue
                cls = int(float(parts[0]))
                nums = [float(v) for v in parts[1:]]
                if len(parts) == 5:
                    x, y, w, h = nums
                    bbox = [
                        (x - w / 2) * width,
                        (y - h / 2) * height,
                        w * width,
                        h * height,
                    ]
                    segmentation: list[list[float]] = []
                else:
                    scaled = [
                        nums[i] * (width if i % 2 == 0 else height)
                        for i in range(len(nums))
                    ]
                    xs = scaled[0::2]
                    ys = scaled[1::2]
                    bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                    segmentation = [scaled]
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls + 1,
                    "bbox": bbox,
                    "area": max(0.0, bbox[2] * bbox[3]),
                    "iscrowd": 0,
                    "segmentation": segmentation,
                })
                ann_id += 1
        ann_dir = output / "annotations"
        ann_dir.mkdir(exist_ok=True)
        (ann_dir / f"instances_{split_name}.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


def convert_coco_to_yolo(source_dir: str | Path, output_dir: str | Path, *, summary: DatasetSummary | None = None) -> Path:
    source = Path(source_dir).resolve()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(source), "format": "yolo", "warnings": []}
    class_names: list[str] = []
    yaml_splits: dict[str, str] = {}
    for split_name, ann_path in find_coco_annotation_files(source).items():
        data = _read_json(ann_path)
        categories = sorted(data.get("categories", []), key=lambda c: c.get("id", 0))
        if not class_names:
            class_names = [str(c.get("name", c.get("id"))) for c in categories]
        cat_to_index = {cat.get("id"): idx for idx, cat in enumerate(categories)}
        images = {image.get("id"): image for image in data.get("images", [])}
        anns_by_image: dict[Any, list[dict[str, Any]]] = {}
        for ann in data.get("annotations", []):
            anns_by_image.setdefault(ann.get("image_id"), []).append(ann)
        img_dir = output / split_name / "images"
        label_dir = output / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        yaml_splits["val" if split_name == "val" else split_name] = f"{split_name}/images"
        for image_id, image in images.items():
            file_name = str(image.get("file_name", ""))
            source_image = source / file_name
            if not source_image.exists():
                source_image = ann_path.parent.parent / file_name
            if source_image.exists():
                shutil.copy2(source_image, img_dir / Path(file_name).name)
            else:
                manifest["warnings"].append(f"Missing image referenced by COCO JSON: {file_name}")
            rows: list[str] = []
            width = float(image.get("width") or 1)
            height = float(image.get("height") or 1)
            for ann in anns_by_image.get(image_id, []):
                if ann.get("category_id") not in cat_to_index:
                    continue
                cls = cat_to_index[ann.get("category_id")]
                segmentation = ann.get("segmentation")
                if isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], list) and len(segmentation[0]) >= 6:
                    coords = segmentation[0]
                    norm = [str(max(0.0, min(1.0, coords[i] / (width if i % 2 == 0 else height)))) for i in range(len(coords))]
                    rows.append(" ".join([str(cls), *norm]))
                else:
                    bbox = ann.get("bbox") or []
                    if len(bbox) != 4:
                        manifest["warnings"].append(f"Skipped invalid bbox for image id {image_id}")
                        continue
                    x, y, w, h = [float(v) for v in bbox]
                    rows.append(f"{cls} {(x + w / 2) / width} {(y + h / 2) / height} {w / width} {h / height}")
            (label_dir / f"{Path(file_name).stem}.txt").write_text("\n".join(rows), encoding="utf-8")
    data_yaml = {"path": str(output.resolve()), "train": yaml_splits.get("train", "train/images"), "val": yaml_splits.get("val", "val/images"), "test": yaml_splits.get("test", "test/images"), "nc": len(class_names), "names": class_names}
    (output / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    manifest["classes"] = class_names
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output
