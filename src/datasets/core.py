from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.utils.project import project_root
from src.datasets.cache import is_dataset_runtime_cache

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation"),
    "test": ("test", "testing"),
}


def read_yaml_file(file_path: str | Path) -> dict[str, Any] | None:
    with open(file_path, "r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    return payload if isinstance(payload, dict) else None


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
    runtime_cache_file_count: int = 0
    runtime_cache_size_bytes: int = 0
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


def _resolve_dataset_paths(base: Path, value: Any, *, path_root: Path | None = None) -> list[Path]:
    """Resolve a data.yaml split value (str or list of str) into image directories.

    Per entry the first existing interpretation wins: absolute path, relative to
    the yaml ``path:`` root, relative to the yaml's own directory, or — as a
    Roboflow-export quirk — relative to the yaml's directory with all leading
    ``../`` segments stripped (their yamls say ``../train/images`` but mean
    ``./train/images``). When nothing exists the yaml-relative interpretation is
    still returned so callers can warn about a concrete missing path.
    """
    if value is None:
        return []
    entries = list(value) if isinstance(value, (list, tuple)) else [value]
    resolved: list[Path] = []
    for entry in entries:
        if not isinstance(entry, (str, Path)) or not str(entry).strip():
            continue
        normalized = str(entry).replace("\\", "/")
        path = Path(normalized)
        if path.is_absolute():
            candidates = [path]
        else:
            candidates = []
            if path_root is not None:
                candidates.append((path_root / path).resolve())
            candidates.append((base / path).resolve())
            stripped = normalized
            while stripped.startswith("../"):
                stripped = stripped[3:]
            if stripped != normalized and stripped:
                candidates.append((base / stripped).resolve())
        chosen = next((c for c in candidates if c.exists()), None)
        if chosen is None:
            chosen = path if path.is_absolute() else (base / path).resolve()
        if chosen not in resolved:
            resolved.append(chosen)
    return resolved


def _resolve_dataset_path(base: Path, value: str | None) -> Path | None:
    paths = _resolve_dataset_paths(base, value)
    return paths[0] if paths else None


def _resolve_label_dir(image_dir: Path) -> Path | None:
    """Find the labels directory that mirrors an images directory.

    Swaps the rightmost 'images' path component with 'labels', then falls back
    to a sibling 'labels/' directory, a 'labels/' directory inside image_dir
    (bare split folders), and finally image_dir itself when label .txt files
    sit next to the images.
    """
    parts = list(image_dir.parts)
    for i in reversed(range(len(parts))):
        if parts[i] == "images":
            tail = parts[i + 1:]
            candidate = Path(*parts[:i], "labels", *tail) if tail else Path(*parts[:i], "labels")
            if candidate.exists():
                return candidate
            break
    sibling = image_dir.parent / "labels"
    if sibling.exists():
        return sibling
    nested = image_dir / "labels"
    if nested.exists():
        return nested
    try:
        has_side_by_side = any(
            p.is_file() and p.suffix.lower() == ".txt" for p in image_dir.iterdir()
        )
    except OSError:
        has_side_by_side = False
    return image_dir if has_side_by_side else None


def _label_path_for(image_path: Path, image_dir: Path, label_dir: Path) -> Path:
    """Return the label file path for an image, mirroring any subdirectory structure."""
    try:
        rel = image_path.relative_to(image_dir)
        mirrored = label_dir / rel.with_suffix(".txt")
        if mirrored.exists():
            return mirrored
    except ValueError:
        pass
    return label_dir / f"{image_path.stem}.txt"


def _normalize_kpt_shape(value: Any) -> tuple[int, int] | None:
    """Coerce a data.yaml ``kpt_shape`` value into ``(num_keypoints, ndim)``."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        k = int(value[0])
        ndim = int(value[1])
    except (TypeError, ValueError):
        return None
    if k <= 0 or ndim not in (2, 3):
        return None
    return k, ndim


def _has_direct_images(path: Path) -> bool:
    try:
        return any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in path.iterdir())
    except OSError:
        return False


def discover_split_dirs(
    root: Path,
    data: dict[str, Any] | None = None,
    *,
    warnings: list[str] | None = None,
) -> dict[str, list[Path]]:
    """Discover split image directories as the union of data.yaml keys and disk layout.

    ``data`` is the parsed data.yaml located at ``root`` (or ``{}``/None when the
    dataset has no yaml). Returns canonical split name -> resolved image dirs.

    Sources:
      A. every present alias key in SPLIT_ALIASES (both ``val:`` and ``valid:``
         count), resolved via _resolve_dataset_paths honoring the ``path:`` root;
      B. on-disk conventions matched case-insensitively against the aliases:
         ``<root>/<alias>/images``, ``<root>/images/<alias>``, and ``<root>/<alias>``
         containing image files directly.

    Candidates are deduped by resolved path, and a candidate nested inside (or
    containing) an already accepted directory is rejected so recursive image
    scans never double-count. Warnings are appended for yaml keys pointing at
    missing directories and for disk-only split folders pulled in alongside a
    yaml that did not reference them.
    """
    root = Path(root).resolve()
    data = data or {}
    result: dict[str, list[Path]] = {name: [] for name in SPLIT_ALIASES}
    accepted: list[Path] = []

    def _accept(canonical: str, candidate: Path) -> bool:
        resolved = candidate.resolve()
        for existing in accepted:
            if resolved == existing or resolved.is_relative_to(existing) or existing.is_relative_to(resolved):
                return False
        accepted.append(resolved)
        result[canonical].append(resolved)
        return True

    raw_root = data.get("path")
    path_root: Path | None = None
    if isinstance(raw_root, (str, Path)) and str(raw_root).strip():
        candidate_root = Path(str(raw_root))
        path_root = candidate_root if candidate_root.is_absolute() else (root / candidate_root).resolve()

    declared_any = False
    for canonical, aliases in SPLIT_ALIASES.items():
        for alias in aliases:
            value = data.get(alias)
            if value is None:
                continue
            declared_any = True
            for path in _resolve_dataset_paths(root, value, path_root=path_root):
                if not path.exists():
                    if warnings is not None:
                        warnings.append(f"data.yaml references missing {alias} path: {value}")
                    continue
                _accept(canonical, path)

    alias_to_canonical = {
        alias: canonical for canonical, aliases in SPLIT_ALIASES.items() for alias in aliases
    }
    try:
        entries = [e for e in root.iterdir() if e.is_dir() and not e.name.startswith(".")]
    except OSError:
        entries = []
    disk_candidates: list[Path] = []
    canonical_by_candidate: dict[Path, str] = {}
    for entry in entries:
        canonical = alias_to_canonical.get(entry.name.lower())
        if canonical is None:
            continue
        images_sub = entry / "images"
        candidate = images_sub if images_sub.is_dir() else entry if _has_direct_images(entry) else None
        if candidate is not None:
            disk_candidates.append(candidate)
            canonical_by_candidate[candidate] = canonical
    images_root = next((e for e in entries if e.name.lower() == "images"), None)
    if images_root is not None:
        try:
            subdirs = [s for s in images_root.iterdir() if s.is_dir()]
        except OSError:
            subdirs = []
        for sub in subdirs:
            canonical = alias_to_canonical.get(sub.name.lower())
            if canonical is not None:
                disk_candidates.append(sub)
                canonical_by_candidate[sub] = canonical
    for candidate in disk_candidates:
        if _accept(canonical_by_candidate[candidate], candidate) and declared_any and warnings is not None:
            warnings.append(
                f"Found split folder on disk not referenced in data.yaml: {candidate}; its images will be included."
            )
    return result


def _has_yolo_disk_layout(root: Path) -> bool:
    """True when the directory looks like a YOLO dataset without a data.yaml."""
    root = Path(root)
    if (root / "images").is_dir() and (root / "labels").is_dir():
        return True
    for dirs in discover_split_dirs(root, {}).values():
        for image_dir in dirs:
            label_dir = _resolve_label_dir(image_dir)
            if label_dir is not None and any(label_dir.rglob("*.txt")):
                return True
    return False


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
    has_yolo = (
        (root / "data.yaml").exists()
        or (root / "dataset.yaml").exists()
        or _has_yolo_disk_layout(root)
    )
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


def _stat_aggregate(path: Path, *, max_depth: int = 2) -> tuple[int, int, int]:
    """Bounded stat-only sweep of a directory: (file_count, max_mtime_ns, total_size).

    Cheap enough for cache signatures — no file contents are read — yet adding,
    removing, or editing any file within ``max_depth`` levels changes the result.
    """
    file_count = 0
    latest_mtime = 0
    total_size = 0
    base_depth = len(Path(path).parts)
    try:
        for dirpath, dirnames, filenames, dirfd in os.fwalk(str(path)):
            if len(Path(dirpath).parts) - base_depth >= max_depth:
                dirnames[:] = []
            sibling_names = {name.lower() for name in filenames}
            for fname in filenames:
                try:
                    file_path = Path(dirpath) / fname
                    if is_dataset_runtime_cache(file_path, sibling_names):
                        continue
                    stat = os.stat(fname, dir_fd=dirfd)
                except OSError:
                    continue
                file_count += 1
                total_size += stat.st_size
                if stat.st_mtime_ns > latest_mtime:
                    latest_mtime = stat.st_mtime_ns
    except OSError:
        pass
    return file_count, latest_mtime, total_size


def _dataset_signature(root: Path) -> str:
    """Generate a quick signature for the dataset directory without scanning images/labels."""
    hasher = hashlib.sha256()
    hasher.update(str(root.resolve()).encode("utf-8"))

    queue = [(root, 0)]
    while queue:
        curr, depth = queue.pop(0)
        if depth > 3:
            continue
        try:
            with os.scandir(curr) as it:
                for entry in it:
                    if entry.name == ".yolomatic_cache" or entry.name.startswith("."):
                        continue
                    try:
                        stat = entry.stat()
                        if entry.is_dir():
                            hasher.update(f"dir:{entry.path}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8"))
                            if entry.name.lower() in {"images", "labels"}:
                                # Directory mtimes miss edits and changes in
                                # subdirs (images/train/...), so fold in a
                                # bounded stat aggregate of the contents.
                                count, mtime, size = _stat_aggregate(Path(entry.path))
                                hasher.update(f"agg:{entry.path}:{count}:{mtime}:{size}".encode("utf-8"))
                            else:
                                queue.append((Path(entry.path), depth + 1))
                        elif entry.is_file():
                            if entry.name.lower().endswith((".yaml", ".yml", ".json")):
                                hasher.update(f"file:{entry.path}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8"))
                    except OSError:
                        pass
        except OSError:
            pass

    return hasher.hexdigest()[:16]


def dataset_summary_from_dict(data: dict[str, Any]) -> DatasetSummary:
    splits_data = data.pop("splits", {})
    splits = {}
    for key, val in splits_data.items():
        splits[key] = SplitSummary(**val)
    return DatasetSummary(splits=splits, **data)


def summarize_dataset(dataset_path: str | Path, *, sample_limit: int = 5000) -> DatasetSummary:
    root = Path(dataset_path).resolve()
    if not root.exists():
        summary = DatasetSummary(path=str(root), name=root.name, format=detect_dataset_format(root))
        summary.errors.append("Dataset path does not exist.")
        return summary

    # Cache check
    sig = _dataset_signature(root)
    cache_dir = project_root() / "datasets" / ".yolomatic_cache" / "summaries"
    cache_path = cache_dir / f"{root.name}_{sig}_limit{sample_limit}.json"

    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            cached_summary = dataset_summary_from_dict(cached_data)
            cached_summary.path = str(root)
            return cached_summary
        except Exception:
            pass

    # Original logic
    summary = DatasetSummary(path=str(root), name=root.name, format=detect_dataset_format(root))
    try:
        total = 0
        for _dirpath, _dirnames, filenames, dirfd in os.fwalk(str(root)):
            sibling_names = {name.lower() for name in filenames}
            for fname in filenames:
                try:
                    stat = os.stat(fname, dir_fd=dirfd)
                    file_path = Path(_dirpath) / fname
                    if is_dataset_runtime_cache(file_path, sibling_names):
                        summary.runtime_cache_file_count += 1
                        summary.runtime_cache_size_bytes += stat.st_size
                        continue
                    total += stat.st_size
                except OSError:
                    pass
        summary.total_size_bytes = total
    except OSError:
        summary.warnings.append("Some files could not be inspected.")

    if summary.format in {"yolo", "mixed"}:
        _summarize_yolo(root, summary, sample_limit)
    if summary.format in {"coco", "mixed"}:
        _summarize_coco(root, summary, sample_limit)
    if summary.format == "unknown":
        summary.image_count = len(_iter_images(root))
        summary.errors.append("No data.yaml or COCO annotations were found.")

    if summary.annotation_count == 0:
        summary.task = "empty" if summary.image_count else "unknown"
    summary.compatibility = {
        "yolo": "native" if summary.format in {"yolo", "mixed"} else "conversion required",
        "rfdetr": "native" if summary.format in {"yolo", "mixed"} else "conversion required",
        "detectron2": "native" if summary.format in {"coco", "mixed"} else "conversion required",
    }

    # Save to cache
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)
    except Exception:
        pass

    return summary


def _summarize_yolo(root: Path, summary: DatasetSummary, sample_limit: int) -> None:
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        yaml_path = root / "dataset.yaml"
    data = _read_yaml(yaml_path) if yaml_path.exists() else {}
    summary.classes = summary.classes or _normalize_names(data.get("names"))
    # Authoritative pose marker: Ultralytics tags keypoint datasets with
    # ``kpt_shape``. Pose label rows (class + bbox + K*ndim) otherwise look like
    # neither a 5-col bbox nor an odd-length polygon, so the column heuristic
    # alone would leave the task unclassified.
    kpt_shape = _normalize_kpt_shape(data.get("kpt_shape"))
    is_pose = bool(data.get("kpt_shape"))
    pose_values = 4 + kpt_shape[0] * kpt_shape[1] if kpt_shape else 0
    detection_hits = segmentation_hits = 0

    def _summarize_dir(split: SplitSummary, images_path: Path) -> None:
        nonlocal detection_hits, segmentation_hits
        labels_path = _resolve_label_dir(images_path) or images_path.parent / "labels"
        if split.images_path is None:
            split.images_path = str(images_path)
            split.labels_path = str(labels_path)
        all_images = _iter_images(images_path)
        split.image_count += len(all_images)
        if len(all_images) > sample_limit:
            summary.warnings.append(
                f"{split.name}: label statistics sampled from the first {sample_limit} of {len(all_images)} images."
            )
        # Pair-driven counting: resolve each image's own label file so nested
        # layouts and >sample_limit splits keep labeled/unlabeled counts honest.
        for idx, image_path in enumerate(all_images):
            label_path = _label_path_for(image_path, images_path, labels_path)
            if not label_path.exists():
                split.unlabeled_image_count += 1
                continue
            
            if idx < sample_limit:
                try:
                    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                except OSError:
                    split.missing_file_count += 1
                    split.unlabeled_image_count += 1
                    continue
                if not lines:
                    split.empty_label_count += 1
                    split.unlabeled_image_count += 1
                    continue
                split.labeled_image_count += 1
                for line in lines:
                    parts = line.split()
                    if pose_values and len(parts) == 1 + pose_values:
                        pass  # pose row: neither detection nor segmentation evidence
                    elif len(parts) == 5:
                        detection_hits += 1
                    elif len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                        segmentation_hits += 1
                    split.annotation_count += 1
            else:
                try:
                    with open(label_path, "rb") as f:
                        content = f.read()
                    if not content.strip():
                        split.empty_label_count += 1
                        split.unlabeled_image_count += 1
                        continue
                    split.labeled_image_count += 1
                    lines_count = sum(1 for line in content.split(b"\n") if line.strip())
                    split.annotation_count += lines_count
                except OSError:
                    split.missing_file_count += 1
                    split.unlabeled_image_count += 1
                    continue

    # Union of data.yaml split keys and conventional split folders found on disk.
    split_dirs = discover_split_dirs(root, data, warnings=summary.warnings)
    for canonical, image_dirs in split_dirs.items():
        split = SplitSummary(canonical)
        for images_path in image_dirs:
            _summarize_dir(split, images_path)
        if len(image_dirs) > 1:
            summary.warnings.append(
                f"{canonical}: counts aggregated across {len(image_dirs)} image directories."
            )
        split.status = "valid" if split.image_count and (split.annotation_count or split.empty_label_count) else "warning"
        summary.splits[canonical] = split

    # Flat-structure fallback for datasets with bare images/ + labels/ at root
    # (e.g. NDJSON-converted datasets that have no train/val/test split keys in data.yaml)
    if not any(s.image_count > 0 for s in summary.splits.values()):
        flat_images = root / "images"
        if flat_images.exists():
            flat_split = SplitSummary("train")
            _summarize_dir(flat_split, flat_images)
            flat_split.status = "valid" if flat_split.image_count else "warning"
            summary.splits["train"] = flat_split

    _rollup(summary)
    # Task priority: explicit task key in data.yaml (prepare writes one), then
    # kpt_shape, then the label line-shape heuristic.
    declared_task = str(data.get("task", "")).lower()
    task_map = {
        "pose": "pose",
        "segment": "segmentation",
        "segmentation": "segmentation",
        "detect": "detection",
        "detection": "detection",
    }
    if declared_task in task_map:
        summary.task = task_map[declared_task]
    elif is_pose:
        summary.task = "pose"
    elif segmentation_hits and detection_hits:
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
        
        all_images = data.get("images", [])
        all_annotations = data.get("annotations", [])
        total_images = len(all_images)
        total_annotations = len(all_annotations)
        
        image_ids = {image.get("id") for image in all_images}
        labeled = {ann.get("image_id") for ann in all_annotations}
        
        split = summary.splits.get(split_name, SplitSummary(split_name))
        split.annotations_path = str(ann_path)
        split.image_count = total_images
        split.annotation_count = total_annotations
        split.labeled_image_count = len(image_ids & labeled)
        split.unlabeled_image_count = max(0, split.image_count - split.labeled_image_count)
        split.status = "valid" if split.image_count else "warning"
        summary.splits[split_name] = split
        
        has_pose = any(isinstance(category.get("keypoints"), list) and category["keypoints"] for category in categories)
        has_pose = has_pose or any(isinstance(ann.get("keypoints"), list) and ann["keypoints"] for ann in all_annotations)
        if has_pose:
            summary.task = "pose"
        elif any(ann.get("segmentation") for ann in all_annotations):
            summary.task = "segmentation" if summary.task in {"unknown", "empty"} else "mixed"
        elif all_annotations and summary.task == "unknown":
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
    return project_root() / "datasets" / ".yolomatic_cache" / family / source.name / digest


def prepared_format_for_family(family: str | None, task: str | None = None) -> str:
    """Return the dataset format a model family expects for a given task.

    - Detectron2 always trains from COCO instance annotations.
    - RF-DETR detection/segmentation train from YOLO (the rfdetr library also
      auto-detects COCO), but RF-DETR **pose** (RFDETRKeypointPreview) only
      accepts COCO keypoint JSON.
    - YOLO/Ultralytics always trains from YOLO format.
    """
    fam = (family or "").strip().lower()
    task_norm = (task or "").strip().lower()
    is_pose = task_norm in {"pose", "keypoint", "keypoints"}
    if fam.startswith("detectron2"):
        return "coco"
    if fam.startswith("rfdetr") or fam.startswith("rf-detr"):
        return "coco" if is_pose else "yolo"
    return "yolo"


def prepare_dataset_for_family(dataset_path: str | Path, family: str, *, task: str | None = None) -> dict[str, Any]:
    source = Path(dataset_path).resolve()
    summary = summarize_dataset(source)
    prepared_format = prepared_format_for_family(family, task)
    # Cache conversions by the target format so families that need the same
    # output (e.g. Detectron2 and RF-DETR pose both need COCO) reuse one cache.
    family_key = prepared_format
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
    yaml_path = source / "data.yaml"
    if not yaml_path.exists():
        yaml_path = source / "dataset.yaml"
    data = _read_yaml(yaml_path) if yaml_path.exists() else {}
    classes = _normalize_names(data.get("names"))
    kpt_shape = data.get("kpt_shape")
    num_kpts = ndim = 0
    if isinstance(kpt_shape, (list, tuple)) and len(kpt_shape) == 2:
        try:
            num_kpts, ndim = int(kpt_shape[0]), int(kpt_shape[1])
        except (TypeError, ValueError):
            num_kpts = ndim = 0
    kpt_names = [f"kpt_{i}" for i in range(num_kpts)]
    manifest = {"source": str(source), "format": "coco", "warnings": [], "classes": classes}
    # Same union discovery as prepare/summarize so the exported COCO dataset
    # covers split folders that data.yaml does not reference (or has no yaml).
    split_dirs = discover_split_dirs(source, data, warnings=manifest["warnings"])
    for split_name, image_dirs in split_dirs.items():
        if not image_dirs:
            continue
        target_images = output / split_name / "images"
        target_images.mkdir(parents=True, exist_ok=True)
        used_names: set[str] = set()
        images: list[tuple[Path, Path, str]] = []
        for images_path in image_dirs:
            label_dir = _resolve_label_dir(images_path) or images_path.parent / "labels"
            for image_path in _iter_images(images_path):
                file_name = image_path.name
                if file_name in used_names:
                    counter = 1
                    while f"{image_path.stem}_{counter}{image_path.suffix}" in used_names:
                        counter += 1
                    file_name = f"{image_path.stem}_{counter}{image_path.suffix}"
                used_names.add(file_name)
                images.append((image_path, _label_path_for(image_path, images_path, label_dir), file_name))
        category_list: list[dict[str, Any]] = []
        for i, name in enumerate(classes):
            cat: dict[str, Any] = {"id": i + 1, "name": name}
            if num_kpts:
                cat["keypoints"] = kpt_names
                cat["skeleton"] = []
            category_list.append(cat)
        coco = {
            "images": [],
            "annotations": [],
            "categories": category_list,
        }
        ann_id = 1
        for image_id, (image_path, label_path, file_name) in enumerate(images, start=1):
            width, height = _image_size(image_path)
            target = target_images / file_name
            if not target.exists():
                shutil.copy2(image_path, target)
            coco["images"].append(
                {
                    "id": image_id,
                    "file_name": str(Path(split_name) / "images" / file_name),
                    "width": int(width),
                    "height": int(height),
                }
            )
            if not label_path.exists():
                continue
            pose_len = 4 + num_kpts * ndim if num_kpts else -1
            for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                parts = line.strip().split()
                is_pose_row = num_kpts and len(parts) - 1 == pose_len
                if not is_pose_row and len(parts) not in {5} and not (len(parts) >= 7 and (len(parts) - 1) % 2 == 0):
                    manifest["warnings"].append(f"Skipped invalid YOLO row {label_path}:{line_no}")
                    continue
                cls = int(float(parts[0]))
                nums = [float(v) for v in parts[1:]]
                keypoints: list[float] = []
                num_keypoints = 0
                if is_pose_row:
                    x, y, w, h = nums[:4]
                    bbox = [
                        (x - w / 2) * width,
                        (y - h / 2) * height,
                        w * width,
                        h * height,
                    ]
                    segmentation: list[list[float]] = []
                    for i in range(num_kpts):
                        base = 4 + i * ndim
                        kx = nums[base] * width
                        ky = nums[base + 1] * height
                        kv = int(nums[base + 2]) if ndim == 3 else 2
                        keypoints.extend([kx, ky, kv])
                        if kv > 0:
                            num_keypoints += 1
                elif len(parts) == 5:
                    x, y, w, h = nums
                    bbox = [
                        (x - w / 2) * width,
                        (y - h / 2) * height,
                        w * width,
                        h * height,
                    ]
                    segmentation = []
                else:
                    scaled = [
                        nums[i] * (width if i % 2 == 0 else height)
                        for i in range(len(nums))
                    ]
                    xs = scaled[0::2]
                    ys = scaled[1::2]
                    bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                    segmentation = [scaled]
                coco_ann: dict[str, Any] = {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls + 1,
                    "bbox": bbox,
                    "area": max(0.0, bbox[2] * bbox[3]),
                    "iscrowd": 0,
                    "segmentation": segmentation,
                }
                if is_pose_row:
                    coco_ann["keypoints"] = keypoints
                    coco_ann["num_keypoints"] = num_keypoints
                coco["annotations"].append(coco_ann)
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
    num_kpts = 0
    yaml_splits: dict[str, str] = {}
    for split_name, ann_path in find_coco_annotation_files(source).items():
        data = _read_json(ann_path)
        categories = sorted(data.get("categories", []), key=lambda c: c.get("id", 0))
        if not class_names:
            class_names = [str(c.get("name", c.get("id"))) for c in categories]
        if not num_kpts:
            for cat in categories:
                names = cat.get("keypoints")
                if isinstance(names, list) and names:
                    num_kpts = len(names)
                    break
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
                raw_kpts = ann.get("keypoints")
                if num_kpts and isinstance(raw_kpts, list) and len(raw_kpts) == num_kpts * 3:
                    bbox = ann.get("bbox") or []
                    if len(bbox) != 4:
                        manifest["warnings"].append(f"Skipped pose ann without bbox for image id {image_id}")
                        continue
                    x, y, w, h = [float(v) for v in bbox]
                    row = [str(cls), f"{(x + w / 2) / width}", f"{(y + h / 2) / height}", f"{w / width}", f"{h / height}"]
                    for i in range(0, len(raw_kpts), 3):
                        row.append(f"{max(0.0, min(1.0, float(raw_kpts[i]) / width))}")
                        row.append(f"{max(0.0, min(1.0, float(raw_kpts[i + 1]) / height))}")
                        row.append(f"{int(raw_kpts[i + 2])}")
                    rows.append(" ".join(row))
                    continue
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
    if num_kpts:
        data_yaml["task"] = "pose"
        data_yaml["kpt_shape"] = [num_kpts, 3]
    (output / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    manifest["classes"] = class_names
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output
