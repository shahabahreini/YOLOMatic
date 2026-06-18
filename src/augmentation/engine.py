"""
Augmentation engine: annotation I/O, Albumentations pipeline, pool+redistribute runner.

Supports:
  - YOLO bbox format  (class_id cx cy w h)
  - YOLO seg format   (class_id x1 y1 x2 y2 … xn yn)
  - Auto-detection of source format
  - Pool-all-images → augment → redistribute to train/val/test splits
  - Output as YOLO Detection, YOLO Segmentation, or COCO
"""
from __future__ import annotations

import logging
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation"),
    "test": ("test", "testing"),
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SplitConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.20
    test_ratio: float = 0.10
    include_originals: bool = True  # overridden by profile.include_originals at call site


@dataclass
class AugmentationStats:
    source_dataset: str
    output_path: str
    profile_name: str
    annotation_format: str
    output_format: str
    total_source_images: int
    total_output_images: int
    split_counts: dict[str, int] = field(default_factory=dict)
    images_skipped: int = 0
    annotations_discarded: int = 0
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Annotation format detection
# ---------------------------------------------------------------------------

def detect_annotation_format(dataset_path: Path) -> str:
    """
    Determine the annotation format of a dataset.

    Detection priority:
      1. COCO JSON annotations directory → 'coco'
      2. data.yaml kpt_shape / task field → 'yolo_pose', 'yolo_seg', or 'yolo_bbox'
      3. Non-empty label file sampling  → 'yolo_seg' or 'yolo_bbox'

    Returns: 'yolo_bbox', 'yolo_seg', 'yolo_pose', or 'coco'
    """
    # 1. COCO JSON
    ann_dir = dataset_path / "annotations"
    if ann_dir.exists() and any(ann_dir.glob("*.json")):
        return "coco"

    # 2. data.yaml kpt_shape (authoritative pose marker) / task field
    data_yaml = dataset_path / "data.yaml"
    if data_yaml.exists():
        try:
            with open(data_yaml, encoding="utf-8") as f:
                meta = yaml.safe_load(f) or {}
            if meta.get("kpt_shape"):
                return "yolo_pose"
            task = str(meta.get("task", "")).lower()
            if "pose" in task:
                return "yolo_pose"
            if "segment" in task:
                return "yolo_seg"
            if "detect" in task:
                return "yolo_bbox"
        except Exception:
            pass

    # 3. Sample non-empty label files (skip background/empty tiles)
    label_files: list[Path] = []
    for split in ("train", "valid", "val", "test"):
        for labels_dir in (
            dataset_path / split / "labels",
            dataset_path / "labels" / split,
        ):
            if labels_dir.exists():
                label_files.extend(sorted(labels_dir.glob("*.txt"))[:25])
                break
    # Flat labels/ fallback
    flat_labels = dataset_path / "labels"
    if flat_labels.exists() and not label_files:
        label_files.extend(sorted(flat_labels.glob("*.txt"))[:50])

    seg_count = 0
    bbox_count = 0
    sampled = 0
    for lf in label_files:
        if sampled >= 60:
            break
        try:
            text = lf.read_text(encoding="utf-8").strip()
            if not text:
                continue  # skip empty background tiles
            sampled += 1
            for line in text.splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    bbox_count += 1
                elif len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                    seg_count += 1
        except OSError:
            pass

    if seg_count > 0 and seg_count >= bbox_count:
        return "yolo_seg"
    if bbox_count > 0:
        return "yolo_bbox"
    return "yolo_bbox"  # safe default


# ---------------------------------------------------------------------------
# Annotation I/O
# ---------------------------------------------------------------------------

def read_yolo_bbox(label_path: Path | None) -> tuple[list[list[float]], list[int]]:
    """
    Returns (bboxes, class_ids) where bboxes = [[cx, cy, w, h], ...] normalized.
    """
    if label_path is None or not label_path.exists():
        return [], []
    bboxes: list[list[float]] = []
    class_ids: list[int] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            bboxes.append([cx, cy, w, h])
            class_ids.append(cls)
        except ValueError:
            pass
    return bboxes, class_ids


def read_yolo_seg(label_path: Path | None) -> tuple[list[list[float]], list[int]]:
    """
    Returns (polygons, class_ids) where polygons = [[x1,y1,x2,y2,...], ...] normalized.
    """
    if label_path is None or not label_path.exists():
        return [], []
    polygons: list[list[float]] = []
    class_ids: list[int] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            continue
        try:
            cls = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
            polygons.append(coords)
            class_ids.append(cls)
        except ValueError:
            pass
    return polygons, class_ids


def write_yolo_bbox(label_path: Path, bboxes: list[list[float]], class_ids: list[int]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    seen: set[str] = set()
    for cls, (cx, cy, w, h) in zip(class_ids, bboxes):
        line = f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    label_path.write_text("\n".join(lines), encoding="utf-8")


def write_yolo_seg(label_path: Path, polygons: list[list[float]], class_ids: list[int]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    seen: set[str] = set()
    for cls, poly in zip(class_ids, polygons):
        coords = " ".join(f"{v:.6f}" for v in poly)
        line = f"{cls} {coords}"
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    label_path.write_text("\n".join(lines), encoding="utf-8")


def read_kpt_shape(dataset_path: Path) -> tuple[int, int] | None:
    """Return ``(num_keypoints, ndim)`` from a dataset's data.yaml ``kpt_shape``, else None."""
    for name in ("data.yaml", "dataset.yaml"):
        yaml_path = dataset_path / name
        if not yaml_path.exists():
            continue
        try:
            with open(yaml_path, encoding="utf-8") as handle:
                meta = yaml.safe_load(handle) or {}
        except Exception:
            return None
        shape = meta.get("kpt_shape")
        if isinstance(shape, (list, tuple)) and len(shape) == 2:
            try:
                k, ndim = int(shape[0]), int(shape[1])
            except (TypeError, ValueError):
                return None
            if k > 0 and ndim in (2, 3):
                return k, ndim
        return None
    return None


def read_yolo_pose(
    label_path: Path | None, kpt_shape: tuple[int, int]
) -> tuple[list[list[float]], list[list[float]], list[int]]:
    """Read a YOLO pose label file.

    Returns ``(bboxes, keypoints, class_ids)`` where ``bboxes`` are ``[cx, cy, w, h]``
    (normalized) and ``keypoints`` is a per-object flattened list of length ``K*ndim``.
    """
    k, ndim = kpt_shape
    expected = 4 + k * ndim
    bboxes: list[list[float]] = []
    keypoints: list[list[float]] = []
    class_ids: list[int] = []
    if label_path is None or not label_path.exists():
        return bboxes, keypoints, class_ids
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 1 + expected:
            continue
        try:
            cls = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        bboxes.append(values[:4])
        keypoints.append(values[4:])
        class_ids.append(cls)
    return bboxes, keypoints, class_ids


def write_yolo_pose(
    label_path: Path,
    bboxes: list[list[float]],
    keypoints: list[list[float]],
    class_ids: list[int],
) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    seen: set[str] = set()
    for cls, bbox, kpts in zip(class_ids, bboxes, keypoints):
        coords = " ".join(f"{v:.6f}" for v in [*bbox, *kpts])
        line = f"{cls} {coords}"
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    label_path.write_text("\n".join(lines), encoding="utf-8")


def polygon_to_bbox(polygon: list[float]) -> list[float]:
    """Normalized polygon [x1,y1,...] → YOLO bbox [cx, cy, w, h]."""
    pts = np.array(polygon, dtype=np.float32).reshape(-1, 2)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    return [
        float((x_min + x_max) / 2),
        float((y_min + y_max) / 2),
        float(x_max - x_min),
        float(y_max - y_min),
    ]


def bbox_to_polygon(bbox: list[float]) -> list[float]:
    """YOLO bbox [cx, cy, w, h] → normalized rectangle polygon [x1,y1, x2,y1, x2,y2, x1,y2]."""
    cx, cy, w, h = bbox
    x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
    return [x1, y1, x2, y1, x2, y2, x1, y2]


# ---------------------------------------------------------------------------
# Polygon ↔ mask conversion
# ---------------------------------------------------------------------------

def polygon_to_mask(polygon: list[float], W: int, H: int) -> np.ndarray:
    """Normalized polygon [x1,y1,...] → binary uint8 mask of shape (H, W)."""
    pts = np.array(polygon, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] = np.clip(pts[:, 0] * W, 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1] * H, 0, H - 1)
    pts = pts.astype(np.int32)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def mask_to_polygon(
    mask: np.ndarray,
    min_area: float = 4.0,
    epsilon_factor: float = 0.003,
) -> list[float] | None:
    """
    Binary mask → normalized polygon [x1/W, y1/H, ...].
    Returns None if mask is empty or below min_area threshold.
    """
    H, W = mask.shape
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return None
    epsilon = epsilon_factor * cv2.arcLength(cnt, closed=True)
    approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
    if len(approx) < 3:
        return None
    pts = approx.reshape(-1, 2).astype(np.float32)
    pts[:, 0] /= W
    pts[:, 1] /= H
    pts = np.clip(pts, 0.0, 1.0)
    return pts.flatten().tolist()


# ---------------------------------------------------------------------------
# Albumentations pipeline builders
# ---------------------------------------------------------------------------

def _import_albumentations():
    try:
        os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
        import albumentations as A
        return A
    except ImportError as exc:
        raise ImportError(
            "albumentations is required for dataset augmentation.\n"
            "Install it with:  uv add albumentations>=1.4"
        ) from exc


def _instantiate_transform(A, t_cfg: dict[str, Any]):
    """Instantiate a single albumentations transform from a profile entry."""
    from src.augmentation.transforms import build_albu_kwargs
    if not t_cfg.get("enabled", False):
        return None
    name = t_cfg.get("name", "")
    cls = getattr(A, name, None)
    if cls is None:
        logger.warning("Unknown albumentations transform: %s — skipping.", name)
        return None
    kwargs = build_albu_kwargs(t_cfg)
    try:
        return cls(**kwargs)
    except Exception as exc:
        logger.warning("Failed to instantiate %s(%s): %s — skipping.", name, kwargs, exc)
        return None


def build_seg_pipeline(profile):
    """A.Compose pipeline for YOLO seg (mask-based annotation handling)."""
    A = _import_albumentations()
    transforms = [
        t for cfg in profile.transforms
        if (t := _instantiate_transform(A, cfg)) is not None
    ]
    if not transforms:
        return A.Compose([A.NoOp()])
    return A.Compose(transforms, is_check_shapes=False)


def build_bbox_pipeline(profile):
    """A.Compose pipeline for YOLO bbox."""
    A = _import_albumentations()
    transforms = [
        t for cfg in profile.transforms
        if (t := _instantiate_transform(A, cfg)) is not None
    ]
    if not transforms:
        import albumentations as _A
        return _A.Compose([_A.NoOp()],
                          bbox_params=_A.BboxParams(format="yolo",
                                                    label_fields=["class_labels"],
                                                    min_visibility=0.1,
                                                    min_area=1))
    import albumentations as _A
    return _A.Compose(transforms,
                      bbox_params=_A.BboxParams(format="yolo",
                                                label_fields=["class_labels"],
                                                min_visibility=0.1,
                                                min_area=1))


# Geometric flips/transposes reorder left/right keypoints, but albumentations does
# not permute keypoints by ``flip_idx``. Applying them would corrupt symmetric pose
# keypoints, so they are stripped from pose pipelines.
_POSE_UNSAFE_TRANSFORMS = {"HorizontalFlip", "VerticalFlip", "Flip", "Transpose"}


def build_pose_pipeline(profile):
    """A.Compose pipeline for YOLO pose (bbox + keypoint handling).

    Flip/transpose transforms are dropped because keypoint index reindexing
    (``flip_idx``) is not tracked; all other transforms are kept.
    """
    A = _import_albumentations()
    safe_cfgs = []
    stripped = False
    for cfg in profile.transforms:
        if cfg.get("enabled", False) and cfg.get("name", "") in _POSE_UNSAFE_TRANSFORMS:
            stripped = True
            continue
        safe_cfgs.append(cfg)
    if stripped:
        logger.warning(
            "Pose augmentation: skipping flip/transpose transforms — keypoint "
            "left/right reindexing (flip_idx) is not supported."
        )
    transforms = [
        t for cfg in safe_cfgs
        if (t := _instantiate_transform(A, cfg)) is not None
    ]
    import albumentations as _A
    bbox_params = _A.BboxParams(format="yolo", label_fields=["class_labels"],
                                min_visibility=0.1, min_area=1)
    keypoint_params = _A.KeypointParams(format="xy", label_fields=["kpt_obj_idx"],
                                        remove_invisible=False)
    if not transforms:
        transforms = [_A.NoOp()]
    return _A.Compose(transforms, bbox_params=bbox_params,
                      keypoint_params=keypoint_params, is_check_shapes=False)


# ---------------------------------------------------------------------------
# Per-image augmentation workers
# ---------------------------------------------------------------------------

def _augment_seg(
    img: np.ndarray,
    polygons: list[list[float]],
    class_ids: list[int],
    pipeline,
    multiplier: int,
) -> list[tuple[np.ndarray, list[list[float]], list[int]]]:
    """
    Apply pipeline `multiplier` times to a seg-format image.
    Returns list of (aug_img, aug_polygons, aug_class_ids).
    """
    H, W = img.shape[:2]
    results = []
    masks = [polygon_to_mask(poly, W, H) for poly in polygons]

    for _ in range(multiplier):
        try:
            if masks:
                out = pipeline(image=img, masks=masks)
                aug_img = out["image"]
                aug_masks = out["masks"]
            else:
                out = pipeline(image=img)
                aug_img = out["image"]
                aug_masks = []

            new_polygons: list[list[float]] = []
            new_cls: list[int] = []
            for cls, aug_mask in zip(class_ids, aug_masks):
                poly = mask_to_polygon(aug_mask)
                if poly is not None and len(poly) >= 6:
                    new_polygons.append(poly)
                    new_cls.append(cls)
            results.append((aug_img, new_polygons, new_cls))
        except Exception as exc:
            logger.debug("Seg augmentation error: %s", exc)
    return results


def _augment_bbox(
    img: np.ndarray,
    bboxes: list[list[float]],
    class_ids: list[int],
    pipeline,
    multiplier: int,
) -> list[tuple[np.ndarray, list[list[float]], list[int]]]:
    """
    Apply pipeline `multiplier` times to a bbox-format image.
    Returns list of (aug_img, aug_bboxes, aug_class_ids).
    """
    results = []
    for _ in range(multiplier):
        try:
            out = pipeline(image=img, bboxes=bboxes, class_labels=class_ids)
            aug_img = out["image"]
            aug_bboxes = [list(b) for b in out.get("bboxes", [])]
            aug_cls = list(out.get("class_labels", []))
            results.append((aug_img, aug_bboxes, aug_cls))
        except Exception as exc:
            logger.debug("BBox augmentation error: %s", exc)
    return results


def _augment_pose(
    img: np.ndarray,
    bboxes: list[list[float]],
    keypoints: list[list[float]],
    class_ids: list[int],
    pipeline,
    multiplier: int,
    kpt_shape: tuple[int, int],
) -> list[tuple[np.ndarray, list[list[float]], list[list[float]], list[int]]]:
    """Apply pipeline `multiplier` times to a pose-format image.

    Returns list of ``(aug_img, aug_bboxes, aug_keypoints, aug_class_ids)`` where each
    keypoints entry is flattened length ``K*ndim`` and normalized.
    """
    k, ndim = kpt_shape
    H, W = img.shape[:2]
    results: list[tuple[np.ndarray, list[list[float]], list[list[float]], list[int]]] = []

    # Flatten keypoints to absolute (x, y) with parallel visibility + object index.
    flat_xy: list[tuple[float, float]] = []
    flat_vis: list[float] = []
    flat_obj: list[int] = []
    for obj_idx, kpts in enumerate(keypoints):
        for j in range(k):
            base = j * ndim
            x = kpts[base] * W
            y = kpts[base + 1] * H
            v = kpts[base + 2] if ndim == 3 else 2.0
            flat_xy.append((x, y))
            flat_vis.append(v)
            flat_obj.append(obj_idx)

    for _ in range(multiplier):
        try:
            if bboxes:
                out = pipeline(
                    image=img,
                    bboxes=bboxes,
                    class_labels=list(range(len(bboxes))),
                    keypoints=flat_xy,
                    kpt_obj_idx=flat_obj,
                )
            else:
                out = pipeline(image=img, bboxes=[], class_labels=[], keypoints=[], kpt_obj_idx=[])
            aug_img = out["image"]
            aH, aW = aug_img.shape[:2]
            out_bboxes = [list(b) for b in out.get("bboxes", [])]
            surviving = list(out.get("class_labels", []))  # original object indices that survived
            out_kpts = out.get("keypoints", [])

            # Keypoints kept in input order (remove_invisible=False), so reshape by object.
            kpts_by_obj: dict[int, list[float]] = {}
            for idx, (x, y) in enumerate(out_kpts):
                obj_idx = flat_obj[idx] if idx < len(flat_obj) else 0
                inside = 0.0 <= x <= aW and 0.0 <= y <= aH
                nx = min(1.0, max(0.0, x / aW)) if aW else 0.0
                ny = min(1.0, max(0.0, y / aH)) if aH else 0.0
                entry = kpts_by_obj.setdefault(obj_idx, [])
                if ndim == 3:
                    v = flat_vis[idx] if inside else 0.0
                    entry.extend([nx, ny, v])
                else:
                    entry.extend([nx, ny])

            aug_bboxes: list[list[float]] = []
            aug_keypoints: list[list[float]] = []
            aug_cls: list[int] = []
            for bbox, obj_idx in zip(out_bboxes, surviving):
                obj_idx = int(obj_idx)
                kpts = kpts_by_obj.get(obj_idx)
                if kpts is None or len(kpts) != k * ndim:
                    continue
                aug_bboxes.append(bbox)
                aug_keypoints.append(kpts)
                aug_cls.append(class_ids[obj_idx])
            results.append((aug_img, aug_bboxes, aug_keypoints, aug_cls))
        except Exception as exc:
            logger.debug("Pose augmentation error: %s", exc)
    return results


# ---------------------------------------------------------------------------
# Image collection
# ---------------------------------------------------------------------------

def _read_dataset_yaml(dataset_path: Path) -> dict[str, Any]:
    for name in ("data.yaml", "dataset.yaml"):
        yaml_path = dataset_path / name
        if yaml_path.exists():
            try:
                with open(yaml_path, encoding="utf-8") as handle:
                    loaded = yaml.safe_load(handle) or {}
                return loaded if isinstance(loaded, dict) else {}
            except Exception:
                return {}
    return {}


def _split_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [item for item in value if item is not None]
    return [value]


def _resolve_dataset_base(dataset_path: Path, data: dict[str, Any]) -> Path:
    raw_base = data.get("path")
    if raw_base in (None, ""):
        return dataset_path
    base = Path(str(raw_base))
    if base.is_absolute():
        return base.resolve()
    normalized = str(base).replace("\\", "/")
    if normalized.startswith("../"):
        return (dataset_path / normalized).resolve()
    return (dataset_path / base).resolve()


def _resolve_dataset_path(dataset_path: Path, value: Any, *, base_path: Path | None = None) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    root = base_path or dataset_path
    resolved = (root / path).resolve()
    if resolved.exists():
        return resolved
    normalized = str(path).replace("\\", "/")
    if normalized.startswith("../"):
        roboflow_resolved = (dataset_path / normalized[3:]).resolve()
        if roboflow_resolved.exists():
            return roboflow_resolved
    return resolved


def _has_direct_images(path: Path) -> bool:
    try:
        return any(
            item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
            for item in path.iterdir()
        )
    except OSError:
        return False


def _normalize_image_dir(path: Path | None) -> Path | None:
    if path is None or not path.exists() or not path.is_dir():
        return None
    if _has_direct_images(path):
        return path.resolve()
    nested_images = path / "images"
    if nested_images.exists() and nested_images.is_dir() and _has_direct_images(nested_images):
        return nested_images.resolve()
    return None


def _iter_split_image_dirs(dataset_path: Path) -> list[tuple[str, Path]]:
    data = _read_dataset_yaml(dataset_path)
    yaml_base = _resolve_dataset_base(dataset_path, data)
    dirs: list[tuple[str, Path]] = []
    seen: set[Path] = set()

    def add(split_name: str, path: Path | None) -> None:
        image_dir = _normalize_image_dir(path)
        if image_dir is None:
            return
        if image_dir in seen:
            return
        seen.add(image_dir)
        dirs.append((split_name, image_dir))

    for canonical, aliases in SPLIT_ALIASES.items():
        value = next((data.get(alias) for alias in aliases if data.get(alias)), None)
        for split_value in _split_values(value):
            add(canonical, _resolve_dataset_path(dataset_path, split_value, base_path=yaml_base))

    for split_name in ("train", "valid", "val", "test"):
        canonical = "val" if split_name == "valid" else split_name
        add(canonical, dataset_path / split_name / "images")
        add(canonical, dataset_path / "images" / split_name)
        add(canonical, dataset_path / split_name)

    return dirs


def _label_candidates(
    dataset_path: Path,
    image_dir: Path,
    img_path: Path,
    split_name: str,
) -> list[Path]:
    stem = img_path.stem + ".txt"
    candidates: list[Path] = []
    image_dir_text = str(image_dir).replace("\\", "/")

    if "/images/" in image_dir_text:
        candidates.append(Path(image_dir_text.replace("/images/", "/labels/")) / stem)
    if image_dir.name == "images":
        candidates.append(image_dir.parent / "labels" / stem)
    candidates.append(dataset_path / split_name / "labels" / stem)
    candidates.append(dataset_path / "labels" / split_name / stem)

    if split_name == "val":
        candidates.append(dataset_path / "valid" / "labels" / stem)
        candidates.append(dataset_path / "labels" / "valid" / stem)
    elif split_name == "valid":
        candidates.append(dataset_path / "val" / "labels" / stem)
        candidates.append(dataset_path / "labels" / "val" / stem)

    candidates.append(dataset_path / "labels" / stem)
    return list(dict.fromkeys(candidates))


def _find_label_path(
    dataset_path: Path,
    image_dir: Path,
    img_path: Path,
    split_name: str,
) -> Path | None:
    for candidate in _label_candidates(dataset_path, image_dir, img_path, split_name):
        if candidate.exists():
            return candidate
    return None


def collect_all_images(dataset_path: Path) -> list[tuple[Path, Path | None]]:
    """
    Walk all known splits and return a flat list of (image_path, label_path | None).
    Handles YOLO split/images, images/split, and flat split-dir structures.
    label_path is None for COCO datasets or images without a label file.
    """
    pairs: list[tuple[Path, Path | None]] = []
    seen_images: set[Path] = set()
    for split_name, img_dir in _iter_split_image_dirs(dataset_path):
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            resolved_img = img_path.resolve()
            if resolved_img in seen_images:
                continue
            seen_images.add(resolved_img)
            pairs.append((
                img_path,
                _find_label_path(dataset_path, img_dir, img_path, split_name),
            ))
    return pairs


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_augmentation(
    source_dataset_path: Path,
    output_name: str,
    profile,
    split_config: SplitConfig,
    output_format: str = "YOLO Segmentation",
    progress_callback: Callable[[int, int, str], None] | None = None,
    max_workers: int = 4,
) -> AugmentationStats:
    """
    Non-destructive augmentation runner.

    Steps:
      1. Detect annotation format from source
      2. Collect all images from all splits
      3. Augment every image (multiplier × each)
      4. Group each source image with its augmented variants (and its original,
         if include_originals)
      5. Shuffle groups with seed, then assign whole groups to splits by ratio so
         a source image and all its variants always stay in the same split (no leakage)
      6. Write output dataset
      7. If COCO: convert using existing convert_yolo_to_coco()
    """
    t0 = time.time()

    try:
        from src.datasets.core import convert_yolo_to_coco, read_yaml_file
    except ImportError:
        from datasets.core import convert_yolo_to_coco, read_yaml_file  # type: ignore

    ann_format = detect_annotation_format(source_dataset_path)
    all_pairs = collect_all_images(source_dataset_path)
    total_source = len(all_pairs)

    if total_source == 0:
        return AugmentationStats(
            source_dataset=source_dataset_path.name,
            output_path=output_name,
            profile_name=profile.name,
            annotation_format=ann_format,
            output_format=output_format,
            total_source_images=0,
            total_output_images=0,
        )

    if progress_callback:
        progress_callback(0, total_source, "Building pipeline...")

    kpt_shape = read_kpt_shape(source_dataset_path) if ann_format == "yolo_pose" else None
    if ann_format == "yolo_pose" and kpt_shape is None:
        # kpt_shape is required to parse pose rows; fall back to bbox handling.
        logger.warning("Pose dataset detected but data.yaml has no usable kpt_shape — treating as bbox.")
        ann_format = "yolo_bbox"

    if output_format == "YOLO Pose" and ann_format != "yolo_pose":
        raise ValueError(
            "YOLO Pose output requires a pose source dataset (data.yaml with kpt_shape). "
            "Keypoints cannot be synthesized from boxes or polygons."
        )

    # Build pipeline
    if ann_format == "yolo_pose":
        pipeline = build_pose_pipeline(profile)
    elif ann_format == "yolo_seg":
        pipeline = build_seg_pipeline(profile)
    else:
        pipeline = build_bbox_pipeline(profile)

    # Output directory (under datasets/ by default)
    out_root = source_dataset_path.parent / output_name
    if out_root.resolve() == source_dataset_path.resolve():
        raise ValueError("Augmentation output path must be different from the source dataset path.")
    if out_root.exists():
        shutil.rmtree(out_root)
    # Use a temporary YOLO structure for augmented images, then optionally convert
    tmp_root = out_root if output_format != "COCO" else out_root / "_yolo_tmp"

    # Augmented pool: list of (img_bgr_bytes, polygons_or_bboxes, class_ids, is_seg)
    # We store encoded bytes to avoid holding all images in RAM simultaneously.
    # Instead, write immediately to disk to keep memory bounded.

    skipped = 0
    discarded = 0

    def _process(pair: tuple[Path, Path | None]) -> tuple[
        list[tuple[bytes, list, list]],   # augmented
        tuple[bytes, list, list] | None,   # original
        int,                               # discarded annotations
    ]:
        img_path, lbl_path = pair
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return [], None, 0

        orig_discarded = 0
        if ann_format == "yolo_pose":
            bboxes, keypoints, cls_ids = read_yolo_pose(lbl_path, kpt_shape)
            aug_results = _augment_pose(img_bgr, bboxes, keypoints, cls_ids, pipeline, profile.multiplier, kpt_shape)
            aug_items = []
            for aug_img, aug_bb, aug_kp, aug_cls in aug_results:
                orig_discarded += max(0, len(bboxes) - len(aug_bb))
                _, buf = cv2.imencode(".jpg", aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                aug_items.append((buf.tobytes(), (aug_bb, aug_kp), aug_cls))
            _, orig_buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return aug_items, (orig_buf.tobytes(), (bboxes, keypoints), cls_ids), orig_discarded

        elif ann_format == "yolo_seg":
            polygons, cls_ids = read_yolo_seg(lbl_path)
            aug_results = _augment_seg(img_bgr, polygons, cls_ids, pipeline, profile.multiplier)
            aug_items = []
            for aug_img, aug_poly, aug_cls in aug_results:
                orig_discarded += len(polygons) - len(aug_poly)
                _, buf = cv2.imencode(".jpg", aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                aug_items.append((buf.tobytes(), aug_poly, aug_cls))
            # Original
            _, orig_buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return aug_items, (orig_buf.tobytes(), polygons, cls_ids), orig_discarded

        else:
            bboxes, cls_ids = read_yolo_bbox(lbl_path)
            aug_results = _augment_bbox(img_bgr, bboxes, cls_ids, pipeline, profile.multiplier)
            aug_items = []
            for aug_img, aug_bb, aug_cls in aug_results:
                orig_discarded += max(0, len(bboxes) - len(aug_bb))
                _, buf = cv2.imencode(".jpg", aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                aug_items.append((buf.tobytes(), aug_bb, aug_cls))
            _, orig_buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return aug_items, (orig_buf.tobytes(), bboxes, cls_ids), orig_discarded

    # One group per source image: [original?] + its augmented variants. Splitting at
    # the group level (rather than per-item) keeps every variant of an image in the
    # same split, preventing train/val/test leakage.
    groups: list[list[tuple[bytes, list, list]]] = []
    done_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process, pair): pair for pair in all_pairs}
        for future in as_completed(futures):
            pair = futures[future]
            done_count += 1
            try:
                aug_items, orig_item, n_disc = future.result()
                if orig_item is None:
                    # Unreadable image — nothing produced.
                    skipped += 1
                else:
                    group: list[tuple[bytes, list, list]] = []
                    if profile.include_originals:
                        group.append(orig_item)
                    group.extend(aug_items)
                    if group:
                        groups.append(group)
                discarded += n_disc
            except Exception as exc:
                logger.warning("Failed to augment %s: %s", pair[0].name, exc)
                skipped += 1

            if progress_callback and done_count % max(1, total_source // 100) == 0:
                progress_callback(done_count, total_source, pair[0].name)

    # Shuffle groups, then assign whole groups to splits by ratio.
    rng = random.Random(profile.seed)
    rng.shuffle(groups)

    total_out = sum(len(g) for g in groups)
    n_val_target = int(total_out * split_config.val_ratio)
    n_test_target = int(total_out * split_config.test_ratio)

    # Fill test then val up to their targets (whole groups only); the remainder goes
    # to train, so a 0.0 test ratio never receives leftover images.
    split_data: dict[str, list[tuple[bytes, list, list]]] = {
        "train": [], "valid": [], "test": [],
    }
    val_count = test_count = 0
    for g in groups:
        if split_config.test_ratio > 0 and test_count < n_test_target:
            split_data["test"].extend(g)
            test_count += len(g)
        elif split_config.val_ratio > 0 and val_count < n_val_target:
            split_data["valid"].extend(g)
            val_count += len(g)
        else:
            split_data["train"].extend(g)

    if progress_callback:
        progress_callback(total_source, total_source, "Writing output dataset...")

    # Read source class info
    data_yaml_path = source_dataset_path / "data.yaml"
    class_names: list[str] = []
    if data_yaml_path.exists():
        data = read_yaml_file(str(data_yaml_path))
        if data:
            class_names = data.get("names", [])

    # Determine output label writer based on requested output_format
    # (independent of the source annotation format). Pose source + COCO output keeps
    # keypoints by writing pose rows to the tmp YOLO dataset before conversion.
    write_as_pose = output_format == "YOLO Pose" or (ann_format == "yolo_pose" and output_format == "COCO")
    write_as_seg = output_format in ("YOLO Segmentation", "COCO") and not write_as_pose

    # Write files
    split_counts: dict[str, int] = {}
    for split_name, items in split_data.items():
        if split_name == "test" and not items:
            split_counts[split_name] = 0
            continue
        img_dir = tmp_root / split_name / "images"
        lbl_dir = tmp_root / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for idx, (img_bytes, anns, cls_ids) in enumerate(items):
            stem = f"aug_{split_name}_{idx:06d}"
            img_path = img_dir / f"{stem}.jpg"
            img_path.write_bytes(img_bytes)
            lbl_path = lbl_dir / f"{stem}.txt"
            if ann_format == "yolo_pose":
                bboxes, keypoints = anns
                if write_as_pose:
                    write_yolo_pose(lbl_path, bboxes, keypoints, cls_ids)
                elif write_as_seg:
                    write_yolo_seg(lbl_path, [bbox_to_polygon(bb) for bb in bboxes], cls_ids)
                else:
                    write_yolo_bbox(lbl_path, bboxes, cls_ids)
            elif write_as_seg:
                # Output wants polygons
                if ann_format == "yolo_bbox":
                    polys = [bbox_to_polygon(bb) for bb in anns]
                else:
                    polys = anns
                write_yolo_seg(lbl_path, polys, cls_ids)
            else:
                # Output wants bboxes
                if ann_format == "yolo_seg":
                    bboxes = [polygon_to_bbox(poly) for poly in anns]
                else:
                    bboxes = anns
                write_yolo_bbox(lbl_path, bboxes, cls_ids)
        split_counts[split_name] = len(items)

    # Write data.yaml (include task field so future detection is instant)
    if write_as_pose:
        task_field = "pose"
    elif write_as_seg:
        task_field = "segment"
    else:
        task_field = "detect"
    data_yaml_content = {
        "train": "train/images",
        "val":   "valid/images",
        "nc":    len(class_names),
        "names": class_names,
        "task":  task_field,
    }
    if split_data["test"]:
        data_yaml_content["test"] = "test/images"
    if write_as_pose and kpt_shape is not None:
        data_yaml_content["kpt_shape"] = [kpt_shape[0], kpt_shape[1]]
    with open(tmp_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False, allow_unicode=True)

    # COCO conversion
    if output_format == "COCO":
        convert_yolo_to_coco(tmp_root, out_root)
        shutil.rmtree(tmp_root, ignore_errors=True)

    return AugmentationStats(
        source_dataset=source_dataset_path.name,
        output_path=str(out_root),
        profile_name=profile.name,
        annotation_format=ann_format,
        output_format=output_format,
        total_source_images=total_source,
        total_output_images=total_out,
        split_counts=split_counts,
        images_skipped=skipped,
        annotations_discarded=discarded,
        elapsed_seconds=time.time() - t0,
    )
