"""
Augmentation engine: annotation I/O, Albumentations pipeline, pool+redistribute runner.

Supports:
  - YOLO bbox format  (class_id cx cy w h)
  - YOLO seg format   (class_id x1 y1 x2 y2 … xn yn)
  - Auto-detection of source format
  - Pool-all-images → group-aware split → augment → write train/val/test
  - Output as YOLO Detection (box), YOLO Segmentation (instance polygon),
    YOLO Semantic (Polygon) (polygons declared as task "semantic"),
    YOLO Segmentation (Semantic) (dense per-pixel class mask), or COCO
"""
from __future__ import annotations

import json
import logging
import math
import multiprocessing
import os
import random
import re
import shutil
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from src.datasets.cache import clean_dataset_image_cache
from src.utils.ml_dependencies import import_cv2
from src.utils.semantic import (
    semantic_background_index,
    semantic_class_names,
    semantic_pixel_value,
)


def _get_cv2() -> Any:
    """Return OpenCV through the serialized dependency loader."""
    return import_cv2()

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation"),
    "test": ("test", "testing"),
}

_DEFAULT_MAX_WORKERS = 4
_AUTO_MAX_WORKERS = 8
_WORKER_STATE: tuple[Any, Any, str, tuple[int, int] | None, int] | None = None

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SplitConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.20
    test_ratio: float = 0.10
    include_originals: bool = True  # overridden by profile.include_originals at call site
    # Regex matched against each source image stem to derive a spatial/provenance
    # group key. Every image sharing a key lands in the same split. Tiled aerial
    # datasets need this: neighbouring tiles from one raster overlap, so splitting
    # per-image leaks near-duplicate pixels into val/test. ``None`` keys each image
    # by its own stem, which reproduces the per-image behaviour.
    group_key_pattern: str | None = None
    # Output splits that receive augmented variants. ``None`` augments every split.
    # ("train",) is the honest default for evaluation: val/test stay pristine
    # originals instead of augmented twins of the training images.
    augment_splits: tuple[str, ...] | None = None


_SPLIT_NAMES = ("train", "valid", "test")


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
    cache_files_removed: int = 0
    cache_bytes_reclaimed: int = 0


def resolve_augmentation_workers(requested: int) -> int:
    """Resolve ``0`` to a conservative process-worker count."""
    if requested < 0:
        raise ValueError("max_workers must be 0 (Auto) or a positive integer.")
    if requested == 0:
        return max(1, min(_AUTO_MAX_WORKERS, os.cpu_count() or _DEFAULT_MAX_WORKERS))
    return requested


def _validate_split_config(split_config: SplitConfig) -> None:
    ratios = (split_config.train_ratio, split_config.val_ratio, split_config.test_ratio)
    if any(
        not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio < 0
        for ratio in ratios
    ):
        raise ValueError("Split ratios must be finite, non-negative numbers.")
    if abs(sum(ratios) - 1.0) > 0.02:
        raise ValueError(f"Split ratios must sum to 1.0 (got {sum(ratios):.3f}).")
    if split_config.augment_splits is not None:
        unknown = set(split_config.augment_splits) - set(_SPLIT_NAMES)
        if unknown:
            raise ValueError(
                f"augment_splits contains unknown split(s): {sorted(unknown)}. "
                f"Valid names are {list(_SPLIT_NAMES)}."
            )
    if split_config.group_key_pattern is not None:
        try:
            re.compile(split_config.group_key_pattern)
        except re.error as exc:
            raise ValueError(f"group_key_pattern is not a valid regex: {exc}") from exc


def derive_group_key(stem: str, pattern: str | None) -> str:
    """Return the split-grouping key for one source image stem.

    With no pattern each image is its own group. With a pattern, the first capture
    group is used when the regex defines one, otherwise the whole match. A stem the
    regex does not match keeps its own stem as key, so unmatched files are never
    silently merged into one giant group.
    """
    if not pattern:
        return stem
    match = re.search(pattern, stem)
    if match is None:
        return stem
    if match.re.groups:
        captured = match.group(1)
        if captured:
            return captured
    return match.group(0)


def assign_groups_to_splits(
    group_sizes: dict[str, int],
    split_config: SplitConfig,
    seed: int,
) -> dict[str, str]:
    """Assign whole groups to splits, largest group first into the neediest split.

    Largest-first bin packing keeps the realised ratios tight even when group sizes
    are wildly uneven (real tiled datasets range from a couple of tiles to several
    hundred). The shuffle before the size sort makes equal-sized groups order
    deterministically from ``seed`` rather than from filesystem order.

    Returns a mapping of group key -> split name. Splits with a zero ratio are never
    candidates, so a 0.0 test ratio cannot pick up remainder.
    """
    ratios = {
        "train": split_config.train_ratio,
        "valid": split_config.val_ratio,
        "test": split_config.test_ratio,
    }
    candidates = {name: ratio for name, ratio in ratios.items() if ratio > 0}
    if not candidates:
        raise ValueError("At least one split ratio must be greater than zero.")

    keys = sorted(group_sizes)
    random.Random(seed).shuffle(keys)
    keys.sort(key=lambda key: -group_sizes[key])

    total = sum(group_sizes.values())
    targets = {name: total * ratio for name, ratio in candidates.items()}
    filled = dict.fromkeys(candidates, 0)
    assignment: dict[str, str] = {}
    for key in keys:
        split_name = max(candidates, key=lambda name: targets[name] - filled[name])
        assignment[key] = split_name
        filled[split_name] += group_sizes[key]
    return assignment


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
    cv2 = _get_cv2()
    pts = np.array(polygon, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] = np.clip(pts[:, 0] * W, 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1] * H, 0, H - 1)
    pts = pts.astype(np.int32)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def polygons_to_semantic_mask(
    polygons: list[list[float]], class_ids: list[int], width: int, height: int,
    num_classes: int = 1,
) -> np.ndarray:
    """Combine per-instance polygons into a dense uint8 class-index mask.

    Pixels hold raw class indices under the convention in ``src.utils.semantic``:
    background is ``0`` and foreground ``1`` for a single-class dataset, while a
    multi-class dataset puts background at ``num_classes`` and keeps class ids as
    they are. Ultralytics runs these values straight through CrossEntropyLoss, so
    anything outside ``[0, nc)`` aborts training with a device-side assert.

    When instances overlap, later polygons draw over earlier ones.
    """
    background = semantic_background_index(num_classes)
    mask = np.full((height, width), background, dtype=np.uint8)
    for polygon, cls in zip(polygons, class_ids):
        if not polygon:
            continue
        instance_mask = polygon_to_mask(polygon, width, height)
        mask[instance_mask.astype(bool)] = semantic_pixel_value(cls, num_classes)
    return mask


def write_semantic_mask(mask_path: Path, mask: np.ndarray) -> None:
    """Write a dense class-index mask as a single-channel PNG."""
    cv2 = _get_cv2()
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), mask)


def mask_to_polygon(
    mask: np.ndarray,
    min_area: float = 4.0,
    epsilon_factor: float = 0.003,
) -> list[float] | None:
    """
    Binary mask → normalized polygon [x1/W, y1/H, ...].
    Returns None if mask is empty or below min_area threshold.
    """
    cv2 = _get_cv2()
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


def _initialize_augmentation_worker(
    profile: Any,
    annotation_format: str,
    kpt_shape: tuple[int, int] | None,
) -> None:
    """Initialize one isolated OpenCV/Albumentations pipeline per process."""
    global _WORKER_STATE
    cv2 = _get_cv2()
    cv2.setNumThreads(1)
    if annotation_format == "yolo_pose":
        pipeline = build_pose_pipeline(profile)
    elif annotation_format == "yolo_seg":
        pipeline = build_seg_pipeline(profile)
    else:
        pipeline = build_bbox_pipeline(profile)
    _WORKER_STATE = (cv2, pipeline, annotation_format, kpt_shape, int(profile.multiplier))


def _write_staged_item(
    group_dir: Path,
    name: str,
    image: np.ndarray,
    annotations: Any,
    class_ids: list[int],
    cv2: Any,
) -> None:
    # 4:4:4 — OpenCV defaults to 4:2:0, which halves the resolution of both chroma
    # planes after RGB->YCbCr. On fused/false-colour imagery the channels carry
    # independent measurements (e.g. a vegetation index in channel 0), not colour,
    # so subsampling would blur the very signal the annotations describe.
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
        ],
    )
    if not success:
        raise RuntimeError("OpenCV could not encode an augmented image as JPEG.")
    (group_dir / f"{name}.jpg").write_bytes(encoded.tobytes())
    (group_dir / f"{name}.json").write_text(
        json.dumps({"annotations": annotations, "class_ids": class_ids}), encoding="utf-8"
    )


def _process_augmentation_job(
    index: int,
    image_path_text: str,
    label_path_text: str | None,
    stage_root_text: str,
    seed: int,
    n_variants: int | None = None,
) -> tuple[int, str | None, int]:
    """Augment one source image and stage its variants without returning image bytes.

    ``n_variants`` overrides the profile multiplier for this image; ``0`` stages the
    original only, which is how non-augmented splits are carried through.
    """
    if _WORKER_STATE is None:
        raise RuntimeError("Augmentation worker was not initialized.")
    cv2, pipeline, annotation_format, kpt_shape, multiplier = _WORKER_STATE
    if n_variants is not None:
        multiplier = int(n_variants)
    image_path = Path(image_path_text)
    label_path = Path(label_path_text) if label_path_text else None
    image = cv2.imread(str(image_path))
    if image is None:
        return index, None, 0

    if hasattr(pipeline, "set_random_seed"):
        pipeline.set_random_seed(seed)
    else:  # Albumentations 1.x compatibility
        random.seed(seed)
        np.random.seed(seed)
    group_dir = Path(stage_root_text) / f"group_{index:08d}"
    group_dir.mkdir(parents=True, exist_ok=False)
    discarded = 0

    if annotation_format == "yolo_pose":
        if kpt_shape is None:
            raise RuntimeError("Pose worker requires kpt_shape.")
        bboxes, keypoints, class_ids = read_yolo_pose(label_path, kpt_shape)
        _write_staged_item(group_dir, "original", image, [bboxes, keypoints], class_ids, cv2)
        for variant, (aug_image, aug_boxes, aug_keypoints, aug_classes) in enumerate(
            _augment_pose(image, bboxes, keypoints, class_ids, pipeline, multiplier, kpt_shape)
        ):
            discarded += max(0, len(bboxes) - len(aug_boxes))
            _write_staged_item(
                group_dir, f"aug_{variant:04d}", aug_image, [aug_boxes, aug_keypoints], aug_classes, cv2
            )
    elif annotation_format == "yolo_seg":
        polygons, class_ids = read_yolo_seg(label_path)
        _write_staged_item(group_dir, "original", image, polygons, class_ids, cv2)
        for variant, (aug_image, aug_polygons, aug_classes) in enumerate(
            _augment_seg(image, polygons, class_ids, pipeline, multiplier)
        ):
            discarded += max(0, len(polygons) - len(aug_polygons))
            _write_staged_item(group_dir, f"aug_{variant:04d}", aug_image, aug_polygons, aug_classes, cv2)
    else:
        bboxes, class_ids = read_yolo_bbox(label_path)
        _write_staged_item(group_dir, "original", image, bboxes, class_ids, cv2)
        for variant, (aug_image, aug_boxes, aug_classes) in enumerate(
            _augment_bbox(image, bboxes, class_ids, pipeline, multiplier)
        ):
            discarded += max(0, len(bboxes) - len(aug_boxes))
            _write_staged_item(group_dir, f"aug_{variant:04d}", aug_image, aug_boxes, aug_classes, cv2)
    return index, str(group_dir), discarded


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
    max_workers: int = 0,
) -> AugmentationStats:
    """
    Non-destructive augmentation runner.

    Steps:
      1. Detect annotation format from source
      2. Collect all images from all splits
      3. Augment every image (multiplier × each)
      4. Group each source image with its augmented variants (and its original,
         if include_originals)
      5. Assign whole groups to splits by ratio *before* augmenting, so a source
         image and all its variants always stay in the same split (no leakage), and
         so splits excluded from ``augment_splits`` never pay the augmentation cost
      6. Write output dataset
      7. If COCO: convert using existing convert_yolo_to_coco()

    A "group" is one source image plus its variants by default. Set
    ``split_config.group_key_pattern`` to widen it to every image sharing a
    provenance key (e.g. all tiles cut from one raster), which is what keeps
    overlapping tiles out of both train and val.
    """
    t0 = time.time()
    _validate_split_config(split_config)
    max_workers = resolve_augmentation_workers(max_workers)

    cache_cleanup = clean_dataset_image_cache(source_dataset_path)
    if cache_cleanup.removed_files:
        logger.info(
            "Removed %d Ultralytics image-cache files (%.2f GiB) before augmentation.",
            cache_cleanup.removed_files,
            cache_cleanup.reclaimed_bytes / 1024**3,
        )
    for error in cache_cleanup.errors:
        logger.warning("Could not remove dataset cache artifact: %s", error)

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
            cache_files_removed=cache_cleanup.removed_files,
            cache_bytes_reclaimed=cache_cleanup.reclaimed_bytes,
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

    # Decide the split for every source image up front. Doing this before the worker
    # pool runs lets splits outside augment_splits skip augmentation entirely.
    group_keys = [
        derive_group_key(image_path.stem, split_config.group_key_pattern)
        for image_path, _label_path in all_pairs
    ]
    group_sizes: dict[str, int] = {}
    for key in group_keys:
        group_sizes[key] = group_sizes.get(key, 0) + 1
    group_split = assign_groups_to_splits(group_sizes, split_config, int(profile.seed))
    pair_splits = [group_split[key] for key in group_keys]

    augment_splits = split_config.augment_splits
    variants_for_split = {
        name: (
            int(profile.multiplier)
            if augment_splits is None or name in augment_splits
            else 0
        )
        for name in _SPLIT_NAMES
    }
    # A split that is neither augmented nor allowed to keep its originals would be
    # written out empty. Carrying the original through is the only sane reading of
    # "don't augment this split".
    keep_original_for_split = {
        name: bool(profile.include_originals) or variants_for_split[name] == 0
        for name in _SPLIT_NAMES
    }

    # Output directory (under datasets/ by default). Build in a sibling staging
    # directory so a failed run never destroys an existing augmented dataset.
    out_root = source_dataset_path.parent / output_name
    if out_root.resolve() == source_dataset_path.resolve():
        raise ValueError("Augmentation output path must be different from the source dataset path.")
    stage_root = source_dataset_path.parent / f".{output_name}.augmenting-{uuid.uuid4().hex}"
    groups_root = stage_root / "groups"
    build_root = stage_root / "output"
    tmp_root = build_root if output_format != "COCO" else stage_root / "yolo"
    groups_root.mkdir(parents=True)
    skipped = discarded = 0
    completed_groups: dict[int, list[Path]] = {}
    worker_context = multiprocessing.get_context("spawn")
    split_data: dict[str, list[Path]] = {name: [] for name in _SPLIT_NAMES}

    try:
        # Keep at most two jobs per process outstanding: large images remain on disk,
        # not in the parent process or an unbounded executor queue.
        job_iter = iter(enumerate(all_pairs))
        pending: dict[Any, tuple[int, Path]] = {}
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=worker_context,
            initializer=_initialize_augmentation_worker,
            initargs=(profile, ann_format, kpt_shape),
        ) as executor:
            def submit_next() -> bool:
                try:
                    index, (image_path, label_path) = next(job_iter)
                except StopIteration:
                    return False
                future = executor.submit(
                    _process_augmentation_job,
                    index,
                    str(image_path),
                    str(label_path) if label_path else None,
                    str(groups_root),
                    int(profile.seed) + index,
                    variants_for_split[pair_splits[index]],
                )
                pending[future] = (index, image_path)
                return True

            for _ in range(min(total_source, max_workers * 2)):
                submit_next()
            done_count = 0
            while pending:
                future = next(as_completed(pending))
                _index, image_path = pending.pop(future)
                done_count += 1
                try:
                    result_index, group_text, n_disc = future.result()
                    discarded += n_disc
                    if group_text is None:
                        skipped += 1
                    else:
                        group_dir = Path(group_text)
                        item_paths = sorted(group_dir.glob("*.jpg"))
                        if not keep_original_for_split[pair_splits[result_index]]:
                            item_paths = [path for path in item_paths if path.stem != "original"]
                        if item_paths:
                            completed_groups[result_index] = item_paths
                except Exception as exc:
                    logger.warning("Failed to augment %s: %s", image_path.name, exc)
                    skipped += 1
                submit_next()
                if progress_callback and done_count % max(1, total_source // 100) == 0:
                    progress_callback(done_count, total_source, image_path.name)
    except Exception:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise

    # Collect each source image's staged variants into the split chosen before
    # augmentation. Iterating in source order keeps output naming reproducible
    # regardless of the order workers happened to finish in.
    for index in sorted(completed_groups):
        split_data[pair_splits[index]].extend(completed_groups[index])
    total_out = sum(len(items) for items in split_data.values())

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
    write_as_semantic = output_format == "YOLO Segmentation (Semantic)" and not write_as_pose
    # Polygon-encoded semantic segmentation: identical files to instance segmentation,
    # but data.yaml declares task "semantic". Ultralytics rasterizes the polygons into
    # a dense target and adds a background class, so no mask images are needed.
    write_as_semantic_polygon = output_format == "YOLO Semantic (Polygon)" and not write_as_pose
    write_as_seg = (
        output_format in ("YOLO Segmentation", "COCO", "YOLO Semantic (Polygon)")
        and not write_as_pose
        and not write_as_semantic
    )

    # Write files
    split_counts: dict[str, int] = {}
    for split_name, items in split_data.items():
        if split_name == "test" and not items:
            split_counts[split_name] = 0
            continue
        img_dir = tmp_root / split_name / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir = tmp_root / split_name / "masks"
        lbl_dir = tmp_root / split_name / "labels"
        if write_as_semantic:
            mask_dir.mkdir(parents=True, exist_ok=True)
        else:
            lbl_dir.mkdir(parents=True, exist_ok=True)
        for idx, staged_image_path in enumerate(items):
            stem = f"aug_{split_name}_{idx:06d}"
            img_path = img_dir / f"{stem}.jpg"
            shutil.copy2(staged_image_path, img_path)
            staged_metadata = json.loads(staged_image_path.with_suffix(".json").read_text(encoding="utf-8"))
            anns = staged_metadata["annotations"]
            cls_ids = staged_metadata["class_ids"]

            if write_as_semantic:
                # Dense per-pixel class mask (semantic segmentation), not a polygon
                # label file. Instance boundaries are merged; last-drawn polygon wins
                # where instances overlap (mirrors benchmark's semantic ground truth).
                if ann_format == "yolo_pose":
                    bboxes, _keypoints = anns
                    polys = [bbox_to_polygon(bb) for bb in bboxes]
                elif ann_format == "yolo_seg":
                    polys = anns
                else:
                    polys = [bbox_to_polygon(bb) for bb in anns]
                cv2 = _get_cv2()
                decoded = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if decoded is None:
                    raise RuntimeError(f"Could not read staged image {img_path.name}.")
                h, w = decoded.shape[:2]
                mask = polygons_to_semantic_mask(polys, cls_ids, w, h, len(class_names))
                write_semantic_mask(mask_dir / f"{stem}.png", mask)
                continue

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
    elif write_as_semantic or write_as_semantic_polygon:
        task_field = "semantic"
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
    if write_as_semantic:
        # Dense masks live in a sibling "masks/" directory next to each split's
        # "images/" directory. Pixels are raw class indices (see src.utils.semantic):
        # a multi-class dataset gains an explicit trailing "background" class, so nc
        # and names must grow to match or CrossEntropyLoss sees an out-of-range target.
        # The key must be "masks_dir": that is what Ultralytics' SemanticDataset
        # reads, and its absence is what selects the polygon-rasterizing path
        # instead — so a misspelling silently produces an unloadable dataset.
        mask_names = semantic_class_names(class_names)
        data_yaml_content["names"] = mask_names
        data_yaml_content["nc"] = len(mask_names)
        data_yaml_content["bg_class_idx"] = semantic_background_index(len(class_names))
        data_yaml_content["masks_dir"] = "masks"
        data_yaml_content["train_masks"] = "train/masks"
        data_yaml_content["val_masks"] = "valid/masks"
        if split_data["test"]:
            data_yaml_content["test_masks"] = "test/masks"
    with open(tmp_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False, allow_unicode=True)

    # Record which source images landed in which split. Output files are renamed to
    # aug_<split>_<n>, so this is the only way to audit the split after the fact —
    # in particular to prove that no group key spans two splits.
    if output_format != "COCO":
        assignment_report = {
            "seed": int(profile.seed),
            "group_key_pattern": split_config.group_key_pattern,
            "augment_splits": list(augment_splits) if augment_splits is not None else None,
            "multiplier": int(profile.multiplier),
            "include_originals": bool(profile.include_originals),
            "ratios": {
                "train": split_config.train_ratio,
                "valid": split_config.val_ratio,
                "test": split_config.test_ratio,
            },
            "group_splits": group_split,
            "sources": {
                str(image_path.name): {"group": group_keys[index], "split": pair_splits[index]}
                for index, (image_path, _label) in enumerate(all_pairs)
            },
        }
        (tmp_root / "split_assignment.json").write_text(
            json.dumps(assignment_report, indent=2, sort_keys=True), encoding="utf-8"
        )

        # COCO conversion writes its final dataset inside the staging directory.
    if output_format == "COCO":
        convert_yolo_to_coco(tmp_root, build_root)

        # Swap only a fully written dataset into the user-visible location. Directory
        # replacement is recoverable on all supported platforms.
    backup_root: Path | None = None
    if out_root.exists():
        backup_root = source_dataset_path.parent / f".{output_name}.backup-{uuid.uuid4().hex}"
        out_root.rename(backup_root)
    try:
        build_root.rename(out_root)
    except Exception:
        if backup_root is not None and backup_root.exists():
            backup_root.rename(out_root)
        raise
    if backup_root is not None:
        shutil.rmtree(backup_root)
    shutil.rmtree(stage_root, ignore_errors=True)

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
        cache_files_removed=cache_cleanup.removed_files,
        cache_bytes_reclaimed=cache_cleanup.reclaimed_bytes,
    )
