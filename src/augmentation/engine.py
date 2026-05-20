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
      2. data.yaml task field           → 'yolo_seg' or 'yolo_bbox'
      3. Non-empty label file sampling  → 'yolo_seg' or 'yolo_bbox'

    Returns: 'yolo_bbox', 'yolo_seg', or 'coco'
    """
    # 1. COCO JSON
    ann_dir = dataset_path / "annotations"
    if ann_dir.exists() and any(ann_dir.glob("*.json")):
        return "coco"

    # 2. data.yaml task field
    data_yaml = dataset_path / "data.yaml"
    if data_yaml.exists():
        try:
            with open(data_yaml, encoding="utf-8") as f:
                meta = yaml.safe_load(f) or {}
            task = str(meta.get("task", "")).lower()
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
    for cls, (cx, cy, w, h) in zip(class_ids, bboxes):
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def write_yolo_seg(label_path: Path, polygons: list[list[float]], class_ids: list[int]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for cls, poly in zip(class_ids, polygons):
        coords = " ".join(f"{v:.6f}" for v in poly)
        lines.append(f"{cls} {coords}")
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


def _resolve_dataset_path(dataset_path: Path, value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    resolved = (dataset_path / path).resolve()
    if resolved.exists():
        return resolved
    normalized = str(path).replace("\\", "/")
    if normalized.startswith("../"):
        roboflow_resolved = (dataset_path / normalized[3:]).resolve()
        if roboflow_resolved.exists():
            return roboflow_resolved
    return resolved


def _iter_split_image_dirs(dataset_path: Path) -> list[tuple[str, Path]]:
    data = _read_dataset_yaml(dataset_path)
    dirs: list[tuple[str, Path]] = []
    seen: set[Path] = set()

    def add(split_name: str, path: Path | None) -> None:
        if path is None or not path.exists():
            return
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        dirs.append((split_name, resolved))

    for canonical, aliases in SPLIT_ALIASES.items():
        value = next((data.get(alias) for alias in aliases if data.get(alias)), None)
        add(canonical, _resolve_dataset_path(dataset_path, value))

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
      4. Pool originals + augmented (if include_originals)
      5. Shuffle with seed, split by ratios
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

    # Build pipeline
    if ann_format == "yolo_seg":
        pipeline = build_seg_pipeline(profile)
    else:
        pipeline = build_bbox_pipeline(profile)

    # Output directory (under datasets/ by default)
    out_root = source_dataset_path.parent / output_name
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
        if ann_format == "yolo_seg":
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

    aug_pool: list[tuple[bytes, list, list]] = []
    orig_pool: list[tuple[bytes, list, list]] = []
    done_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process, pair): pair for pair in all_pairs}
        for future in as_completed(futures):
            pair = futures[future]
            done_count += 1
            try:
                aug_items, orig_item, n_disc = future.result()
                aug_pool.extend(aug_items)
                if orig_item is not None:
                    orig_pool.append(orig_item)
                else:
                    skipped += 1
                discarded += n_disc
            except Exception as exc:
                logger.warning("Failed to augment %s: %s", pair[0].name, exc)
                skipped += 1

            if progress_callback and done_count % max(1, total_source // 100) == 0:
                progress_callback(done_count, total_source, pair[0].name)

    # Assemble final pool
    final_pool: list[tuple[bytes, list, list]] = []
    if profile.include_originals:
        final_pool.extend(orig_pool)
    final_pool.extend(aug_pool)

    # Shuffle
    rng = random.Random(profile.seed)
    rng.shuffle(final_pool)

    total_out = len(final_pool)
    n_train = int(total_out * split_config.train_ratio)
    n_val = int(total_out * split_config.val_ratio)

    split_data = {
        "train": final_pool[:n_train],
        "valid": final_pool[n_train:n_train + n_val],
        "test":  final_pool[n_train + n_val:],
    }

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
    # (independent of the source annotation format)
    write_as_seg = output_format in ("YOLO Segmentation", "COCO")

    # Write files
    split_counts: dict[str, int] = {}
    for split_name, items in split_data.items():
        img_dir = tmp_root / split_name / "images"
        lbl_dir = tmp_root / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for idx, (img_bytes, anns, cls_ids) in enumerate(items):
            stem = f"aug_{split_name}_{idx:06d}"
            img_path = img_dir / f"{stem}.jpg"
            img_path.write_bytes(img_bytes)
            lbl_path = lbl_dir / f"{stem}.txt"
            if write_as_seg:
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
    task_field = "segment" if write_as_seg else "detect"
    data_yaml_content = {
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    len(class_names),
        "names": class_names,
        "task":  task_field,
    }
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
