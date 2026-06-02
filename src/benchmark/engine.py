"""Core benchmark evaluation engine."""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import BenchmarkConfig
from .metrics import (
    SizeBucketMetrics,
    _mask_iou_for_pred_gt,
    _box_iou_for_pred_gt,
    _safe_div,
    aggregate_metrics,
    greedy_match,
    polygon_to_mask,
    size_bucket,
)

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ---------------------------------------------------------------------------
# Internal prediction/GT objects
# ---------------------------------------------------------------------------

@dataclass
class PredObject:
    conf: float
    cls: int
    box_xyxy: tuple[float, float, float, float]
    mask: np.ndarray | None = None


@dataclass
class GTObject:
    cls: int
    box_xyxy: tuple[float, float, float, float]
    mask: np.ndarray | None = None
    area: float = 0.0


# ---------------------------------------------------------------------------
# Per-image result
# ---------------------------------------------------------------------------

@dataclass
class ImageResult:
    image_id: int
    image_path: Path
    task: str
    gt_count: int
    pred_count: int
    tp: int
    fp: int
    fn: int
    matched_ious: list[float]
    precision: float
    recall: float
    f1: float
    dominant_bucket: str = "medium"
    raw_preds: list[PredObject] = field(default_factory=list, repr=False)
    raw_gts: list[GTObject] = field(default_factory=list, repr=False)
    mean_iou: float = 0.0


# ---------------------------------------------------------------------------
# Per-model result
# ---------------------------------------------------------------------------

@dataclass
class ModelMetrics:
    weights_path: Path
    task: str
    precision: float
    recall: float
    f1: float
    map50: float
    map75: float
    map50_95: float
    small: SizeBucketMetrics
    medium: SizeBucketMetrics
    large: SizeBucketMetrics
    per_image: list[ImageResult]
    inference_time_ms: float = 0.0
    fps: float = 0.0

@dataclass
class BenchmarkResult:
    models: list[ModelMetrics]
    config: BenchmarkConfig


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_annotation_format(validation_dir: Path) -> str:
    """Return 'coco', 'yolo', or 'none'."""
    if list(validation_dir.glob("_annotations.coco.json")) or list(validation_dir.glob("*.json")):
        return "coco"
    if _find_yolo_labels_dir(validation_dir) is not None:
        return "yolo"
    return "none"


def _find_yolo_labels_dir(validation_dir: Path) -> Path | None:
    """Locate the labels/ directory that pairs with the images in validation_dir.

    Handles two common layouts:
      <split>/images/  →  labels/ is <split>/labels/
      <split>/          →  labels/ is <split>/labels/
    """
    # Case 1: validation_dir IS the images/ directory
    if validation_dir.name == "images":
        sibling = validation_dir.parent / "labels"
        if sibling.exists():
            return sibling

    # Case 2: validation_dir has an images/ subdirectory
    images_sub = validation_dir / "images"
    labels_sub = validation_dir / "labels"
    if images_sub.exists() and labels_sub.exists():
        return labels_sub

    # Case 3: validation_dir directly contains .txt label files alongside images
    if any(validation_dir.glob("*.txt")):
        return validation_dir

    return None


# ---------------------------------------------------------------------------
# YOLO label loading
# ---------------------------------------------------------------------------

def _load_gt_from_yolo_txt(label_path: Path, img_w: int, img_h: int) -> list[GTObject]:
    """Parse a YOLO segmentation label file into GTObjects."""
    gts: list[GTObject] = []
    try:
        lines = label_path.read_text().splitlines()
    except OSError:
        return gts

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            continue

        # Denormalise
        xs = [coords[i] * img_w for i in range(0, len(coords), 2)]
        ys = [coords[i] * img_h for i in range(1, len(coords), 2)]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        box_xyxy = (x1, y1, x2, y2)
        area = (x2 - x1) * (y2 - y1)

        mask: np.ndarray | None = None
        if len(xs) >= 3:
            try:
                flat = [v for pair in zip(xs, ys) for v in pair]
                from .metrics import polygon_to_mask
                mask = polygon_to_mask(flat, img_w, img_h)
                area = float(mask.sum())
            except Exception:
                mask = None

        gts.append(GTObject(cls=cls, box_xyxy=box_xyxy, mask=mask, area=area))
    return gts


def _load_yolo_data(
    validation_dir: Path,
    all_images: list[Path],
) -> dict[str, list[GTObject]]:
    """Load YOLO .txt labels → filename-keyed GT dict."""
    labels_dir = _find_yolo_labels_dir(validation_dir)
    fname_to_gts: dict[str, list[GTObject]] = {}

    for img_path in all_images:
        if not img_path.exists():
            continue
        if labels_dir is not None:
            label_path = labels_dir / (img_path.stem + ".txt")
        else:
            label_path = img_path.with_suffix(".txt")

        try:
            from PIL import Image as _PILImage
            with _PILImage.open(img_path) as _im:
                img_w, img_h = _im.size
        except Exception:
            img_w, img_h = 640, 640

        fname_to_gts[img_path.name] = (
            _load_gt_from_yolo_txt(label_path, img_w, img_h)
            if label_path.exists()
            else []
        )
    return fname_to_gts


# ---------------------------------------------------------------------------
# COCO annotation loading
# ---------------------------------------------------------------------------

def _auto_detect_annotations(validation_dir: Path) -> Path:
    candidates = list(validation_dir.glob("_annotations.coco.json"))
    if candidates:
        return candidates[0]
    candidates = list(validation_dir.glob("*.json"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"No annotation JSON found in {validation_dir}. "
        "Pass annotations_file explicitly in BenchmarkConfig."
    )


def _load_coco_data(annotations_file: Path) -> tuple[dict[int, Path], dict[int, list[GTObject]]]:
    """Load COCO JSON → (image_id→path, image_id→[GTObject])."""
    with annotations_file.open() as f:
        coco = json.load(f)

    img_dir = annotations_file.parent

    id_to_path: dict[int, Path] = {}
    for img_info in coco.get("images", []):
        img_path = img_dir / img_info["file_name"]
        # Also check inside an images/ subdirectory
        if not img_path.exists():
            img_path = img_dir / "images" / img_info["file_name"]
        id_to_path[img_info["id"]] = img_path

    # Index widths/heights for polygon rasterisation
    id_to_size: dict[int, tuple[int, int]] = {
        img["id"]: (img["width"], img["height"])
        for img in coco.get("images", [])
    }

    id_to_gts: dict[int, list[GTObject]] = {img_id: [] for img_id in id_to_path}

    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in id_to_path:
            continue
        w, h = id_to_size.get(img_id, (640, 640))
        cls = ann.get("category_id", 0)

        # Bounding box (COCO format: x, y, width, height)
        bbox = ann.get("bbox", [0, 0, 0, 0])
        box_xyxy = (float(bbox[0]), float(bbox[1]),
                    float(bbox[0] + bbox[2]), float(bbox[1] + bbox[3]))
        area = float(ann.get("area", bbox[2] * bbox[3]))

        # Segmentation mask
        segs = ann.get("segmentation", [])
        mask: np.ndarray | None = None
        if segs and isinstance(segs, list) and len(segs) > 0 and isinstance(segs[0], list):
            try:
                combined = np.zeros((h, w), dtype=bool)
                for poly in segs:
                    if len(poly) >= 6:
                        combined |= polygon_to_mask(poly, w, h)
                mask = combined
                area = float(mask.sum())
            except Exception:
                mask = None

        id_to_gts[img_id].append(GTObject(cls=cls, box_xyxy=box_xyxy, mask=mask, area=area))

    return id_to_path, id_to_gts


# ---------------------------------------------------------------------------
# Task auto-detection
# ---------------------------------------------------------------------------

def _detect_task(model, probe_image: Path, conf: float, device: str) -> str:
    """Run one inference to decide if the model returns masks (segmentation) or boxes only."""
    try:
        results = model(str(probe_image), conf=conf, device=device, verbose=False)
        if results and results[0].masks is not None and len(results[0].masks) > 0:
            return "segmentation"
    except Exception:
        pass
    return "detection"


# ---------------------------------------------------------------------------
# Prediction extraction
# ---------------------------------------------------------------------------

def _extract_preds(result, task: str, img_w: int, img_h: int) -> list[PredObject]:
    preds: list[PredObject] = []
    boxes = result.boxes
    if boxes is None:
        return preds

    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
    clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)

    masks_data = None
    if task == "segmentation" and result.masks is not None:
        raw = result.masks.data
        masks_data = raw.cpu().numpy() if hasattr(raw, "cpu") else np.array(raw)

    for i in range(len(confs)):
        box = tuple(float(v) for v in xyxy[i])
        mask: np.ndarray | None = None
        if masks_data is not None and i < len(masks_data):
            m = masks_data[i]
            # Resize mask to original image size if needed
            if m.shape != (img_h, img_w):
                import cv2
                m = cv2.resize(m.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            mask = m.astype(bool)
        preds.append(PredObject(
            conf=float(confs[i]),
            cls=int(clss[i]),
            box_xyxy=box,
            mask=mask,
        ))
    return preds


# ---------------------------------------------------------------------------
# Per-image evaluation
# ---------------------------------------------------------------------------

def _dominant_bucket(gts: list[GTObject]) -> str:
    if not gts:
        return "medium"
    from collections import Counter
    buckets = [size_bucket(gt.area) for gt in gts]
    return Counter(buckets).most_common(1)[0][0]


def _evaluate_image(
    image_id: int,
    image_path: Path,
    preds: list[PredObject],
    gts: list[GTObject],
    task: str,
    iou_threshold: float,
) -> ImageResult:
    if task == "segmentation":
        iou_fn = _mask_iou_for_pred_gt
    else:
        iou_fn = _box_iou_for_pred_gt

    match = greedy_match(preds, gts, iou_fn, iou_threshold)
    precision = _safe_div(match.tp, match.tp + match.fp)
    recall = _safe_div(match.tp, match.tp + match.fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    mean_iou = float(np.mean(match.matched_ious)) if match.matched_ious else 0.0
    dominant = _dominant_bucket(gts)

    return ImageResult(
        image_id=image_id,
        image_path=image_path,
        task=task,
        gt_count=len(gts),
        pred_count=len(preds),
        tp=match.tp,
        fp=match.fp,
        fn=match.fn,
        matched_ious=match.matched_ious,
        precision=precision,
        recall=recall,
        f1=f1,
        dominant_bucket=dominant,
        raw_preds=preds,
        raw_gts=gts,
        mean_iou=mean_iou,
    )


# ---------------------------------------------------------------------------
# Performance helpers
# ---------------------------------------------------------------------------

def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _evaluate_model_worker(
    weights: Path,
    all_images: list[Path],
    ann_format: str,
    annotations_file: "Path | None",
    validation_dir: Path,
    conf_threshold: float,
    iou_threshold: float,
    device: str,
    batch_size: int,
) -> "tuple[ModelMetrics | None, list[str]]":
    """Evaluate one model over all validation images.

    Returns (metrics, log_lines). Re-loads GT data internally so the
    function is safe to dispatch to a subprocess worker.
    """
    logs: list[str] = []
    log = logs.append

    # Re-load GT data inside the worker to avoid shipping large arrays
    # across process boundaries.
    if ann_format == "yolo":
        fname_to_gts = _load_yolo_data(validation_dir, all_images)
        id_to_path: dict[int, Path] = {}
        id_to_gts: dict[int, list[GTObject]] = {}
        fname_to_id: dict[str, int] = {}
    else:
        ann_file = annotations_file or _auto_detect_annotations(validation_dir)
        log(f"Loading annotations from {ann_file}")
        id_to_path, id_to_gts = _load_coco_data(ann_file)
        fname_to_gts = {}
        # O(1) reverse index: filename → image_id
        fname_to_id = {p.name: img_id for img_id, p in id_to_path.items()}

    log(f"\nLoading model: {weights.name}")
    try:
        from ultralytics import YOLO
        model = YOLO(str(weights))
    except Exception as exc:
        log(f"  [ERROR] Failed to load {weights}: {exc}")
        return None, logs

    task = "detection"
    probe = [p for p in all_images if p.exists()]
    if probe:
        log("  Detecting task type...")
        task = _detect_task(model, probe[0], conf_threshold, device)
    log(f"  Task: {task}")

    import time
    valid_images = [p for p in all_images if p.exists()]
    total = len(valid_images)
    per_image: list[ImageResult] = []

    total_inference_time = 0.0
    total_images_inferred = 0

    for batch_start in range(0, total, batch_size):
        batch_paths = valid_images[batch_start: batch_start + batch_size]
        batch_end = batch_start + len(batch_paths)
        if batch_end % 50 < batch_size or batch_end >= total:
            log(f"  Processing image {batch_end}/{total}...")

        # Read image dimensions from headers (lazy, fast — no pixel decoding)
        import PIL.Image as _PIL
        img_sizes: list[tuple[int, int]] = []
        for p in batch_paths:
            try:
                with _PIL.open(p) as im:
                    img_sizes.append(im.size)
            except Exception:
                img_sizes.append((640, 640))

        try:
            t0 = time.perf_counter()
            batch_results = model(
                [str(p) for p in batch_paths],
                conf=conf_threshold,
                device=device,
                verbose=False,
            )
            t1 = time.perf_counter()
            total_inference_time += (t1 - t0)
            total_images_inferred += len(batch_paths)
        except Exception as exc:
            log(f"  [WARN] Batch inference failed at offset {batch_start}: {exc}")
            batch_results = [None] * len(batch_paths)

        for img_path, result, (iw, ih) in zip(batch_paths, batch_results, img_sizes):
            if ann_format == "yolo":
                gts = fname_to_gts.get(img_path.name, [])
                img_id = hash(img_path.name) & 0x7FFFFFFF
            else:
                img_id = fname_to_id.get(img_path.name)
                gts = id_to_gts.get(img_id, []) if img_id is not None else []
                img_id = img_id or 0

            try:
                preds = _extract_preds(result, task, iw, ih) if result is not None else []
            except Exception as exc:
                log(f"  [WARN] Pred extraction failed for {img_path.name}: {exc}")
                preds = []

            per_image.append(
                _evaluate_image(img_id or 0, img_path, preds, gts, task, iou_threshold)
            )

    agg = aggregate_metrics(per_image, task)
    fps = total_images_inferred / total_inference_time if total_inference_time > 0 else 0.0
    inference_time_ms = (total_inference_time / total_images_inferred * 1000) if total_images_inferred > 0 else 0.0
    log(
        f"  mAP@50={agg['map50']:.3f}  mAP@50:95={agg['map50_95']:.3f}"
        f"  F1={agg['f1']:.3f}  P={agg['precision']:.3f}  R={agg['recall']:.3f}"
        f"  FPS={fps:.1f}  Latency={inference_time_ms:.1f}ms"
    )
    return ModelMetrics(
        weights_path=weights,
        task=task,
        precision=agg["precision"],
        recall=agg["recall"],
        f1=agg["f1"],
        map50=agg["map50"],
        map75=agg["map75"],
        map50_95=agg["map50_95"],
        small=agg["small"],
        medium=agg["medium"],
        large=agg["large"],
        per_image=per_image,
        inference_time_ms=inference_time_ms,
        fps=fps,
    ), logs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _resolve_workers(max_workers: int, n_models: int, is_gpu: bool, cpu_count: int) -> int:
    if max_workers != 0:
        return max(1, max_workers)
    if is_gpu:
        return 1
    return max(1, min(n_models, cpu_count // 2))


def run_benchmark(
    config: BenchmarkConfig,
    logger_fn: Callable[[str], None] = print,
) -> BenchmarkResult:
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing

    all_images = sorted(
        p for p in config.validation_dir.rglob("*") if p.suffix.lower() in _IMAGE_EXTS
    )

    ann_format = detect_annotation_format(config.validation_dir)
    if config.annotations_file is not None:
        ann_format = "coco"

    raw_device = config.device if config.device != "auto" else ""
    is_gpu = _has_gpu() and raw_device.lower() not in ("cpu",)

    workers = _resolve_workers(
        max_workers=config.max_workers,
        n_models=len(config.weights),
        is_gpu=is_gpu,
        cpu_count=os.cpu_count() or 2,
    )

    worker_kwargs = dict(
        all_images=all_images,
        ann_format=ann_format,
        annotations_file=config.annotations_file,
        validation_dir=config.validation_dir,
        conf_threshold=config.conf_threshold,
        iou_threshold=config.iou_threshold,
        device=raw_device,
        batch_size=config.batch_size,
    )

    model_results: list[ModelMetrics] = []

    if workers <= 1 or len(config.weights) <= 1:
        for weights in config.weights:
            result, logs = _evaluate_model_worker(weights=weights, **worker_kwargs)
            for line in logs:
                logger_fn(line)
            if result is not None:
                model_results.append(result)
    else:
        # spawn context avoids CUDA/fork-related issues in child processes
        mp_ctx = multiprocessing.get_context("spawn")
        ordered: dict[Path, ModelMetrics] = {}
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as executor:
            futures = {
                executor.submit(_evaluate_model_worker, weights=w, **worker_kwargs): w
                for w in config.weights
            }
            for future in as_completed(futures):
                w = futures[future]
                try:
                    result, logs = future.result()
                    for line in logs:
                        logger_fn(line)
                    if result is not None:
                        ordered[w] = result
                except Exception as exc:
                    logger_fn(f"  [ERROR] Worker for {w.name} failed: {exc}")
        # Restore original submission order
        for w in config.weights:
            if w in ordered:
                model_results.append(ordered[w])

    return BenchmarkResult(models=model_results, config=config)


def _find_image_id(img_path: Path, id_to_path: dict[int, Path]) -> int | None:
    name = img_path.name
    for img_id, p in id_to_path.items():
        if p.name == name or p == img_path:
            return img_id
    return None
