"""Pure metric computation — no ML framework imports."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def polygon_to_mask(polygon: list[float], width: int, height: int) -> np.ndarray:
    """Rasterise a COCO flat polygon [x1,y1,x2,y2,...] to a binary mask."""
    import cv2  # only import when called; cv2 is always available via ultralytics

    pts = np.array(polygon, dtype=np.float32).reshape(-1, 2).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(intersection / union) if union > 0 else 0.0


def box_iou(pred: tuple[float, float, float, float],
            gt: tuple[float, float, float, float]) -> float:
    """Axis-aligned bounding box IoU. Boxes are (x1, y1, x2, y2)."""
    px1, py1, px2, py2 = pred
    gx1, gy1, gx2, gy2 = gt
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    gt_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = pred_area + gt_area - inter
    return float(inter / union) if union > 0 else 0.0


def mask_to_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return (0, 0, 0, 0)
    y1, y2 = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    x1, x2 = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
    return (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Greedy matching
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    matched_ious: list[float] = field(default_factory=list)
    prediction_matches: list[bool] = field(default_factory=list)


def greedy_match(
    preds: list,           # list of objects with .conf and .(mask|box_xyxy)
    gts: list,             # list of GT objects with .(mask|box_xyxy)
    iou_fn: Callable,      # mask_iou or box_iou — takes (pred_obj, gt_obj) → float
    iou_threshold: float = 0.5,
) -> MatchResult:
    if not preds and not gts:
        return MatchResult()
    if not preds:
        return MatchResult(fn=len(gts))
    if not gts:
        return MatchResult(fp=len(preds))

    sorted_preds = sorted(preds, key=lambda p: p.conf, reverse=True)
    matched_gt = set()
    tp, fp = 0, 0
    matched_ious: list[float] = []
    prediction_matches: list[bool] = []

    for pred in sorted_preds:
        best_iou, best_idx = 0.0, -1
        for gi, gt in enumerate(gts):
            if gi in matched_gt:
                continue
            iou = iou_fn(pred, gt)
            if iou > best_iou:
                best_iou, best_idx = iou, gi
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_idx)
            matched_ious.append(best_iou)
            prediction_matches.append(True)
        else:
            fp += 1
            prediction_matches.append(False)

    fn = len(gts) - len(matched_gt)
    return MatchResult(
        tp=tp,
        fp=fp,
        fn=fn,
        matched_ious=matched_ious,
        prediction_matches=prediction_matches,
    )


# ---------------------------------------------------------------------------
# AP / mAP
# ---------------------------------------------------------------------------

def compute_ap(precisions: list[float], recalls: list[float]) -> float:
    """101-point interpolated AP."""
    p = np.array(precisions, dtype=float)
    r = np.array(recalls, dtype=float)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        mask = r >= t
        ap += (p[mask].max() if mask.any() else 0.0)
    return float(ap / 101)


def _build_pr_curve(
    per_image_preds: list[list],   # per image: list of (conf, is_tp) tuples
    total_gt: int,
) -> tuple[list[float], list[float]]:
    flat = sorted(
        [(conf, is_tp) for img in per_image_preds for conf, is_tp in img],
        key=lambda x: x[0],
        reverse=True,
    )
    tp_cum, fp_cum = 0, 0
    precisions, recalls = [], []
    for _, is_tp in flat:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / total_gt if total_gt > 0 else 0.0)
    return precisions, recalls


def compute_map_at_threshold(
    per_image_results: list,   # list[ImageResult]
    iou_threshold: float,
    task: str,
) -> float:

    total_gt = sum(r.gt_count for r in per_image_results)
    if total_gt == 0:
        return 0.0

    per_image_preds: list[list] = []
    for result in per_image_results:
        img_preds: list[tuple[float, bool]] = []
        # Re-run matching at this threshold using stored predictions
        if task == "segmentation":
            iou_fn = _mask_iou_for_pred_gt
        else:
            iou_fn = _box_iou_for_pred_gt
        match = greedy_match(result.raw_preds, result.raw_gts, iou_fn, iou_threshold)
        sorted_preds = sorted(result.raw_preds, key=lambda p: p.conf, reverse=True)
        for pred, is_tp in zip(sorted_preds, match.prediction_matches):
            img_preds.append((pred.conf, is_tp))
        per_image_preds.append(img_preds)

    precisions, recalls = _build_pr_curve(per_image_preds, total_gt)
    return compute_ap(precisions, recalls)


def _mask_iou_for_pred_gt(pred, gt) -> float:
    if pred.mask is None or gt.mask is None:
        return 0.0
    return mask_iou(pred.mask, gt.mask)


def _box_iou_for_pred_gt(pred, gt) -> float:
    return box_iou(pred.box_xyxy, gt.box_xyxy)


def compute_map_range(
    per_image_results: list,
    thresholds: list[float],
    task: str,
) -> float:
    aps = [compute_map_at_threshold(per_image_results, t, task) for t in thresholds]
    return float(np.mean(aps)) if aps else 0.0


# ---------------------------------------------------------------------------
# Object size bucketing
# ---------------------------------------------------------------------------

def size_bucket(area_px: float) -> str:
    if area_px < 1024:
        return "small"
    if area_px < 9216:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

@dataclass
class SizeBucketMetrics:
    name: str
    count: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    map50: float = 0.0
    map50_95: float = 0.0


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def aggregate_metrics(per_image_results: list, task: str) -> dict:
    """Returns dict with global + per-bucket metrics."""
    total_tp = sum(r.tp for r in per_image_results)
    total_fp = sum(r.fp for r in per_image_results)
    total_fn = sum(r.fn for r in per_image_results)

    precision = _safe_div(total_tp, total_tp + total_fp)
    recall = _safe_div(total_tp, total_tp + total_fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    thresholds_50_95 = [round(t, 2) for t in np.arange(0.50, 1.00, 0.05).tolist()]
    map50 = compute_map_at_threshold(per_image_results, 0.50, task)
    map75 = compute_map_at_threshold(per_image_results, 0.75, task)
    map50_95 = compute_map_range(per_image_results, thresholds_50_95, task)

    bucket_metrics: dict[str, SizeBucketMetrics] = {
        "small": SizeBucketMetrics("small"),
        "medium": SizeBucketMetrics("medium"),
        "large": SizeBucketMetrics("large"),
    }

    for bname in ("small", "medium", "large"):
        bucket_results = [r for r in per_image_results if _result_in_bucket(r, bname)]
        if not bucket_results:
            continue
        btp = sum(r.tp for r in bucket_results)
        bfp = sum(r.fp for r in bucket_results)
        bfn = sum(r.fn for r in bucket_results)
        bp = _safe_div(btp, btp + bfp)
        br = _safe_div(btp, btp + bfn)
        bf = _safe_div(2 * bp * br, bp + br)
        bm50 = compute_map_at_threshold(bucket_results, 0.50, task)
        bm50_95 = compute_map_range(bucket_results, thresholds_50_95, task)
        bucket_metrics[bname] = SizeBucketMetrics(
            name=bname,
            count=sum(r.gt_count for r in bucket_results),
            precision=bp, recall=br, f1=bf, map50=bm50,
            map50_95=bm50_95,
        )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map75": map75,
        "map50_95": map50_95,
        "small": bucket_metrics["small"],
        "medium": bucket_metrics["medium"],
        "large": bucket_metrics["large"],
    }


def _result_in_bucket(result, bucket_name: str) -> bool:
    """Check if an ImageResult has GT objects in the given size bucket.
    For simplicity, assigns an image to a bucket if its dominant GT size class matches.
    """
    # Use the bucket stored on the result if available
    return getattr(result, "dominant_bucket", "medium") == bucket_name
