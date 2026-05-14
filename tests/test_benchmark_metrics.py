"""Tests for benchmark metric computation."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

from src.benchmark.metrics import (
    _safe_div,
    box_iou,
    compute_ap,
    compute_map_at_threshold,
    greedy_match,
    mask_iou,
    polygon_to_mask,
    size_bucket,
)
from src.benchmark.engine import GTObject, ImageResult, PredObject

_HAS_CV2 = importlib.util.find_spec("cv2") is not None



# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class TestMaskIoU(unittest.TestCase):
    def test_perfect_overlap(self):
        mask = np.ones((10, 10), dtype=bool)
        self.assertAlmostEqual(mask_iou(mask, mask), 1.0)

    def test_no_overlap(self):
        a = np.zeros((10, 10), dtype=bool)
        b = np.zeros((10, 10), dtype=bool)
        a[:5, :5] = True
        b[5:, 5:] = True
        self.assertAlmostEqual(mask_iou(a, b), 0.0)

    def test_partial_overlap(self):
        a = np.zeros((4, 4), dtype=bool)
        b = np.zeros((4, 4), dtype=bool)
        a[:, :2] = True   # left half
        b[:, 1:3] = True  # middle
        # intersection = col 1 (4 px), union = cols 0-2 (12 px)
        expected = 4 / 12
        self.assertAlmostEqual(mask_iou(a, b), expected, places=5)

    def test_empty_masks(self):
        a = np.zeros((5, 5), dtype=bool)
        b = np.zeros((5, 5), dtype=bool)
        self.assertAlmostEqual(mask_iou(a, b), 0.0)


class TestBoxIoU(unittest.TestCase):
    def test_perfect_overlap(self):
        box = (0.0, 0.0, 10.0, 10.0)
        self.assertAlmostEqual(box_iou(box, box), 1.0)

    def test_no_overlap(self):
        self.assertAlmostEqual(box_iou((0, 0, 5, 5), (6, 6, 10, 10)), 0.0)

    def test_partial_overlap(self):
        # Two 10x10 boxes offset by 5 → intersection 5x10=50, union 150
        iou = box_iou((0, 0, 10, 10), (5, 0, 15, 10))
        self.assertAlmostEqual(iou, 50 / 150, places=5)

    def test_zero_area_box(self):
        self.assertAlmostEqual(box_iou((0, 0, 0, 0), (0, 0, 5, 5)), 0.0)


@unittest.skipUnless(_HAS_CV2, "cv2 not available")
class TestPolygonToMask(unittest.TestCase):
    def test_full_square(self):
        poly = [0, 0, 10, 0, 10, 10, 0, 10]
        mask = polygon_to_mask(poly, 10, 10)
        self.assertEqual(mask.dtype, bool)
        self.assertGreater(mask.sum(), 0)

    def test_output_shape(self):
        poly = [0, 0, 5, 0, 5, 5, 0, 5]
        mask = polygon_to_mask(poly, 20, 15)
        self.assertEqual(mask.shape, (15, 20))


# ---------------------------------------------------------------------------
# Greedy matching — segmentation
# ---------------------------------------------------------------------------

def _seg_pred(conf, mask):
    return PredObject(conf=conf, cls=0, box_xyxy=(0, 0, 1, 1), mask=mask)


def _seg_gt(mask):
    return GTObject(cls=0, box_xyxy=(0, 0, 1, 1), mask=mask, area=float(mask.sum()))


def _iou_fn_seg(pred, gt):
    if pred.mask is None or gt.mask is None:
        return 0.0
    return mask_iou(pred.mask, gt.mask)


class TestGreedyMatchSegmentation(unittest.TestCase):
    def _make_mask(self, rows, cols, shape=(10, 10)):
        m = np.zeros(shape, dtype=bool)
        m[rows[0]:rows[1], cols[0]:cols[1]] = True
        return m

    def test_two_match(self):
        m1 = self._make_mask((0, 5), (0, 5))
        m2 = self._make_mask((5, 10), (5, 10))
        preds = [_seg_pred(0.9, m1), _seg_pred(0.8, m2)]
        gts = [_seg_gt(m1), _seg_gt(m2)]
        r = greedy_match(preds, gts, _iou_fn_seg, iou_threshold=0.5)
        self.assertEqual(r.tp, 2)
        self.assertEqual(r.fp, 0)
        self.assertEqual(r.fn, 0)

    def test_extra_pred_is_fp(self):
        m1 = self._make_mask((0, 5), (0, 5))
        m2 = self._make_mask((0, 3), (0, 3))  # overlaps m1 heavily
        preds = [_seg_pred(0.9, m1), _seg_pred(0.5, m2)]
        gts = [_seg_gt(m1)]
        r = greedy_match(preds, gts, _iou_fn_seg, iou_threshold=0.5)
        self.assertEqual(r.tp, 1)
        self.assertEqual(r.fp, 1)
        self.assertEqual(r.fn, 0)

    def test_unmatched_gt_is_fn(self):
        m1 = self._make_mask((0, 5), (0, 5))
        m2 = self._make_mask((5, 10), (5, 10))
        preds = [_seg_pred(0.9, m1)]
        gts = [_seg_gt(m1), _seg_gt(m2)]
        r = greedy_match(preds, gts, _iou_fn_seg, iou_threshold=0.5)
        self.assertEqual(r.tp, 1)
        self.assertEqual(r.fp, 0)
        self.assertEqual(r.fn, 1)

    def test_empty_preds(self):
        m = self._make_mask((0, 5), (0, 5))
        r = greedy_match([], [_seg_gt(m)], _iou_fn_seg)
        self.assertEqual(r.fn, 1)
        self.assertEqual(r.tp, 0)

    def test_empty_gts(self):
        m = self._make_mask((0, 5), (0, 5))
        r = greedy_match([_seg_pred(0.9, m)], [], _iou_fn_seg)
        self.assertEqual(r.fp, 1)
        self.assertEqual(r.tp, 0)


# ---------------------------------------------------------------------------
# Greedy matching — detection (boxes)
# ---------------------------------------------------------------------------

def _det_pred(conf, box):
    return PredObject(conf=conf, cls=0, box_xyxy=box, mask=None)


def _det_gt(box):
    area = (box[2] - box[0]) * (box[3] - box[1])
    return GTObject(cls=0, box_xyxy=box, mask=None, area=float(area))


def _iou_fn_det(pred, gt):
    return box_iou(pred.box_xyxy, gt.box_xyxy)


class TestGreedyMatchDetection(unittest.TestCase):
    def test_two_match(self):
        preds = [_det_pred(0.9, (0, 0, 5, 5)), _det_pred(0.8, (6, 6, 10, 10))]
        gts = [_det_gt((0, 0, 5, 5)), _det_gt((6, 6, 10, 10))]
        r = greedy_match(preds, gts, _iou_fn_det, iou_threshold=0.5)
        self.assertEqual(r.tp, 2)
        self.assertEqual(r.fp, 0)
        self.assertEqual(r.fn, 0)

    def test_below_threshold_is_fp(self):
        # pred barely overlaps gt
        preds = [_det_pred(0.9, (0, 0, 10, 1))]
        gts = [_det_gt((0, 0, 10, 10))]
        r = greedy_match(preds, gts, _iou_fn_det, iou_threshold=0.5)
        self.assertEqual(r.fp, 1)
        self.assertEqual(r.fn, 1)


# ---------------------------------------------------------------------------
# AP computation
# ---------------------------------------------------------------------------

class TestComputeAP(unittest.TestCase):
    def test_perfect_curve(self):
        precisions = [1.0] * 101
        recalls = [i / 100 for i in range(101)]
        self.assertAlmostEqual(compute_ap(precisions, recalls), 1.0, places=5)

    def test_zero_curve(self):
        ap = compute_ap([0.0] * 10, [0.1 * i for i in range(10)])
        self.assertAlmostEqual(ap, 0.0, places=5)

    def test_map_uses_actual_confidence_ordered_matches(self):
        gt = _det_gt((0, 0, 10, 10))
        high_conf_fp = _det_pred(0.95, (20, 20, 30, 30))
        low_conf_tp = _det_pred(0.50, (0, 0, 10, 10))
        image_result = ImageResult(
            image_id=1,
            image_path=Path("image.jpg"),
            task="detection",
            gt_count=1,
            pred_count=2,
            tp=1,
            fp=1,
            fn=0,
            matched_ious=[1.0],
            precision=0.5,
            recall=1.0,
            f1=2 / 3,
            raw_preds=[high_conf_fp, low_conf_tp],
            raw_gts=[gt],
            mean_iou=1.0,
        )

        ap = compute_map_at_threshold([image_result], 0.5, "detection")

        self.assertAlmostEqual(ap, 0.5, places=5)


# ---------------------------------------------------------------------------
# Size bucketing
# ---------------------------------------------------------------------------

class TestSizeBucket(unittest.TestCase):
    def test_small(self):
        self.assertEqual(size_bucket(0), "small")
        self.assertEqual(size_bucket(1023), "small")

    def test_medium_boundary(self):
        self.assertEqual(size_bucket(1024), "medium")
        self.assertEqual(size_bucket(9215), "medium")

    def test_large_boundary(self):
        self.assertEqual(size_bucket(9216), "large")
        self.assertEqual(size_bucket(100000), "large")


# ---------------------------------------------------------------------------
# _safe_div
# ---------------------------------------------------------------------------

class TestSafeDiv(unittest.TestCase):
    def test_normal(self):
        self.assertAlmostEqual(_safe_div(1, 2), 0.5)

    def test_zero_denominator(self):
        self.assertAlmostEqual(_safe_div(5, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
