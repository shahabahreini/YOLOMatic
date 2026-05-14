"""Tests for the benchmark evaluation engine."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.benchmark.engine import (
    GTObject,
    PredObject,
    _auto_detect_annotations,
    _dominant_bucket,
    _evaluate_image,
    _find_image_id,
    _load_coco_data,
)

_HAS_CV2 = importlib.util.find_spec("cv2") is not None



def _write_coco(path: Path, images, annotations, categories=None):
    cats = categories or [{"id": 0, "name": "object", "supercategory": "none"}]
    coco = {"images": images, "annotations": annotations, "categories": cats}
    path.write_text(json.dumps(coco))


class TestAutoDetectAnnotations(unittest.TestCase):
    def test_finds_standard_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            ann = d / "_annotations.coco.json"
            ann.write_text("{}")
            result = _auto_detect_annotations(d)
            self.assertEqual(result, ann)

    def test_falls_back_to_any_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            j = d / "labels.json"
            j.write_text("{}")
            result = _auto_detect_annotations(d)
            self.assertEqual(result, j)

    def test_raises_when_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                _auto_detect_annotations(Path(tmp))


@unittest.skipUnless(_HAS_CV2, "cv2 not available")
class TestLoadCocoDataSegmentation(unittest.TestCase):
    def test_loads_polygon_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            # Create a dummy image file
            img_path = d / "img1.jpg"
            img_path.write_bytes(b"")

            ann_file = d / "_annotations.coco.json"
            images = [{"id": 1, "file_name": "img1.jpg", "width": 20, "height": 20}]
            annotations = [{
                "id": 1, "image_id": 1, "category_id": 0, "area": 25.0,
                "bbox": [0, 0, 5, 5],
                "segmentation": [[0, 0, 5, 0, 5, 5, 0, 5]],
            }]
            _write_coco(ann_file, images, annotations)

            id_to_path, id_to_gts = _load_coco_data(ann_file)
            self.assertIn(1, id_to_gts)
            gt = id_to_gts[1][0]
            self.assertIsNotNone(gt.mask)
            self.assertEqual(gt.mask.shape, (20, 20))

    def test_loads_bbox_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            ann_file = d / "_annotations.coco.json"
            images = [{"id": 2, "file_name": "img2.jpg", "width": 50, "height": 50}]
            annotations = [{
                "id": 2, "image_id": 2, "category_id": 0, "area": 100.0,
                "bbox": [5, 5, 10, 10],
                "segmentation": [],
            }]
            _write_coco(ann_file, images, annotations)

            _, id_to_gts = _load_coco_data(ann_file)
            gt = id_to_gts[2][0]
            self.assertIsNone(gt.mask)
            self.assertAlmostEqual(gt.box_xyxy[2], 15.0)  # x2 = x + w


class TestFindImageId(unittest.TestCase):
    def test_match_by_name(self):
        id_to_path = {1: Path("/some/dir/img.jpg"), 2: Path("/other/dir/foo.png")}
        self.assertEqual(_find_image_id(Path("/different/img.jpg"), id_to_path), 1)

    def test_no_match_returns_none(self):
        id_to_path = {1: Path("/a/b.jpg")}
        self.assertIsNone(_find_image_id(Path("/a/c.jpg"), id_to_path))


class TestEvaluateImage(unittest.TestCase):
    def _mask(self, r0, r1, c0, c1, shape=(10, 10)):
        m = np.zeros(shape, dtype=bool)
        m[r0:r1, c0:c1] = True
        return m

    def test_perfect_segmentation(self):
        m = self._mask(0, 5, 0, 5)
        preds = [PredObject(conf=0.9, cls=0, box_xyxy=(0, 0, 5, 5), mask=m)]
        gts = [GTObject(cls=0, box_xyxy=(0, 0, 5, 5), mask=m, area=25.0)]
        result = _evaluate_image(1, Path("img.jpg"), preds, gts, "segmentation", 0.5)
        self.assertEqual(result.tp, 1)
        self.assertEqual(result.fp, 0)
        self.assertEqual(result.fn, 0)
        self.assertAlmostEqual(result.f1, 1.0)

    def test_no_predictions(self):
        m = self._mask(0, 5, 0, 5)
        gts = [GTObject(cls=0, box_xyxy=(0, 0, 5, 5), mask=m, area=25.0)]
        result = _evaluate_image(1, Path("img.jpg"), [], gts, "segmentation", 0.5)
        self.assertEqual(result.fn, 1)
        self.assertAlmostEqual(result.f1, 0.0)

    def test_detection_mode(self):
        preds = [PredObject(conf=0.9, cls=0, box_xyxy=(0, 0, 10, 10), mask=None)]
        gts = [GTObject(cls=0, box_xyxy=(0, 0, 10, 10), mask=None, area=100.0)]
        result = _evaluate_image(1, Path("img.jpg"), preds, gts, "detection", 0.5)
        self.assertEqual(result.tp, 1)
        self.assertAlmostEqual(result.f1, 1.0)


class TestDominantBucket(unittest.TestCase):
    def test_small_objects(self):
        gts = [GTObject(0, (0,0,1,1), None, area=100.0)] * 3
        self.assertEqual(_dominant_bucket(gts), "small")

    def test_large_objects(self):
        gts = [GTObject(0, (0,0,100,100), None, area=20000.0)] * 2
        self.assertEqual(_dominant_bucket(gts), "large")

    def test_empty(self):
        self.assertEqual(_dominant_bucket([]), "medium")


class TestEvaluateModelWorker(unittest.TestCase):
    def test_logs_model_name_on_load_failure(self):
        """Worker must log 'Loading model: <name>' even when YOLO load fails."""
        from src.benchmark.engine import _evaluate_model_worker
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            fake_weights = Path(tmp) / "my_model.pt"
            fake_weights.touch()
            _, logs = _evaluate_model_worker(
                weights=fake_weights,
                all_images=[],
                ann_format="yolo",
                annotations_file=None,
                validation_dir=Path(tmp),
                conf_threshold=0.25,
                iou_threshold=0.5,
                device="cpu",
                batch_size=16,
            )
        self.assertTrue(any("my_model.pt" in line for line in logs))


class TestResolveWorkers(unittest.TestCase):
    def test_gpu_always_one(self):
        from src.benchmark.engine import _resolve_workers
        self.assertEqual(_resolve_workers(max_workers=0, n_models=4, is_gpu=True, cpu_count=8), 1)

    def test_cpu_scales_with_models(self):
        from src.benchmark.engine import _resolve_workers
        w = _resolve_workers(max_workers=0, n_models=3, is_gpu=False, cpu_count=4)
        self.assertGreaterEqual(w, 2)
        self.assertLessEqual(w, 3)

    def test_explicit_override(self):
        from src.benchmark.engine import _resolve_workers
        self.assertEqual(_resolve_workers(max_workers=3, n_models=10, is_gpu=True, cpu_count=2), 3)

    def test_min_one(self):
        from src.benchmark.engine import _resolve_workers
        self.assertEqual(_resolve_workers(max_workers=0, n_models=1, is_gpu=False, cpu_count=1), 1)


class TestRunBenchmarkSequential(unittest.TestCase):
    def test_sequential_path_single_weight(self):
        from unittest.mock import patch
        from src.benchmark.config import BenchmarkConfig
        from src.benchmark.engine import run_benchmark, ModelMetrics, BenchmarkResult
        from src.benchmark.metrics import SizeBucketMetrics
        import tempfile

        called: list[Path] = []

        def fake_worker(weights, **_kwargs):
            called.append(weights)
            mm = ModelMetrics(
                weights_path=weights, task="detection",
                precision=1.0, recall=1.0, f1=1.0,
                map50=1.0, map75=1.0, map50_95=1.0,
                small=SizeBucketMetrics("small"),
                medium=SizeBucketMetrics("medium"),
                large=SizeBucketMetrics("large"),
                per_image=[],
            )
            return mm, [f"\nLoading model: {weights.name}",
                        "  mAP@50=1.000  mAP@50:95=1.000  F1=1.000  P=1.000  R=1.000"]

        with tempfile.TemporaryDirectory() as tmp:
            fake_pt = Path(tmp) / "model.pt"
            fake_pt.touch()
            cfg = BenchmarkConfig(weights=[fake_pt], validation_dir=Path(tmp))
            with patch("src.benchmark.engine._evaluate_model_worker", side_effect=fake_worker), \
                 patch("src.benchmark.engine.detect_annotation_format", return_value="yolo"):
                result = run_benchmark(cfg)

        self.assertEqual(len(called), 1)
        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(len(result.models), 1)


class TestBenchmarkConfigNewFields(unittest.TestCase):
    def test_defaults(self):
        from src.benchmark.config import BenchmarkConfig
        cfg = BenchmarkConfig(weights=[], validation_dir=Path("."))
        self.assertEqual(cfg.batch_size, 16)
        self.assertEqual(cfg.max_workers, 0)

    def test_custom(self):
        from src.benchmark.config import BenchmarkConfig
        cfg = BenchmarkConfig(weights=[], validation_dir=Path("."), batch_size=8, max_workers=2)
        self.assertEqual(cfg.batch_size, 8)
        self.assertEqual(cfg.max_workers, 2)


if __name__ == "__main__":
    unittest.main()
