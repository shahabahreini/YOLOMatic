"""Tests for HTML report generation."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


from src.benchmark.config import BenchmarkConfig
from src.benchmark.engine import BenchmarkResult, ImageResult, ModelMetrics
from src.benchmark.metrics import SizeBucketMetrics
from src.benchmark.report import write_benchmark_report


def _make_image_result(image_id: int, f1: float = 0.8, task: str = "detection") -> ImageResult:
    return ImageResult(
        image_id=image_id,
        image_path=Path(f"img{image_id}.jpg"),
        task=task,
        gt_count=2,
        pred_count=2,
        tp=1, fp=1, fn=1,
        matched_ious=[0.75],
        precision=0.5,
        recall=0.5,
        f1=f1,
        dominant_bucket="medium",
        raw_preds=[],
        raw_gts=[],
        mean_iou=0.75,
    )


def _make_model_metrics(name: str = "best_model", task: str = "detection") -> ModelMetrics:
    per_image = [_make_image_result(i, f1=0.5 + 0.1 * i, task=task) for i in range(5)]
    bucket = SizeBucketMetrics(name="medium", count=5, precision=0.8, recall=0.7, f1=0.75, map50=0.6)
    return ModelMetrics(
        weights_path=Path(f"runs/{name}.pt"),
        task=task,
        precision=0.8,
        recall=0.7,
        f1=0.75,
        map50=0.65,
        map75=0.45,
        map50_95=0.40,
        small=SizeBucketMetrics("small", 0, 0.0, 0.0, 0.0, 0.0),
        medium=bucket,
        large=SizeBucketMetrics("large", 0, 0.0, 0.0, 0.0, 0.0),
        per_image=per_image,
    )


def _make_model_metrics_for_path(path: Path, task: str = "detection") -> ModelMetrics:
    model = _make_model_metrics(path.stem, task)
    model.weights_path = path
    return model


def _make_result(tasks=("detection",)) -> BenchmarkResult:
    with tempfile.TemporaryDirectory() as tmp:
        val_dir = Path(tmp)
    config = BenchmarkConfig(
        weights=[Path(f"model_{t}.pt") for t in tasks],
        validation_dir=val_dir,
        generate_thumbnails=False,
    )
    models = [_make_model_metrics(f"model_{t}", task=t) for t in tasks]
    return BenchmarkResult(models=models, config=config)


class TestWriteReport(unittest.TestCase):
    def test_creates_html_file(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_benchmark_report(result, Path(tmp))
            self.assertTrue(path.exists())
            self.assertEqual(path.suffix, ".html")

    def test_rejects_empty_model_results(self):
        result = BenchmarkResult(
            models=[],
            config=BenchmarkConfig(weights=[], validation_dir=Path("valid")),
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                ValueError,
                "no model metrics were produced",
            ):
                write_benchmark_report(result, Path(tmp))

    def test_report_handles_model_with_no_per_image_rows(self):
        model = _make_model_metrics()
        model.per_image = []
        result = BenchmarkResult(
            models=[model],
            config=BenchmarkConfig(
                weights=[model.weights_path],
                validation_dir=Path("valid"),
                generate_thumbnails=False,
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_benchmark_report(result, Path(tmp))
            content = path.read_text()

        self.assertIn("No per-image results were available", content)

    def test_report_contains_plotly(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_benchmark_report(result, Path(tmp))
            content = path.read_text()
            self.assertIn("plotly", content.lower())

    def test_report_sections_present(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_benchmark_report(result, Path(tmp))
            content = path.read_text()
            # Check structural markers
            self.assertIn("scatter-plot", content)
            self.assertIn("gallery-panel", content)
            self.assertIn("YOLOMatic", content)

    def test_multi_model_report(self):
        result = _make_result(tasks=("detection", "segmentation"))
        with tempfile.TemporaryDirectory() as tmp:
            path = write_benchmark_report(result, Path(tmp))
            content = path.read_text()
            self.assertIn("model_detection", content)
            self.assertIn("model_segmentation", content)

    def test_best_and_last_weights_use_distinguishable_names(self):
        config = BenchmarkConfig(
            weights=[
                Path("runs/detect/train_a/weights/best.pt"),
                Path("runs/detect/train_a/weights/last.pt"),
            ],
            validation_dir=Path("valid"),
            generate_thumbnails=False,
        )
        result = BenchmarkResult(
            models=[
                _make_model_metrics_for_path(Path("runs/detect/train_a/weights/best.pt")),
                _make_model_metrics_for_path(Path("runs/detect/train_a/weights/last.pt")),
            ],
            config=config,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_benchmark_report(result, Path(tmp))
            content = path.read_text()

        # Plotly JSON-encodes "/" as \u002f in chart data, so check for
        # either the literal form (used in the header bar) or the escaped
        # form (used inside Plotly traces).
        self.assertTrue(
            "train_a / best.pt" in content or "train_a \\u002f best.pt" in content,
            "Expected distinguishable name for best.pt in report",
        )
        self.assertTrue(
            "train_a / last.pt" in content or "train_a \\u002f last.pt" in content,
            "Expected distinguishable name for last.pt in report",
        )

    def test_report_is_self_contained(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_benchmark_report(result, Path(tmp))
            content = path.read_text()
            # gallery JS is inline, not an external src
            self.assertIn("scatter.on", content)


if __name__ == "__main__":
    unittest.main()
