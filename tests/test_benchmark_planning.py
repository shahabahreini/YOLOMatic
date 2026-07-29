"""Tests for task-aware benchmark planning and dense semantic metrics."""
from __future__ import annotations

import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from src.benchmark.metrics import semantic_metrics
from src.benchmark.planning import (
    BenchmarkCompatibilityError,
    resolve_dataset_splits,
    resolve_selected_model_task,
)


def _dataset(root: Path, task: str = "detection") -> None:
    for split in ("train", "val", "test"):
        images = root / "images" / split
        labels = root / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        (images / f"{split}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        label = "0 0.5 0.5 0.2 0.2\n"
        if task == "semantic":
            label = "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"
        (labels / f"{split}.txt").write_text(label, encoding="utf-8")
    (root / "data.yaml").write_text(
        textwrap.dedent(
            f"""\
            path: {root}
            train: images/train
            val: images/val
            test: images/test
            names: [object]
            task: {task}
            """
        ),
        encoding="utf-8",
    )


class DatasetSplitPlanningTests(unittest.TestCase):
    def test_all_normalizes_val_and_returns_every_split(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _dataset(root)
            splits = resolve_dataset_splits(root, "detection", "all")
        self.assertEqual([split.name for split in splits], ["train", "valid", "test"])

    def test_rejects_incompatible_task(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _dataset(root, "semantic")
            with self.assertRaisesRegex(BenchmarkCompatibilityError, "semantic"):
                resolve_dataset_splits(root, "detection", "valid")


class ModelTaskPlanningTests(unittest.TestCase):
    def test_rejects_mixed_verified_tasks(self) -> None:
        weights = [Path("detect.pt"), Path("segment.pt")]
        with patch(
            "src.benchmark.planning.resolve_ultralytics_model_task",
            side_effect=["detection", "segmentation"],
        ):
            with self.assertRaisesRegex(BenchmarkCompatibilityError, "different tasks"):
                resolve_selected_model_task(weights)


class SemanticMetricsTests(unittest.TestCase):
    def test_perfect_prediction_scores_one(self) -> None:
        target = np.array([[0, 1], [1, 0]], dtype=np.int32)
        metrics = semantic_metrics([target], [target], [0, 1])
        self.assertEqual(metrics["miou"], 1.0)
        self.assertEqual(metrics["pixel_accuracy"], 1.0)
        self.assertEqual(metrics["dice"], 1.0)

    def test_ignored_pixels_do_not_affect_accuracy(self) -> None:
        prediction = np.array([[0, 1]], dtype=np.int32)
        target = np.array([[0, -1]], dtype=np.int32)
        metrics = semantic_metrics([prediction], [target], [0, 1])
        self.assertEqual(metrics["pixel_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
