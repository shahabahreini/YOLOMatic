"""Tests for dataset summary accuracy: split discovery, pair-driven label
counting, task detection, and sample-limit behavior."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.datasets.core import summarize_dataset


class DatasetSummaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        # Keep summary caches inside the temp dir, not the real project.
        self.patcher = patch("src.datasets.core.project_root", return_value=self.tmp)
        self.patcher.start()

    def tearDown(self) -> None:
        self.patcher.stop()
        shutil.rmtree(self.tmp)

    def _write_image(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\xff\xd8\xff\xd9")  # tiny JPEG stub

    def _write_label(self, path: Path, row: str = "0 0.5 0.5 0.25 0.25\n") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(row, encoding="utf-8")

    def test_nested_labels_counted_recursively(self) -> None:
        root = self.tmp / "nested"
        root.mkdir()
        (root / "data.yaml").write_text(
            "train: train/images\nnc: 1\nnames: [cat]\n", encoding="utf-8"
        )
        self._write_image(root / "train" / "images" / "sub" / "a.jpg")
        self._write_label(root / "train" / "labels" / "sub" / "a.txt")

        summary = summarize_dataset(root)

        split = summary.splits["train"]
        self.assertEqual(split.image_count, 1)
        self.assertEqual(split.labeled_image_count, 1)
        self.assertEqual(split.unlabeled_image_count, 0)
        self.assertEqual(split.annotation_count, 1)

    def test_disk_only_split_included_in_summary_and_rollup(self) -> None:
        root = self.tmp / "diskonly"
        root.mkdir()
        (root / "data.yaml").write_text(
            "train: train/images\nnc: 1\nnames: [cat]\n", encoding="utf-8"
        )
        self._write_image(root / "train" / "images" / "a.jpg")
        self._write_label(root / "train" / "labels" / "a.txt")
        for name in ("b.jpg", "c.jpg"):
            self._write_image(root / "test" / "images" / name)
            self._write_label(root / "test" / "labels" / Path(name).with_suffix(".txt").name)

        summary = summarize_dataset(root)

        self.assertEqual(summary.splits["test"].image_count, 2)
        self.assertEqual(summary.image_count, 3)
        self.assertTrue(any("not referenced" in w for w in summary.warnings))

    def test_summary_without_data_yaml(self) -> None:
        root = self.tmp / "noyaml"
        self._write_image(root / "train" / "images" / "a.jpg")
        self._write_label(root / "train" / "labels" / "a.txt")
        self._write_image(root / "valid" / "images" / "b.jpg")
        self._write_label(root / "valid" / "labels" / "b.txt")

        summary = summarize_dataset(root)

        self.assertEqual(summary.format, "yolo")
        self.assertEqual(summary.image_count, 2)
        self.assertEqual(summary.splits["train"].image_count, 1)
        self.assertEqual(summary.splits["val"].image_count, 1)

    def test_task_key_is_authoritative(self) -> None:
        root = self.tmp / "tasked"
        root.mkdir()
        # Detection-shaped rows, but the yaml declares segmentation.
        (root / "data.yaml").write_text(
            "train: train/images\nnc: 1\nnames: [cat]\ntask: segment\n",
            encoding="utf-8",
        )
        self._write_image(root / "train" / "images" / "a.jpg")
        self._write_label(root / "train" / "labels" / "a.txt")

        summary = summarize_dataset(root)

        self.assertEqual(summary.task, "segmentation")

    def test_sample_limit_keeps_counts_honest_and_warns(self) -> None:
        root = self.tmp / "sampled"
        root.mkdir()
        (root / "data.yaml").write_text(
            "train: train/images\nnc: 1\nnames: [cat]\n", encoding="utf-8"
        )
        for idx in range(5):
            self._write_image(root / "train" / "images" / f"img_{idx}.jpg")
            self._write_label(root / "train" / "labels" / f"img_{idx}.txt")

        summary = summarize_dataset(root, sample_limit=3)

        # Full image count and labeled/unlabeled counts stay accurate even when labels are sampled.
        self.assertEqual(split.image_count, 5)
        self.assertEqual(split.labeled_image_count + split.unlabeled_image_count, 5)
        self.assertEqual(split.annotation_count, 5)
        self.assertTrue(any("sampled" in w for w in summary.warnings))


if __name__ == "__main__":
    unittest.main()
