"""Tests for dataset summary accuracy: split discovery, pair-driven label
counting, task detection, and sample-limit behavior."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yolomatic.datasets.core import summarize_dataset


class DatasetSummaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        # Keep summary caches inside the temp dir, not the real project.
        self.patcher = patch("yolomatic.datasets.core.project_root", return_value=self.tmp)
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

        split = summary.splits["train"]
        # Full image count and labeled/unlabeled counts stay accurate even when labels are sampled.
        self.assertEqual(split.image_count, 5)
        self.assertEqual(split.labeled_image_count + split.unlabeled_image_count, 5)
        self.assertEqual(split.annotation_count, 5)
        self.assertTrue(any("sampled" in w for w in summary.warnings))

    def test_summary_and_size_without_os_fwalk(self) -> None:
        import os

        from yolomatic.datasets.core import _stat_aggregate
        from yolomatic.utils.project import calculate_folder_size

        root = self.tmp / "no_fwalk"
        root.mkdir()
        self._write_image(root / "img.jpg")
        self._write_label(root / "img.txt")

        # Simulate Windows environment where os.fwalk does not exist
        saved_fwalk = getattr(os, "fwalk", None)
        if saved_fwalk is not None:
            del os.fwalk
        try:
            summary = summarize_dataset(root)
            self.assertGreater(summary.total_size_bytes, 0)
            file_count, _, total_size = _stat_aggregate(root)
            self.assertEqual(file_count, 2)
            self.assertGreater(total_size, 0)
            folder_size = calculate_folder_size(root)
            self.assertGreater(folder_size, 0)
        finally:
            if saved_fwalk is not None:
                os.fwalk = saved_fwalk

    def test_converted_coco_split_records_image_root_and_images_dir(self) -> None:
        # Regression: _summarize_coco never set images_path, so Detectron2 fell back to
        # <root>/train/images and joined it with file_name "train/images/x.jpg"
        # (doubling the path), and the benchmark planner rejected every COCO dataset.
        import cv2
        import numpy as np

        from yolomatic.benchmark.engine import _auto_detect_annotations, _load_coco_data
        from yolomatic.datasets.core import convert_yolo_to_coco

        source = self.tmp / "yolo_src"
        (source / "data.yaml").parent.mkdir(parents=True)
        (source / "data.yaml").write_text(
            "train: train/images\nval: valid/images\nnc: 1\nnames: [cat]\n", encoding="utf-8"
        )
        for split in ("train", "valid"):
            image = source / split / "images" / f"{split}_a.jpg"
            image.parent.mkdir(parents=True)
            cv2.imwrite(str(image), np.zeros((8, 8, 3), dtype=np.uint8))
            self._write_label(source / split / "labels" / f"{split}_a.txt")

        converted = self.tmp / "coco_out"
        convert_yolo_to_coco(source, converted)
        summary = summarize_dataset(converted)

        train = summary.splits["train"]
        ann = json.loads(Path(train.annotations_path).read_text(encoding="utf-8"))
        file_name = ann["images"][0]["file_name"]
        self.assertTrue((Path(train.image_root) / file_name).is_file())
        self.assertEqual(Path(train.images_path), (converted / "train" / "images").resolve())

        val = summary.splits["val"]
        self.assertIsNotNone(val.images_path)
        ann_file = _auto_detect_annotations(Path(val.images_path))
        id_to_path, _ = _load_coco_data(ann_file)
        self.assertTrue(all(path.is_file() for path in id_to_path.values()))


if __name__ == "__main__":
    unittest.main()

