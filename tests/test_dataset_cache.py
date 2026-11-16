import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from yolomatic.config.generator import YOLOConfigGenerator
from yolomatic.datasets.cache import (
    clean_dataset_image_cache,
    inspect_dataset_cache,
    normalize_yolo_cache_setting,
)
from yolomatic.datasets.core import summarize_dataset
from yolomatic.utils.project import list_dataset_directories


class DatasetCacheTest(unittest.TestCase):
    def test_cleanup_removes_only_verified_image_caches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            images = root / "train" / "images"
            labels = root / "train" / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            image = np.zeros((6, 10, 3), dtype=np.uint8)
            Image.fromarray(image).save(images / "sample.jpg")
            np.save(images / "sample.npy", image)  # what Ultralytics cache=disk writes
            cache_size = (images / "sample.npy").stat().st_size
            (images / "user-array.npy").write_bytes(b"keep")
            # Regression: user arrays sitting beside an image used to be deleted.
            Image.fromarray(image).save(images / "depth.png")
            np.save(images / "depth.npy", np.zeros((6, 10), dtype=np.float32))
            Image.fromarray(image).save(images / "other.png")
            np.save(images / "other.npy", np.zeros((3, 3, 3), dtype=np.uint8))
            (labels / "labels.cache").write_bytes(b"metadata")

            before = inspect_dataset_cache(root)
            result = clean_dataset_image_cache(root)

            self.assertEqual(before.image_cache_files, 1)
            self.assertEqual(before.metadata_cache_files, 1)
            self.assertEqual(result.removed_files, 1)
            self.assertEqual(result.reclaimed_bytes, cache_size)
            self.assertFalse((images / "sample.npy").exists())
            self.assertTrue((images / "user-array.npy").exists())
            self.assertTrue((images / "depth.npy").exists())
            self.assertTrue((images / "other.npy").exists())
            self.assertTrue((labels / "labels.cache").exists())

    def test_dataset_summary_excludes_runtime_cache_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            images = root / "train" / "images"
            labels = root / "train" / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            (root / "data.yaml").write_text(
                "train: train/images\nnames: [item]\n", encoding="utf-8"
            )
            (images / "sample.jpg").write_bytes(b"image")
            (images / "sample.npy").write_bytes(b"large-cache")
            (labels / "labels.cache").write_bytes(b"label-cache")

            summary = summarize_dataset(root)

            expected_content = len((root / "data.yaml").read_bytes()) + len(b"image")
            self.assertEqual(summary.total_size_bytes, expected_content)
            self.assertEqual(summary.runtime_cache_file_count, 2)
            self.assertEqual(
                summary.runtime_cache_size_bytes,
                len(b"large-cache") + len(b"label-cache"),
            )

    def test_disk_cache_setting_is_disabled_but_ram_is_preserved(self) -> None:
        self.assertEqual(normalize_yolo_cache_setting("disk"), (False, True))
        self.assertEqual(normalize_yolo_cache_setting("DISK"), (False, True))
        self.assertEqual(normalize_yolo_cache_setting("ram"), ("ram", False))
        self.assertEqual(normalize_yolo_cache_setting(True), (True, False))

    def test_hidden_conversion_cache_is_not_listed_as_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "datasets"
            (root / ".yolomatic_cache").mkdir(parents=True)
            (root / "real_dataset").mkdir()

            datasets = list_dataset_directories(root, include_size=False)

            self.assertEqual([dataset["name"] for dataset in datasets], ["real_dataset"])

    def test_config_metrics_exclude_runtime_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            images = root / "train" / "images"
            images.mkdir(parents=True)
            (images / "sample.jpg").write_bytes(b"image")
            (images / "sample.npy").write_bytes(b"large-cache")

            metrics = YOLOConfigGenerator(root)._collect_dataset_metrics()

            self.assertEqual(metrics["total_size_bytes"], len(b"image"))
            self.assertEqual(metrics["image_count"], 1)
            self.assertEqual(metrics["runtime_cache_file_count"], 1)
            self.assertEqual(metrics["runtime_cache_size_bytes"], len(b"large-cache"))


if __name__ == "__main__":
    unittest.main()
