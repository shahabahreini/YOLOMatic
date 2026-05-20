import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from src.augmentation.engine import SplitConfig, collect_all_images, run_augmentation


class AugmentationEngineCollectionTest(unittest.TestCase):
    def _touch(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")

    def _write_label(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("0 0.500000 0.500000 0.250000 0.250000\n", encoding="utf-8")

    def test_collects_standard_split_images_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._touch(root / "train" / "images" / "a.jpg")
            self._write_label(root / "train" / "labels" / "a.txt")

            pairs = collect_all_images(root)

            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0][0], root / "train" / "images" / "a.jpg")
            self.assertEqual(pairs[0][1], root / "train" / "labels" / "a.txt")

    def test_collects_root_images_split_and_labels_split_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data.yaml").write_text(
                "train: images/train\nval: images/val\ntest: images/test\n",
                encoding="utf-8",
            )
            for split in ("train", "val", "test"):
                self._touch(root / "images" / split / f"{split}.jpg")
                self._write_label(root / "labels" / split / f"{split}.txt")

            pairs = collect_all_images(root)

            self.assertEqual(
                [path.name for path, _ in pairs],
                ["train.jpg", "val.jpg", "test.jpg"],
            )
            self.assertEqual(
                [label for _, label in pairs],
                [
                    root / "labels" / "train" / "train.txt",
                    root / "labels" / "val" / "val.txt",
                    root / "labels" / "test" / "test.txt",
                ],
            )

    def test_collects_missing_label_as_none_without_skipping_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._touch(root / "images" / "val" / "background.jpg")

            pairs = collect_all_images(root)

            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0][0], root / "images" / "val" / "background.jpg")
            self.assertIsNone(pairs[0][1])

    def test_run_augmentation_pools_all_source_splits_then_redistributes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "source"
            root.mkdir()
            (root / "data.yaml").write_text(
                "task: detect\n"
                "names: [item]\n"
                "train: images/train\n"
                "val: images/val\n"
                "test: images/test\n",
                encoding="utf-8",
            )
            for split in ("train", "val", "test"):
                image_path = root / "images" / split / f"{split}.jpg"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image = np.full((12, 12, 3), 120, dtype=np.uint8)
                cv2.imwrite(str(image_path), image)
                self._write_label(root / "labels" / split / f"{split}.txt")

            stats = run_augmentation(
                root,
                "augmented",
                SimpleNamespace(
                    name="noop",
                    multiplier=1,
                    include_originals=True,
                    seed=7,
                    transforms=[],
                ),
                SplitConfig(train_ratio=0.50, val_ratio=0.25, test_ratio=0.25),
                output_format="YOLO Detection",
                max_workers=1,
            )

            out_root = root.parent / "augmented"
            self.assertEqual(stats.total_source_images, 3)
            self.assertEqual(stats.total_output_images, 6)
            self.assertEqual(stats.split_counts, {"train": 3, "valid": 1, "test": 2})
            self.assertEqual(
                len(list((out_root / "train" / "images").glob("*.jpg"))),
                3,
            )
            self.assertEqual(
                len(list((out_root / "valid" / "images").glob("*.jpg"))),
                1,
            )
            self.assertEqual(
                len(list((out_root / "test" / "images").glob("*.jpg"))),
                2,
            )

            data_yaml = yaml.safe_load(
                (out_root / "data.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(data_yaml["train"], "train/images")
            self.assertEqual(data_yaml["val"], "valid/images")
            self.assertEqual(data_yaml["test"], "test/images")


if __name__ == "__main__":
    unittest.main()
