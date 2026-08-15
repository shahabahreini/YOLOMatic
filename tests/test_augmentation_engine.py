import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from src.augmentation.engine import (
    SplitConfig,
    _write_staged_item,
    assign_groups_to_splits,
    collect_all_images,
    derive_group_key,
    resolve_augmentation_workers,
    run_augmentation,
)

# Tiles named "<raster>_r<row>_c<col>" — the convention the augment wizard offers.
TILE_PATTERN = r"^(.+?)_r\d+_c\d+$"


class AugmentationEngineCollectionTest(unittest.TestCase):
    def _touch(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")

    def _write_label(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("0 0.500000 0.500000 0.250000 0.250000\n", encoding="utf-8")

    def _write_seg_label(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "0 0.100000 0.100000 0.900000 0.100000 0.900000 0.900000 0.100000 0.900000\n",
            encoding="utf-8",
        )

    def _write_image(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((12, 12, 3), 120, dtype=np.uint8)
        cv2.imwrite(str(path), image)

    def _noop_profile(self, *, multiplier: int = 1, include_originals: bool = True) -> SimpleNamespace:
        return SimpleNamespace(
            name="noop",
            multiplier=multiplier,
            include_originals=include_originals,
            seed=7,
            transforms=[],
        )

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

    def test_collects_yaml_path_base_without_duplicate_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "source"
            root.mkdir()
            (root / "data.yaml").write_text(
                "path: assets\n"
                "task: detect\n"
                "names: [item]\n"
                "train: images/train\n",
                encoding="utf-8",
            )
            self._touch(root / "assets" / "images" / "train" / "a.jpg")
            self._write_label(root / "assets" / "labels" / "train" / "a.txt")

            pairs = collect_all_images(root)

            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0][0], (root / "assets" / "images" / "train" / "a.jpg").resolve())
            self.assertEqual(pairs[0][1], root / "assets" / "labels" / "train" / "a.txt")

    def test_collects_split_root_as_nested_images_dir_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data.yaml").write_text(
                "task: detect\n"
                "names: [item]\n"
                "train: train\n",
                encoding="utf-8",
            )
            self._touch(root / "train" / "images" / "a.jpg")
            self._write_label(root / "train" / "labels" / "a.txt")

            pairs = collect_all_images(root)

            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0][0], (root / "train" / "images" / "a.jpg").resolve())
            self.assertEqual(pairs[0][1], root / "train" / "labels" / "a.txt")

    def test_collects_valid_alias_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data.yaml").write_text(
                "task: detect\n"
                "names: [item]\n"
                "val: valid/images\n",
                encoding="utf-8",
            )
            self._touch(root / "valid" / "images" / "a.jpg")
            self._write_label(root / "valid" / "labels" / "a.txt")

            pairs = collect_all_images(root)

            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0][1], root / "valid" / "labels" / "a.txt")

    def test_run_augmentation_zero_test_ratio_leaves_test_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "source"
            root.mkdir()
            (root / "data.yaml").write_text(
                "task: detect\nnames: [item]\ntrain: train/images\n",
                encoding="utf-8",
            )
            # 9 images at 0.8/0.2/0.0: int truncation leaves a remainder that
            # must land in train, never test.
            for idx in range(9):
                self._write_image(root / "train" / "images" / f"img_{idx}.jpg")
                self._write_label(root / "train" / "labels" / f"img_{idx}.txt")

            stats = run_augmentation(
                root,
                "augmented",
                self._noop_profile(),
                SplitConfig(train_ratio=0.8, val_ratio=0.2, test_ratio=0.0),
                output_format="YOLO Detection",
                max_workers=1,
            )

            out_root = root.parent / "augmented"
            self.assertEqual(stats.split_counts["test"], 0)
            # 9 originals + 9 augmented copies, all split between train/valid.
            self.assertEqual(
                stats.split_counts["train"] + stats.split_counts["valid"],
                stats.total_output_images,
            )
            self.assertEqual(stats.total_output_images, 18)
            self.assertFalse((out_root / "test").exists())
            data_yaml = yaml.safe_load(
                (out_root / "data.yaml").read_text(encoding="utf-8")
            )
            self.assertNotIn("test", data_yaml)

    def test_resolves_auto_and_rejects_negative_worker_counts(self) -> None:
        self.assertGreaterEqual(resolve_augmentation_workers(0), 1)
        self.assertEqual(resolve_augmentation_workers(2), 2)
        with self.assertRaises(ValueError):
            resolve_augmentation_workers(-1)

    def test_rejects_invalid_split_before_replacing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "source"
            root.mkdir()
            existing = root.parent / "augmented"
            existing.mkdir()
            marker = existing / "keep.txt"
            marker.write_text("old output", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Split ratios"):
                run_augmentation(
                    root,
                    "augmented",
                    self._noop_profile(),
                    SplitConfig(train_ratio=0.9, val_ratio=0.2, test_ratio=0.0),
                )

            self.assertEqual(marker.read_text(encoding="utf-8"), "old output")

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
                max_workers=2,
            )

            out_root = root.parent / "augmented"
            self.assertEqual(stats.total_source_images, 3)
            self.assertEqual(stats.total_output_images, 6)
            # Group-aware split: each source image (original + its 1 augmented copy)
            # is a 2-item group kept intact, so the 3 groups land one per split —
            # {2, 2, 2}, never the per-item {4, 1, 1}. The even distribution itself
            # proves no variant of a source image leaks across splits.
            self.assertEqual(stats.split_counts, {"train": 2, "valid": 2, "test": 2})
            self.assertEqual(
                len(list((out_root / "train" / "images").glob("*.jpg"))),
                2,
            )
            self.assertEqual(
                len(list((out_root / "valid" / "images").glob("*.jpg"))),
                2,
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

    def test_fixed_seed_is_deterministic_across_worker_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "source"
            root.mkdir()
            (root / "data.yaml").write_text(
                "task: detect\nnames: [item]\ntrain: train/images\n", encoding="utf-8"
            )
            for index in range(3):
                self._write_image(root / "train" / "images" / f"{index}.jpg")
                self._write_label(root / "train" / "labels" / f"{index}.txt")
            profile = SimpleNamespace(
                name="seeded",
                multiplier=1,
                include_originals=True,
                seed=42,
                transforms=[{"name": "RandomBrightnessContrast", "enabled": True, "p": 1.0}],
            )
            split = SplitConfig(train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)
            run_augmentation(root, "single", profile, split, "YOLO Detection", max_workers=1)
            run_augmentation(root, "parallel", profile, split, "YOLO Detection", max_workers=2)

            def contents(dataset: Path) -> dict[str, bytes]:
                return {
                    path.relative_to(dataset).as_posix(): path.read_bytes()
                    for path in sorted(dataset.rglob("*"))
                    if path.is_file()
                }

            self.assertEqual(contents(root.parent / "single"), contents(root.parent / "parallel"))

    def test_failed_coco_conversion_keeps_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "source"
            root.mkdir()
            (root / "data.yaml").write_text(
                "task: detect\nnames: [item]\ntrain: train/images\n", encoding="utf-8"
            )
            self._write_image(root / "train" / "images" / "a.jpg")
            self._write_label(root / "train" / "labels" / "a.txt")
            existing = root.parent / "augmented"
            existing.mkdir()
            marker = existing / "keep.txt"
            marker.write_text("old output", encoding="utf-8")

            with patch("src.datasets.core.convert_yolo_to_coco", side_effect=RuntimeError("conversion failed")):
                with self.assertRaisesRegex(RuntimeError, "conversion failed"):
                    run_augmentation(
                        root,
                        "augmented",
                        self._noop_profile(multiplier=0, include_originals=True),
                        SplitConfig(train_ratio=1.0, val_ratio=0.0, test_ratio=0.0),
                        output_format="COCO",
                        max_workers=1,
                    )

            self.assertEqual(marker.read_text(encoding="utf-8"), "old output")

    def test_run_augmentation_removes_stale_output_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "source"
            root.mkdir()
            (root / "data.yaml").write_text(
                "task: detect\nnames: [item]\ntrain: train/images\n",
                encoding="utf-8",
            )
            self._write_image(root / "train" / "images" / "a.jpg")
            self._write_label(root / "train" / "labels" / "a.txt")

            stale_label = root.parent / "augmented" / "train" / "labels" / "stale.txt"
            stale_label.parent.mkdir(parents=True, exist_ok=True)
            stale_label.write_text("0 0.1 0.1 0.1 0.1\n", encoding="utf-8")

            stats = run_augmentation(
                root,
                "augmented",
                self._noop_profile(multiplier=0, include_originals=True),
                SplitConfig(train_ratio=1.0, val_ratio=0.0, test_ratio=0.0),
                output_format="YOLO Detection",
                max_workers=1,
            )

            out_root = root.parent / "augmented"
            self.assertEqual(stats.total_output_images, 1)
            self.assertFalse(stale_label.exists())
            self.assertEqual(
                len(list((out_root / "train" / "labels").glob("*.txt"))),
                1,
            )

    def test_run_augmentation_removes_source_image_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "source"
            root.mkdir()
            (root / "data.yaml").write_text(
                "task: detect\nnames: [item]\ntrain: train/images\n",
                encoding="utf-8",
            )
            image_path = root / "train" / "images" / "a.jpg"
            self._write_image(image_path)
            self._write_label(root / "train" / "labels" / "a.txt")
            cache_path = image_path.with_suffix(".npy")
            cache_path.write_bytes(b"uncompressed-cache")

            stats = run_augmentation(
                root,
                "augmented",
                self._noop_profile(multiplier=0, include_originals=True),
                SplitConfig(train_ratio=1.0, val_ratio=0.0, test_ratio=0.0),
                output_format="YOLO Detection",
                max_workers=1,
            )

            self.assertFalse(cache_path.exists())
            self.assertEqual(stats.cache_files_removed, 1)
            self.assertEqual(stats.cache_bytes_reclaimed, len(b"uncompressed-cache"))

    def test_run_augmentation_deduplicates_identical_label_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "source"
            root.mkdir()
            (root / "data.yaml").write_text(
                "task: detect\nnames: [item]\ntrain: train/images\n",
                encoding="utf-8",
            )
            self._write_image(root / "train" / "images" / "a.jpg")
            label_path = root / "train" / "labels" / "a.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(
                "0 0.500000 0.500000 0.250000 0.250000\n"
                "0 0.500000 0.500000 0.250000 0.250000\n",
                encoding="utf-8",
            )

            run_augmentation(
                root,
                "augmented",
                self._noop_profile(multiplier=0, include_originals=True),
                SplitConfig(train_ratio=1.0, val_ratio=0.0, test_ratio=0.0),
                output_format="YOLO Detection",
                max_workers=1,
            )

            output_labels = list((root.parent / "augmented" / "train" / "labels").glob("*.txt"))
            self.assertEqual(len(output_labels), 1)
            rows = [
                line
                for line in output_labels[0].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(rows, ["0 0.500000 0.500000 0.250000 0.250000"])

    def test_run_augmentation_semantic_output_writes_dense_class_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "source"
            root.mkdir()
            (root / "data.yaml").write_text(
                "task: segment\nnames: [item]\ntrain: train/images\n",
                encoding="utf-8",
            )
            self._write_image(root / "train" / "images" / "a.jpg")
            self._write_seg_label(root / "train" / "labels" / "a.txt")

            stats = run_augmentation(
                root,
                "augmented",
                self._noop_profile(multiplier=0, include_originals=True),
                SplitConfig(train_ratio=1.0, val_ratio=0.0, test_ratio=0.0),
                output_format="YOLO Segmentation (Semantic)",
                max_workers=1,
            )

            out_root = root.parent / "augmented"
            self.assertEqual(stats.total_output_images, 1)
            self.assertFalse((out_root / "train" / "labels").exists())

            mask_files = list((out_root / "train" / "masks").glob("*.png"))
            self.assertEqual(len(mask_files), 1)
            mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_UNCHANGED)
            self.assertIsNotNone(mask)
            # Class 0 occupies value 1 (0 stays reserved for background).
            self.assertEqual(set(np.unique(mask).tolist()), {0, 1})
            self.assertGreater(int((mask == 1).sum()), 0)

            data_yaml = yaml.safe_load(
                (out_root / "data.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(data_yaml["task"], "semantic")
            self.assertEqual(data_yaml["train_masks"], "train/masks")
            # "masks_dir" is the key Ultralytics' SemanticDataset reads. Spelling it
            # "mask_dir" silently selects the polygon path, which finds no labels.
            self.assertEqual(data_yaml["masks_dir"], "masks")


class SemanticMaskOutputTest(unittest.TestCase):
    """Dense-mask output must use the class indices Ultralytics feeds to its loss."""

    def _build_source(self, root: Path, class_names: list[str]) -> None:
        root.mkdir(parents=True)
        (root / "data.yaml").write_text(
            yaml.dump({
                "names": class_names,
                "nc": len(class_names),
                "task": "segment",
                "train": "images/train",
                "val": "images/train",
            }),
            encoding="utf-8",
        )
        # Each class gets its own horizontal band. Overlapping polygons would let a
        # later class overdraw an earlier one, hiding its value from the mask.
        rows = ""
        for cls in range(len(class_names)):
            top = 0.02 + cls * (0.96 / len(class_names))
            bottom = top + (0.96 / len(class_names)) - 0.02
            rows += (
                f"{cls} 0.02 {top:.4f} 0.98 {top:.4f} "
                f"0.98 {bottom:.4f} 0.02 {bottom:.4f}\n"
            )
        for index in range(4):
            stem = f"img{index:02d}_r0000_c0000"
            image_path = root / "images" / "train" / f"{stem}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), np.full((32, 32, 3), 120, dtype=np.uint8))
            label_path = root / "labels" / "train" / f"{stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(rows, encoding="utf-8")

    def _run(self, root: Path, class_names: list[str]):
        self._build_source(root, class_names)
        run_augmentation(
            root, "augmented",
            SimpleNamespace(name="noop", multiplier=1, include_originals=True,
                            seed=5, transforms=[]),
            SplitConfig(0.75, 0.25, 0.0, augment_splits=("train",)),
            output_format="YOLO Segmentation (Semantic)",
            max_workers=1,
        )
        out_root = root.parent / "augmented"
        data_yaml = yaml.safe_load((out_root / "data.yaml").read_text(encoding="utf-8"))
        masks = sorted((out_root / "train" / "masks").glob("*.png"))
        values: set[int] = set()
        for mask_path in masks:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            values |= {int(v) for v in np.unique(mask)}
        return data_yaml, values

    def test_binary_dataset_writes_background_zero_and_foreground_one(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_yaml, values = self._run(Path(temp_dir) / "source", ["veg"])
            self.assertEqual(values, {0, 1})
            self.assertEqual(data_yaml["nc"], 1)
            self.assertEqual(data_yaml["names"], ["veg"])
            self.assertEqual(data_yaml["bg_class_idx"], 0)

    def test_multiclass_appends_background_and_keeps_class_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_yaml, values = self._run(Path(temp_dir) / "source", ["alpha", "beta"])
            # Foreground ids are NOT shifted; background takes the next free index.
            self.assertEqual(values, {0, 1, 2})
            self.assertEqual(data_yaml["names"], ["alpha", "beta", "background"])
            self.assertEqual(data_yaml["nc"], 3)
            self.assertEqual(data_yaml["bg_class_idx"], 2)

    def test_no_mask_pixel_exceeds_the_declared_class_count(self) -> None:
        # Regression: writing class_id + 1 with an unwidened nc put a target of `nc`
        # into CrossEntropyLoss, which aborts training with a device-side assert.
        for class_names in (["a"], ["a", "b"], ["a", "b", "c"]):
            with self.subTest(classes=len(class_names)), tempfile.TemporaryDirectory() as temp_dir:
                data_yaml, values = self._run(Path(temp_dir) / "source", class_names)
                limit = data_yaml["nc"] if len(class_names) > 1 else 2
                self.assertLess(max(values), limit)


class GroupKeyTest(unittest.TestCase):
    def test_no_pattern_keys_each_image_by_its_own_stem(self) -> None:
        self.assertEqual(derive_group_key("block_r01_c02", None), "block_r01_c02")

    def test_capture_group_becomes_the_key(self) -> None:
        self.assertEqual(derive_group_key("block12_r0640_c1280", TILE_PATTERN), "block12")

    def test_whole_match_used_when_regex_has_no_capture_group(self) -> None:
        self.assertEqual(derive_group_key("tile_14N_604_5522_r01_c02", r"14N_\d+_\d+"), "14N_604_5522")

    def test_non_matching_stem_keeps_its_own_key(self) -> None:
        # Never collapse unmatched files into one shared group: that would silently
        # force unrelated images into the same split.
        self.assertEqual(derive_group_key("README_thumb", TILE_PATTERN), "README_thumb")


class AssignGroupsToSplitsTest(unittest.TestCase):
    def test_hits_target_ratios_with_uneven_group_sizes(self) -> None:
        # Deliberately lopsided: one group holds 40% of the images.
        sizes = {"a": 400, "b": 300, "c": 200, "d": 60, "e": 30, "f": 8, "g": 2}
        total = sum(sizes.values())
        assignment = assign_groups_to_splits(
            sizes, SplitConfig(0.85, 0.10, 0.05), seed=42
        )

        counts = {"train": 0, "valid": 0, "test": 0}
        for key, split in assignment.items():
            counts[split] += sizes[key]

        self.assertEqual(sum(counts.values()), total)
        # Largest-first packing cannot beat the granularity of the biggest group it
        # must place, so allow that much slack rather than an exact ratio.
        slack = max(sizes.values()) / total
        for split, ratio in (("train", 0.85), ("valid", 0.10), ("test", 0.05)):
            self.assertAlmostEqual(counts[split] / total, ratio, delta=slack)

    def test_zero_ratio_split_never_receives_a_group(self) -> None:
        sizes = {chr(ord("a") + i): i + 1 for i in range(10)}
        assignment = assign_groups_to_splits(
            sizes, SplitConfig(0.80, 0.20, 0.0), seed=1
        )
        self.assertNotIn("test", set(assignment.values()))

    def test_assignment_is_deterministic_for_a_given_seed(self) -> None:
        sizes = {chr(ord("a") + i): (i % 4) + 1 for i in range(20)}
        config = SplitConfig(0.70, 0.20, 0.10)
        self.assertEqual(
            assign_groups_to_splits(sizes, config, seed=5),
            assign_groups_to_splits(sizes, config, seed=5),
        )

    def test_all_zero_ratios_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            assign_groups_to_splits({"a": 1}, SplitConfig(0.0, 0.0, 0.0), seed=0)


class SplitGroupingAndTrainOnlyAugmentationTest(unittest.TestCase):
    """End-to-end checks for the leakage controls on a tiled source dataset."""

    def _build_tiled_dataset(self, root: Path, rasters: int, tiles_per_raster: int) -> None:
        root.mkdir(parents=True)
        (root / "data.yaml").write_text(
            "task: segment\nnames: [item]\ntrain: images/train\nval: images/val\n",
            encoding="utf-8",
        )
        for raster in range(rasters):
            for tile in range(tiles_per_raster):
                stem = f"block{raster:02d}_r{tile:04d}_c0000"
                image_path = root / "images" / "train" / f"{stem}.jpg"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(image_path), np.full((12, 12, 3), 120, dtype=np.uint8))
                label_path = root / "labels" / "train" / f"{stem}.txt"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text(
                    "0 0.100000 0.100000 0.900000 0.100000 0.900000 0.900000 0.100000 0.900000\n",
                    encoding="utf-8",
                )

    def _profile(self, multiplier: int) -> SimpleNamespace:
        return SimpleNamespace(
            name="noop",
            multiplier=multiplier,
            include_originals=True,
            seed=42,
            transforms=[],
        )

    def test_group_pattern_keeps_every_tile_of_a_raster_in_one_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "source"
            self._build_tiled_dataset(root, rasters=10, tiles_per_raster=4)

            run_augmentation(
                root,
                "augmented",
                self._profile(multiplier=1),
                SplitConfig(
                    0.60, 0.20, 0.20,
                    group_key_pattern=TILE_PATTERN,
                    augment_splits=("train",),
                ),
                output_format="YOLO Segmentation",
                max_workers=2,
            )

            report = json.loads(
                (root.parent / "augmented" / "split_assignment.json").read_text(encoding="utf-8")
            )
            raster_splits: dict[str, set[str]] = {}
            for entry in report["sources"].values():
                raster_splits.setdefault(entry["group"], set()).add(entry["split"])
            self.assertEqual(len(raster_splits), 10)
            for raster, splits in raster_splits.items():
                self.assertEqual(len(splits), 1, f"raster {raster} was split across {splits}")

    def test_augment_splits_leaves_val_and_test_unmultiplied(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "source"
            self._build_tiled_dataset(root, rasters=10, tiles_per_raster=4)

            stats = run_augmentation(
                root,
                "augmented",
                self._profile(multiplier=3),
                SplitConfig(
                    0.60, 0.20, 0.20,
                    group_key_pattern=TILE_PATTERN,
                    augment_splits=("train",),
                ),
                output_format="YOLO Segmentation",
                max_workers=2,
            )

            report = json.loads(
                (root.parent / "augmented" / "split_assignment.json").read_text(encoding="utf-8")
            )
            source_per_split = {"train": 0, "valid": 0, "test": 0}
            for entry in report["sources"].values():
                source_per_split[entry["split"]] += 1

            # Train is multiplied (originals + 3 variants); val/test pass through 1:1.
            self.assertEqual(stats.split_counts["train"], source_per_split["train"] * 4)
            self.assertEqual(stats.split_counts["valid"], source_per_split["valid"])
            self.assertEqual(stats.split_counts["test"], source_per_split["test"])

    def test_non_augmented_split_keeps_originals_even_without_include_originals(self) -> None:
        # include_originals=False would otherwise leave val/test with nothing at all,
        # since they produce no augmented variants either.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "source"
            self._build_tiled_dataset(root, rasters=8, tiles_per_raster=2)

            profile = self._profile(multiplier=2)
            profile.include_originals = False
            stats = run_augmentation(
                root,
                "augmented",
                profile,
                SplitConfig(
                    0.50, 0.25, 0.25,
                    group_key_pattern=TILE_PATTERN,
                    augment_splits=("train",),
                ),
                output_format="YOLO Segmentation",
                max_workers=1,
            )

            self.assertGreater(stats.split_counts["valid"], 0)
            self.assertGreater(stats.split_counts["test"], 0)

    def test_semantic_polygon_output_writes_labels_and_omits_masks_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "source"
            self._build_tiled_dataset(root, rasters=4, tiles_per_raster=2)

            run_augmentation(
                root,
                "augmented",
                self._profile(multiplier=1),
                SplitConfig(0.50, 0.25, 0.25, augment_splits=("train",)),
                output_format="YOLO Semantic (Polygon)",
                max_workers=1,
            )

            out_root = root.parent / "augmented"
            labels = sorted((out_root / "train" / "labels").glob("*.txt"))
            self.assertTrue(labels)
            # Polygon rows, not boxes: class id followed by an even number of coords.
            fields = labels[0].read_text(encoding="utf-8").split()
            self.assertGreaterEqual(len(fields), 7)
            self.assertEqual(len(fields) % 2, 1)
            self.assertFalse((out_root / "train" / "masks").exists())

            data_yaml = yaml.safe_load((out_root / "data.yaml").read_text(encoding="utf-8"))
            self.assertEqual(data_yaml["task"], "semantic")
            # Absence of masks_dir is what makes Ultralytics rasterize the polygons
            # and add a background class instead of hunting for mask PNGs.
            self.assertNotIn("masks_dir", data_yaml)

    def test_default_split_config_reproduces_per_image_grouping(self) -> None:
        # Regression guard: the defaults must behave exactly as they did before
        # group keys and augment_splits existed.
        config = SplitConfig()
        self.assertIsNone(config.group_key_pattern)
        self.assertIsNone(config.augment_splits)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "source"
            self._build_tiled_dataset(root, rasters=4, tiles_per_raster=1)

            stats = run_augmentation(
                root,
                "augmented",
                self._profile(multiplier=1),
                SplitConfig(0.50, 0.25, 0.25),
                output_format="YOLO Segmentation",
                max_workers=1,
            )

            # Every split is augmented, so all 4 sources yield 2 items each.
            self.assertEqual(stats.total_output_images, 8)
            self.assertEqual(sum(stats.split_counts.values()), 8)


class StagedImageEncodingTest(unittest.TestCase):
    def test_staged_jpeg_uses_444_chroma_sampling(self) -> None:
        # Fused imagery carries independent measurements per channel, so the default
        # 4:2:0 subsampling would blur two of the three bands.
        with tempfile.TemporaryDirectory() as temp_dir:
            group_dir = Path(temp_dir)
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            image[:, ::2, 0] = 255  # 1px-wide vertical stripes in channel 0 only
            _write_staged_item(group_dir, "original", image, [], [], cv2)

            with open(group_dir / "original.jpg", "rb") as handle:
                data = handle.read()
            # SOF0 marker: [id, precision, h(2), w(2), ncomp, (cid, sampling, qtable)*]
            sof = data.index(b"\xff\xc0")
            n_components = data[sof + 9]
            self.assertEqual(n_components, 3)
            sampling_factors = {data[sof + 10 + 3 * i + 1] for i in range(n_components)}
            # 0x11 = 1x1 sampling on every component, i.e. no subsampling.
            self.assertEqual(sampling_factors, {0x11})


if __name__ == "__main__":
    unittest.main()
