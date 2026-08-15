import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.datasets.validate import validate_dataset
from src.utils.semantic import (
    semantic_background_index,
    semantic_class_names,
    semantic_max_pixel_value,
    semantic_pixel_value,
)

BOX = "0 0.5 0.5 0.25 0.25\n"
POLYGON = "0 0.10 0.10 0.90 0.10 0.90 0.90 0.10 0.90\n"
POSE = "0 0.5 0.5 0.4 0.4 0.40 0.40 2 0.60 0.40 2 0.50 0.60 1\n"


def _build(
    root: Path,
    task: str,
    rows: list[str],
    nc: int = 1,
    extra: dict | None = None,
    mask_values: list[int] | None = None,
) -> Path:
    dataset = root / "ds"
    (dataset / "train" / "images").mkdir(parents=True)
    meta = {
        "names": [f"c{i}" for i in range(nc)],
        "nc": nc,
        "task": task,
        "train": "train/images",
    }
    meta.update(extra or {})
    for index, row in enumerate(rows):
        cv2.imwrite(
            str(dataset / "train" / "images" / f"i{index}.jpg"),
            np.zeros((32, 32, 3), np.uint8),
        )
        if mask_values is not None:
            mask_dir = dataset / "train" / "masks"
            mask_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(mask_dir / f"i{index}.png"),
                np.full((32, 32), mask_values[index], np.uint8),
            )
        else:
            label_dir = dataset / "train" / "labels"
            label_dir.mkdir(parents=True, exist_ok=True)
            (label_dir / f"i{index}.txt").write_text(row, encoding="utf-8")
    (dataset / "data.yaml").write_text(yaml.dump(meta), encoding="utf-8")
    return dataset


class SemanticConventionTest(unittest.TestCase):
    """The pixel convention Ultralytics' CrossEntropyLoss requires."""

    def test_binary_dataset_uses_background_zero_foreground_one(self) -> None:
        self.assertEqual(semantic_background_index(1), 0)
        self.assertEqual(semantic_pixel_value(0, 1), 1)
        self.assertEqual(semantic_class_names(["veg"]), ["veg"])
        self.assertEqual(semantic_max_pixel_value(1), 1)

    def test_multiclass_appends_background_after_the_last_class(self) -> None:
        self.assertEqual(semantic_background_index(3), 3)
        # Foreground ids stay as they are — no shifting.
        self.assertEqual(semantic_pixel_value(0, 3), 0)
        self.assertEqual(semantic_pixel_value(2, 3), 2)
        self.assertEqual(semantic_class_names(["a", "b", "c"]), ["a", "b", "c", "background"])
        self.assertEqual(semantic_max_pixel_value(3), 3)

    def test_binary_values_are_a_zero_one_target(self) -> None:
        # nc == 1 drives a single output channel trained with BCE, so the mask is a
        # {0, 1} indicator rather than an index into a class table.
        self.assertEqual(
            {semantic_pixel_value(0, 1), semantic_background_index(1)}, {0, 1}
        )

    def test_multiclass_values_stay_inside_the_widened_class_range(self) -> None:
        # An out-of-range target is what aborts training inside CrossEntropyLoss, so
        # this is the property that actually matters.
        for num_classes in range(2, 6):
            values = [semantic_pixel_value(c, num_classes) for c in range(num_classes)]
            values.append(semantic_background_index(num_classes))
            widened_nc = len(semantic_class_names([f"c{i}" for i in range(num_classes)]))
            self.assertEqual(widened_nc, num_classes + 1)
            for value in values:
                self.assertGreaterEqual(value, 0)
                self.assertLess(value, widened_nc)
            # Background must not collide with any foreground class.
            self.assertEqual(len(set(values)), num_classes + 1)


class ValidateWellFormedDatasetTest(unittest.TestCase):
    def test_accepts_each_task_with_matching_labels(self) -> None:
        cases = [
            ("detect", [BOX], 1, None),
            ("segment", [POLYGON], 1, None),
            ("semantic", [POLYGON], 1, None),
            ("pose", [POSE], 1, {"kpt_shape": [3, 3]}),
        ]
        for task, rows, nc, extra in cases:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as temp_dir:
                report = validate_dataset(_build(Path(temp_dir), task, rows, nc, extra))
                self.assertTrue(report.ok, report.errors)
                self.assertEqual(report.task, task)

    def test_accepts_a_dense_mask_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = _build(
                Path(temp_dir), "semantic", [""], nc=3,
                extra={"masks_dir": "masks"}, mask_values=[2],
            )
            report = validate_dataset(dataset)
            self.assertTrue(report.ok, report.errors)
            self.assertEqual(report.label_style, "mask")


class ValidateMalformedDatasetTest(unittest.TestCase):
    def test_rejects_polygon_rows_in_a_detection_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = validate_dataset(_build(Path(temp_dir), "detect", [POLYGON]))
            self.assertFalse(report.ok)
            self.assertIn("expected 4 box values", report.errors[0])

    def test_rejects_box_rows_in_a_segmentation_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = validate_dataset(_build(Path(temp_dir), "segment", [BOX]))
            self.assertFalse(report.ok)
            self.assertIn("polygon needs at least 3 points", report.errors[0])

    def test_rejects_class_id_beyond_nc(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = validate_dataset(_build(Path(temp_dir), "detect", ["3 0.5 0.5 0.2 0.2\n"]))
            self.assertFalse(report.ok)
            self.assertIn("class id 3 outside", report.errors[0])

    def test_rejects_pixel_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = validate_dataset(_build(Path(temp_dir), "detect", ["0 320 240 100 80\n"]))
            self.assertFalse(report.ok)
            self.assertIn("outside [0, 1]", report.errors[0])

    def test_rejects_odd_polygon_coordinate_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            row = "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1\n"  # 7 coords
            report = validate_dataset(_build(Path(temp_dir), "segment", [row]))
            self.assertFalse(report.ok)
            self.assertIn("odd number of coordinates", report.errors[0])

    def test_rejects_wrong_keypoint_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = validate_dataset(
                _build(Path(temp_dir), "pose", ["0 0.5 0.5 0.4 0.4 0.4 0.4 2\n"],
                       extra={"kpt_shape": [3, 3]})
            )
            self.assertFalse(report.ok)
            self.assertIn("expected 13 values", report.errors[0])

    def test_accepts_visibility_flag_of_two_as_not_a_coordinate(self) -> None:
        # Regression: visibility is 0/1/2 and must not be range-checked as a coordinate.
        with tempfile.TemporaryDirectory() as temp_dir:
            report = validate_dataset(
                _build(Path(temp_dir), "pose", [POSE], extra={"kpt_shape": [3, 3]})
            )
            self.assertTrue(report.ok, report.errors)

    def test_rejects_invalid_visibility_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            row = "0 0.5 0.5 0.4 0.4 0.4 0.4 5 0.6 0.4 2 0.5 0.6 1\n"
            report = validate_dataset(
                _build(Path(temp_dir), "pose", [row], extra={"kpt_shape": [3, 3]})
            )
            self.assertFalse(report.ok)
            self.assertIn("visibility flags", report.errors[0])

    def test_rejects_mask_pixels_beyond_the_class_range(self) -> None:
        # This is precisely the state that aborts training inside a CUDA kernel.
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = _build(
                Path(temp_dir), "semantic", [""], nc=2,
                extra={"masks_dir": "masks"}, mask_values=[7],
            )
            report = validate_dataset(dataset)
            self.assertFalse(report.ok)
            self.assertIn("exceed the maximum class index", report.errors[0])

    def test_rejects_nc_and_names_disagreement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = _build(Path(temp_dir), "detect", [BOX], nc=1)
            meta = yaml.safe_load((dataset / "data.yaml").read_text())
            meta["nc"] = 5
            (dataset / "data.yaml").write_text(yaml.dump(meta), encoding="utf-8")
            report = validate_dataset(dataset)
            self.assertFalse(report.ok)
            self.assertTrue(any("names" in message for message in report.errors))

    def test_rejects_expected_task_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = _build(Path(temp_dir), "detect", [BOX])
            report = validate_dataset(dataset, expected_task="segment")
            self.assertFalse(report.ok)
            self.assertIn("expected", report.errors[0])

    def test_reports_missing_label_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = _build(Path(temp_dir), "detect", [BOX])
            next((dataset / "train" / "labels").glob("*.txt")).unlink()
            report = validate_dataset(dataset)
            self.assertFalse(report.ok)
            self.assertIn("no label file", report.errors[0])


if __name__ == "__main__":
    unittest.main()
