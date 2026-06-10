"""Pose-task support: model families, family-key resolution, dataset detection,
and YOLO config generation for keypoint datasets.

A standard Ultralytics pose dataset is identified by ``kpt_shape`` in data.yaml
plus ``class + bbox + K*ndim`` label rows. These tests build a minimal one in a
temp dir and assert YOLOmatic detects it as ``pose`` and routes a ``-pose``
model to the matching family.
"""

import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.config.generator import YOLOConfigGenerator
from src.datasets.core import summarize_dataset
from src.models.data import model_data_dict


POSE_FAMILIES = ("yolo26-pose", "yolov11-pose", "yolov8-pose")


def _make_pose_dataset(root: Path) -> Path:
    """Write a minimal standard YOLO pose dataset under ``root`` and return it."""
    for split in ("train", "val", "test"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        (img_dir / f"{split}_0.jpg").write_bytes(b"\xff\xd8\xff\xd9")  # tiny stub
        # class + bbox (4) + one keypoint (x, y, visibility) = 8 columns.
        (lbl_dir / f"{split}_0.txt").write_text(
            "0 0.5 0.5 0.1 0.1 0.5 0.5 2\n", encoding="utf-8"
        )
    (root / "data.yaml").write_text(
        textwrap.dedent(
            f"""\
            path: {root}
            train: images/train
            val: images/val
            test: images/test
            nc: 1
            names: ['pole']
            kpt_shape: [1, 3]
            flip_idx: [0]
            """
        ),
        encoding="utf-8",
    )
    return root


class PoseFamilyRegistryTests(unittest.TestCase):
    def test_pose_families_registered(self) -> None:
        for family in POSE_FAMILIES:
            self.assertIn(family, model_data_dict)
            variants = [row["Model"] for row in model_data_dict[family]]
            self.assertEqual(len(variants), 5)  # n/s/m/l/x
            self.assertTrue(all("-pose" in v.lower() for v in variants))

    def test_family_key_resolution_for_pose_variants(self) -> None:
        gen = YOLOConfigGenerator(".")
        cases = {
            "yolo26l-pose": "yolo26-pose",
            "yolo11n-pose": "yolov11-pose",
            "yolov8x-pose": "yolov8-pose",
            "yolo26s": "yolo26",          # non-pose still resolves normally
            "yolo26m-seg": "yolo26-seg",
        }
        for variant, expected in cases.items():
            self.assertEqual(gen._resolve_family_key(variant), expected)


class PoseDatasetDetectionTests(unittest.TestCase):
    def test_summarize_dataset_reports_pose(self) -> None:
        with TemporaryDirectory() as tmp:
            root = _make_pose_dataset(Path(tmp))
            summary = summarize_dataset(root)
            self.assertEqual(summary.task, "pose")
            self.assertEqual(summary.format, "yolo")
            self.assertEqual(summary.compatibility.get("yolo"), "native")
            self.assertEqual(summary.classes, ["pole"])

    def test_generator_detects_pose_task_type(self) -> None:
        with TemporaryDirectory() as tmp:
            root = _make_pose_dataset(Path(tmp))
            gen = YOLOConfigGenerator(str(root))
            self.assertTrue(gen.extract_dataset_info())
            self.assertEqual(gen.dataset_info.get("task_type"), "pose")

    def test_generated_config_is_pose(self) -> None:
        with TemporaryDirectory() as tmp:
            root = _make_pose_dataset(Path(tmp))
            gen = YOLOConfigGenerator(str(root))
            gen.extract_dataset_info()
            config = gen.generate_config("yolo26l-pose")
            self.assertEqual(config["settings"]["task"], "pose")
            self.assertEqual(config["settings"]["model_type"], "yolo26l-pose")
            self.assertEqual(config["settings"]["model_family"], "yolo")
            self.assertEqual(config["dataset"]["task"], "pose")


if __name__ == "__main__":
    unittest.main()
