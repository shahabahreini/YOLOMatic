"""Per-family / per-task training parameter catalogs and RF-DETR pose support.

Covers ``parameters_for`` task filtering, ``prepared_format_for_family`` format
selection (RF-DETR pose must prepare COCO, others YOLO/Detectron2 COCO), and the
RF-DETR keypoint/pose registry + config generation.
"""

import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from src.config.parameters import (
    DETECTRON2_TRAINING_PARAMETERS,
    RFDETR_TRAINING_PARAMETERS,
    YOLO_TRAINING_PARAMETERS,
    parameters_for,
)
from src.datasets.core import prepared_format_for_family
from src.models.data import model_data_dict
from src.models.rfdetr import RFDETR_FAMILY_KEYS, get_rfdetr_variant, rfdetr_table_rows


def _names(params) -> set[str]:
    return {p.name for p in params}


class ParametersForTests(unittest.TestCase):
    def test_yolo_detection_excludes_seg_and_pose_params(self) -> None:
        names = _names(parameters_for("yolov11", "detection"))
        self.assertNotIn("overlap_mask", names)
        self.assertNotIn("mask_ratio", names)
        self.assertNotIn("pose", names)
        self.assertNotIn("kobj", names)
        self.assertIn("epochs", names)

    def test_yolo_segmentation_includes_mask_excludes_pose(self) -> None:
        names = _names(parameters_for("yolo26-seg", "segmentation"))
        self.assertIn("overlap_mask", names)
        self.assertIn("mask_ratio", names)
        self.assertNotIn("pose", names)

    def test_yolo_pose_includes_pose_excludes_mask(self) -> None:
        names = _names(parameters_for("yolo26-pose", "pose"))
        self.assertIn("pose", names)
        self.assertIn("kobj", names)
        self.assertNotIn("overlap_mask", names)

    def test_rfdetr_detection_excludes_keypoint_params(self) -> None:
        names = _names(parameters_for("rfdetr", "detection"))
        self.assertNotIn("num_keypoints_per_class", names)
        self.assertIn("resolution", names)
        # The catalog is the RF-DETR one, not the YOLO one.
        self.assertNotIn("imgsz", names)

    def test_rfdetr_pose_includes_keypoint_params(self) -> None:
        names = _names(parameters_for("rfdetr-pose", "pose"))
        self.assertIn("num_keypoints_per_class", names)
        self.assertIn("keypoint_l1_loss_coef", names)

    def test_detectron2_uses_detectron2_catalog(self) -> None:
        names = _names(parameters_for("detectron2", "detection"))
        self.assertEqual(names, _names(DETECTRON2_TRAINING_PARAMETERS))
        self.assertIn("max_iter", names)

    def test_catalogs_are_nonempty_and_distinct(self) -> None:
        self.assertTrue(YOLO_TRAINING_PARAMETERS)
        self.assertTrue(RFDETR_TRAINING_PARAMETERS)
        self.assertTrue(DETECTRON2_TRAINING_PARAMETERS)


class PreparedFormatTests(unittest.TestCase):
    def test_detectron2_always_coco(self) -> None:
        self.assertEqual(prepared_format_for_family("detectron2", "detection"), "coco")
        self.assertEqual(prepared_format_for_family("detectron2-seg", "segmentation"), "coco")

    def test_rfdetr_detection_and_seg_yolo(self) -> None:
        self.assertEqual(prepared_format_for_family("rfdetr", "detection"), "yolo")
        self.assertEqual(prepared_format_for_family("rfdetr", "segmentation"), "yolo")

    def test_rfdetr_pose_coco(self) -> None:
        self.assertEqual(prepared_format_for_family("rfdetr", "pose"), "coco")

    def test_yolo_always_yolo(self) -> None:
        self.assertEqual(prepared_format_for_family("yolo", "pose"), "yolo")
        self.assertEqual(prepared_format_for_family("yolov11", "detection"), "yolo")


class RFDETRPoseRegistryTests(unittest.TestCase):
    def test_pose_family_registered(self) -> None:
        self.assertIn("rfdetr-pose", RFDETR_FAMILY_KEYS)
        self.assertIn("rfdetr-pose", model_data_dict)

    def test_keypoint_variant_metadata(self) -> None:
        variant = get_rfdetr_variant("RF-DETR-Keypoint")
        self.assertEqual(variant.task, "pose")
        self.assertEqual(variant.class_name, "RFDETRKeypointPreview")

    def test_pose_table_rows_only_pose(self) -> None:
        rows = rfdetr_table_rows("pose")
        self.assertTrue(rows)
        self.assertEqual(rows[0]["Model"], "RF-DETR-Keypoint")


def _make_pose_dataset(root: Path) -> Path:
    for split in ("train", "val"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        (img_dir / f"{split}_0.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        (lbl_dir / f"{split}_0.txt").write_text(
            "0 0.5 0.5 0.1 0.1 0.5 0.5 2\n", encoding="utf-8"
        )
    (root / "data.yaml").write_text(
        textwrap.dedent(
            f"""\
            path: {root}
            train: images/train
            val: images/val
            nc: 1
            names: ['pole']
            kpt_shape: [1, 3]
            flip_idx: [0]
            """
        ),
        encoding="utf-8",
    )
    return root


class RFDETRPoseConfigTests(unittest.TestCase):
    def test_pose_config_uses_coco_and_keypoint_params(self) -> None:
        from src.config.generator import RFDETRConfigGenerator

        with TemporaryDirectory() as tmp:
            dataset = _make_pose_dataset(Path(tmp) / "poses")
            generator = RFDETRConfigGenerator(str(dataset))
            config = generator.generate_config("RF-DETR-Keypoint")

        self.assertEqual(config["settings"]["task"], "pose")
        self.assertEqual(config["settings"]["model_family"], "rfdetr")
        self.assertEqual(config["dataset"]["format"], "coco")
        training = config["training"]
        self.assertEqual(training["num_keypoints_per_class"], [0, 17])
        self.assertIn("keypoint_l1_loss_coef", training)


if __name__ == "__main__":
    unittest.main()
