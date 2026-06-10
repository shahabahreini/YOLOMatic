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

import cv2
import numpy as np
import yaml

from src.config.generator import YOLOConfigGenerator
from src.datasets.core import convert_coco_to_yolo, convert_yolo_to_coco, summarize_dataset
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


def _make_real_pose_dataset(root: Path, kpt_shape=(2, 3), count: int = 3) -> Path:
    """Pose dataset with real (decodable) images for conversion/augmentation tests."""
    k, ndim = kpt_shape
    kpts = " ".join(["0.40 0.40 2" if ndim == 3 else "0.40 0.40"] * k)
    for split in ("train", "val"):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(count):
            cv2.imwrite(str(img_dir / f"{split}_{idx}.jpg"), np.full((16, 16, 3), 100, dtype=np.uint8))
            (lbl_dir / f"{split}_{idx}.txt").write_text(
                f"0 0.5 0.5 0.25 0.25 {kpts}\n", encoding="utf-8"
            )
    (root / "data.yaml").write_text(
        f"path: {root}\ntrain: train/images\nval: val/images\nnc: 1\nnames: [person]\n"
        f"task: pose\nkpt_shape: [{k}, {ndim}]\n",
        encoding="utf-8",
    )
    return root


class PoseConversionRoundTripTests(unittest.TestCase):
    def test_yolo_to_coco_to_yolo_preserves_keypoints(self) -> None:
        with TemporaryDirectory() as tmp:
            root = _make_real_pose_dataset(Path(tmp) / "src")
            coco_out = Path(tmp) / "coco"
            yolo_out = Path(tmp) / "yolo"

            convert_yolo_to_coco(root, coco_out)
            import json
            payload = json.loads((coco_out / "annotations" / "instances_train.json").read_text())
            self.assertIn("keypoints", payload["categories"][0])
            self.assertEqual(payload["annotations"][0]["num_keypoints"], 2)

            convert_coco_to_yolo(coco_out, yolo_out)
            data = yaml.safe_load((yolo_out / "data.yaml").read_text())
            self.assertEqual(data.get("task"), "pose")
            self.assertEqual(data.get("kpt_shape"), [2, 3])
            label = next((yolo_out / "train" / "labels").glob("*.txt"))
            tokens = label.read_text().split()
            self.assertEqual(len(tokens), 11)  # class + 4 bbox + 2*3 keypoints


class BenchmarkPoseDetectionTests(unittest.TestCase):
    def test_detect_task_returns_pose_for_keypoint_output(self) -> None:
        from src.benchmark.engine import _detect_task

        class _Result:
            masks = None
            keypoints = [object()]  # non-empty → pose

        def _fake_model(*_args, **_kwargs):
            return [_Result()]

        task = _detect_task(_fake_model, Path("probe.jpg"), conf=0.25, device="cpu")
        self.assertEqual(task, "pose")

    def test_detect_task_prefers_segmentation_over_pose(self) -> None:
        from src.benchmark.engine import _detect_task

        class _Result:
            masks = [object()]
            keypoints = [object()]

        task = _detect_task(lambda *a, **k: [_Result()], Path("p.jpg"), conf=0.25, device="cpu")
        self.assertEqual(task, "segmentation")


class AugmentationPoseTests(unittest.TestCase):
    def test_read_write_yolo_pose_round_trip(self) -> None:
        from src.augmentation.engine import read_yolo_pose, write_yolo_pose

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "lbl.txt"
            path.write_text("0 0.5 0.5 0.2 0.2 0.4 0.4 2 0.6 0.6 1\n", encoding="utf-8")
            bboxes, kpts, cls = read_yolo_pose(path, (2, 3))
            self.assertEqual(len(bboxes), 1)
            self.assertEqual(len(kpts[0]), 6)
            self.assertEqual(cls, [0])

            out = Path(tmp) / "out.txt"
            write_yolo_pose(out, bboxes, kpts, cls)
            tokens = out.read_text().split()
            self.assertEqual(len(tokens), 11)

    def test_build_pose_pipeline_strips_flips(self) -> None:
        from src.augmentation.engine import build_pose_pipeline

        class _Profile:
            transforms = [
                {"name": "HorizontalFlip", "enabled": True, "p": 0.5},
                {"name": "RandomBrightnessContrast", "enabled": True, "p": 0.5},
            ]

        pipeline = build_pose_pipeline(_Profile())
        names = [type(t).__name__ for t in pipeline.transforms]
        self.assertNotIn("HorizontalFlip", names)
        self.assertIn("RandomBrightnessContrast", names)

    def test_augment_pose_preserves_keypoint_count(self) -> None:
        from src.augmentation.engine import _augment_pose, build_pose_pipeline

        class _Profile:
            transforms = [{"name": "RandomBrightnessContrast", "enabled": True, "p": 1.0}]

        pipeline = build_pose_pipeline(_Profile())
        img = np.full((32, 32, 3), 100, dtype=np.uint8)
        bboxes = [[0.5, 0.5, 0.4, 0.4]]
        keypoints = [[0.4, 0.4, 2.0, 0.6, 0.6, 2.0]]
        results = _augment_pose(img, bboxes, keypoints, [0], pipeline, multiplier=2, kpt_shape=(2, 3))
        self.assertEqual(len(results), 2)
        for _img, aug_bb, aug_kp, aug_cls in results:
            self.assertEqual(len(aug_bb), 1)
            self.assertEqual(len(aug_kp[0]), 6)
            self.assertEqual(aug_cls, [0])


if __name__ == "__main__":
    unittest.main()
