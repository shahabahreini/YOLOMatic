"""Semantic segmentation must not fall through the gaps left by detect/segment/pose."""
import tempfile
import unittest
from pathlib import Path

import yaml

from src.cli.run import _base_model_name
from src.config.parameters import parameters_for
from src.utils.export_config import (
    ExportModelDetails,
    TASK_LIMITED_FORMATS,
    supported_formats_for_model,
)
from src.utils.project import infer_ultralytics_task_from_name


class ModelNameTaskTest(unittest.TestCase):
    def test_sem_suffix_resolves_to_the_semantic_task(self) -> None:
        self.assertEqual(infer_ultralytics_task_from_name("yolo26x-sem"), "semantic")

    def test_base_name_strips_a_task_suffix(self) -> None:
        for name, expected in (
            ("yolo26x-sem", "yolo26x"),
            ("yolo26x-seg", "yolo26x"),
            ("yolo26n-pose", "yolo26n"),
            ("yolo11n", "yolo11n"),
        ):
            with self.subTest(name=name):
                self.assertEqual(_base_model_name(name), expected)

    def test_variant_suggestions_never_stack_two_task_suffixes(self) -> None:
        # "yolo26x-sem" + "-seg" used to yield the non-existent "yolo26x-sem-seg".
        for name in ("yolo26x-sem", "yolo26x-seg", "yolo26n-pose"):
            with self.subTest(name=name):
                suggestion = f"{_base_model_name(name)}-seg"
                self.assertEqual(suggestion.count("-"), 1)


class ShortNameTest(unittest.TestCase):
    def test_sem_models_keep_their_size_letter_in_compact_labels(self) -> None:
        from src.utils.tui import model_chart_short_name as short_name

        for name, expected_size in (
            ("YOLO26X-sem", "X"),
            ("YOLO26N-sem", "N"),
            ("YOLO26M-seg", "M"),
        ):
            with self.subTest(name=name):
                # Without "-sem" in the suffix list every -sem row collapsed to "M",
                # the trailing letter of the suffix itself.
                self.assertIn(expected_size.lower(), short_name(name).lower())


class SemanticParameterCatalogTest(unittest.TestCase):
    def test_semantic_drops_box_mask_and_nms_parameters(self) -> None:
        names = {p.name for p in parameters_for("yolo", "semantic")}
        for unused in ("box", "dfl", "overlap_mask", "mask_ratio", "single_cls", "iou"):
            self.assertNotIn(unused, names)

    def test_semantic_keeps_the_shared_training_knobs(self) -> None:
        names = {p.name for p in parameters_for("yolo", "semantic")}
        for kept in ("epochs", "batch", "imgsz", "lr0", "optimizer", "patience"):
            self.assertIn(kept, names)

    def test_segmentation_still_offers_its_mask_parameters(self) -> None:
        names = {p.name for p in parameters_for("yolo", "segmentation")}
        self.assertIn("overlap_mask", names)
        self.assertIn("box", names)


class SemanticExportTest(unittest.TestCase):
    def test_edgetpu_stays_available_for_semantic_models(self) -> None:
        self.assertIn("semantic", TASK_LIMITED_FORMATS["edgetpu"])

    def test_semantic_offers_the_same_formats_as_segmentation(self) -> None:
        format_map = {"ONNX": "onnx", "TensorRT": "engine", "Edge TPU": "edgetpu"}
        semantic = supported_formats_for_model(
            format_map, ExportModelDetails(path="m.pt", task="semantic")
        )
        segmentation = supported_formats_for_model(
            format_map, ExportModelDetails(path="m.pt", task="segment")
        )
        self.assertEqual(set(semantic), set(segmentation))
        # edgetpu was silently dropped for semantic before it joined the allow-list.
        self.assertIn("Edge TPU", semantic)


class SemanticDatasetDetectionTest(unittest.TestCase):
    def _generator_for(self, dataset: Path):
        from src.config.generator import YOLOConfigGenerator

        generator = YOLOConfigGenerator(str(dataset))
        # Loads data.yaml and only then classifies the dataset, which is the order
        # that lets a declared "task:" key win over label-shape counting.
        generator.extract_dataset_info()
        return generator

    def test_declared_semantic_task_is_reported_as_semantic(self) -> None:
        # Polygon labels are byte-identical to instance segmentation, so only the
        # declaration in data.yaml can distinguish the two.
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "ds"
            (dataset / "train" / "images").mkdir(parents=True)
            (dataset / "train" / "labels").mkdir(parents=True)
            (dataset / "train" / "labels" / "a.txt").write_text(
                "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n", encoding="utf-8"
            )
            (dataset / "data.yaml").write_text(
                yaml.dump({
                    "names": ["veg"], "nc": 1, "task": "semantic",
                    "train": "train/images", "val": "train/images",
                }),
                encoding="utf-8",
            )
            generator = self._generator_for(dataset)
            self.assertEqual(generator.dataset_info.get("task_type"), "semantic")

    def test_segmentation_dataset_is_still_reported_as_segmentation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "ds"
            (dataset / "train" / "images").mkdir(parents=True)
            (dataset / "train" / "labels").mkdir(parents=True)
            (dataset / "train" / "labels" / "a.txt").write_text(
                "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n", encoding="utf-8"
            )
            (dataset / "data.yaml").write_text(
                yaml.dump({
                    "names": ["veg"], "nc": 1, "task": "segment",
                    "train": "train/images", "val": "train/images",
                }),
                encoding="utf-8",
            )
            generator = self._generator_for(dataset)
            self.assertEqual(generator.dataset_info.get("task_type"), "segmentation")


class SemanticPrepareOutputTest(unittest.TestCase):
    def test_yolo_semantic_is_an_offered_output_format(self) -> None:
        from src.datasets.prepare import OUTPUT_FORMATS

        self.assertIn("YOLO Semantic", OUTPUT_FORMATS)


class BenchmarkSemanticGroundTruthTest(unittest.TestCase):
    def test_ground_truth_uses_the_models_class_indexing(self) -> None:
        import numpy as np

        from src.benchmark.engine import GTObject, _semantic_ground_truth

        mask = np.zeros((8, 8), dtype=bool)
        mask[2:5, 2:5] = True

        # Binary: background 0, the single foreground class 1.
        gt = GTObject(cls=0, box_xyxy=(2.0, 2.0, 5.0, 5.0), mask=mask)
        binary = _semantic_ground_truth([gt], 8, 8, 1)
        self.assertEqual(set(np.unique(binary).tolist()), {0, 1})

        # Multi-class: class ids are NOT shifted, background sits after them.
        multi = _semantic_ground_truth([gt], 8, 8, 3)
        self.assertEqual(sorted(np.unique(multi).tolist()), [0, 3])
        self.assertEqual(int(multi[3, 3]), 0)


if __name__ == "__main__":
    unittest.main()
