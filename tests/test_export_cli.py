from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

from src.utils.export_config import build_export_kwargs
from src.utils.export_config import (
    ExportModelDetails,
    extract_model_details,
    filter_export_defaults,
    filter_export_definitions,
)


@dataclass(frozen=True)
class Param:
    name: str
    default: Any
    value_type: str


PARAMS = [
    Param("half", False, "bool"),
    Param("workspace", 4.0, "float"),
    Param("dynamic", False, "bool"),
    Param("batch", 1, "int"),
    Param("opset", 17, "int"),
    Param("data", "", "str"),
]


class ExportCliTests(unittest.TestCase):
    def test_load_model_details_reads_task_and_classes_from_loaded_model(self) -> None:
        class Model:
            task = "segment"
            names = {1: "tree", 0: "weed"}
            model_name = "yolo11s-seg"

        details = extract_model_details("best.pt", Model())

        self.assertEqual(details.task, "segment")
        self.assertEqual(details.class_count, 2)
        self.assertEqual(details.class_names, ("weed", "tree"))
        self.assertEqual(details.model_name, "yolo11s-seg")

    def test_tensorrt_workspace_is_preserved_for_all_gpu_sizes(self) -> None:
        kwargs = build_export_kwargs(
            "engine",
            PARAMS,
            {"workspace", "dynamic", "batch", "opset"},
            {"workspace": 18.0, "dynamic": True, "batch": 4, "opset": 12},
        )

        self.assertEqual(kwargs["workspace"], 18.0)

    def test_tensorrt_dynamic_batch_defaults_to_max_batch_when_batch_is_one(self) -> None:
        kwargs = build_export_kwargs(
            "engine",
            PARAMS,
            {"dynamic", "batch", "opset"},
            {"dynamic": True, "batch": 1, "opset": 12},
        )

        self.assertEqual(kwargs["batch"], 16)

    def test_tensorrt_opset_is_preserved_for_non_segmentation_exports(self) -> None:
        kwargs = build_export_kwargs(
            "engine",
            PARAMS,
            {"half", "dynamic", "opset"},
            {"half": True, "dynamic": True, "opset": 17},
            model_details=ExportModelDetails(path="best.pt", task="detect"),
        )

        self.assertEqual(kwargs["opset"], 17)

    def test_tensorrt_opset_is_limited_for_dynamic_fp16_segmentation(self) -> None:
        warnings: list[str] = []

        kwargs = build_export_kwargs(
            "engine",
            PARAMS,
            {"half", "dynamic", "opset"},
            {"half": True, "dynamic": True, "opset": 17},
            warn=warnings.append,
            model_details=ExportModelDetails(path="best.pt", task="segment"),
        )

        self.assertEqual(kwargs["opset"], 12)
        self.assertTrue(any("dynamic FP16 segmentation" in warning for warning in warnings))

    def test_empty_optional_string_values_are_omitted(self) -> None:
        kwargs = build_export_kwargs(
            "onnx",
            PARAMS,
            {"format", "data"},
            {"data": ""},
        )

        self.assertNotIn("data", kwargs)

    def test_engine_options_keep_workspace_and_hide_keras_optimize(self) -> None:
        filtered = filter_export_definitions(
            PARAMS + [Param("keras", False, "bool"), Param("optimize", False, "bool")],
            "engine",
            ExportModelDetails(path="best.pt", task="detect"),
        )

        names = {param.name for param in filtered}
        self.assertIn("workspace", names)
        self.assertNotIn("keras", names)
        self.assertNotIn("optimize", names)

    def test_classification_model_hides_nms(self) -> None:
        filtered = filter_export_definitions(
            PARAMS + [Param("nms", False, "bool")],
            "onnx",
            ExportModelDetails(path="best.pt", task="classify"),
        )

        self.assertNotIn("nms", {param.name for param in filtered})

    def test_format_defaults_are_filtered_to_applicable_options(self) -> None:
        defaults = filter_export_defaults(
            {
                "half": False,
                "workspace": 4.0,
                "keras": False,
                "optimize": False,
                "opset": 17,
            },
            "torchscript",
            ExportModelDetails(path="best.pt", task="detect"),
        )

        self.assertEqual(defaults, {"half": False, "optimize": False})


if __name__ == "__main__":
    unittest.main()
