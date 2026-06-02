from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

from src.utils.export_config import build_export_kwargs


@dataclass(frozen=True)
class Param:
    name: str
    default: Any
    value_type: str


PARAMS = [
    Param("workspace", 4.0, "float"),
    Param("dynamic", False, "bool"),
    Param("batch", 1, "int"),
    Param("opset", 17, "int"),
    Param("data", "", "str"),
]


class ExportCliTests(unittest.TestCase):
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

    def test_tensorrt_opset_is_limited_to_compatible_value(self) -> None:
        kwargs = build_export_kwargs(
            "engine",
            PARAMS,
            {"opset"},
            {"opset": 17},
        )

        self.assertEqual(kwargs["opset"], 12)

    def test_empty_optional_string_values_are_omitted(self) -> None:
        kwargs = build_export_kwargs(
            "onnx",
            PARAMS,
            {"format", "data"},
            {"data": ""},
        )

        self.assertNotIn("data", kwargs)


if __name__ == "__main__":
    unittest.main()
