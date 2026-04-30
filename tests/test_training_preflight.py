from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.utils.training_preflight import (
    _find_uv_project_root,
    validate_export_config,
)


class FindUvProjectRootTests(unittest.TestCase):
    def test_returns_directory_containing_uv_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            (root / "uv.lock").touch()

            self.assertEqual(_find_uv_project_root(root), root)

    def test_walks_up_to_find_uv_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            (root / "uv.lock").touch()
            nested = root / "src" / "deep" / "child"
            nested.mkdir(parents=True)

            self.assertEqual(_find_uv_project_root(nested), root)

    def test_does_not_match_subtree_without_uv_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            nested = root / "a" / "b"
            nested.mkdir(parents=True)

            # The function may still find a uv.lock further up the filesystem,
            # but it must not falsely report `root` when no uv.lock exists there.
            self.assertNotEqual(_find_uv_project_root(nested), root)


class ValidateExportConfigTests(unittest.TestCase):
    def test_onnx_export_fails_preflight_when_slim_dependency_is_missing(self) -> None:
        installed = {"onnx": "1.15.0", "onnxruntime": "1.23.2"}

        def fake_version(package_name: str) -> str:
            if package_name not in installed:
                raise PackageNotFoundError(package_name)
            return installed[package_name]

        from importlib.metadata import PackageNotFoundError

        with patch("src.utils.training_preflight.metadata.version", fake_version):
            is_valid, errors = validate_export_config({"format": "onnx"})

        self.assertFalse(is_valid)
        self.assertTrue(any("onnxslim>=0.1.71" in error for error in errors))

    def test_non_onnx_export_does_not_require_onnx_runtime_dependencies(self) -> None:
        from importlib.metadata import PackageNotFoundError

        with patch(
            "src.utils.training_preflight.metadata.version",
            side_effect=PackageNotFoundError,
        ):
            is_valid, errors = validate_export_config({"format": "torchscript"})

        self.assertTrue(is_valid)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
