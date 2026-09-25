from __future__ import annotations

import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from src.utils import ml_dependencies


class ImportCv2Tests(unittest.TestCase):
    @staticmethod
    def _opencv_distributions() -> list[object]:
        desktop = type("Distribution", (), {"metadata": {"Name": "opencv-python"}})()
        headless = type(
            "Distribution", (), {"metadata": {"Name": "opencv-python-headless"}}
        )()
        return [desktop, headless]

    def test_accepts_multiple_opencv_distributions_when_cv2_is_healthy(self) -> None:
        healthy = ModuleType("cv2")
        for attr in ml_dependencies._CV2_REQUIRED_ATTRS:
            setattr(healthy, attr, object())

        with (
            patch.object(ml_dependencies, "prepare_ml_runtime"),
            patch.object(
                ml_dependencies.metadata,
                "distributions",
                return_value=self._opencv_distributions(),
            ),
            patch.object(
                ml_dependencies.importlib, "import_module", return_value=healthy
            ) as import_module,
        ):
            self.assertIs(ml_dependencies.import_cv2(), healthy)

        import_module.assert_called_once_with("cv2")

    def test_retries_known_partial_cv2_import_after_purging_submodules(self) -> None:
        stale_cv2 = ModuleType("cv2")
        stale_gapi = ModuleType("cv2.gapi")
        replacement = ModuleType("cv2")
        for attr in ml_dependencies._CV2_REQUIRED_ATTRS:
            setattr(replacement, attr, object())
        original_modules = {
            name: sys.modules.get(name) for name in ("cv2", "cv2.gapi")
        }
        sys.modules["cv2"] = stale_cv2
        sys.modules["cv2.gapi"] = stale_gapi
        try:
            with (
                patch.object(ml_dependencies, "prepare_ml_runtime"),
                patch.object(ml_dependencies.metadata, "distributions", return_value=[]),
                patch.object(
                    ml_dependencies.importlib,
                    "import_module",
                    side_effect=[
                        AttributeError(
                            "partially initialized module 'cv2' has no attribute "
                            "'gapi_wip_gst_GStreamerPipeline'"
                        ),
                        replacement,
                    ],
                ) as import_module,
            ):
                self.assertIs(ml_dependencies.import_cv2(), replacement)

            self.assertEqual(import_module.call_count, 2)
            self.assertNotIn("cv2.gapi", sys.modules)
        finally:
            for name, module in original_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_rejects_cv2_module_missing_core_attributes(self) -> None:
        broken = ModuleType("cv2")  # simulates a corrupted install: imports fine, no real bindings

        with (
            patch.object(ml_dependencies, "prepare_ml_runtime"),
            patch.object(ml_dependencies.metadata, "distributions", return_value=[]),
            patch.object(ml_dependencies.importlib, "import_module", return_value=broken),
        ):
            with self.assertRaisesRegex(ml_dependencies.MLDependencyError, "missing core attributes"):
                ml_dependencies.import_cv2()

    def test_corrupt_cv2_error_reports_multiple_distributions(self) -> None:
        broken = ModuleType("cv2")

        with (
            patch.object(ml_dependencies, "prepare_ml_runtime"),
            patch.object(
                ml_dependencies.metadata,
                "distributions",
                return_value=self._opencv_distributions(),
            ),
            patch.object(ml_dependencies.importlib, "import_module", return_value=broken),
        ):
            with self.assertRaisesRegex(
                ml_dependencies.MLDependencyError, "Multiple OpenCV wheels are installed"
            ):
                ml_dependencies.import_cv2()


class RuntimeLibraryEnvironmentTests(unittest.TestCase):
    def test_uses_dyld_library_path_on_macos(self) -> None:
        with (
            patch.object(ml_dependencies, "_IS_WINDOWS", False),
            patch.object(ml_dependencies.platform, "system", return_value="Darwin"),
        ):
            self.assertEqual(
                ml_dependencies._runtime_library_environment_name(),
                "DYLD_LIBRARY_PATH",
            )


if __name__ == "__main__":
    unittest.main()
