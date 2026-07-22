from __future__ import annotations

import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from src.utils import ml_dependencies


class ImportCv2Tests(unittest.TestCase):
    def test_retries_known_partial_cv2_import_after_purging_submodules(self) -> None:
        stale_cv2 = ModuleType("cv2")
        stale_gapi = ModuleType("cv2.gapi")
        replacement = ModuleType("cv2")
        original_modules = {
            name: sys.modules.get(name) for name in ("cv2", "cv2.gapi")
        }
        sys.modules["cv2"] = stale_cv2
        sys.modules["cv2.gapi"] = stale_gapi
        try:
            with (
                patch.object(ml_dependencies, "prepare_ml_runtime"),
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


if __name__ == "__main__":
    unittest.main()
