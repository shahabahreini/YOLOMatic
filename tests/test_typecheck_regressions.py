"""Runtime bugs found by enabling pyrefly in CI."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TypecheckRegressionTest(unittest.TestCase):
    def test_upload_dataset_picker_lists_dataset_folders(self) -> None:
        # Regression: entry dicts from list_dataset_directories were used as Paths
        # (AttributeError as soon as any dataset existed), and the project root was
        # scanned instead of datasets/.
        from yolomatic.cli import upload_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "datasets" / "leaves").mkdir(parents=True)
            (root / "datasets" / "leaves" / "data.yaml").write_text("names: [a]\n", encoding="utf-8")
            (root / "src").mkdir()
            seen: dict = {}

            def choose(options, **kwargs):
                seen["options"] = options
                return options[0]

            with patch.object(upload_dataset, "get_user_choice", side_effect=choose):
                chosen = upload_dataset.step_select_dataset(root, None)
        self.assertEqual(chosen, root / "datasets" / "leaves")
        self.assertNotIn("src", seen["options"])

    def test_benchmark_custom_dataset_prompt_builds(self) -> None:
        # Regression: the "Enter custom dataset root" prompt omitted the required
        # help_text and raised TypeError.
        from yolomatic.cli import benchmark

        with patch.object(benchmark, "_collect_val_candidates", return_value=[]), \
             patch.object(benchmark, "get_user_choice", return_value="Enter custom dataset root..."), \
             patch.object(benchmark, "get_parameter_value_input", return_value=None) as prompt:
            result = benchmark._select_dataset_splits("detection")
        prompt.assert_called_once()
        self.assertTrue(prompt.call_args.args[0].help_text)
        self.assertNotIsInstance(result, dict)

    def test_augmentation_cleanup_survives_pool_start_failure(self) -> None:
        # Regression: if the worker pool failed to start, the cleanup path hit an
        # unbound `pending` and raised NameError instead of the real error.
        from yolomatic.augmentation import engine
        from yolomatic.augmentation.engine import SplitConfig

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "src"
            (source / "train" / "images").mkdir(parents=True)
            (source / "train" / "labels").mkdir(parents=True)
            (source / "data.yaml").write_text("names: [a]\ntrain: train/images\n", encoding="utf-8")
            import cv2
            import numpy as np

            cv2.imwrite(str(source / "train" / "images" / "a.jpg"), np.zeros((8, 8, 3), np.uint8))
            (source / "train" / "labels" / "a.txt").write_text("0 .5 .5 .2 .2\n", encoding="utf-8")
            profile = type("P", (), {"name": "p", "multiplier": 1, "include_originals": True, "seed": 0, "transforms": []})()
            with patch.object(engine, "ProcessPoolExecutor", side_effect=OSError("pool failed")):
                with self.assertRaises(OSError):
                    engine.run_augmentation(source, "out", profile, SplitConfig(1.0, 0.0, 0.0), max_workers=1)
            leftovers = [p.name for p in source.parent.iterdir() if ".augmenting-" in p.name]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
