import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from src.augmentation.engine import SplitConfig, run_augmentation


class AugmentationProgressAndMoveTest(unittest.TestCase):
    def _create_source_dataset(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "data.yaml").write_text("names: [tree, rock]\nnc: 2\n", encoding="utf-8")
        train_img_dir = root / "train" / "images"
        train_lbl_dir = root / "train" / "labels"
        train_img_dir.mkdir(parents=True)
        train_lbl_dir.mkdir(parents=True)

        for i in range(5):
            img = np.full((20, 20, 3), 100 + i * 10, dtype=np.uint8)
            cv2.imwrite(str(train_img_dir / f"img_{i:02d}.jpg"), img)
            (train_lbl_dir / f"img_{i:02d}.txt").write_text(f"0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    def test_run_augmentation_reports_granular_writing_progress_and_moves_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_dir = temp_path / "source"
            self._create_source_dataset(source_dir)

            profile = SimpleNamespace(
                name="test_profile",
                multiplier=1,
                include_originals=True,
                seed=42,
                transforms=[],
            )
            split_config = SplitConfig(train_ratio=0.8, val_ratio=0.2, test_ratio=0.0)

            callbacks = []

            def track_progress(done: int, total: int, msg: str) -> None:
                callbacks.append((done, total, msg))

            # Mock worker execution to return pre-staged items without needing Albumentations
            def mock_process_job(index, image_path_text, label_path_text, stage_root_text, seed, n_variants=None):
                group_dir = Path(stage_root_text) / f"source_{index:06d}"
                group_dir.mkdir(parents=True, exist_ok=True)
                img = cv2.imread(image_path_text)
                cv2.imwrite(str(group_dir / "original.jpg"), img)
                (group_dir / "original.json").write_text(
                    json.dumps({"annotations": [[0.5, 0.5, 0.2, 0.2]], "class_ids": [0]}),
                    encoding="utf-8",
                )
                return index, str(group_dir), 0

            from concurrent.futures import Future

            with patch("src.augmentation.engine._process_augmentation_job", side_effect=mock_process_job):
                with patch("src.augmentation.engine.ProcessPoolExecutor") as mock_executor_cls:
                    # Make executor run synchronously in test using real completed Future
                    mock_executor = MagicMock()
                    mock_executor.__enter__.return_value = mock_executor
                    mock_executor.__exit__.return_value = None

                    def fake_submit(fn, *args, **kwargs):
                        fut = Future()
                        res = fn(*args, **kwargs)
                        fut.set_result(res)
                        return fut

                    mock_executor.submit.side_effect = fake_submit
                    mock_executor_cls.return_value = mock_executor

                    stats = run_augmentation(
                        source_dataset_path=source_dir,
                        output_name="output_augmented",
                        profile=profile,
                        split_config=split_config,
                        output_format="YOLO Detection",
                        progress_callback=track_progress,
                        max_workers=1,
                    )

            output_dir = temp_path / "output_augmented"
            self.assertTrue(output_dir.exists())
            self.assertEqual(stats.total_output_images, 5)

            # Check that writing progress messages were emitted
            writing_messages = [msg for _, _, msg in callbacks if "Writing" in msg]
            self.assertTrue(any("Writing train" in msg or "Writing valid" in msg for msg in writing_messages))
            self.assertTrue(any("data.yaml" in msg for msg in writing_messages))

            # Check that finalization progress message was emitted
            finalizing_messages = [msg for _, _, msg in callbacks if "Finalizing" in msg]
            self.assertEqual(len(finalizing_messages), 1)

            # Verify that output directory has all images and labels
            train_images = list((output_dir / "train" / "images").glob("*.jpg"))
            val_images = list((output_dir / "valid" / "images").glob("*.jpg"))
            self.assertEqual(len(train_images) + len(val_images), 5)
            self.assertTrue((output_dir / "data.yaml").exists())
            self.assertTrue((output_dir / "split_assignment.json").exists())


if __name__ == "__main__":
    unittest.main()
