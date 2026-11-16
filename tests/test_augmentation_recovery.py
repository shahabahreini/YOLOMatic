import tempfile
import unittest
from pathlib import Path

from yolomatic.augmentation.recovery import find_recoverable_augmentations, recover_augmentation
from yolomatic.utils.fs import COMPLETION_MARKER


class AugmentationRecoveryTest(unittest.TestCase):
    def test_detects_and_recovers_orphaned_augmentation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            staged = root / ".my_dataset_augmented.augmenting-63d5fb64ccba436897653d1f6e06c4a6"
            output_dir = staged / "output"
            train_images = output_dir / "train" / "images"
            train_images.mkdir(parents=True)
            (train_images / "aug_train_000000.jpg").write_text("img0", encoding="utf-8")
            (train_images / "aug_train_000001.jpg").write_text("img1", encoding="utf-8")
            (output_dir / "data.yaml").write_text("names: [cat, dog]\n", encoding="utf-8")
            (output_dir / COMPLETION_MARKER).write_text("ok\n", encoding="utf-8")

            recoverable = find_recoverable_augmentations(root)
            self.assertEqual(len(recoverable), 1)
            rec = recoverable[0]
            self.assertEqual(rec.target_name, "my_dataset_augmented")
            self.assertEqual(rec.image_count, 2)
            self.assertTrue(rec.has_data_yaml)

            success = recover_augmentation(rec)
            self.assertTrue(success)

            target = root / "my_dataset_augmented"
            self.assertTrue(target.exists())
            self.assertTrue((target / "data.yaml").exists())
            self.assertEqual((target / "train" / "images" / "aug_train_000000.jpg").read_text(encoding="utf-8"), "img0")
            self.assertFalse(staged.exists())
            self.assertFalse((target / COMPLETION_MARKER).exists())

    def test_partial_runs_are_not_offered_and_existing_targets_are_protected(self) -> None:
        # Regression: a run interrupted while writing (images but no completion
        # marker) was offered for recovery, and recovery silently replaced an
        # existing dataset of the same name.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            partial = root / ".half.augmenting-" "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" / "output" / "train" / "images"
            partial.mkdir(parents=True)
            (partial / "aug_train_000000.jpg").write_text("x", encoding="utf-8")
            (partial.parents[1] / "data.yaml").write_text("names: [a]\n", encoding="utf-8")
            self.assertEqual(find_recoverable_augmentations(root), [])

            (partial.parents[1] / COMPLETION_MARKER).write_text("ok\n", encoding="utf-8")
            (root / "half").mkdir()
            (root / "half" / "keep.txt").write_text("keep", encoding="utf-8")
            rec = find_recoverable_augmentations(root)[0]
            with self.assertRaises(FileExistsError):
                recover_augmentation(rec)
            self.assertTrue((root / "half" / "keep.txt").exists())
            self.assertTrue(recover_augmentation(rec, overwrite=True))
            self.assertTrue((root / "half" / "train" / "images" / "aug_train_000000.jpg").exists())

    def test_ignores_non_matching_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            (root / "regular_dataset").mkdir()
            (root / ".other_hidden_dir").mkdir()
            (root / ".invalid_name.augmenting-short").mkdir()

            recoverable = find_recoverable_augmentations(root)
            self.assertEqual(len(recoverable), 0)


if __name__ == "__main__":
    unittest.main()
