import tempfile
import unittest
from pathlib import Path

from src.augmentation.recovery import find_recoverable_augmentations, recover_augmentation


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
