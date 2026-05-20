import unittest
import tempfile
from pathlib import Path

from src.augmentation.profiles import AugmentationProfile
from src.cli.augment import (
    _apply_profile_editor_result,
    _build_profile_editor_state,
    _ensure_transform_entries,
    _quick_dataset_description,
)


class AugmentationTUIEditorTest(unittest.TestCase):
    def _profile(self) -> AugmentationProfile:
        return AugmentationProfile(
            name="test_profile",
            description="Test profile",
            multiplier=2,
            seed=7,
            include_originals=True,
            transforms=[
                {"name": "HorizontalFlip", "enabled": True, "p": 0.5},
                {
                    "name": "RandomBrightnessContrast",
                    "enabled": False,
                    "p": 0.2,
                    "brightness_limit_low": -0.1,
                    "brightness_limit_high": 0.1,
                },
            ],
            created_at="2026-05-16T00:00:00",
            modified_at="2026-05-16T00:00:00",
        )

    def test_flat_editor_state_contains_profile_and_transform_parameters(self) -> None:
        profile = self._profile()
        transform_by_name = _ensure_transform_entries(profile)

        parameters, selected, values, field_map = _build_profile_editor_state(
            profile,
            transform_by_name,
        )
        names = [p.name for p in parameters]

        self.assertIn("multiplier", names)
        self.assertIn("HorizontalFlip.enabled", names)
        self.assertIn("HorizontalFlip.p", names)
        self.assertIn("RandomBrightnessContrast.brightness_limit_low", names)
        self.assertIn("HorizontalFlip.enabled", selected)
        self.assertNotIn("RandomBrightnessContrast.enabled", selected)
        self.assertEqual(values["multiplier"], 2)
        self.assertEqual(field_map["HorizontalFlip.p"], ("HorizontalFlip", "p"))

        enable_param = next(p for p in parameters if p.name == "HorizontalFlip.enabled")
        self.assertIn("When to use:", enable_param.help_text)
        self.assertIn("Watch out:", enable_param.help_text)

    def test_editor_result_applies_profile_values_and_enabled_transforms(self) -> None:
        profile = self._profile()
        transform_by_name = _ensure_transform_entries(profile)
        _, selected, values, field_map = _build_profile_editor_state(
            profile,
            transform_by_name,
        )
        selected.discard("HorizontalFlip.enabled")
        selected.update(
            {
                "RandomBrightnessContrast.enabled",
                "RandomBrightnessContrast.p",
                "RandomBrightnessContrast.brightness_limit_low",
                "RandomBrightnessContrast.brightness_limit_high",
            }
        )
        values["multiplier"] = 4
        values["seed"] = 99
        values["include_originals"] = False
        values["RandomBrightnessContrast.p"] = 0.7
        values["RandomBrightnessContrast.brightness_limit_low"] = -0.2

        updated = _apply_profile_editor_result(
            profile,
            transform_by_name,
            selected,
            values,
            field_map,
        )
        by_name = {t["name"]: t for t in updated.transforms}

        self.assertEqual(updated.multiplier, 4)
        self.assertEqual(updated.seed, 99)
        self.assertFalse(updated.include_originals)
        self.assertFalse(by_name["HorizontalFlip"]["enabled"])
        self.assertTrue(by_name["RandomBrightnessContrast"]["enabled"])
        self.assertEqual(by_name["RandomBrightnessContrast"]["p"], 0.7)
        self.assertEqual(by_name["RandomBrightnessContrast"]["brightness_limit_low"], -0.2)

    def test_quick_dataset_description_uses_yaml_without_full_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "dataset-82c47944"
            dataset.mkdir()
            (dataset / "data.yaml").write_text(
                "task: segment\n"
                "names:\n"
                "  0: vegetation\n"
                "train: images/train\n"
                "val: images/val\n"
                "test: images/test\n",
                encoding="utf-8",
            )

            description = _quick_dataset_description(dataset.name, dataset)

            self.assertIn("Format: [yellow]yolo[/yellow]", description)
            self.assertIn("Task: [yellow]segment[/yellow]", description)
            self.assertIn("Classes: [yellow]1[/yellow]", description)
            self.assertIn("vegetation", description)


if __name__ == "__main__":
    unittest.main()
