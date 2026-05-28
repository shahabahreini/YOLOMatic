from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from src.config.settings import (
    DEFAULT_SETTINGS,
    load_settings,
    reset_settings,
    save_settings,
    snapshot_clearml_settings,
    snapshot_roboflow_settings,
)


class SettingsTests(unittest.TestCase):
    def test_load_settings_merges_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "settings.yaml"
            path.write_text("clearml:\n  enabled: false\n", encoding="utf-8")

            settings = load_settings(path)

            self.assertFalse(settings["clearml"]["enabled"])
            self.assertEqual(settings["clearml"]["task_name_format"], "%Y-%m-%d-%H-%M")
            self.assertIn("roboflow", settings)
            self.assertIn("ultralytics", settings)

    def test_invalid_narrative_mode_falls_back_to_guided(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "settings.yaml"
            path.write_text("narratives:\n  mode: noisy\n", encoding="utf-8")

            self.assertEqual(load_settings(path)["narratives"]["mode"], "guided")

    def test_save_and_reset_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "settings.yaml"
            settings = load_settings(path)
            settings["roboflow"]["auto_upload_after_training"] = True
            save_settings(settings, path)

            persisted = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertTrue(persisted["roboflow"]["auto_upload_after_training"])

            reset = reset_settings(path)
            self.assertEqual(reset, DEFAULT_SETTINGS)
            self.assertFalse(load_settings(path)["roboflow"]["auto_upload_after_training"])

    def test_snapshots_do_not_include_roboflow_secrets(self) -> None:
        settings = load_settings(Path("/tmp/does-not-exist-yolomatic-settings.yaml"))

        clearml = snapshot_clearml_settings(settings, "YOLO", "YOLO11N")
        roboflow = snapshot_roboflow_settings(settings)

        self.assertEqual(clearml["project_name"], "YOLO Training - YOLO11N")
        self.assertNotIn("api_key", roboflow)
        self.assertIn("upload", roboflow)

    def test_ultralytics_settings_do_not_include_api_key(self) -> None:
        settings = load_settings(Path("/tmp/does-not-exist-yolomatic-settings.yaml"))

        self.assertIn("ultralytics", settings)
        self.assertNotIn("api_key", settings["ultralytics"])


if __name__ == "__main__":
    unittest.main()
