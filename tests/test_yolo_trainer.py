from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.cli.upload import EnvConfig
from src.trainers.yolo_trainer import (
    configure_ultralytics_runtime,
    disable_ultralytics_clearml_callbacks,
    normalize_class_names,
    print_config_summary,
    upload_to_roboflow_if_configured,
)


def local_callback():
    pass


def clearml_callback():
    pass


clearml_callback.__module__ = "ultralytics.utils.callbacks.clearml"


class FakeModel:
    def __init__(self) -> None:
        self.callbacks = {
            "on_train_end": [local_callback, clearml_callback],
            "on_fit_epoch_end": [clearml_callback],
            "on_val_end": [local_callback],
        }


class YoloTrainerTests(unittest.TestCase):
    @patch("src.trainers.yolo_trainer.import_ultralytics_settings")
    def test_configure_ultralytics_runtime_disables_optional_mlflow(self, mock_settings) -> None:
        settings = {}
        mock_settings.return_value = settings

        configure_ultralytics_runtime()

        self.assertTrue(settings["tensorboard"])
        self.assertFalse(settings["clearml"])
        self.assertFalse(settings["mlflow"])

    def test_disable_ultralytics_clearml_callbacks_preserves_other_callbacks(self) -> None:
        model = FakeModel()

        disable_ultralytics_clearml_callbacks(model)

        self.assertEqual(model.callbacks["on_train_end"], [local_callback])
        self.assertEqual(model.callbacks["on_fit_epoch_end"], [])
        self.assertEqual(model.callbacks["on_val_end"], [local_callback])

    def test_normalize_class_names_handles_mapping_and_non_string_values(self) -> None:
        self.assertEqual(
            normalize_class_names({1: "tree", 0: "vegetation", 2: 7}),
            ["vegetation", "tree", "7"],
        )

    @patch("src.trainers.yolo_trainer.console")
    def test_print_config_summary_accepts_yolo_names_mapping(self, mock_console) -> None:
        config = {
            "settings": {"model_type": "yolo11l", "dataset": "dataset-v12"},
            "clearml": {"project_name": "YOLOmatic"},
            "training": {"batch": 4, "epochs": 1, "imgsz": 640},
        }
        dataset_config = {
            "names": {0: "vegetation"},
        }

        print_config_summary(config, dataset_config)

        self.assertGreaterEqual(mock_console.print.call_count, 2)

    @patch("src.trainers.yolo_trainer.console")
    def test_print_config_summary_accepts_yolo_names_integer_list(self, mock_console) -> None:
        config = {
            "settings": {"model_type": "yolo11l", "dataset": "dataset-v12"},
            "clearml": {"project_name": "YOLOmatic"},
            "training": {"batch": 4, "epochs": 1, "imgsz": 640},
        }
        dataset_config = {
            "names": [0, 1],
        }

        print_config_summary(config, dataset_config)

        self.assertGreaterEqual(mock_console.print.call_count, 2)



class TestRoboflowUpload(unittest.TestCase):
    @patch("src.trainers.yolo_trainer.upload_model")
    @patch("src.trainers.yolo_trainer.build_candidate")
    def test_upload_skipped_if_not_configured(self, mock_build, mock_upload):
        config = {"roboflow": {"upload": False}}
        upload_to_roboflow_if_configured(config, {}, Path("/tmp/run"), "yolo11", MagicMock())
        mock_build.assert_not_called()
        mock_upload.assert_not_called()

    def test_upload_called_with_real_upload_model_signature(self):
        # Regression: the trainer used to mutate a frozen UploadCandidate and call
        # upload_model() with 2 of its 6 arguments, so auto-upload never worked.
        # autospec=True makes the mock enforce the real signature.
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "train"
            (run_dir / "weights").mkdir(parents=True)
            (run_dir / "weights" / "best.pt").write_bytes(b"weights")
            (run_dir / "args.yaml").write_text("model: yolo11n.pt\ntask: detect\n", encoding="utf-8")

            config = {"roboflow": {"upload": True, "weight": "best.pt"}}
            dataset_config = {"roboflow": {"workspace": "ws", "project": "proj-a, proj-b"}}
            env = EnvConfig(api_key="secret", workspace=None, project_ids=[])
            console = MagicMock()

            with patch(
                "src.trainers.yolo_trainer.upload_model", autospec=True
            ) as mock_upload, patch(
                "src.trainers.yolo_trainer.load_env_config", autospec=True, return_value=env
            ):
                upload_to_roboflow_if_configured(config, dataset_config, run_dir, "yolo11n", console)

            mock_upload.assert_called_once()
            api_key, candidate, workspace, project_ids, model_type, model_name = mock_upload.call_args.args
            self.assertEqual(api_key, "secret")
            self.assertEqual(candidate.weight_path, run_dir / "weights" / "best.pt")
            self.assertEqual(workspace, "ws")
            self.assertEqual(project_ids, ["proj-a", "proj-b"])
            self.assertEqual(model_type, "yolov11")
            self.assertEqual(model_name, "train-best")
            printed = " ".join(str(call.args[0]) for call in console.print.call_args_list)
            self.assertIn("completed successfully", printed)
            self.assertNotIn("upload failed", printed)

    def test_upload_reports_missing_api_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "train"
            (run_dir / "weights").mkdir(parents=True)
            (run_dir / "weights" / "best.pt").write_bytes(b"weights")
            config = {"roboflow": {"upload": True, "weight": "best.pt"}}
            dataset_config = {"roboflow": {"workspace": "ws", "project": "proj"}}
            console = MagicMock()
            with patch(
                "src.trainers.yolo_trainer.upload_model", autospec=True
            ) as mock_upload, patch(
                "src.trainers.yolo_trainer.load_env_config",
                autospec=True,
                side_effect=RuntimeError("ROBOFLOW_API_KEY is missing."),
            ):
                upload_to_roboflow_if_configured(config, dataset_config, run_dir, "yolo11n", console)
            mock_upload.assert_not_called()
            printed = " ".join(str(call.args[0]) for call in console.print.call_args_list)
            self.assertIn("ROBOFLOW_API_KEY is missing", printed)

if __name__ == "__main__":
    unittest.main()
