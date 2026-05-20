from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.trainers.yolo_trainer import (
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

    @patch("src.trainers.yolo_trainer.upload_model")
    @patch("src.trainers.yolo_trainer.build_candidate")
    @patch("src.trainers.yolo_trainer.stage_upload_candidate")
    @patch("pathlib.Path.exists", return_value=True)
    def test_upload_called_if_configured(self, mock_exists, mock_stage, mock_build, mock_upload):
        config = {"roboflow": {"upload": True, "weight": "best.pt"}}
        dataset_config = {"roboflow": {"workspace": "ws", "project": "proj", "version": "1"}}
        run_dir = Path("/tmp/run")
        
        # Setup mock candidate
        mock_candidate = MagicMock()
        mock_build.return_value = mock_candidate
        mock_stage.return_value = mock_candidate
        
        upload_to_roboflow_if_configured(config, dataset_config, run_dir, "yolov11", MagicMock())
        
        # Verify the candidate is built with the right path
        expected_path = run_dir / "weights" / "best.pt"
        mock_build.assert_called_once_with(expected_path)
        
        # Verify candidate overrides are set
        self.assertEqual(mock_candidate.workspace, "ws")
        self.assertEqual(mock_candidate.project_ids, "proj")
        
        mock_stage.assert_called_once_with(mock_candidate, "yolov11")
        mock_upload.assert_called_once_with(mock_candidate, "yolov11")

if __name__ == "__main__":
    unittest.main()
