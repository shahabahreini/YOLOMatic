from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.trainers.rfdetr_trainer import (
    find_rfdetr_checkpoint,
    instantiate_rfdetr_model,
    prepare_training_params,
)


class RFDETRTrainerTests(unittest.TestCase):
    def test_prepare_training_params_keeps_auto_download_out_of_train_kwargs(self) -> None:
        config = {
            "training": {
                "epochs": 10,
                "pretrain_weights": "checkpoint.pth",
                "output_dir": "runs/rfdetr",
            }
        }

        self.assertEqual(prepare_training_params(config), {"epochs": 10})

    def test_find_rfdetr_checkpoint_prefers_best_total(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "checkpoint.pth").touch()
            best = run_dir / "nested" / "checkpoint_best_total.pth"
            best.parent.mkdir()
            best.touch()

            self.assertEqual(find_rfdetr_checkpoint(run_dir), best)

    @patch("src.trainers.rfdetr_trainer.import_rfdetr_model_class")
    def test_instantiate_rfdetr_model_omits_pretrain_for_auto_download(self, mock_import) -> None:
        class FakeModel:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        mock_import.return_value = FakeModel
        config = {
            "settings": {"model_type": "RF-DETR-Small"},
            "training": {},
        }

        model = instantiate_rfdetr_model(config)

        mock_import.assert_called_once_with("RFDETRSmall")
        self.assertEqual(model.kwargs, {})

    @patch("src.trainers.rfdetr_trainer._validate_local_weights")
    @patch("src.trainers.rfdetr_trainer.import_rfdetr_model_class")
    def test_instantiate_rfdetr_model_passes_finetune_checkpoint(self, mock_import, mock_validate) -> None:
        class FakeModel:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        mock_import.return_value = FakeModel
        config = {
            "settings": {"model_type": "RF-DETR-Medium"},
            "training": {"pretrain_weights": "checkpoint.pth"},
        }

        model = instantiate_rfdetr_model(config)

        mock_import.assert_called_once_with("RFDETRMedium")
        self.assertEqual(model.kwargs, {"pretrain_weights": "checkpoint.pth"})


if __name__ == "__main__":
    unittest.main()
