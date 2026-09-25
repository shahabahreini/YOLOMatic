from __future__ import annotations

import unittest
from unittest.mock import patch


class Detectron2TrainerTests(unittest.TestCase):
    @patch("src.trainers.detectron2_trainer.train_from_config")
    def test_main_raises_system_exit_on_failure(self, mock_train) -> None:
        from src.trainers.detectron2_trainer import main

        mock_train.side_effect = RuntimeError("Training failed")

        with self.assertRaises(SystemExit) as cm:
            main("dummy_config.yaml")

        self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
