from __future__ import annotations

import unittest
from unittest.mock import patch


class Detectron2TrainerTests(unittest.TestCase):
    @patch("yolomatic.trainers.detectron2_trainer.train_from_config")
    def test_main_reports_failure_result(self, mock_train) -> None:
        from yolomatic.trainers.detectron2_trainer import main

        mock_train.side_effect = RuntimeError("Training failed")

        # Failures are reported as a FAILED CommandResult (exit code 1 when run
        # from the shell) instead of raising SystemExit inside the TUI process.
        result = main("dummy_config.yaml")

        self.assertEqual(result.exit_code, 1)
        self.assertIn("Training failed", result.message)


if __name__ == "__main__":
    unittest.main()
