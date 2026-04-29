from __future__ import annotations

import unittest

from src.trainers.yolo_trainer import disable_ultralytics_clearml_callbacks


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


if __name__ == "__main__":
    unittest.main()
