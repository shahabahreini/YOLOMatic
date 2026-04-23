from __future__ import annotations

import importlib
import subprocess
import sys
import unittest


MODULES = [
    "src",
    "src.cli.run",
    "src.cli.predict",
    "src.cli.upload",
    "src.cli.tensorboard_launcher",
    "src.trainers.yolo_trainer",
    "src.trainers.nas_trainer",
    "src.utils.tensorboard",
]


class ImportSmokeTests(unittest.TestCase):
    def test_modules_import_cleanly(self) -> None:
        for module_name in MODULES:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    def test_cli_help_commands(self) -> None:
        commands = [
            [sys.executable, "-m", "src.cli.predict", "--help"],
            [sys.executable, "-m", "src.cli.upload", "--help"],
            [sys.executable, "-m", "src.trainers.yolo_trainer", "--help"],
        ]
        for command in commands:
            with self.subTest(command=" ".join(command)):
                process = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertIn("usage:", process.stdout.lower())
