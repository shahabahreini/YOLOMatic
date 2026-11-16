from __future__ import annotations

import sys
import unittest
from unittest.mock import Mock, patch


class UpdateFlowTest(unittest.TestCase):
    def test_upgrade_uses_uv_pip_with_active_python(self) -> None:
        from yolomatic.cli import run

        completed = Mock(returncode=0)
        with patch("yolomatic.cli.updates.shutil.which", return_value="/usr/bin/uv"), \
             patch("subprocess.run", return_value=completed) as run_mock:
            self.assertTrue(run._run_pip_upgrade(["ultralytics"]))

        command = run_mock.call_args.args[0]
        self.assertEqual(command[:5], ["uv", "pip", "install", "--python", sys.executable])
        self.assertIn("--upgrade", command)
        self.assertIn("ultralytics", command)
        # The old hard-coded UV_CACHE_DIR=/tmp/uv-cache was wrong on Windows.
        self.assertNotIn("UV_CACHE_DIR", run_mock.call_args.kwargs.get("env", {}) or {})
        # Output streams to the terminal instead of being captured behind a spinner.
        self.assertNotIn("capture_output", run_mock.call_args.kwargs)

    def test_falls_back_to_pip_without_uv(self) -> None:
        from yolomatic.cli.updates import build_upgrade_command

        with patch("yolomatic.cli.updates.shutil.which", return_value=None):
            command = build_upgrade_command(["torch"])
        self.assertEqual(command[:5], [sys.executable, "-m", "pip", "install", "--upgrade"])

    def test_failed_launch_is_reported_not_raised(self) -> None:
        from yolomatic.cli import updates

        with patch("subprocess.run", side_effect=FileNotFoundError("uv")):
            self.assertFalse(updates._run_pip_upgrade(["ultralytics"]))


if __name__ == "__main__":
    unittest.main()
