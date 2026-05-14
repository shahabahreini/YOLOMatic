from __future__ import annotations

import sys
import unittest
from unittest.mock import Mock, patch


class UpdateFlowTest(unittest.TestCase):
    def test_upgrade_uses_uv_pip_with_active_python(self) -> None:
        from src.cli import run

        completed = Mock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=completed) as run_mock:
            self.assertTrue(run._run_pip_upgrade(["ultralytics"]))

        command = run_mock.call_args.args[0]
        kwargs = run_mock.call_args.kwargs

        self.assertEqual(command[:5], ["uv", "pip", "install", "--python", sys.executable])
        self.assertIn("--upgrade", command)
        self.assertIn("ultralytics", command)
        self.assertEqual(kwargs["env"]["UV_CACHE_DIR"], "/tmp/uv-cache")


if __name__ == "__main__":
    unittest.main()
