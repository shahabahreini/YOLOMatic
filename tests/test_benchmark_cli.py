"""Tests for benchmark TUI integration."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path


class TestBenchmarkMenuIntegration(unittest.TestCase):
    """Verify the benchmark menu item is present in src/cli/run.py without importing ML deps."""

    _RUN_PY = Path(__file__).parent.parent / "src" / "cli" / "run.py"

    def _read_source(self) -> str:
        return self._RUN_PY.read_text(encoding="utf-8")

    def test_menu_item_exists(self):
        src = self._read_source()
        self.assertIn('"Benchmark Models"', src)

    def test_menu_description_exists(self):
        src = self._read_source()
        self.assertIn('"Benchmark Models":', src)
        # Description should mention key metrics
        self.assertIn("mAP", src)

    def test_elif_handler_exists(self):
        src = self._read_source()
        self.assertIn('main_choice == "Benchmark Models"', src)
        self.assertIn("src.cli.benchmark", src)

    def test_safe_subcommand_called(self):
        src = self._read_source()
        # The handler should use _safe_subcommand
        self.assertIn('_safe_subcommand("Benchmark"', src)


class TestBenchmarkModuleImports(unittest.TestCase):
    """Verify the benchmark package public API is correct."""

    def test_public_api_importable(self):
        from src.benchmark import BenchmarkConfig, BenchmarkResult, ModelMetrics
        from src.benchmark import run_benchmark, write_benchmark_report
        self.assertTrue(callable(run_benchmark))
        self.assertTrue(callable(write_benchmark_report))

    def test_benchmark_config_defaults(self):
        from src.benchmark import BenchmarkConfig
        from pathlib import Path
        config = BenchmarkConfig(
            weights=[Path("model.pt")],
            validation_dir=Path("data/valid"),
        )
        self.assertAlmostEqual(config.conf_threshold, 0.25)
        self.assertAlmostEqual(config.iou_threshold, 0.5)
        self.assertEqual(config.device, "auto")
        self.assertTrue(config.generate_thumbnails)
        self.assertFalse(config.open_in_browser)


if __name__ == "__main__":
    unittest.main()
