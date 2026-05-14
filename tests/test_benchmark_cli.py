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


class TestBenchmarkProgressRendering(unittest.TestCase):
    """Verify live benchmark progress parsing stays useful for long model lists."""

    def test_progress_rows_show_done_running_and_next_models(self):
        from src.cli.benchmark import _benchmark_progress_rows

        weights = [
            Path("runs/detect/train_a/weights/best.pt"),
            Path("runs/detect/train_a/weights/last.pt"),
            Path("runs/detect/train_b/weights/best.pt"),
        ]
        rows = _benchmark_progress_rows(
            weights,
            [
                "Loading model: best.pt",
                "  mAP@50=0.810  mAP@50:95=0.620  F1=0.700  P=0.750  R=0.660",
                "Loading model: last.pt",
            ],
        )

        self.assertEqual(rows[0]["name"], "train_a / best.pt")
        self.assertEqual(rows[0]["status"], "Done")
        self.assertEqual(rows[0]["map50"], "0.810")
        self.assertEqual(rows[1]["name"], "train_a / last.pt")
        self.assertEqual(rows[1]["status"], "Running")
        self.assertEqual(rows[2]["name"], "train_b / best.pt")
        self.assertEqual(rows[2]["status"], "Pending")

    def test_progress_rows_mark_next_after_completed_model(self):
        from src.cli.benchmark import _benchmark_progress_rows

        rows = _benchmark_progress_rows(
            [Path("a.pt"), Path("b.pt")],
            [
                "Loading model: a.pt",
                "  mAP@50=0.500  mAP@50:95=0.300  F1=0.400  P=0.450  R=0.350",
            ],
        )

        self.assertEqual(rows[0]["status"], "Done")
        self.assertEqual(rows[1]["status"], "Next")

    def test_progress_renderer_adapts_to_narrow_terminal(self):
        from rich.console import Console
        from src.cli import benchmark

        renderable = benchmark._render_benchmark_progress(
            [Path("runs/detect/train_a/weights/best.pt")],
            [
                "Loading model: best.pt",
                "  mAP@50=0.810  mAP@50:95=0.620  F1=0.700  P=0.750  R=0.660",
            ],
            width=90,
            height=28,
        )
        console = Console(record=True, width=90, height=28, color_system=None)
        console.print(renderable)
        output = console.export_text()

        self.assertIn("Evaluation Summary", output)
        self.assertIn("Live Log", output)
        self.assertNotIn("mAP50:95", output)

    def test_progress_renderer_keeps_full_metrics_on_wide_terminal(self):
        from rich.console import Console
        from src.cli import benchmark

        renderable = benchmark._render_benchmark_progress(
            [Path("runs/detect/train_a/weights/best.pt")],
            [
                "Loading model: best.pt",
                "  mAP@50=0.810  mAP@50:95=0.620  F1=0.700  P=0.750  R=0.660",
            ],
            width=150,
            height=36,
        )
        console = Console(record=True, width=150, height=36, color_system=None)
        console.print(renderable)
        output = console.export_text()

        self.assertIn("mAP5", output)
        self.assertIn("0.620", output)

    def test_live_screen_contains_single_refreshable_dashboard(self):
        from rich.console import Console
        from src.cli import benchmark

        renderable = benchmark._render_benchmark_live_screen(
            [Path("runs/detect/train_a/weights/best.pt")],
            ["Loading model: best.pt"],
            width=120,
            height=30,
        )
        console = Console(record=True, width=120, height=30, color_system=None)
        console.print(renderable)
        output = console.export_text()

        self.assertIn("Running Benchmark", output)
        self.assertIn("Evaluation Summary", output)
        self.assertIn("Live Log", output)


if __name__ == "__main__":
    unittest.main()
