from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.cli.predict import (
    BatchPredictionResult,
    build_batch_summary,
    discover_images,
    parse_args,
    validate_worker_count,
)


class PredictBatchTests(unittest.TestCase):
    def test_discover_images_returns_supported_files_sorted_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("b.png", "a.JPG", "notes.txt", "c.webp"):
                (root / name).touch()

            self.assertEqual(
                [path.name for path in discover_images(root)],
                ["a.JPG", "b.png", "c.webp"],
            )

    def test_validate_worker_count_rejects_zero(self) -> None:
        with self.assertRaises(ValueError):
            validate_worker_count(0)

    def test_build_batch_summary_counts_successes_failures_and_outputs(self) -> None:
        results = [
            BatchPredictionResult(Path("a.jpg"), Path("runs/predict")),
            BatchPredictionResult(Path("b.jpg"), Path("runs/predict")),
            BatchPredictionResult(Path("c.jpg"), None, "failed"),
        ]

        summary = build_batch_summary(results, elapsed_seconds=1.25, workers=3)

        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["succeeded"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["elapsed_seconds"], 1.25)
        self.assertEqual(summary["workers"], 3)
        self.assertEqual(summary["output_dirs"], ["runs/predict"])

    def test_input_dir_alias_populates_source(self) -> None:
        args = parse_args(["--mode", "folder", "--input-dir", "images"])

        self.assertEqual(args.source, "images")


if __name__ == "__main__":
    unittest.main()
