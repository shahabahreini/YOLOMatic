from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config.generator import YOLOConfigGenerator
from src.utils.project import (
    find_available_weights,
    find_run_directories,
    format_size,
    format_weight_label,
    list_dataset_directories,
    load_dataset_config,
    verify_dataset_directories,
)


class ProjectUtilsTests(unittest.TestCase):
    def test_format_weight_label_prefers_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            weight_path = root / "runs" / "detect" / "weights" / "best.pt"
            weight_path.parent.mkdir(parents=True)
            weight_path.touch()
            self.assertEqual(
                format_weight_label(root, weight_path),
                "runs/detect/weights/best.pt",
            )

    def test_find_available_weights_includes_root_and_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            root_weight = root / "root.pt"
            run_weight = root / "runs" / "segment" / "weights" / "best.pt"
            run_weight.parent.mkdir(parents=True)
            root_weight.touch()
            run_weight.touch()

            results = find_available_weights(root)

            self.assertEqual({path.name for path in results}, {"root.pt", "best.pt"})

    def test_find_run_directories_uses_args_yaml_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "runs" / "detect" / "train"
            run_dir.mkdir(parents=True)
            (run_dir / "args.yaml").write_text("model: yolo\n", encoding="utf-8")

            self.assertEqual(find_run_directories(root / "runs"), [run_dir])

    def test_list_dataset_directories_reports_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            datasets_root = Path(temp_dir) / "datasets"
            dataset_dir = datasets_root / "demo"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "sample.txt").write_text("hello", encoding="utf-8")

            datasets = list_dataset_directories(datasets_root)

            self.assertEqual(len(datasets), 1)
            self.assertEqual(datasets[0]["name"], "demo")
            self.assertEqual(datasets[0]["path"], dataset_dir.resolve())

    def test_load_dataset_config_resolves_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            datasets_root = Path(temp_dir) / "datasets"
            dataset_dir = datasets_root / "demo"
            train_dir = dataset_dir / "train" / "images"
            valid_dir = dataset_dir / "valid" / "images"
            test_dir = dataset_dir / "test" / "images"
            train_dir.mkdir(parents=True)
            valid_dir.mkdir(parents=True)
            test_dir.mkdir(parents=True)

            (dataset_dir / "data.yaml").write_text(
                "train: train/images\nval: valid/images\ntest: test/images\nnames: [demo]\n",
                encoding="utf-8",
            )

            dataset_config, data_yaml_path, dataset_path = load_dataset_config(
                "demo",
                datasets_root=datasets_root,
            )

            self.assertEqual(dataset_path, str(dataset_dir.resolve()))
            self.assertEqual(data_yaml_path, str((dataset_dir / "data.yaml").resolve()))
            self.assertEqual(dataset_config["train"], str(train_dir.resolve()))
            self.assertEqual(dataset_config["val"], str(valid_dir.resolve()))
            self.assertEqual(dataset_config["test"], str(test_dir.resolve()))

    def test_verify_dataset_directories_reports_missing_paths(self) -> None:
        missing = verify_dataset_directories(
            {
                "train": "/missing/train",
                "val": "/missing/val",
                "test": "/missing/test",
            }
        )

        self.assertEqual(len(missing), 3)

    def test_format_size(self) -> None:
        self.assertEqual(format_size(1024), "1.00 KB")

    def test_worker_recommendation_is_conservative_on_tight_systems(self) -> None:
        generator = YOLOConfigGenerator("/tmp/nonexistent-dataset")
        profile, reason = generator._recommend_worker_profile(
            model_metrics={"heaviness": "heavy"},
            dataset_metrics={
                "total_size_bytes": 24 * 1024**3,
                "total_file_count": 150000,
                "image_count": 80000,
            },
            system_metrics={
                "cpu_count": 8,
                "available_ram_bytes": 12 * 1024**3,
                "available_gpu_memory_bytes": 8 * 1024**3,
                "device": "cuda",
            },
        )

        self.assertEqual(profile, "light")
        self.assertIn("limited RAM headroom", reason)

    def test_worker_recommendation_only_goes_heavy_with_clear_headroom(self) -> None:
        generator = YOLOConfigGenerator("/tmp/nonexistent-dataset")
        profile, reason = generator._recommend_worker_profile(
            model_metrics={"heaviness": "light"},
            dataset_metrics={
                "total_size_bytes": 2 * 1024**3,
                "total_file_count": 18000,
                "image_count": 12000,
            },
            system_metrics={
                "cpu_count": 24,
                "available_ram_bytes": 64 * 1024**3,
                "available_gpu_memory_bytes": 16 * 1024**3,
                "device": "cuda",
            },
        )

        self.assertEqual(profile, "heavy")
        self.assertIn("strong CPU core count", reason)
