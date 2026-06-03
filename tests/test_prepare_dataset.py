from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import yaml

from src.cli.prepare_dataset import _discover_ndjson_files
from src.datasets.prepare import (
    PrepareDatasetConfig,
    PrepareSplitConfig,
    prepare_dataset,
    resolve_versioned_output,
    split_records,
    ImageRecord,
    Annotation,
    object_size_bucket,
)


class PrepareDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def _write_image(self, path: Path, value: int = 120) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((16, 16, 3), value, dtype=np.uint8)
        cv2.imwrite(str(path), image)

    def _write_yolo_dataset(self, root: Path, count: int = 6) -> None:
        (root / "data.yaml").write_text(
            "path: .\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 2\nnames: [cat, dog]\ntask: detect\n",
            encoding="utf-8",
        )
        for idx in range(count):
            split = "train" if idx < count else "valid"
            self._write_image(root / split / "images" / f"img_{idx}.jpg", value=80 + idx)
            label = root / split / "labels" / f"img_{idx}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            cls = idx % 2
            label.write_text(f"{cls} 0.500000 0.500000 0.250000 0.250000\n", encoding="utf-8")

    def test_resolve_versioned_output_auto_increments(self) -> None:
        output_root = self.tmp / "datasets"
        (output_root / "plants_v001").mkdir(parents=True)

        path, version = resolve_versioned_output(output_root, "plants")

        self.assertEqual(path, output_root / "plants_v002")
        self.assertEqual(version, 2)

    def test_prepare_yolo_to_yolo_detection_with_manifest(self) -> None:
        source = self.tmp / "source"
        source.mkdir()
        self._write_yolo_dataset(source, count=6)

        stats = prepare_dataset(
            PrepareDatasetConfig(
                source_path=source,
                output_root=self.tmp / "datasets",
                output_slug="prepared",
                output_format="YOLO Detection",
                split_config=PrepareSplitConfig(0.50, 0.25, 0.25),
                seed=3,
            )
        )

        out = Path(stats.output_path)
        self.assertEqual(out.name, "prepared_v001")
        self.assertTrue((out / "data.yaml").exists())
        self.assertTrue((out / "manifest.json").exists())
        self.assertTrue((out / "README.md").exists())
        data = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
        self.assertEqual(data["train"], "train/images")
        self.assertEqual(data["val"], "valid/images")
        self.assertEqual(data["task"], "detect")
        self.assertEqual(stats.split_counts, {"train": 3, "valid": 1, "test": 2})
        self.assertEqual(len(list((out / "train" / "labels").glob("*.txt"))), 3)

    def test_prepare_yolo_to_coco(self) -> None:
        source = self.tmp / "source"
        source.mkdir()
        self._write_yolo_dataset(source, count=4)

        stats = prepare_dataset(
            PrepareDatasetConfig(
                source_path=source,
                output_root=self.tmp / "datasets",
                output_slug="coco_ready",
                output_format="COCO",
                split_config=PrepareSplitConfig(0.50, 0.50, 0.0),
            )
        )

        out = Path(stats.output_path)
        train_json = out / "annotations" / "instances_train.json"
        self.assertTrue(train_json.exists())
        payload = json.loads(train_json.read_text(encoding="utf-8"))
        self.assertEqual(payload["categories"][0]["id"], 1)
        self.assertTrue((out / "valid" / "images").exists())

    def test_prepare_coco_to_yolo_normalizes_non_contiguous_category_ids(self) -> None:
        source = self.tmp / "coco"
        self._write_image(source / "train" / "images" / "a.jpg")
        ann_dir = source / "annotations"
        ann_dir.mkdir(parents=True)
        coco = {
            "images": [{"id": 10, "file_name": "train/images/a.jpg", "width": 16, "height": 16}],
            "categories": [{"id": 5, "name": "cat"}, {"id": 9, "name": "dog"}],
            "annotations": [{"id": 1, "image_id": 10, "category_id": 9, "bbox": [4, 4, 4, 4], "area": 16, "iscrowd": 0}],
        }
        (ann_dir / "instances_train.json").write_text(json.dumps(coco), encoding="utf-8")

        stats = prepare_dataset(
            PrepareDatasetConfig(
                source_path=source,
                output_root=self.tmp / "datasets",
                output_slug="from_coco",
                output_format="YOLO Detection",
                split_config=PrepareSplitConfig(1.0, 0.0, 0.0),
            )
        )

        label = next((Path(stats.output_path) / "train" / "labels").glob("*.txt"))
        self.assertTrue(label.read_text(encoding="utf-8").startswith("1 "))
        data = yaml.safe_load((Path(stats.output_path) / "data.yaml").read_text(encoding="utf-8"))
        self.assertEqual(data["names"], ["cat", "dog"])

    def test_prepare_ndjson_with_mocked_download(self) -> None:
        ndjson = self.tmp / "labels.ndjson"
        row = {
            "data_row": {"global_key": "test.jpg", "row_data": "https://example.com/test.jpg"},
            "projects": {
                "proj": {
                    "labels": [
                        {
                            "annotations": {
                                "objects": [
                                    {
                                        "name": "cat",
                                        "bounding_box": {"top": 4, "left": 4, "height": 4, "width": 4},
                                    }
                                ]
                            }
                        }
                    ]
                }
            },
        }
        ndjson.write_text(json.dumps(row), encoding="utf-8")

        with patch("requests.get") as mock_get:
            response = MagicMock()
            response.content = cv2.imencode(".jpg", np.full((16, 16, 3), 120, dtype=np.uint8))[1].tobytes()
            response.raise_for_status.return_value = None
            mock_get.return_value = response

            stats = prepare_dataset(
                PrepareDatasetConfig(
                    source_path=ndjson,
                    output_root=self.tmp / "datasets",
                    output_slug="ndjson_ready",
                    output_format="YOLO Detection",
                    split_config=PrepareSplitConfig(1.0, 0.0, 0.0),
                )
            )

        self.assertEqual(stats.source_format, "ndjson")
        self.assertEqual(stats.classes, ["cat"])
        self.assertTrue((Path(stats.output_path) / "train" / "labels" / "test.txt").exists())

    def test_prepare_ultralytics_ndjson_export_with_mocked_download(self) -> None:
        ndjson = self.tmp / "ultralytics.ndjson"
        rows = [
            {
                "type": "dataset",
                "task": "segment",
                "class_names": {"0": "vegetation"},
                "version": 4,
            },
            {
                "type": "image",
                "file": "tile.jpg",
                "url": "https://cdn.ul.run/example/tile.jpg",
                "width": 16,
                "height": 16,
                "annotations": {
                    "segments": [[0, 0.25, 0.25, 0.75, 0.25, 0.75, 0.75, 0.25, 0.75]]
                },
            },
        ]
        ndjson.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

        with patch("requests.get") as mock_get:
            response = MagicMock()
            response.content = cv2.imencode(".jpg", np.full((16, 16, 3), 120, dtype=np.uint8))[1].tobytes()
            response.raise_for_status.return_value = None
            mock_get.return_value = response

            stats = prepare_dataset(
                PrepareDatasetConfig(
                    source_path=ndjson,
                    output_root=self.tmp / "datasets",
                    output_slug="ultralytics_ready",
                    output_format="YOLO Segmentation",
                    split_config=PrepareSplitConfig(1.0, 0.0, 0.0),
                    max_workers=2,
                    use_multiprocessing=True,
                )
            )

        self.assertEqual(stats.source_format, "ndjson")
        self.assertEqual(stats.classes, ["vegetation"])
        manifest = json.loads((Path(stats.output_path) / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["use_multiprocessing"])
        self.assertEqual(manifest["max_workers"], 2)
        label = Path(stats.output_path) / "train" / "labels" / "tile.txt"
        self.assertTrue(label.exists())
        self.assertTrue(label.read_text(encoding="utf-8").startswith("0 0.250000 0.250000"))

    def test_split_records_is_deterministic_and_preserves_counts(self) -> None:
        records = [
            ImageRecord(Path(f"img_{idx}.jpg"), f"img_{idx}.jpg", 10, 10, [Annotation(idx % 2, bbox=[0.5, 0.5, 0.1, 0.1])])
            for idx in range(10)
        ]

        first = split_records(records, PrepareSplitConfig(0.60, 0.20, 0.20), seed=99)
        second = split_records(records, PrepareSplitConfig(0.60, 0.20, 0.20), seed=99)

        self.assertEqual({key: [r.file_name for r in value] for key, value in first.items()}, {key: [r.file_name for r in value] for key, value in second.items()})
        self.assertEqual({key: len(value) for key, value in first.items()}, {"train": 6, "valid": 2, "test": 2})

    def test_smart_split_reports_progress(self) -> None:
        records = [
            ImageRecord(Path(f"img_{idx}.jpg"), f"img_{idx}.jpg", 10, 10, [Annotation(idx % 2, bbox=[0.5, 0.5, 0.1, 0.1])])
            for idx in range(20)
        ]
        events: list[tuple[int, int, str]] = []

        split_records(
            records,
            PrepareSplitConfig(0.60, 0.20, 0.20),
            seed=99,
            strategy="smart_balanced",
            progress_callback=lambda current, total, message: events.append((current, total, message)),
        )

        self.assertTrue(any("Preparing smart split" in message for _, _, message in events))
        self.assertEqual(events[-1][0], 20)
        self.assertEqual(events[-1][1], 20)
        self.assertIn("Smart balanced split assignment", events[-1][2])

    def test_smart_split_writes_object_size_diagnostics(self) -> None:
        source = self.tmp / "smart"
        source.mkdir()
        (source / "data.yaml").write_text(
            "path: .\ntrain: train/images\nnc: 2\nnames: [small, large]\ntask: detect\n",
            encoding="utf-8",
        )
        for idx in range(12):
            self._write_image(source / "train" / "images" / f"img_{idx}.jpg", value=70 + idx)
            label = source / "train" / "labels" / f"img_{idx}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            if idx < 6:
                label.write_text("0 0.500000 0.500000 0.050000 0.050000\n", encoding="utf-8")
            else:
                label.write_text("1 0.500000 0.500000 0.500000 0.500000\n", encoding="utf-8")

        stats = prepare_dataset(
            PrepareDatasetConfig(
                source_path=source,
                output_root=self.tmp / "datasets",
                output_slug="smart_ready",
                split_config=PrepareSplitConfig(0.50, 0.25, 0.25),
                split_strategy="smart_balanced",
            )
        )

        manifest = json.loads((Path(stats.output_path) / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["split_strategy"], "smart_balanced")
        self.assertEqual(manifest["split_diagnostics"]["strategy"], "smart_balanced")
        self.assertEqual(manifest["split_diagnostics"]["train"]["image_count"], 6)
        self.assertEqual(manifest["split_diagnostics"]["valid"]["image_count"], 3)
        self.assertEqual(manifest["split_diagnostics"]["test"]["image_count"], 3)
        for split in ("valid", "test"):
            counts = manifest["split_diagnostics"][split]["object_size_counts"]
            self.assertGreater(counts["small"], 0)
            self.assertGreater(counts["large"], 0)

    def test_object_size_bucket_uses_normalized_area(self) -> None:
        self.assertEqual(object_size_bucket(Annotation(0, bbox=[0.5, 0.5, 0.05, 0.05]), 100, 100), "small")
        self.assertEqual(object_size_bucket(Annotation(0, bbox=[0.5, 0.5, 0.15, 0.15]), 100, 100), "medium")
        self.assertEqual(object_size_bucket(Annotation(0, bbox=[0.5, 0.5, 0.40, 0.40]), 100, 100), "large")

    def test_discover_ndjson_files_checks_project_root_and_datasets(self) -> None:
        root_export = self.tmp / "labelbox.ndjson"
        nested_export = self.tmp / "datasets" / "weeds" / "export.ndjson"
        ignored_export = self.tmp / "other" / "export.ndjson"
        hidden_export = self.tmp / "datasets" / ".download.ndjson"
        for path in (root_export, nested_export, ignored_export, hidden_export):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")

        discovered = _discover_ndjson_files(self.tmp)

        self.assertEqual(discovered, [nested_export, root_export])

    def test_data_yaml_omits_test_when_test_split_is_zero(self) -> None:
        source = self.tmp / "source"
        source.mkdir()
        self._write_yolo_dataset(source, count=4)

        stats = prepare_dataset(
            PrepareDatasetConfig(
                source_path=source,
                output_root=self.tmp / "datasets",
                output_slug="no_test",
                output_format="YOLO Detection",
                split_config=PrepareSplitConfig(0.75, 0.25, 0.0),
            )
        )

        data = yaml.safe_load((Path(stats.output_path) / "data.yaml").read_text(encoding="utf-8"))
        self.assertIn("train", data)
        self.assertIn("val", data)
        self.assertNotIn("test", data)
        self.assertFalse((Path(stats.output_path) / "test" / "images").exists())

    def test_ndjson_rejects_non_http_url_scheme(self) -> None:
        ndjson = self.tmp / "labels.ndjson"
        row = {
            "data_row": {"global_key": "leak.jpg", "row_data": "file:///etc/passwd"},
            "projects": {},
        }
        ndjson.write_text(json.dumps(row), encoding="utf-8")

        with patch("requests.get") as mock_get:
            # Should never be called because scheme is rejected.
            mock_get.side_effect = AssertionError("file:// must not be requested")
            with self.assertRaises(Exception):
                prepare_dataset(
                    PrepareDatasetConfig(
                        source_path=ndjson,
                        output_root=self.tmp / "datasets",
                        output_slug="leak",
                        output_format="YOLO Detection",
                        split_config=PrepareSplitConfig(1.0, 0.0, 0.0),
                    )
                )

    def test_ndjson_download_retries_on_transient_failure(self) -> None:
        from src.datasets.prepare import _download_ndjson_image

        target = self.tmp / "img.jpg"
        good = MagicMock()
        good.content = b"jpeg"
        good.raise_for_status.return_value = None
        bad = MagicMock()
        bad.raise_for_status.side_effect = __import__("requests").ConnectionError("boom")

        with patch("requests.get", side_effect=[bad, good]) as mock_get, patch("time.sleep"):
            ok, reason = _download_ndjson_image("https://example.com/img.jpg", target, retries=2)

        self.assertTrue(ok)
        self.assertIsNone(reason)
        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(target.read_bytes(), b"jpeg")


class DatasetSummaryCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        from pathlib import Path as RealPath
        def mock_path(*args, **kwargs):
            if args and args[0] == "datasets":
                return RealPath(self.tmp / "datasets")
            return RealPath(*args, **kwargs)
        self.patcher = patch("src.datasets.core.Path", side_effect=mock_path)
        self.patcher.start()

    def tearDown(self) -> None:
        self.patcher.stop()
        shutil.rmtree(self.tmp)

    def _write_yolo_dataset(self, root: Path) -> None:
        (root / "data.yaml").write_text(
            "path: .\ntrain: train/images\nnc: 1\nnames: [cat]\ntask: detect\n",
            encoding="utf-8",
        )
        img = root / "train" / "images" / "img.jpg"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_text("dummy image", encoding="utf-8")
        label = root / "train" / "labels" / "img.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    def test_summarize_dataset_cache_creation_and_hit(self) -> None:
        from src.datasets.core import summarize_dataset

        dataset_path = self.tmp / "my_test_dataset"
        dataset_path.mkdir()
        self._write_yolo_dataset(dataset_path)

        # First call: computes and caches
        summary1 = summarize_dataset(dataset_path)
        self.assertEqual(summary1.image_count, 1)
        self.assertEqual(summary1.annotation_count, 1)

        # Verify cache file exists
        cache_files = list((self.tmp / "datasets" / ".yolomatic_cache" / "summaries").glob("*.json"))
        self.assertEqual(len(cache_files), 1)

        # Patch os.fwalk to raise error to prove it's not called on second run
        with patch("os.fwalk", side_effect=AssertionError("Should load from cache")):
            summary2 = summarize_dataset(dataset_path)
            self.assertEqual(summary2.image_count, 1)
            self.assertEqual(summary2.annotation_count, 1)
            self.assertEqual(summary2.total_size_bytes, summary1.total_size_bytes)


if __name__ == "__main__":
    unittest.main()
