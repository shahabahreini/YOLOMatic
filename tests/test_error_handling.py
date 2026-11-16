"""Regression tests for the Phase-4 error-handling sweep."""
from __future__ import annotations

import errno
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests

from yolomatic.cli.result import EXIT_FAILED, CommandResult, CommandStatus


class ExportCliTest(unittest.TestCase):
    def test_missing_weight_is_a_failure_not_a_traceback(self) -> None:
        from yolomatic.cli import export

        result = export.main(["--weight", "definitely/missing.pt"])
        self.assertEqual(result.status, CommandStatus.FAILED)
        self.assertEqual(result.exit_code, EXIT_FAILED)
        self.assertIn("not found", result.message)

    def test_default_device_is_cpu_without_cuda(self) -> None:
        from yolomatic.cli import export

        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.object(export, "import_torch", return_value=torch):
            self.assertEqual(export.default_export_device(), "cpu")
        torch_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        with patch.object(export, "import_torch", return_value=torch_cuda):
            self.assertEqual(export.default_export_device(), "0")

    def test_custom_engine_carries_ultralytics_metadata(self) -> None:
        from yolomatic.cli.export import build_engine_metadata
        from yolomatic.utils.trt_compiler import read_engine_metadata, write_engine_file

        yolo = SimpleNamespace(
            task="segment",
            names={0: "leaf", 1: "stem"},
            model=SimpleNamespace(stride=[8, 16, 32], yaml={"channels": 3}),
        )
        meta = build_engine_metadata(yolo, {"imgsz": 640, "half": True}, max_batch=16)
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / "m.engine"
            write_engine_file(engine, b"ENGINEBYTES", meta)
            restored = read_engine_metadata(engine)
            self.assertEqual(engine.read_bytes()[-11:], b"ENGINEBYTES")
        self.assertEqual(restored["task"], "segment")
        self.assertEqual(restored["names"], {"0": "leaf", "1": "stem"})
        self.assertEqual(restored["stride"], 32)
        self.assertEqual(restored["imgsz"], [640, 640])


class UploadDatasetLayoutTest(unittest.TestCase):
    def test_labels_found_for_split_images_layout_and_masks_skipped(self) -> None:
        # Regression: train/images + train/labels (Roboflow/Ultralytics layout)
        # uploaded every image unlabeled, and semantic masks were sent as images.
        from yolomatic.cli.upload_dataset import collect_upload_items

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "valid"):
                (root / split / "images").mkdir(parents=True)
                (root / split / "labels").mkdir(parents=True)
                (root / split / "masks").mkdir(parents=True)
                (root / split / "images" / f"{split}.jpg").write_bytes(b"x")
                (root / split / "labels" / f"{split}.txt").write_text("0 .5 .5 .1 .1\n", encoding="utf-8")
                (root / split / "masks" / f"{split}.png").write_bytes(b"m")
            (root / "images" / "test").mkdir(parents=True)
            (root / "labels" / "test").mkdir(parents=True)
            (root / "images" / "test" / "t.jpg").write_bytes(b"x")
            (root / "labels" / "test" / "t.txt").write_text("0 .5 .5 .1 .1\n", encoding="utf-8")

            items = collect_upload_items(root, ["leaf"])

        by_name = {Path(item["file"]).name: item for item in items}
        self.assertEqual(sorted(by_name), ["t.jpg", "train.jpg", "valid.jpg"])
        self.assertEqual(by_name["valid.jpg"]["split"], "valid")
        self.assertEqual(by_name["t.jpg"]["split"], "test")
        for item in items:
            self.assertIsNotNone(item["annotationfile"], item["file"])

    def test_coco_split_annotations_are_packaged(self) -> None:
        from yolomatic.integrations.roboflow_dataset import get_dataset_root_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "train").mkdir()
            (root / "train" / "_annotations.coco.json").write_text("{}", encoding="utf-8")
            (root / "data.yaml").write_text("names: [a]\n", encoding="utf-8")
            names = sorted(p.relative_to(root).as_posix() for p in get_dataset_root_files(root))
        self.assertEqual(names, ["data.yaml", "train/_annotations.coco.json"])


class UltralyticsClientTest(unittest.TestCase):
    def _client(self, session: MagicMock):
        from yolomatic.integrations.ultralytics_platform import UltralyticsPlatformClient

        return UltralyticsPlatformClient("token", base_url="https://platform.example/api/", session=session)

    def test_refuses_to_send_token_to_another_host(self) -> None:
        from yolomatic.integrations.ultralytics_platform import UltralyticsPlatformError

        session = MagicMock()
        with self.assertRaises(UltralyticsPlatformError):
            self._client(session).request("GET", "https://evil.example/next?page=2")
        session.request.assert_not_called()

    def test_retries_idempotent_5xx_and_wraps_connection_errors(self) -> None:
        from yolomatic.integrations.ultralytics_platform import UltralyticsPlatformError

        ok = MagicMock(status_code=200, content=b"{}")
        bad = MagicMock(status_code=503, headers={}, text="busy")
        session = MagicMock()
        session.request.side_effect = [bad, ok]
        with patch("yolomatic.integrations.ultralytics_platform.time.sleep"):
            self.assertIs(self._client(session).request("GET", "datasets"), ok)
        self.assertEqual(session.request.call_count, 2)

        session = MagicMock()
        session.request.side_effect = requests.ConnectionError("down")
        with patch("yolomatic.integrations.ultralytics_platform.time.sleep"):
            with self.assertRaises(UltralyticsPlatformError):
                self._client(session).request("GET", "datasets")

        # POST is not retried on 5xx (it could create duplicates).
        session = MagicMock()
        session.request.side_effect = [bad, ok]
        with self.assertRaises(UltralyticsPlatformError):
            self._client(session).request("POST", "upload/complete")
        self.assertEqual(session.request.call_count, 1)

    def test_interrupted_download_leaves_no_file(self) -> None:
        from yolomatic.integrations.ultralytics_platform import UltralyticsPlatformError

        response = MagicMock(status_code=200, headers={"Content-Length": "100"})
        response.iter_content.return_value = [b"x" * 10]
        response.__enter__ = lambda self: self
        response.__exit__ = lambda self, *a: None
        with tempfile.TemporaryDirectory() as tmp, patch("requests.get", return_value=response):
            target = Path(tmp) / "dataset.zip"
            with self.assertRaises(UltralyticsPlatformError):
                self._client(MagicMock()).download_signed_url("https://cdn.example/x", target)
            self.assertEqual(list(Path(tmp).iterdir()), [])


class BenchmarkScoringTest(unittest.TestCase):
    def test_matching_is_class_aware(self) -> None:
        from yolomatic.benchmark.metrics import greedy_match

        pred = SimpleNamespace(conf=0.9, cls=1, box=(0, 0, 10, 10))
        gt = SimpleNamespace(cls=0, box=(0, 0, 10, 10))
        match = greedy_match([pred], [gt], lambda p, g: 1.0, 0.5)
        self.assertEqual((match.tp, match.fp, match.fn), (0, 1, 1))
        agnostic = greedy_match([pred], [gt], lambda p, g: 1.0, 0.5, class_aware=False)
        self.assertEqual(agnostic.tp, 1)

    def test_coco_categories_map_to_contiguous_zero_based_ids(self) -> None:
        from yolomatic.benchmark.engine import _load_coco_data

        with tempfile.TemporaryDirectory() as tmp:
            ann = Path(tmp) / "_annotations.coco.json"
            (Path(tmp) / "a.jpg").write_bytes(b"x")
            ann.write_text(json.dumps({
                "images": [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
                "categories": [{"id": 1, "name": "cat"}, {"id": 3, "name": "dog"}],
                "annotations": [
                    {"image_id": 1, "category_id": 3, "bbox": [0, 0, 5, 5]},
                    {"image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5]},
                ],
            }), encoding="utf-8")
            _paths, gts = _load_coco_data(ann)
        self.assertEqual(sorted(g.cls for g in gts[1]), [0, 1])

    def test_task_comes_from_model_metadata(self) -> None:
        from yolomatic.benchmark.engine import _detect_task

        model = MagicMock(task="segment")
        self.assertEqual(_detect_task(model, Path("x.jpg"), 0.25, "cpu"), "segmentation")
        model.assert_not_called()


class MiscRegressionTest(unittest.TestCase):
    def test_rfdetr_class_ignores_unrelated_path_parts(self) -> None:
        from yolomatic.models.rfdetr import infer_rfdetr_class

        self.assertEqual(
            infer_rfdetr_class("/pixel_data/segments/runs/RF-DETR-Large-2026/checkpoint.pth"),
            "RFDETRLarge",
        )
        self.assertEqual(infer_rfdetr_class("/data/runs/RF-DETR-Seg-Nano-1/best.pth"), "RFDETRSegNano")

    def test_small_augmentation_runs_always_get_a_validation_split(self) -> None:
        from yolomatic.augmentation.engine import SplitConfig, assign_groups_to_splits

        assignment = assign_groups_to_splits({"a": 1, "b": 1}, SplitConfig(0.7, 0.2, 0.1), seed=0)
        self.assertIn("valid", assignment.values())

    def test_cuda_index_devices_trigger_cuda_preflight(self) -> None:
        from yolomatic.utils.training_preflight import _device_requests_cuda

        for device in ("cuda", "0", "0,1", "cuda:0", "cuda:0,1"):
            self.assertTrue(_device_requests_cuda(device), device)
        for device in ("cpu", "mps", None):
            self.assertFalse(_device_requests_cuda(device), device)

    def test_secrets_are_redacted_from_error_text(self) -> None:
        from yolomatic.utils.ai_client import redact_secrets

        text = redact_secrets("GET https://x/v1?key=AIzaSyA1234567890abcdefghijk failed; Bearer abc.def")
        self.assertNotIn("AIzaSy", text)
        self.assertNotIn("abc.def", text)

    def test_local_yolo_checkpoint_is_not_rewritten_as_hub_name(self) -> None:
        from yolomatic.trainers import yolo_trainer

        yolo_class = MagicMock()
        with tempfile.TemporaryDirectory() as tmp:
            weight = Path(tmp) / "yolov8n_finetuned.pt"
            weight.write_bytes(b"w")
            with patch.object(yolo_trainer, "import_ultralytics_yolo", return_value=yolo_class):
                yolo_trainer.verify_model_file(str(weight))
        yolo_class.assert_called_once_with(str(weight))

    def test_predict_source_prompt_stops_on_eof_and_back(self) -> None:
        from yolomatic.cli.predict import prompt_source

        for side_effect in (EOFError(), [":b"]):
            with self.subTest(side_effect=side_effect), patch("builtins.input", side_effect=side_effect):
                with self.assertRaises(KeyboardInterrupt):
                    prompt_source("single")

    def test_transient_check_rejects_disk_full(self) -> None:
        from yolomatic.utils.fs import is_transient_fs_error

        self.assertFalse(is_transient_fs_error(OSError(errno.ENOSPC, "full")))
        self.assertTrue(is_transient_fs_error(PermissionError(errno.EACCES, "locked")))


class RunnerTest(unittest.TestCase):
    def test_failures_pause_and_cancellations_do_not(self) -> None:
        from yolomatic.cli import run

        with patch.object(run, "clear_screen"), patch("builtins.input", return_value="") as pause:
            result = run._safe_subcommand("Export", lambda: CommandResult.failed("boom", hint="fix it"))
            self.assertEqual(result.status, CommandStatus.FAILED)
            self.assertEqual(pause.call_count, 1)

            pause.reset_mock()
            result = run._safe_subcommand("Export", lambda: CommandResult.cancelled())
            self.assertEqual(result.status, CommandStatus.CANCELLED)
            pause.assert_not_called()

            pause.reset_mock()
            result = run._safe_subcommand("Export", lambda: (_ for _ in ()).throw(RuntimeError("bad")))
            self.assertEqual(result.status, CommandStatus.FAILED)
            self.assertEqual(pause.call_count, 1)

            pause.reset_mock()
            run._safe_subcommand("Export", lambda: CommandResult.ok(outputs={"Export": "a.onnx"}))
            self.assertEqual(pause.call_count, 1)

    def test_combine_with_no_datasets_reports_failure(self) -> None:
        from yolomatic.utils import combine_datasets

        with tempfile.TemporaryDirectory() as tmp, patch.object(combine_datasets, "find_datasets", return_value=[]):
            with patch("yolomatic.utils.combine_datasets.Path.is_dir", return_value=False):
                result = combine_datasets.main()
            self.assertEqual(result.status, CommandStatus.FAILED)
            del tmp


if __name__ == "__main__":
    unittest.main()
