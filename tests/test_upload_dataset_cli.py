"""Tests for Roboflow dataset upload CLI parser and arguments."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


from src.cli import upload_dataset
from src.cli.upload_dataset import parse_args


def test_upload_dataset_cli_defaults():
    test_args = ["yolomatic-upload-dataset"]
    with patch.object(sys, "argv", test_args):
        args = parse_args()
        assert args.zip is True
        assert args.type == "instance-segmentation"
        assert args.license == "private"
        assert args.batch_name == "dataset-upload"
        assert args.workers == 10
        assert args.max_batch_size_mb == 500.0
        assert args.chunk_images == 1000
        assert args.upload_timeout == 1800
        assert args.max_retries == 3
        assert args.yes is False


def test_upload_dataset_cli_custom_flags():
    test_args = [
        "yolomatic-upload-dataset",
        "--api-key", "rf_test_key_123",
        "--workspace", "my-ws",
        "--project", "my-proj",
        "--dataset", "datasets/custom_ds",
        "--type", "object-detection",
        "--license", "MIT",
        "--batch-name", "release-v1",
        "--no-zip",
        "--workers", "16",
        "--max-batch-size-mb", "250.0",
        "--chunk-images", "500",
        "--upload-timeout", "3600",
        "--max-retries", "5",
        "--yes",
    ]
    with patch.object(sys, "argv", test_args):
        args = parse_args()
        assert args.api_key == "rf_test_key_123"
        assert args.workspace == "my-ws"
        assert args.project == "my-proj"
        assert args.dataset == "datasets/custom_ds"
        assert args.type == "object-detection"
        assert args.license == "MIT"
        assert args.batch_name == "release-v1"
        assert args.zip is False
        assert args.workers == 16
        assert args.max_batch_size_mb == 250.0
        assert args.chunk_images == 500
        assert args.upload_timeout == 3600
        assert args.max_retries == 5
        assert args.yes is True


def _make_image(path: Path, payload: bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0d") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_execute_zip_upload_completes_full_flow(tmp_path: Path, monkeypatch):
    """The default zip flow must run compress -> sign -> PUT -> poll without crashing.

    Regression guard: `poll_zip_status` was referenced but never imported, so this
    path raised NameError after the archive had already been PUT to Roboflow.
    """
    ds_dir = tmp_path / "ds"
    (ds_dir).mkdir()
    (ds_dir / "data.yaml").write_text("names: [cat]\nnc: 1\n")
    img = ds_dir / "images" / "train" / "a.png"
    _make_image(img)
    lbl = ds_dir / "labels" / "train" / "a.txt"
    lbl.parent.mkdir(parents=True, exist_ok=True)
    lbl.write_text("0 0.5 0.5 0.2 0.2\n")

    parsed_images = [{
        "file": str(img.resolve()),
        "split": "train",
        "annotationfile": {"file": str(lbl.resolve()), "labelmap": {0: "cat"}},
        "index": 0,
    }]

    polled: list[str] = []

    monkeypatch.setattr(
        upload_dataset.rfapi,
        "init_zip_upload",
        lambda *a, **k: {"taskId": "task-1", "signedUrl": "https://example.invalid/put"},
    )
    monkeypatch.setattr(
        upload_dataset,
        "upload_zip_stream",
        lambda **kwargs: None,
    )

    def _fake_get_status(api_key, workspace_url, task_id):
        polled.append(task_id)
        return {"status": "completed", "progress": {"current": "done"}}

    monkeypatch.setattr(upload_dataset.rfapi, "get_zip_upload_status", _fake_get_status)

    ws = SimpleNamespace(url="my-ws")
    project = SimpleNamespace(id="my-ws/my-proj")

    stats = upload_dataset.execute_zip_upload(
        api_key="rf_key",
        ws=ws,
        project=project,
        dataset_path=ds_dir,
        parsed_images=parsed_images,
        batch_name="test-batch",
        max_batch_size_mb=500.0,
        chunk_images=1000,
        timeout_secs=60,
        max_retries=1,
    )

    assert stats["uploaded"] == 1
    assert stats["batches"] == 1
    assert stats["server_task_ids"] == ["task-1"]
    assert polled == ["task-1"]
