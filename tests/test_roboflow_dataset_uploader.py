"""Tests for Roboflow dataset batch partitioning and headless upload utilities."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
import requests
from roboflow.adapters.rfapi import RoboflowError

from src.integrations import roboflow_dataset
from src.integrations.roboflow_dataset import (
    ProgressFileReader,
    UploadBatchInfo,
    create_batch_zip_archive,
    partition_dataset_into_batches,
)


def test_progress_file_reader(tmp_path: Path):
    sample_file = tmp_path / "sample.bin"
    sample_file.write_bytes(b"A" * 5000)

    bytes_reported = 0

    def on_progress(chunk_len: int) -> None:
        nonlocal bytes_reported
        bytes_reported += chunk_len

    with ProgressFileReader(sample_file, progress_callback=on_progress) as reader:
        assert len(reader) == 5000
        chunk1 = reader.read(2000)
        assert len(chunk1) == 2000
        assert bytes_reported == 2000

        pos = reader.tell()
        assert pos == 2000

        reader.seek(0)
        assert reader.tell() == 0

        chunk2 = reader.read()
        assert len(chunk2) == 5000
        assert bytes_reported == 7000


def test_partition_dataset_into_batches(tmp_path: Path):
    ds_dir = tmp_path / "fake_dataset"
    ds_dir.mkdir()
    (ds_dir / "data.yaml").write_text("names: [test]\nnc: 1\n")

    # Generate 15 dummy image files (each ~100 KB)
    images = []
    for i in range(15):
        img_p = ds_dir / f"img_{i}.png"
        img_p.write_bytes(b"X" * 102400)  # 100 KB
        images.append({"file": str(img_p), "split": "train", "index": i})

    # Partition with max 5 images per batch
    batches = partition_dataset_into_batches(
        dataset_path=ds_dir,
        parsed_images=images,
        max_batch_size_mb=100.0,
        max_images_per_batch=5,
    )

    assert len(batches) == 3
    assert len(batches[0].images) == 5
    assert len(batches[1].images) == 5
    assert len(batches[2].images) == 5

    # Check that root files (data.yaml) are preserved in each batch
    for b in batches:
        assert any(rf.name == "data.yaml" for rf in b.root_files)


def test_create_batch_zip_archive(tmp_path: Path):
    ds_dir = tmp_path / "zip_test_dataset"
    (ds_dir / "images" / "train").mkdir(parents=True)
    (ds_dir / "labels" / "train").mkdir(parents=True)

    (ds_dir / "data.yaml").write_text("names: [sample]\nnc: 1\n")

    img_p = ds_dir / "images" / "train" / "img_0.png"
    img_p.write_bytes(b"dummy image data" * 100)
    lbl_p = ds_dir / "labels" / "train" / "img_0.txt"
    lbl_p.write_text("0 0.5 0.5 0.2 0.2\n")

    images = [{
        "file": str(img_p),
        "split": "train",
        "annotationfile": {"file": str(lbl_p)},
    }]

    batch_info = UploadBatchInfo(
        batch_idx=1,
        images=images,
        root_files=[ds_dir / "data.yaml"],
    )

    bytes_zipped = 0

    def on_zip(adv: int) -> None:
        nonlocal bytes_zipped
        bytes_zipped += adv

    zip_path = create_batch_zip_archive(ds_dir, batch_info, progress_callback=on_zip)
    assert zip_path.exists()
    assert zip_path.stat().st_size > 0
    assert bytes_zipped > 0

    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        assert "data.yaml" in namelist
        assert "images/train/img_0.png" in namelist
        assert "labels/train/img_0.txt" in namelist

    zip_path.unlink()


def test_poll_zip_status_tolerates_transient_errors(monkeypatch):
    """A dropped status poll must not discard an archive that is already uploaded."""
    calls = {"n": 0}

    def _flaky(api_key, workspace_url, task_id):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise requests.exceptions.ConnectionError("transient")
        return {"status": "completed", "progress": {"current": "done"}}

    monkeypatch.setattr(roboflow_dataset.rfapi, "get_zip_upload_status", _flaky)
    monkeypatch.setattr(roboflow_dataset.time, "sleep", lambda *_: None)

    result = roboflow_dataset.poll_zip_status(
        api_key="k", workspace_url="ws", task_id="t1", poll_interval=0,
    )
    assert result["status"] == "completed"
    assert calls["n"] == 4


def test_poll_zip_status_gives_up_on_sustained_outage(monkeypatch):
    def _always_fails(api_key, workspace_url, task_id):
        raise requests.exceptions.ConnectionError("down")

    monkeypatch.setattr(roboflow_dataset.rfapi, "get_zip_upload_status", _always_fails)
    monkeypatch.setattr(roboflow_dataset.time, "sleep", lambda *_: None)

    with pytest.raises(RoboflowError, match="Lost contact with Roboflow"):
        roboflow_dataset.poll_zip_status(
            api_key="k", workspace_url="ws", task_id="t1",
            poll_interval=0, max_poll_errors=2,
        )


def test_create_batch_zip_archive_skips_paths_outside_root(tmp_path: Path):
    """A descriptor pointing outside the dataset root must be skipped, not crash."""
    ds_dir = tmp_path / "ds"
    ds_dir.mkdir()
    inside = ds_dir / "in.png"
    inside.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0d")
    outside = tmp_path / "elsewhere.png"
    outside.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0d")

    batch = UploadBatchInfo(
        batch_idx=1,
        images=[{"file": str(inside)}, {"file": str(outside)}],
        root_files=[],
    )
    zip_path = create_batch_zip_archive(ds_dir, batch)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            assert zf.namelist() == ["in.png"]
    finally:
        zip_path.unlink(missing_ok=True)
