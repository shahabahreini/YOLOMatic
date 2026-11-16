"""Headless integration layer for Roboflow dataset uploads.

Handles:
1. Resilient batch partitioning for large datasets / slow networks.
2. Streamed zip upload to Google Cloud Storage signed URLs.
3. Server-side processing status polling.
4. Concurrent per-image upload fallback.
5. Zero emojis in logging or status messages.
"""

from __future__ import annotations

import os
import tempfile
import time
import zipfile
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import requests
from roboflow.adapters import rfapi
from roboflow.adapters.rfapi import RoboflowError


logger = logging.getLogger(__name__)


class ProgressFileReader:
    """File-like wrapper that reports byte read progress to a callback."""

    def __init__(self, path: Path, progress_callback: Optional[Callable[[int], None]] = None):
        self._path = path
        self._f = open(path, "rb")
        self._progress_callback = progress_callback
        self._total = os.path.getsize(path)

    def read(self, size: int = -1) -> bytes:
        data = self._f.read(size)
        if data and self._progress_callback:
            self._progress_callback(len(data))
        return data

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._f.seek(offset, whence)

    def tell(self) -> int:
        return self._f.tell()

    def __len__(self) -> int:
        return self._total

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> ProgressFileReader:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


@dataclass
class UploadBatchInfo:
    batch_idx: int
    images: List[Dict[str, Any]]
    root_files: List[Path] = field(default_factory=list)
    estimated_bytes: int = 0


def get_dataset_root_files(dataset_path: Path) -> List[Path]:
    """Find top-level configuration/label files (e.g. data.yaml, dataset.yaml, README)."""
    dataset_path = dataset_path.resolve()
    root_files: List[Path] = []
    if not dataset_path.is_dir():
        return root_files

    for item in dataset_path.iterdir():
        if item.is_file() and item.suffix.lower() in {".yaml", ".yml", ".json", ".txt", ".csv", ".names", ".md"}:
            root_files.append(item)
    return root_files


def partition_dataset_into_batches(
    dataset_path: Path,
    parsed_images: Sequence[Dict[str, Any]],
    max_batch_size_mb: float = 500.0,
    max_images_per_batch: int = 1000,
) -> List[UploadBatchInfo]:
    """Partition dataset into smaller chunks to prevent GCS gateway timeouts on slow connections."""
    dataset_path = dataset_path.resolve()
    root_files = get_dataset_root_files(dataset_path)

    if not parsed_images:
        return []

    # If chunking is disabled, return a single unified batch
    if max_batch_size_mb <= 0 and max_images_per_batch <= 0:
        total_size = 0
        for img in parsed_images:
            p = Path(img["file"])
            if p.exists():
                total_size += p.stat().st_size
        return [
            UploadBatchInfo(
                batch_idx=1,
                images=list(parsed_images),
                root_files=root_files,
                estimated_bytes=total_size,
            )
        ]

    max_bytes = int(max_batch_size_mb * 1024 * 1024) if max_batch_size_mb > 0 else float("inf")
    max_count = max_images_per_batch if max_images_per_batch > 0 else float("inf")

    batches: List[UploadBatchInfo] = []
    current_images: List[Dict[str, Any]] = []
    current_bytes = 0

    for img_desc in parsed_images:
        img_path = Path(img_desc["file"]).resolve()
        img_size = img_path.stat().st_size if img_path.exists() else 0
        annot_size = 0
        annot_desc = img_desc.get("annotationfile")
        if annot_desc and isinstance(annot_desc, dict) and "file" in annot_desc:
            ap = Path(annot_desc["file"]).resolve()
            if ap.exists():
                annot_size = ap.stat().st_size

        item_size = img_size + annot_size

        if current_images and ((current_bytes + item_size > max_bytes) or (len(current_images) >= max_count)):
            batches.append(
                UploadBatchInfo(
                    batch_idx=len(batches) + 1,
                    images=current_images,
                    root_files=root_files,
                    estimated_bytes=current_bytes,
                )
            )
            current_images = []
            current_bytes = 0

        current_images.append(img_desc)
        current_bytes += item_size

    if current_images:
        batches.append(
            UploadBatchInfo(
                batch_idx=len(batches) + 1,
                images=current_images,
                root_files=root_files,
                estimated_bytes=current_bytes,
            )
        )

    return batches


def create_batch_zip_archive(
    dataset_path: Path,
    batch_info: UploadBatchInfo,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Path:
    """Create a temporary zip archive for a specific batch."""
    dataset_path = dataset_path.resolve()
    temp_zip_file = Path(tempfile.mktemp(prefix=f"roboflow_batch_{batch_info.batch_idx}_", suffix=".zip"))

    files_to_zip: List[Tuple[Path, str]] = []

    # Ingest top-level files (data.yaml, etc.)
    for rf in batch_info.root_files:
        rf = rf.resolve()
        if rf.exists():
            files_to_zip.append((rf, rf.name))

    # Ingest images and annotations
    def _archive_name(path: Path) -> Optional[str]:
        """Path relative to the dataset root, or None if it escapes the root."""
        try:
            return str(path.relative_to(dataset_path))
        except ValueError:
            logger.warning("Skipping %s: outside dataset root %s", path, dataset_path)
            return None

    for img_desc in batch_info.images:
        img_p = Path(img_desc["file"]).resolve()
        if img_p.exists():
            rel_img = _archive_name(img_p)
            if rel_img:
                files_to_zip.append((img_p, rel_img))

        annot_desc = img_desc.get("annotationfile")
        if annot_desc and isinstance(annot_desc, dict) and "file" in annot_desc:
            annot_p = Path(annot_desc["file"]).resolve()
            if annot_p.exists():
                rel_annot = _archive_name(annot_p)
                if rel_annot:
                    files_to_zip.append((annot_p, rel_annot))

    with zipfile.ZipFile(temp_zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path, arcname in files_to_zip:
            zf.write(file_path, arcname=arcname)
            if progress_callback:
                progress_callback(file_path.stat().st_size)

    return temp_zip_file


def upload_zip_stream(
    signed_url: str,
    zip_path: Path,
    progress_callback: Optional[Callable[[int], None]] = None,
    timeout_secs: int = 1800,
) -> None:
    """Upload a zip file to the GCS signed URL with progress tracking."""
    total_bytes = zip_path.stat().st_size
    headers = {
        "Content-Type": "application/zip",
        "Content-Length": str(total_bytes),
    }

    with ProgressFileReader(zip_path, progress_callback) as stream:
        response = requests.put(
            signed_url,
            data=stream,
            headers=headers,
            timeout=(60, timeout_secs),
        )

    if not response.ok:
        raise RoboflowError(f"Zip upload to signed URL failed ({response.status_code}): {response.text}")


def poll_zip_status(
    api_key: str,
    workspace_url: str,
    task_id: str,
    status_callback: Optional[Callable[[str], None]] = None,
    poll_interval: float = 4.0,
    poll_timeout: float = 3600.0,
    max_poll_errors: int = 5,
) -> Dict[str, Any]:
    """Poll Roboflow server for zip unpacking, dataset parsing, and annotation validation."""
    deadline = time.monotonic() + poll_timeout
    last_stage = None
    consecutive_errors = 0

    while True:
        try:
            status = rfapi.get_zip_upload_status(api_key, workspace_url, task_id)
            consecutive_errors = 0
        except (requests.exceptions.RequestException, TimeoutError) as exc:
            # The archive is already on the server; losing a status poll is not a
            # reason to discard it and re-upload. Give up only on a sustained outage.
            consecutive_errors += 1
            if consecutive_errors > max_poll_errors:
                raise RoboflowError(
                    f"Lost contact with Roboflow after {consecutive_errors} consecutive "
                    f"status polls (task_id={task_id}): {exc}"
                ) from exc
            if time.monotonic() >= deadline:
                raise RoboflowError(
                    f"Server processing timed out after {poll_timeout}s (task_id={task_id})"
                ) from exc
            time.sleep(poll_interval)
            continue

        state = status.get("status")
        prog_info = status.get("progress") or {}
        current_stage = prog_info.get("current") or state

        if current_stage and current_stage != last_stage:
            last_stage = current_stage
            if status_callback:
                status_callback(str(current_stage))

        if state == "completed":
            return status
        if state == "failed":
            raise RoboflowError(f"Roboflow server processing failed: {status.get('error') or status}")

        if time.monotonic() >= deadline:
            raise RoboflowError(f"Server processing timed out after {poll_timeout}s (task_id={task_id})")

        time.sleep(poll_interval)
