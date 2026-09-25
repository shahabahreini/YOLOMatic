"""
Filesystem utility functions for safe atomic directory swapping and cross-platform file operations.
"""
from __future__ import annotations

import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_filesystem_op(
    op: Callable[[], T],
    max_retries: int = 8,
    backoff_base: float = 0.2,
    max_delay: float = 5.0,
) -> T:
    """
    Execute a filesystem operation with exponential backoff on permission/sharing errors.

    Crucial for Windows NTFS where real-time antivirus (e.g. Windows Defender MsMpEng.exe),
    Windows Search Indexer (SearchIndexer.exe), or shell listeners temporarily hold
    delete-inhibiting read handles on newly created files or directories.
    """
    delay = backoff_base
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return op()
        except (PermissionError, OSError) as exc:
            last_exc = exc
            if attempt == max_retries - 1:
                raise
            logger.debug(
                "Filesystem operation '%s' raised %s (attempt %d/%d). Retrying in %.2fs...",
                op,
                exc,
                attempt + 1,
                max_retries,
                delay,
            )
            time.sleep(delay)
            delay = min(delay * 2.0, max_delay)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Retry loop exited without result or exception.")


def safe_directory_swap(
    source_dir: Path | str,
    target_dir: Path | str,
    max_retries: int = 8,
    backoff_base: float = 0.2,
) -> None:
    """
    Safely swap source_dir into target_dir with Windows NTFS locking resilience.

    Steps:
    1. If target_dir exists, renames target_dir to a unique `.backup-<uuid>` directory with retries.
    2. Attempts atomic directory rename source_dir -> target_dir with exponential backoff retries.
    3. If directory rename fails permanently due to immovable folder node locks, falls back
       to recursive item-by-item migration from source_dir into target_dir.
    4. On unrecoverable failure, rolls back the backup_dir to target_dir.
    5. Safely cleans up backup_dir, suppressing non-critical cleanup errors.
    """
    source = Path(source_dir).resolve()
    target = Path(target_dir).resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source}")

    if source == target:
        raise ValueError(f"Source and target directories cannot be identical: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)

    backup_dir: Path | None = None
    if target.exists():
        backup_dir = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
        try:
            retry_filesystem_op(
                lambda: target.rename(backup_dir),
                max_retries=max_retries,
                backoff_base=backoff_base,
            )
        except Exception as exc:
            logger.error("Failed to backup existing target directory %s: %s", target, exc)
            raise

    swap_succeeded = False
    try:
        retry_filesystem_op(
            lambda: source.rename(target),
            max_retries=max_retries,
            backoff_base=backoff_base,
        )
        swap_succeeded = True
    except (PermissionError, OSError) as rename_exc:
        logger.warning(
            "Atomic directory rename from %s to %s failed (%s). Falling back to recursive migration...",
            source,
            target,
            rename_exc,
        )

    if not swap_succeeded:
        try:
            target.mkdir(parents=True, exist_ok=True)
            for item in list(source.iterdir()):
                dest_item = target / item.name
                retry_filesystem_op(
                    lambda: shutil.move(str(item), str(dest_item)),
                    max_retries=max_retries,
                    backoff_base=backoff_base,
                )
            # Remove the now-empty source container
            retry_filesystem_op(
                lambda: source.rmdir(),
                max_retries=max_retries,
                backoff_base=backoff_base,
            )
            swap_succeeded = True
        except Exception as migration_exc:
            logger.error("Recursive item migration from %s to %s failed: %s", source, target, migration_exc)
            # Attempt rollback if backup directory was created
            if backup_dir is not None and backup_dir.exists():
                logger.info("Rolling back target directory from backup %s...", backup_dir)
                try:
                    if target.exists():
                        shutil.rmtree(target, ignore_errors=True)
                    retry_filesystem_op(
                        lambda: backup_dir.rename(target),
                        max_retries=max_retries,
                        backoff_base=backoff_base,
                    )
                except Exception as rollback_exc:
                    logger.critical("Rollback failed! Target directory %s might be corrupted: %s", target, rollback_exc)
            raise migration_exc

    # Clean up backup directory if swap succeeded
    if backup_dir is not None and backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)
