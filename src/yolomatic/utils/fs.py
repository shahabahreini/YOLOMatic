"""
Filesystem utility functions for safe atomic directory swapping and cross-platform file operations.
"""
from __future__ import annotations

import errno
import logging
import os
import shutil
import sys
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Windows: ERROR_ACCESS_DENIED, ERROR_SHARING_VIOLATION, ERROR_LOCK_VIOLATION.
_TRANSIENT_WINERRORS = frozenset({5, 32, 33})
_TRANSIENT_ERRNOS = frozenset({errno.EACCES, errno.EBUSY, errno.EAGAIN})


def is_transient_fs_error(exc: BaseException) -> bool:
    """Return True for lock/sharing errors that are worth retrying.

    Permanent failures (disk full, missing file, destination exists, read-only
    filesystem) are *not* transient: retrying them only delays the error and can
    make a partially completed ``shutil.move`` nest data inside itself.
    """
    if not isinstance(exc, OSError):
        return False
    if isinstance(exc, (FileNotFoundError, FileExistsError, IsADirectoryError, NotADirectoryError)):
        return False
    if getattr(exc, "winerror", None) in _TRANSIENT_WINERRORS:
        return True
    if isinstance(exc, PermissionError):
        return True
    return exc.errno in _TRANSIENT_ERRNOS


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
        except OSError as exc:
            last_exc = exc
            if not is_transient_fs_error(exc) or attempt == max_retries - 1:
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
        migrated: list[tuple[Path, Path]] = []
        try:
            target.mkdir(parents=True, exist_ok=True)
            for item in list(source.iterdir()):
                dest_item = target / item.name
                retry_filesystem_op(
                    lambda: shutil.move(str(item), str(dest_item)),
                    max_retries=max_retries,
                    backoff_base=backoff_base,
                )
                migrated.append((item, dest_item))
            # Remove the now-empty source container
            retry_filesystem_op(
                lambda: source.rmdir(),
                max_retries=max_retries,
                backoff_base=backoff_base,
            )
            swap_succeeded = True
        except Exception as migration_exc:
            logger.error("Recursive item migration from %s to %s failed: %s", source, target, migration_exc)
            _rollback_partial_migration(source, target, backup_dir, migrated, max_retries, backoff_base)
            raise migration_exc

    # Clean up backup directory if swap succeeded
    if backup_dir is not None and backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)


def _rollback_partial_migration(
    source: Path,
    target: Path,
    backup_dir: Path | None,
    migrated: list[tuple[Path, Path]],
    max_retries: int,
    backoff_base: float,
) -> None:
    """Undo a failed item-by-item migration without ever deleting user data.

    1. Move every already-migrated item back into ``source``.
    2. If ``target`` is then empty, remove it; otherwise park it under a
       ``.partial-<uuid>`` name so nothing is lost.
    3. Restore the original ``target`` from ``backup_dir`` when one exists.
    """
    restore_failures: list[Path] = []
    for original, moved in reversed(migrated):
        if not moved.exists():
            continue
        try:
            source.mkdir(parents=True, exist_ok=True)
            retry_filesystem_op(
                lambda: shutil.move(str(moved), str(original)),
                max_retries=max_retries,
                backoff_base=backoff_base,
            )
        except Exception as exc:  # noqa: BLE001 - keep rolling back the rest
            restore_failures.append(moved)
            logger.error("Could not move %s back to %s during rollback: %s", moved, original, exc)

    if target.exists():
        try:
            target.rmdir()
        except OSError:
            parked = target.parent / f".{target.name}.partial-{uuid.uuid4().hex}"
            try:
                target.rename(parked)
                logger.critical(
                    "Rollback left files in %s (could not be moved back to %s). Nothing was deleted.",
                    parked,
                    source,
                )
            except OSError as exc:
                logger.critical(
                    "Rollback could not clear %s (%s). Original data remains in backup %s.",
                    target,
                    exc,
                    backup_dir,
                )
                return

    if backup_dir is not None and backup_dir.exists():
        logger.info("Rolling back target directory from backup %s...", backup_dir)
        try:
            retry_filesystem_op(
                lambda: backup_dir.rename(target),
                max_retries=max_retries,
                backoff_base=backoff_base,
            )
        except Exception as rollback_exc:  # noqa: BLE001
            logger.critical(
                "Rollback failed! Original target data is preserved at %s: %s", backup_dir, rollback_exc
            )
    if restore_failures:
        logger.critical("%d item(s) could not be restored to %s.", len(restore_failures), source)


class UnsafePathError(ValueError):
    """Raised when a destructive operation targets a path it must never touch."""


def _same_or_inside(path: Path, other: Path) -> bool:
    """True when ``path`` equals ``other`` or lives inside it (case-insensitive where the OS is)."""
    a, b = os.path.normcase(str(path)), os.path.normcase(str(other))
    if sys.platform == "darwin":  # default APFS volumes are case-insensitive
        a, b = a.casefold(), b.casefold()
    return a == b or a.startswith(b.rstrip(os.sep) + os.sep)


def ensure_safe_destructive_target(
    target: Path | str,
    allowed_root: Path | str,
    forbidden: Iterable[Path | str] = (),
) -> Path:
    """Validate a path before it is deleted or overwritten; return it resolved.

    The target must live strictly *inside* ``allowed_root`` (never the root
    itself or an ancestor of it), and must not equal, contain, or sit inside any
    ``forbidden`` path (for example the source datasets being read).
    """
    resolved = Path(target).expanduser().resolve()
    root = Path(allowed_root).expanduser().resolve()
    if not _same_or_inside(resolved, root) or _same_or_inside(root, resolved):
        raise UnsafePathError(f"Refusing to modify {resolved}: it must be a folder inside {root}.")
    for raw in forbidden:
        other = Path(raw).expanduser().resolve()
        if _same_or_inside(resolved, other) or _same_or_inside(other, resolved):
            raise UnsafePathError(
                f"Refusing to modify {resolved}: it overlaps the source folder {other}."
            )
    return resolved


def ensure_overwritable_dir(
    target: Path | str,
    *,
    protected: Iterable[Path | str] = (),
    must_not_contain: Iterable[Path | str] = (),
) -> Path:
    """Validate a user-chosen folder before deleting it to overwrite; return it resolved.

    Refuses filesystem roots, the home directory, the working directory and any
    ``protected`` folder -- including every ancestor of those -- and any folder
    that contains one of ``must_not_contain`` (for example the input file).
    """
    resolved = Path(target).expanduser().resolve()
    if resolved == Path(resolved.anchor):
        raise UnsafePathError(f"Refusing to delete the filesystem root {resolved}.")
    guarded = [Path.home(), Path.cwd(), *protected]
    for raw in guarded:
        other = Path(raw).expanduser().resolve()
        if _same_or_inside(other, resolved):
            raise UnsafePathError(
                f"Refusing to delete {resolved}: it is, or contains, the protected folder {other}."
            )
    for raw in must_not_contain:
        inner = Path(raw).expanduser().resolve()
        if _same_or_inside(inner, resolved):
            raise UnsafePathError(f"Refusing to delete {resolved}: it contains the input {inner}.")
    return resolved
