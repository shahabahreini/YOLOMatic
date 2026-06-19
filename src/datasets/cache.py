"""Inspection and cleanup of dataset-local runtime cache artifacts."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = (".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp")


@dataclass
class DatasetCacheStats:
    image_cache_files: int = 0
    image_cache_bytes: int = 0
    metadata_cache_files: int = 0
    metadata_cache_bytes: int = 0
    removed_files: int = 0
    reclaimed_bytes: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total_files(self) -> int:
        return self.image_cache_files + self.metadata_cache_files

    @property
    def total_bytes(self) -> int:
        return self.image_cache_bytes + self.metadata_cache_bytes


def is_ultralytics_image_cache(
    path: Path,
    sibling_names: set[str] | None = None,
) -> bool:
    """Return True only for a .npy cache with a same-stem source image beside it."""
    if path.suffix.lower() != ".npy" or not path.is_file():
        return False
    if sibling_names is not None:
        return any(path.with_suffix(extension).name.lower() in sibling_names for extension in IMAGE_EXTENSIONS)
    return any(
        candidate.is_file()
        for extension in IMAGE_EXTENSIONS
        for candidate in (path.with_suffix(extension), path.with_suffix(extension.upper()))
    )


def is_ultralytics_metadata_cache(path: Path) -> bool:
    """Recognize Ultralytics label-index caches without treating arbitrary caches as datasets."""
    if path.suffix.lower() != ".cache" or not path.is_file():
        return False
    return path.name == "labels.cache" or path.parent.name.lower() == "labels"


def is_dataset_runtime_cache(path: Path, sibling_names: set[str] | None = None) -> bool:
    return is_ultralytics_image_cache(path, sibling_names) or is_ultralytics_metadata_cache(path)


def inspect_dataset_cache(dataset_path: str | Path) -> DatasetCacheStats:
    root = Path(dataset_path)
    stats = DatasetCacheStats()
    if not root.exists():
        return stats

    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name != ".yolomatic_cache"]
            directory = Path(dirpath)
            sibling_names = {name.lower() for name in filenames}
            for filename in filenames:
                path = directory / filename
                try:
                    size = path.stat().st_size
                    if is_ultralytics_image_cache(path, sibling_names):
                        stats.image_cache_files += 1
                        stats.image_cache_bytes += size
                    elif is_ultralytics_metadata_cache(path):
                        stats.metadata_cache_files += 1
                        stats.metadata_cache_bytes += size
                except OSError as error:
                    stats.errors.append(f"{path}: {error}")
    except OSError as error:
        stats.errors.append(f"{root}: {error}")
    return stats


def clean_dataset_image_cache(dataset_path: str | Path) -> DatasetCacheStats:
    """Remove verified Ultralytics image caches, retaining label metadata caches."""
    root = Path(dataset_path)
    stats = DatasetCacheStats()
    if not root.exists():
        return stats

    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name != ".yolomatic_cache"]
            directory = Path(dirpath)
            sibling_names = {name.lower() for name in filenames}
            for filename in filenames:
                path = directory / filename
                try:
                    size = path.stat().st_size
                    if is_ultralytics_image_cache(path, sibling_names):
                        stats.image_cache_files += 1
                        stats.image_cache_bytes += size
                        path.unlink()
                        stats.removed_files += 1
                        stats.reclaimed_bytes += size
                    elif is_ultralytics_metadata_cache(path):
                        stats.metadata_cache_files += 1
                        stats.metadata_cache_bytes += size
                except OSError as error:
                    stats.errors.append(f"{path}: {error}")
    except OSError as error:
        stats.errors.append(f"{root}: {error}")
    return stats


def normalize_yolo_cache_setting(value: Any) -> tuple[Any, bool]:
    """Disable persistent disk caching while preserving False and RAM modes."""
    if isinstance(value, str) and value.strip().lower() == "disk":
        return False, True
    return value, False
