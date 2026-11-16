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


def _source_image_for(path: Path, sibling_names: set[str] | None) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        for candidate_ext in (extension, extension.upper()):
            candidate = path.with_suffix(candidate_ext)
            if sibling_names is not None:
                if candidate.name.lower() in sibling_names:
                    return candidate
            elif candidate.is_file():
                return candidate
    return None


def has_ultralytics_cache_signature(path: Path, image_path: Path | None = None) -> bool:
    """Return True when ``path`` looks like an array Ultralytics wrote for ``image_path``.

    Ultralytics' ``cache=disk`` stores ``np.save(f, cv2.imread(img))``: a uint8
    ``H x W x C`` (or ``H x W``) array matching the source image size. Only the
    ``.npy`` header and the image header are read, never pixel data. Arrays that
    differ (float depth maps, uint16 NDVI, embeddings, ...) are user data and
    must never be deleted.
    """
    try:
        import numpy as np

        with open(path, "rb") as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, _fortran, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, _fortran, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                return False
    except Exception:  # noqa: BLE001 - unreadable or not an .npy array: keep it
        return False
    if dtype != np.uint8 or len(shape) not in (2, 3):
        return False
    if len(shape) == 3 and shape[2] not in (1, 3, 4):
        return False
    if image_path is None:
        return True
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
    except Exception:  # noqa: BLE001 - cannot read the image header: be conservative
        return False
    return tuple(shape[:2]) == (height, width)


def is_ultralytics_metadata_cache(path: Path) -> bool:
    """Recognize Ultralytics label-index caches without treating arbitrary caches as datasets."""
    if path.suffix.lower() != ".cache" or not path.is_file():
        return False
    return path.name == "labels.cache" or path.parent.name.lower() == "labels"


def is_dataset_runtime_cache(path: Path, sibling_names: set[str] | None = None) -> bool:
    return is_ultralytics_image_cache(path, sibling_names) or is_ultralytics_metadata_cache(path)


def _is_verified_image_cache(path: Path, sibling_names: set[str]) -> bool:
    if not is_ultralytics_image_cache(path, sibling_names):
        return False
    return has_ultralytics_cache_signature(path, _source_image_for(path, sibling_names))


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
                    if _is_verified_image_cache(path, sibling_names):
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
    """Remove verified Ultralytics image caches, retaining label metadata caches.

    A ``.npy`` file is removed only when it sits beside a same-stem image *and*
    its header matches what Ultralytics writes for that image.
    """
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
                    if _is_verified_image_cache(path, sibling_names):
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
