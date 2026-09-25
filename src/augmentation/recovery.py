"""
Dataset augmentation recovery utilities.

Detects and recovers orphaned .augmenting-<uuid> staging directories left behind
by interrupted, crashed, or locked augmentation runs, preventing data loss.
"""
from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.utils.fs import safe_directory_swap

logger = logging.getLogger(__name__)

_AUGMENTING_PATTERN = re.compile(r"^\.(.+)\.augmenting-[0-9a-fA-F]{32}$")


@dataclass
class RecoverableAugmentation:
    stage_path: Path
    output_path: Path
    target_name: str
    target_path: Path
    image_count: int
    has_data_yaml: bool


def find_recoverable_augmentations(datasets_dir: Path | str) -> list[RecoverableAugmentation]:
    """
    Scan a datasets directory for orphaned augmentation staging directories.

    Returns a list of RecoverableAugmentation objects containing output images
    ready to be finalized.
    """
    root = Path(datasets_dir).resolve()
    if not root.exists() or not root.is_dir():
        return []

    recoverable: list[RecoverableAugmentation] = []
    for item in root.iterdir():
        if not item.is_dir():
            continue
        match = _AUGMENTING_PATTERN.match(item.name)
        if not match:
            continue

        target_name = match.group(1)
        target_path = root / target_name
        output_dir = item / "output"

        if not output_dir.exists() or not output_dir.is_dir():
            continue

        # Count completed images across split folders
        image_count = 0
        for split in ("train", "valid", "test"):
            images_dir = output_dir / split / "images"
            if images_dir.exists():
                image_count += sum(1 for _ in images_dir.glob("*.jpg"))

        has_data_yaml = (output_dir / "data.yaml").exists()

        if image_count > 0:
            recoverable.append(
                RecoverableAugmentation(
                    stage_path=item,
                    output_path=output_dir,
                    target_name=target_name,
                    target_path=target_path,
                    image_count=image_count,
                    has_data_yaml=has_data_yaml,
                )
            )

    return sorted(recoverable, key=lambda r: r.stage_path.stat().st_mtime, reverse=True)


def recover_augmentation(rec: RecoverableAugmentation) -> bool:
    """
    Finalize an orphaned augmentation run by moving its output to the intended target.
    """
    logger.info("Recovering orphaned augmentation %s -> %s", rec.output_path, rec.target_path)
    try:
        safe_directory_swap(rec.output_path, rec.target_path)
        shutil.rmtree(rec.stage_path, ignore_errors=True)
        return True
    except Exception as exc:
        logger.error("Failed to recover augmentation from %s: %s", rec.stage_path, exc)
        raise
