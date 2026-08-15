"""Package a prepared dataset directory into one or more size-capped zip archives.

Upload endpoints — the Ultralytics Platform among them — reject archives past a
size limit, and a dataset that has been augmented several times over easily clears
it. Rather than fail at upload time, the packer plans the split up front: every part
is a standalone, valid zip holding a subset of the tree at its original relative
paths, so extracting all parts into one directory reproduces the dataset byte for
byte.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Already-compressed payloads dominate a dataset by volume, and deflating them
# spends minutes to save almost nothing. Storing them keeps the size predictable,
# which is what makes the part planning below trustworthy.
_STORED_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".gif", ".zip", ".gz", ".7z", ".mp4", ".avi"}
)

# Local file header + central directory entry + end-of-central-directory share.
# Deliberately generous: overshooting the cap is a failed upload, undershooting it
# only costs a slightly smaller part.
_PER_ENTRY_OVERHEAD_BYTES = 512
_ARCHIVE_OVERHEAD_BYTES = 64 * 1024

DEFAULT_MAX_ARCHIVE_GIB = 20.0


@dataclass
class ArchiveMember:
    source: Path
    arcname: str
    size: int


@dataclass
class ArchivePart:
    index: int
    path: Path
    members: list[ArchiveMember] = field(default_factory=list)
    bytes_written: int = 0
    sha256: str = ""

    @property
    def planned_bytes(self) -> int:
        return sum(m.size + _PER_ENTRY_OVERHEAD_BYTES for m in self.members)


@dataclass
class ArchiveResult:
    parts: list[ArchivePart]
    manifest_path: Path | None
    total_bytes: int
    total_files: int

    @property
    def is_split(self) -> bool:
        return len(self.parts) > 1


def collect_members(dataset_path: Path) -> list[ArchiveMember]:
    """List every file under ``dataset_path`` with its archive-relative name.

    Sorted so that planning and part contents are reproducible across runs.
    """
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    members: list[ArchiveMember] = []
    for path in sorted(dataset_path.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        members.append(
            ArchiveMember(
                source=path,
                arcname=path.relative_to(dataset_path).as_posix(),
                size=path.stat().st_size,
            )
        )
    if not members:
        raise ValueError(f"Dataset directory is empty: {dataset_path}")
    return members


def plan_parts(members: list[ArchiveMember], max_bytes: int) -> list[list[ArchiveMember]]:
    """Group members into parts that each stay under ``max_bytes``.

    Members keep their sorted order so a part holds a contiguous, human-legible slice
    of the tree (all of train/, then valid/, ...) rather than an arbitrary scatter.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive.")
    budget = max_bytes - _ARCHIVE_OVERHEAD_BYTES
    oversized = [m for m in members if m.size + _PER_ENTRY_OVERHEAD_BYTES > budget]
    if oversized:
        biggest = max(oversized, key=lambda m: m.size)
        raise ValueError(
            f"'{biggest.arcname}' is {biggest.size / 1024**3:.2f} GiB, which cannot fit in a "
            f"{max_bytes / 1024**3:.2f} GiB archive. Raise the size limit to package this dataset."
        )

    parts: list[list[ArchiveMember]] = []
    current: list[ArchiveMember] = []
    current_bytes = 0
    for member in members:
        cost = member.size + _PER_ENTRY_OVERHEAD_BYTES
        if current and current_bytes + cost > budget:
            parts.append(current)
            current, current_bytes = [], 0
        current.append(member)
        current_bytes += cost
    if current:
        parts.append(current)
    return parts


def _compression_for(arcname: str, compress_images: bool) -> int:
    if compress_images:
        return zipfile.ZIP_DEFLATED
    suffix = Path(arcname).suffix.lower()
    return zipfile.ZIP_STORED if suffix in _STORED_SUFFIXES else zipfile.ZIP_DEFLATED


def _part_path(output_dir: Path, stem: str, index: int, total: int) -> Path:
    if total == 1:
        return output_dir / f"{stem}.zip"
    width = max(2, len(str(total)))
    return output_dir / f"{stem}.part{index:0{width}d}of{total:0{width}d}.zip"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_dataset_archives(
    dataset_path: Path,
    output_dir: Path,
    stem: str,
    max_bytes: int | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    compute_checksums: bool = True,
    compress_images: bool = False,
) -> ArchiveResult:
    """Write ``dataset_path`` as one zip, or as a numbered series under ``max_bytes``.

    ``compress_images`` deflates already-compressed payloads too. That typically buys
    only a few percent and costs minutes, but augmented datasets repeat the same
    content under different transforms, so the saving can reach ~10%. Part planning
    uses uncompressed sizes either way, so the cap holds regardless.

    Returns the parts written plus a manifest path when the output was split.
    """
    members = collect_members(dataset_path)
    total_bytes = sum(m.size for m in members)
    limit = max_bytes if max_bytes and max_bytes > 0 else None
    groups = plan_parts(members, limit) if limit else [members]

    output_dir.mkdir(parents=True, exist_ok=True)
    total_parts = len(groups)
    parts: list[ArchivePart] = []
    done = 0

    for index, group in enumerate(groups, start=1):
        part_path = _part_path(output_dir, stem, index, total_parts)
        part = ArchivePart(index=index, path=part_path, members=group)
        # Write to a temporary name so an interrupted run never leaves a truncated
        # archive that looks complete.
        staging = part_path.with_suffix(".zip.partial")
        with zipfile.ZipFile(staging, "w", allowZip64=True) as archive:
            for member in group:
                archive.write(
                    member.source,
                    member.arcname,
                    compress_type=_compression_for(member.arcname, compress_images),
                )
                done += 1
                if progress_callback and (done % 500 == 0 or done == len(members)):
                    progress_callback(done, len(members), part_path.name)
        staging.replace(part_path)
        part.bytes_written = part_path.stat().st_size
        if compute_checksums:
            part.sha256 = _sha256(part_path)
        parts.append(part)

    manifest_path: Path | None = None
    if total_parts > 1:
        manifest_path = output_dir / f"{stem}.parts.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "dataset": dataset_path.name,
                    "parts": total_parts,
                    "total_files": len(members),
                    "total_uncompressed_bytes": total_bytes,
                    "max_part_bytes": limit,
                    "reassemble": (
                        "Extract every part into the same directory, in any order, to "
                        "reconstruct the dataset."
                    ),
                    "files": [
                        {
                            "part": part.index,
                            "name": part.path.name,
                            "files": len(part.members),
                            "bytes": part.bytes_written,
                            "sha256": part.sha256,
                        }
                        for part in parts
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return ArchiveResult(
        parts=parts,
        manifest_path=manifest_path,
        total_bytes=total_bytes,
        total_files=len(members),
    )
