import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.datasets.archive import (
    ArchiveMember,
    collect_members,
    package_dataset_archives,
    plan_parts,
)


def _build_dataset(root: Path, images: int = 30, payload: int = 20_000) -> Path:
    dataset = root / "ds"
    (dataset / "train" / "images").mkdir(parents=True)
    (dataset / "train" / "labels").mkdir(parents=True)
    for index in range(images):
        (dataset / "train" / "images" / f"i{index:03d}.jpg").write_bytes(
            bytes(range(256)) * (payload // 256)
        )
        (dataset / "train" / "labels" / f"i{index:03d}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
    (dataset / "data.yaml").write_text("nc: 1\nnames: [x]\ntask: detect\n", encoding="utf-8")
    return dataset


class PlanPartsTest(unittest.TestCase):
    def _members(self, sizes: list[int]) -> list[ArchiveMember]:
        return [
            ArchiveMember(source=Path(f"/tmp/f{i}"), arcname=f"f{i}", size=size)
            for i, size in enumerate(sizes)
        ]

    def test_single_part_when_everything_fits(self) -> None:
        parts = plan_parts(self._members([100, 100, 100]), max_bytes=10_000_000)
        self.assertEqual(len(parts), 1)

    def test_splits_once_the_budget_is_exceeded(self) -> None:
        members = self._members([400_000] * 10)
        parts = plan_parts(members, max_bytes=1_000_000)
        self.assertGreater(len(parts), 1)
        # No part may exceed the cap, which is the whole point of the exercise.
        for part in parts:
            self.assertLessEqual(sum(m.size for m in part), 1_000_000)
        self.assertEqual(sum(len(p) for p in parts), len(members))

    def test_every_member_lands_in_exactly_one_part(self) -> None:
        members = self._members([i * 1000 + 500 for i in range(50)])
        parts = plan_parts(members, max_bytes=200_000)
        placed = [m.arcname for part in parts for m in part]
        self.assertEqual(sorted(placed), sorted(m.arcname for m in members))
        self.assertEqual(len(placed), len(set(placed)))

    def test_rejects_a_file_larger_than_the_cap(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            plan_parts(self._members([5_000_000]), max_bytes=1_000_000)

    def test_rejects_a_non_positive_cap(self) -> None:
        with self.assertRaises(ValueError):
            plan_parts(self._members([10]), max_bytes=0)


class PackageDatasetArchivesTest(unittest.TestCase):
    def test_writes_a_single_flat_archive_when_under_the_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = _build_dataset(root)
            result = package_dataset_archives(
                dataset, root / "out", "demo", max_bytes=None, compute_checksums=False
            )
            self.assertEqual(len(result.parts), 1)
            self.assertFalse(result.is_split)
            self.assertIsNone(result.manifest_path)
            self.assertEqual(result.parts[0].path.name, "demo.zip")
            with zipfile.ZipFile(result.parts[0].path) as archive:
                names = archive.namelist()
            # Archive root must hold data.yaml directly — no wrapper directory.
            self.assertIn("data.yaml", names)
            self.assertTrue(any(n.startswith("train/images/") for n in names))

    def test_split_parts_each_stay_under_the_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = _build_dataset(root)
            cap = 150_000
            result = package_dataset_archives(
                dataset, root / "out", "demo", max_bytes=cap, compute_checksums=False
            )
            self.assertGreater(len(result.parts), 1)
            for part in result.parts:
                self.assertLessEqual(part.bytes_written, cap)
                self.assertIn("of", part.path.name)

    def test_extracting_all_parts_reproduces_the_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = _build_dataset(root)
            result = package_dataset_archives(
                dataset, root / "out", "demo", max_bytes=150_000, compute_checksums=False
            )
            rebuilt = root / "rebuilt"
            for part in result.parts:
                with zipfile.ZipFile(part.path) as archive:
                    archive.extractall(rebuilt)

            original = {
                p.relative_to(dataset).as_posix(): p.read_bytes()
                for p in dataset.rglob("*") if p.is_file()
            }
            restored = {
                p.relative_to(rebuilt).as_posix(): p.read_bytes()
                for p in rebuilt.rglob("*") if p.is_file()
            }
            self.assertEqual(original, restored)

    def test_manifest_lists_every_part_with_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = _build_dataset(root)
            result = package_dataset_archives(
                dataset, root / "out", "demo", max_bytes=150_000, compute_checksums=True
            )
            self.assertIsNotNone(result.manifest_path)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["parts"], len(result.parts))
            self.assertEqual(manifest["total_files"], result.total_files)
            self.assertEqual(len(manifest["files"]), len(result.parts))
            for entry in manifest["files"]:
                self.assertEqual(len(entry["sha256"]), 64)

    def test_leaves_no_partial_archive_behind(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = _build_dataset(root)
            package_dataset_archives(
                dataset, root / "out", "demo", max_bytes=150_000, compute_checksums=False
            )
            self.assertEqual(list((root / "out").glob("*.partial")), [])

    def test_rejects_a_missing_or_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(FileNotFoundError):
                collect_members(root / "nope")
            (root / "empty").mkdir()
            with self.assertRaises(ValueError):
                collect_members(root / "empty")


if __name__ == "__main__":
    unittest.main()
