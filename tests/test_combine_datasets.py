from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src.utils.combine_datasets import (
    find_datasets,
    merge_datasets,
    resolve_output_path,
    validate_output_name,
)
from src.utils.fs import UnsafePathError


def _make_dataset(root: Path, name: str, classes: list[str], label_class: int) -> None:
    ds = root / name
    for split in ("train", "val"):
        (ds / split / "images").mkdir(parents=True)
        (ds / split / "labels").mkdir(parents=True)
        (ds / split / "images" / f"{split}.jpg").write_bytes(b"jpg")
        (ds / split / "labels" / f"{split}.txt").write_text(
            f"{label_class} 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
    (ds / "data.yaml").write_text(
        yaml.safe_dump({"train": "train/images", "val": "val/images", "names": classes}),
        encoding="utf-8",
    )


class CombineDatasetsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve() / "datasets"
        self.root.mkdir()
        _make_dataset(self.root, "a", ["Car", "person"], label_class=1)
        _make_dataset(self.root, "b", ["car"], label_class=0)
        self.datasets = sorted(find_datasets(str(self.root)), key=lambda d: d["name"])

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_merge_runs_end_to_end_and_remaps_classes(self) -> None:
        # Regression: merge_datasets used rich Progress classes that were never
        # imported, so every merge crashed with NameError.
        out = merge_datasets(self.datasets, "merged", normalize_classes=True, datasets_root=self.root)
        self.assertEqual(out, self.root / "merged")
        data = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
        self.assertEqual(data["names"], ["Car", "person"])
        self.assertEqual(
            (out / "train" / "labels" / "a_train.txt").read_text(encoding="utf-8").split()[0], "1"
        )
        self.assertEqual(
            (out / "train" / "labels" / "b_train.txt").read_text(encoding="utf-8").split()[0], "0"
        )
        self.assertEqual([p.name for p in self.root.iterdir() if p.name.startswith(".")], [])

    def test_rejects_unsafe_output_names(self) -> None:
        for bad in ("", ".", "..", "../escape", "nested/name", ".hidden"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_output_name(bad)
        with self.assertRaises(UnsafePathError):
            resolve_output_path(self.datasets, "a", self.root)

    def test_existing_output_requires_overwrite_and_survives_failures(self) -> None:
        existing = self.root / "merged"
        existing.mkdir()
        (existing / "keep.txt").write_text("keep", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            merge_datasets(self.datasets, "merged", normalize_classes=True, datasets_root=self.root)

        with patch(
            "src.utils.combine_datasets.process_single_image", side_effect=RuntimeError("boom")
        ):
            with self.assertRaises(RuntimeError):
                merge_datasets(
                    self.datasets, "merged", normalize_classes=True,
                    datasets_root=self.root, overwrite=True,
                )
        self.assertEqual((existing / "keep.txt").read_text(encoding="utf-8"), "keep")
        self.assertEqual([p.name for p in self.root.iterdir() if p.name.startswith(".")], [])


if __name__ == "__main__":
    unittest.main()
