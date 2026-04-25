from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.utils.training_preflight import _find_uv_project_root


class FindUvProjectRootTests(unittest.TestCase):
    def test_returns_directory_containing_uv_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            (root / "uv.lock").touch()

            self.assertEqual(_find_uv_project_root(root), root)

    def test_walks_up_to_find_uv_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            (root / "uv.lock").touch()
            nested = root / "src" / "deep" / "child"
            nested.mkdir(parents=True)

            self.assertEqual(_find_uv_project_root(nested), root)

    def test_does_not_match_subtree_without_uv_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            nested = root / "a" / "b"
            nested.mkdir(parents=True)

            # The function may still find a uv.lock further up the filesystem,
            # but it must not falsely report `root` when no uv.lock exists there.
            self.assertNotEqual(_find_uv_project_root(nested), root)


if __name__ == "__main__":
    unittest.main()
