import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.utils.fs import retry_filesystem_op, safe_directory_swap


class SafeDirectorySwapTest(unittest.TestCase):
    def test_normal_swap_creates_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            (source / "file1.txt").write_text("hello", encoding="utf-8")
            (source / "sub").mkdir()
            (source / "sub" / "file2.txt").write_text("world", encoding="utf-8")

            safe_directory_swap(source, target)

            self.assertFalse(source.exists())
            self.assertTrue(target.exists())
            self.assertEqual((target / "file1.txt").read_text(encoding="utf-8"), "hello")
            self.assertEqual((target / "sub" / "file2.txt").read_text(encoding="utf-8"), "world")

    def test_swap_replaces_existing_target_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            (source / "new.txt").write_text("new content", encoding="utf-8")
            target.mkdir()
            (target / "old.txt").write_text("old content", encoding="utf-8")

            safe_directory_swap(source, target)

            self.assertFalse(source.exists())
            self.assertTrue(target.exists())
            self.assertEqual((target / "new.txt").read_text(encoding="utf-8"), "new content")
            self.assertFalse((target / "old.txt").exists())

            # Verify no backup directories remain
            backups = list(root.glob(".target.backup-*"))
            self.assertEqual(len(backups), 0)

    def test_simulated_transient_permission_error_retries_and_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            (source / "data.txt").write_text("payload", encoding="utf-8")

            call_count = 0
            original_rename = Path.rename

            def flaky_rename(self_path, dest_path):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise PermissionError(5, "Access is denied (simulated Windows Defender lock)")
                return original_rename(self_path, dest_path)

            with patch.object(Path, "rename", side_effect=flaky_rename, autospec=True):
                safe_directory_swap(source, target, max_retries=5, backoff_base=0.01)

            self.assertGreaterEqual(call_count, 3)
            self.assertTrue(target.exists())
            self.assertEqual((target / "data.txt").read_text(encoding="utf-8"), "payload")

    def test_fallback_to_recursive_migration_when_rename_permanently_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            (source / "img.jpg").write_text("image bytes", encoding="utf-8")
            (source / "subdir").mkdir()
            (source / "subdir" / "labels.txt").write_text("labels", encoding="utf-8")

            # Force rename to fail permanently with PermissionError
            def fail_rename(self_path, dest_path):
                raise PermissionError(5, "Directory locked permanently by external handle")

            with patch.object(Path, "rename", side_effect=fail_rename, autospec=True):
                safe_directory_swap(source, target, max_retries=2, backoff_base=0.01)

            self.assertTrue(target.exists())
            self.assertEqual((target / "img.jpg").read_text(encoding="utf-8"), "image bytes")
            self.assertEqual((target / "subdir" / "labels.txt").read_text(encoding="utf-8"), "labels")
            self.assertFalse(source.exists())

    def test_rollback_to_backup_when_fallback_migration_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = (root / "source").resolve()
            target = (root / "target").resolve()
            source.mkdir()
            (source / "new.txt").write_text("new content", encoding="utf-8")
            target.mkdir()
            (target / "original.txt").write_text("original content", encoding="utf-8")

            original_rename = Path.rename

            def selective_rename(self_path, dest_path):
                if Path(self_path).resolve() == source:
                    raise PermissionError(5, "Source rename denied")
                return original_rename(self_path, dest_path)

            with patch.object(Path, "rename", side_effect=selective_rename, autospec=True):
                with patch("src.utils.fs.shutil.move", side_effect=RuntimeError("Disk full during move")):
                    with self.assertRaises(RuntimeError):
                        safe_directory_swap(source, target, max_retries=2, backoff_base=0.01)

            # Target should be rolled back and restored
            self.assertTrue(target.exists())
            self.assertEqual((target / "original.txt").read_text(encoding="utf-8"), "original content")

    def test_missing_source_raises_file_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "non_existent"
            target = root / "target"

            with self.assertRaises(FileNotFoundError):
                safe_directory_swap(source, target)

    def test_identical_source_and_target_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "same_dir"
            source.mkdir()

            with self.assertRaises(ValueError):
                safe_directory_swap(source, source)


if __name__ == "__main__":
    unittest.main()
