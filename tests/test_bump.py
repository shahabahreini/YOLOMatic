from __future__ import annotations

import unittest
from pathlib import Path

from src.cli.bump import bump_version, read_project_version, replace_project_version

ROOT = Path(__file__).resolve().parents[1]


class BumpTest(unittest.TestCase):
    def test_only_project_version_changes_in_real_pyproject(self) -> None:
        # Regression: an unanchored regex rewrote target-version, required-version
        # and every pinned [[tool.uv.dependency-metadata]] version too.
        original = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        current = read_project_version(original)
        self.assertIsNotNone(current)

        updated = replace_project_version(original, "99.0.0")

        self.assertEqual(read_project_version(updated), "99.0.0")
        changed = [
            (a, b)
            for a, b in zip(original.splitlines(), updated.splitlines(), strict=True)
            if a != b
        ]
        self.assertEqual(len(changed), 1)
        self.assertIn(current, changed[0][0])
        self.assertEqual(changed[0][1].strip(), 'version = "99.0.0"')

    def test_ignores_versions_outside_project_table(self) -> None:
        content = (
            '[tool.ruff]\ntarget-version = "py312"\n\n'
            '[project]\nname = "x"\nversion = "1.2.3"\n\n'
            '[[tool.uv.dependency-metadata]]\nversion = "11.0.0"\n'
        )
        updated = replace_project_version(content, "1.2.4")
        self.assertIn('target-version = "py312"', updated)
        self.assertIn('version = "11.0.0"', updated)
        self.assertIn('version = "1.2.4"', updated)

    def test_bump_version_semantics(self) -> None:
        self.assertEqual(bump_version("6.4.0", "patch"), "6.4.1")
        self.assertEqual(bump_version("6.4.1", "minor"), "6.5.0")
        self.assertEqual(bump_version("6.4.1", "major"), "7.0.0")
        self.assertEqual(bump_version("6.4.1", "7.0.0"), "7.0.0")


if __name__ == "__main__":
    unittest.main()
