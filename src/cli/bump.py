import argparse
import re
import sys
from pathlib import Path

# Paths relative to project root (where command is usually run)
VERSION_FILE = Path("src/__version__.py")
PYPROJECT_FILE = Path("pyproject.toml")


def get_current_version():
    if not VERSION_FILE.exists():
        print(f"Error: Could not find {VERSION_FILE}", file=sys.stderr)
        sys.exit(1)

    content = VERSION_FILE.read_text("utf-8")
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        print(f"Error: Could not parse version from {VERSION_FILE}", file=sys.stderr)
        sys.exit(1)

    return match.group(1)


def write_version_to_file(
    file_path: Path, pattern: str, replacement_template: str, new_version: str
):
    if not file_path.exists():
        return False

    content = file_path.read_text("utf-8")
    new_content, count = re.subn(
        pattern, replacement_template.format(new_version), content
    )

    if count > 0:
        file_path.write_text(new_content, "utf-8")
        return True
    return False


def bump_version(current_version: str, bump_type: str) -> str:
    """Bump the semantic version string based on type."""
    try:
        parts = [int(p) for p in current_version.split(".")]
    except ValueError:
        print(
            f"Error: Current version '{current_version}' is not pure numeric semantic versioning (e.g. x.y.z).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Pad to 3 parts if needed
    while len(parts) < 3:
        parts.append(0)

    if bump_type == "major":
        parts[0] += 1
        parts[1] = 0
        parts[2] = 0
    elif bump_type == "minor":
        parts[1] += 1
        parts[2] = 0
    elif bump_type == "patch":
        parts[2] += 1
    elif re.match(r"^\d+\.\d+\.\d+$", bump_type):
        return bump_type
    else:
        print(
            f"Error: Invalid bump type '{bump_type}'. Must be 'major', 'minor', 'patch', or a specific version 'x.y.z'.",
            file=sys.stderr,
        )
        sys.exit(1)

    return ".".join(map(str, parts))


def main():
    parser = argparse.ArgumentParser(description="Bump project version.")
    parser.add_argument(
        "type",
        type=str,
        help="Version bump type: major, minor, patch, or a specific version like 1.2.3",
    )

    args = parser.parse_args()

    try:
        from src.utils.tui import TUI_CONSOLE as console
        def print_success(msg): console.print(f"[bold green]{msg}[/bold green]")
        def print_info(msg): console.print(f"[bold cyan]{msg}[/bold cyan]")
    except ImportError:
        def print_success(msg): print(f"SUCCESS: {msg}")
        def print_info(msg): print(msg)

    current_version = get_current_version()
    new_version = bump_version(current_version, args.type.lower())

    if current_version == new_version:
        print_info(f"Version is already {new_version}. No changes made.")
        return

    print_info(f"Bumping version: {current_version} -> {new_version}")

    # 1. Update src/__version__.py
    v_pattern = r'__version__\s*=\s*["\'][^"\']+["\']'
    v_replacement = f'__version__ = "{new_version}"'
    if write_version_to_file(VERSION_FILE, v_pattern, v_replacement, new_version):
        print_info(f"Updated {VERSION_FILE}")
    else:
        print(f"Failed to update {VERSION_FILE}", file=sys.stderr)
        sys.exit(1)

    # 2. Update pyproject.toml
    p_pattern = r'version\s*=\s*["\'][^"\']+["\']'
    p_replacement = f'version = "{new_version}"'
    if write_version_to_file(PYPROJECT_FILE, p_pattern, p_replacement, new_version):
        print_info(f"Updated {PYPROJECT_FILE}")
    else:
        print(f"Failed to update {PYPROJECT_FILE}", file=sys.stderr)
        sys.exit(1)

    print_success(f"✓ Version successfully bumped to {new_version}")
    print_info(
        "Don't forget to run 'uv sync' to lock the new version in your environment if necessary."
    )


if __name__ == "__main__":
    main()
