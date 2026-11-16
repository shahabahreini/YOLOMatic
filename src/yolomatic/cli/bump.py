import argparse
import re
import sys
from pathlib import Path

# Paths relative to project root (where command is usually run)
PYPROJECT_FILE = Path("pyproject.toml")


_PROJECT_VERSION_RE = re.compile(r'^(version\s*=\s*)(["\'])([^"\']+)\2', re.MULTILINE)


def _project_table_span(content: str) -> tuple[int, int] | None:
    """Return the ``[project]`` table's character span in a pyproject.toml."""
    header = re.search(r"^\[project\]\s*$", content, re.MULTILINE)
    if header is None:
        return None
    next_table = re.search(r"^\[", content[header.end():], re.MULTILINE)
    end = header.end() + next_table.start() if next_table else len(content)
    return header.end(), end


def read_project_version(content: str) -> str | None:
    span = _project_table_span(content)
    if span is None:
        return None
    match = _PROJECT_VERSION_RE.search(content, span[0], span[1])
    return match.group(3) if match else None


def replace_project_version(content: str, new_version: str) -> str:
    """Replace only ``[project].version``.

    The previous implementation substituted every ``version = "..."`` in the
    file, which also rewrote ``target-version``, ``required-version`` and the
    pinned ``[[tool.uv.dependency-metadata]]`` versions.
    """
    span = _project_table_span(content)
    if span is None:
        raise ValueError("pyproject.toml has no [project] table.")
    match = _PROJECT_VERSION_RE.search(content, span[0], span[1])
    if match is None:
        raise ValueError("[project] table has no version field.")
    replacement = f"{match.group(1)}{match.group(2)}{new_version}{match.group(2)}"
    return content[: match.start()] + replacement + content[match.end():]


def get_current_version():
    if not PYPROJECT_FILE.exists():
        print(f"Error: Could not find {PYPROJECT_FILE}", file=sys.stderr)
        sys.exit(1)

    version = read_project_version(PYPROJECT_FILE.read_text("utf-8"))
    if version is None:
        print(f"Error: Could not parse [project] version from {PYPROJECT_FILE}", file=sys.stderr)
        sys.exit(1)
    return version


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

        def print_success(msg):
            console.print(f"[bold green]{msg}[/bold green]")

        def print_info(msg):
            console.print(f"[bold cyan]{msg}[/bold cyan]")

    except ImportError:

        def print_success(msg):
            print(f"SUCCESS: {msg}")

        def print_info(msg):
            print(msg)

    current_version = get_current_version()
    new_version = bump_version(current_version, args.type.lower())

    if current_version == new_version:
        print_info(f"Version is already {new_version}. No changes made.")
        return

    print_info(f"Bumping version: {current_version} -> {new_version}")

    # Update only [project].version (single source of truth). `src/__version__.py`
    # reads the installed metadata / pyproject at runtime.
    try:
        content = PYPROJECT_FILE.read_text("utf-8")
        PYPROJECT_FILE.write_text(replace_project_version(content, new_version), "utf-8")
    except (OSError, ValueError) as error:
        print(f"Failed to update {PYPROJECT_FILE}: {error}", file=sys.stderr)
        sys.exit(1)
    print_info(f"Updated {PYPROJECT_FILE}")

    print_success(f"✓ Version successfully bumped to {new_version}")
    print_info(
        "Run 'uv lock' to record the new version in uv.lock."
    )


if __name__ == "__main__":
    main()
