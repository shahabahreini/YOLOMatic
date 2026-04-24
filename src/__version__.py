import re
from pathlib import Path

try:
    # prefer importlib.metadata when available (runtime-installed package)
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _dist_version
except Exception:  # pragma: no cover - fallback
    _dist_version = None
    PackageNotFoundError = Exception


def _get_version() -> str:
    """Return project version. Prefer installed package metadata, fallback to pyproject.toml."""
    if _dist_version is not None:
        try:
            return _dist_version("yolomatic")
        except PackageNotFoundError:
            pass

    # Fallback: read `pyproject.toml` at repository root using a simple regex.
    project_root = Path(__file__).resolve().parent.parent
    pyproject_file = project_root / "pyproject.toml"
    if pyproject_file.exists():
        text = pyproject_file.read_text(encoding="utf-8")
        m = re.search(r'^\s*version\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
        if m:
            return m.group(1)

    return "0.0.0"


__version__ = _get_version()
