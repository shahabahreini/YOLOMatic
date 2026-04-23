from __future__ import annotations

from pathlib import Path

try:
    # prefer importlib.metadata (Python 3.8+)
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # pragma: no cover - very old python
    from pkg_resources import get_distribution, DistributionNotFound

__all__ = ["__version__"]


def _get_version() -> str:
    """Return the current package version by inspecting metadata."""
    try:
        # running from installed package
        return version("yolomatic")
    except NameError:  # importlib not available
        try:
            return get_distribution("yolomatic").version
        except Exception:  # DistributionNotFound or others
            pass
    except PackageNotFoundError:
        # not installed yet; read from pyproject using the standard library
        try:
            import tomllib as _toml
        except ImportError:
            import tomli as _toml  # type: ignore

        project_root = Path(__file__).resolve().parent.parent
        with open(project_root / "pyproject.toml", "rb") as f:
            data = _toml.load(f)
        return data.get("project", {}).get("version", "0.0.0")

    return "0.0.0"


__version__ = _get_version()
