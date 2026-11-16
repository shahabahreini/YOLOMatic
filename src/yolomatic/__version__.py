"""Single source of truth for the package version: the installed distribution metadata."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

try:
    __version__ = _dist_version("yolomatic")
except PackageNotFoundError:  # running from an uninstalled source tree
    __version__ = "0.0.0+unknown"
