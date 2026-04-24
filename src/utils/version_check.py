"""Dependency health checks for YOLOmatic.

Queries PyPI for the latest version of each tracked package, compares against
the installed version, and classifies the gap by semantic-version severity so
the TUI can surface critical outdated dependencies distinctly from cosmetic
patch releases.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as installed_version
from typing import Sequence

from packaging.version import InvalidVersion, Version


PYPI_TIMEOUT_SECONDS = 4.0
PYPI_JSON_URL = "https://pypi.org/pypi/{name}/json"
PYPI_WORKER_LIMIT = 8


@dataclass(frozen=True)
class CriticalPackage:
    """A package YOLOmatic depends on and wants to keep current."""

    name: str           # PyPI distribution name
    display_name: str   # Human-friendly label for the TUI
    description: str    # One-line role summary
    importance: str     # "critical" | "important" | "optional"


# Ordered roughly by how load-bearing the dep is for YOLOmatic's core flows.
CRITICAL_PACKAGES: tuple[CriticalPackage, ...] = (
    CriticalPackage(
        "ultralytics",
        "Ultralytics YOLO",
        "Core trainer for YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOv12, and YOLO26.",
        "critical",
    ),
    CriticalPackage(
        "torch",
        "PyTorch",
        "Deep-learning runtime powering training and inference.",
        "critical",
    ),
    CriticalPackage(
        "torchvision",
        "TorchVision",
        "Image-processing utilities and the transform pipeline.",
        "critical",
    ),
    CriticalPackage(
        "super-gradients",
        "SuperGradients",
        "Required to train YOLO-NAS detection models.",
        "important",
    ),
    CriticalPackage(
        "tensorboard",
        "TensorBoard",
        "Dashboards for inspecting training metrics and artifacts.",
        "important",
    ),
    CriticalPackage(
        "roboflow",
        "Roboflow",
        "Client for dataset downloads and Roboflow model uploads.",
        "important",
    ),
    CriticalPackage(
        "onnx",
        "ONNX",
        "Interchange format for exported inference models.",
        "optional",
    ),
    CriticalPackage(
        "onnxruntime",
        "ONNX Runtime",
        "Accelerated inference engine for exported ONNX models.",
        "optional",
    ),
)


@dataclass
class PackageStatus:
    package: CriticalPackage
    installed: str | None
    latest: str | None
    error: str | None = None

    @property
    def severity(self) -> str:
        """Classify the gap between installed and latest into one of:

        ``up_to_date`` — installed is at or above latest
        ``patch``      — same MAJOR.MINOR, newer PATCH available
        ``minor``      — same MAJOR, newer MINOR available
        ``major``      — newer MAJOR available (breaking-change territory)
        ``missing``    — importance=critical but not installed
        ``uninstalled``— importance!=critical, not installed
        ``unknown``    — couldn't query PyPI or parse the version
        """
        if self.installed is None:
            return "missing" if self.package.importance == "critical" else "uninstalled"
        if self.latest is None:
            return "unknown"
        try:
            current = Version(self.installed)
            target = Version(self.latest)
        except InvalidVersion:
            return "unknown"
        if current >= target:
            return "up_to_date"

        current_release = current.release + (0, 0, 0)
        target_release = target.release + (0, 0, 0)
        if current_release[0] != target_release[0]:
            return "major"
        if current_release[1] != target_release[1]:
            return "minor"
        return "patch"

    @property
    def needs_update(self) -> bool:
        return self.severity in {"patch", "minor", "major"}

    @property
    def is_blocking(self) -> bool:
        """True if a user *must* act on this — missing critical dep or a major bump."""
        return self.severity in {"missing", "major"}


# (color, glyph, label) per severity. Keep glyphs printable + single-width.
SEVERITY_META: dict[str, tuple[str, str, str]] = {
    "up_to_date": ("green", "✓", "Up to date"),
    "patch":      ("yellow", "•", "Patch update available"),
    "minor":      ("bright_yellow", "▲", "Minor update available"),
    "major":      ("red", "⚠", "Major update available"),
    "missing":    ("bold red", "✗", "Not installed"),
    "uninstalled":("dim", "○", "Not installed"),
    "unknown":    ("dim", "?", "PyPI unreachable"),
}

IMPORTANCE_STYLE: dict[str, str] = {
    "critical":  "[bold red]critical[/bold red]",
    "important": "[bold yellow]important[/bold yellow]",
    "optional":  "[dim]optional[/dim]",
}


def _get_installed_version(package_name: str) -> str | None:
    try:
        return installed_version(package_name)
    except PackageNotFoundError:
        return None


def _get_latest_pypi_version(
    package_name: str, timeout: float = PYPI_TIMEOUT_SECONDS
) -> tuple[str | None, str | None]:
    """Return ``(latest_version, error_message)``.

    Either ``latest_version`` is a PEP 440 string, or ``error_message`` is
    populated describing why the lookup failed — never both populated.
    """
    url = PYPI_JSON_URL.format(name=package_name)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "YOLOmatic-DependencyCheck/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        return None, f"HTTP {error.code}"
    except urllib.error.URLError as error:
        return None, f"network ({error.reason})"
    except socket.timeout:
        return None, "network (timeout)"
    except json.JSONDecodeError as error:
        return None, f"parse ({error.msg})"
    except OSError as error:
        return None, f"io ({error})"

    info = payload.get("info") or {}
    version = info.get("version")
    if not version:
        return None, "no version in PyPI response"
    return str(version), None


def check_packages(
    packages: Sequence[CriticalPackage] | None = None,
    timeout: float = PYPI_TIMEOUT_SECONDS,
) -> list[PackageStatus]:
    """Query installed + latest versions for every package in parallel."""
    targets = list(packages) if packages is not None else list(CRITICAL_PACKAGES)
    if not targets:
        return []

    statuses: list[PackageStatus] = [
        PackageStatus(
            package=target,
            installed=_get_installed_version(target.name),
            latest=None,
        )
        for target in targets
    ]

    worker_count = min(PYPI_WORKER_LIMIT, len(statuses))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_idx = {
            executor.submit(_get_latest_pypi_version, status.package.name, timeout): idx
            for idx, status in enumerate(statuses)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                latest, error = future.result()
            except Exception as exc:  # network layer below raised something new
                latest, error = None, f"unexpected ({exc})"
            statuses[idx] = PackageStatus(
                package=statuses[idx].package,
                installed=statuses[idx].installed,
                latest=latest,
                error=error,
            )
    return statuses


def summarize(statuses: Sequence[PackageStatus]) -> dict[str, int]:
    """Bucket counts for a high-level summary banner."""
    buckets = {key: 0 for key in SEVERITY_META}
    for status in statuses:
        buckets[status.severity] = buckets.get(status.severity, 0) + 1
    return buckets
