from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path
from typing import TypeVar

_T = TypeVar("_T")


class MLDependencyError(RuntimeError):
    pass


def _candidate_site_packages() -> list[Path]:
    paths: list[Path] = []
    for raw_path in sys.path:
        if not raw_path:
            continue
        path = Path(raw_path)
        if path.name == "site-packages" and path.exists():
            paths.append(path)
    return paths


def _runtime_library_environment_name() -> str:
    system = platform.system()
    if system == "Windows":
        return "PATH"
    if system == "Darwin":
        return "DYLD_LIBRARY_PATH"
    return "LD_LIBRARY_PATH"


def _candidate_nvidia_lib_dirs() -> list[Path]:
    relative_dirs = [
        Path("nvidia/cudnn/lib"),
        Path("nvidia/cublas/lib"),
        Path("nvidia/cuda_runtime/lib"),
        Path("nvidia/cuda_nvrtc/lib"),
        Path("nvidia/cuda_cupti/lib"),
        Path("nvidia/cufft/lib"),
        Path("nvidia/curand/lib"),
        Path("nvidia/cusolver/lib"),
        Path("nvidia/cusparse/lib"),
        Path("nvidia/nvjitlink/lib"),
        Path("nvidia/nvtx/lib"),
        Path("nvidia/cufile/lib"),
    ]
    discovered: list[Path] = []
    for site_packages in _candidate_site_packages():
        for relative_dir in relative_dirs:
            lib_dir = site_packages / relative_dir
            if lib_dir.exists() and lib_dir.is_dir():
                discovered.append(lib_dir)
    unique_dirs: list[Path] = []
    seen: set[Path] = set()
    for path in discovered:
        resolved_path = path.resolve()
        if resolved_path not in seen:
            seen.add(resolved_path)
            unique_dirs.append(resolved_path)
    return unique_dirs


def prepare_ml_runtime() -> list[Path]:
    env_var_name = _runtime_library_environment_name()
    path_separator = os.pathsep
    existing = [
        item for item in os.environ.get(env_var_name, "").split(path_separator) if item
    ]
    updated = existing[:]
    added_paths: list[Path] = []
    for lib_dir in _candidate_nvidia_lib_dirs():
        lib_dir_str = str(lib_dir)
        if lib_dir_str not in updated:
            updated.append(lib_dir_str)
            added_paths.append(lib_dir)
    if updated != existing:
        os.environ[env_var_name] = path_separator.join(updated)
    return added_paths


def _raise_dependency_error(module_name: str, error: Exception) -> None:
    prepare_ml_runtime()
    message = str(error)
    if "libcudnn.so.9" in message:
        raise MLDependencyError(
            "PyTorch could not load cuDNN (missing libcudnn.so.9). "
            "Your current .venv appears to have an incomplete NVIDIA runtime installation. "
            "Re-sync or reinstall the CUDA/cuDNN runtime packages in this .venv, then retry."
        ) from error
    raise MLDependencyError(
        f"Required ML dependency '{module_name}' could not be imported: {message}"
    ) from error


def import_module_or_raise(module_name: str) -> _T:
    prepare_ml_runtime()
    try:
        return importlib.import_module(module_name)
    except Exception as error:
        _raise_dependency_error(module_name, error)


def import_torch() -> _T:
    return import_module_or_raise("torch")


def import_ultralytics_yolo() -> object:
    ultralytics_module = import_module_or_raise("ultralytics")
    try:
        return getattr(ultralytics_module, "YOLO")
    except AttributeError as error:
        raise MLDependencyError(
            "ultralytics.YOLO is not available in the installed package."
        ) from error


def import_ultralytics_settings() -> object:
    ultralytics_module = import_module_or_raise("ultralytics")
    try:
        return getattr(ultralytics_module, "settings")
    except AttributeError as error:
        raise MLDependencyError(
            "ultralytics.settings is not available in the installed package."
        ) from error
