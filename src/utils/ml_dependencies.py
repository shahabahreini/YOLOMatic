from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path
from typing import TypeVar

_T = TypeVar("_T")

_IS_WINDOWS = platform.system() == "Windows"
_NVIDIA_PACKAGE_NAMES = (
    "cudnn",
    "cublas",
    "cuda_runtime",
    "cuda_nvrtc",
    "cuda_cupti",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "nvjitlink",
    "nvtx",
    "cufile",
)
# Tracks DLL search paths already registered via os.add_dll_directory on Windows
# so we don't leak handles or re-register on every call.
_REGISTERED_DLL_DIRS: set[str] = set()


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
    if _IS_WINDOWS:
        return "PATH"
    if platform.system() == "Darwin":
        return "DYLD_LIBRARY_PATH"
    return "LD_LIBRARY_PATH"


def _candidate_nvidia_lib_dirs() -> list[Path]:
    # Windows wheels put DLLs under `nvidia/<pkg>/bin`; Linux/macOS wheels under
    # `nvidia/<pkg>/lib`. Also include `lib/x64` because some older wheels and
    # tooling stage Windows DLLs there. Scanning both keeps the helper portable.
    subdirs = ("bin", "lib", "lib/x64") if _IS_WINDOWS else ("lib",)
    discovered: list[Path] = []
    for site_packages in _candidate_site_packages():
        for package_name in _NVIDIA_PACKAGE_NAMES:
            for subdir in subdirs:
                lib_dir = site_packages / "nvidia" / package_name / subdir
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


def _torch_bundled_lib_dirs() -> list[Path]:
    # Windows torch wheels bundle their CUDA DLLs in `torch/lib/`. Adding this
    # directory as a DLL search path mirrors what `import torch` already does
    # internally and helps subprocess inspectors that haven't imported torch yet.
    discovered: list[Path] = []
    for site_packages in _candidate_site_packages():
        for relative in (Path("torch/lib"), Path("torch/bin")):
            candidate = site_packages / relative
            if candidate.exists() and candidate.is_dir():
                discovered.append(candidate.resolve())
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in discovered:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _register_windows_dll_dir(directory: Path) -> None:
    # Python >= 3.8 on Windows ignores PATH when resolving DLLs for imported
    # extension modules. `os.add_dll_directory` is the only mechanism that
    # actually works for torch's bundled CUDA libraries.
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:
        return
    key = str(directory)
    if key in _REGISTERED_DLL_DIRS:
        return
    if not directory.exists() or not directory.is_dir():
        return
    try:
        add_dll_directory(key)
    except (OSError, FileNotFoundError):
        return
    _REGISTERED_DLL_DIRS.add(key)


def prepare_ml_runtime() -> list[Path]:
    env_var_name = _runtime_library_environment_name()
    path_separator = os.pathsep
    existing = [
        item for item in os.environ.get(env_var_name, "").split(path_separator) if item
    ]
    updated = existing[:]
    added_paths: list[Path] = []

    candidate_dirs: list[Path] = []
    candidate_dirs.extend(_candidate_nvidia_lib_dirs())
    if _IS_WINDOWS:
        candidate_dirs.extend(_torch_bundled_lib_dirs())

    for lib_dir in candidate_dirs:
        lib_dir_str = str(lib_dir)
        if lib_dir_str not in updated:
            updated.append(lib_dir_str)
            added_paths.append(lib_dir)
        if _IS_WINDOWS:
            _register_windows_dll_dir(lib_dir)

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
