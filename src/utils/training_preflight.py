from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

from src.utils.cli import get_user_choice
from src.utils.ml_dependencies import prepare_ml_runtime

console = Console()
DEFAULT_TORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu128"
DEFAULT_NUMPY_CONSTRAINT = "numpy==1.23.0"
_IS_WINDOWS = platform.system() == "Windows"
# Markers fence the JSON payload so any stray stdout from torch import (warnings
# routed through `print`, deprecation notices, etc.) cannot break json.loads and
# trigger a false "torch not importable" repair prompt.
_INSPECTION_BEGIN = "<<<YOLOMATIC_TORCH_INSPECTION_BEGIN>>>"
_INSPECTION_END = "<<<YOLOMATIC_TORCH_INSPECTION_END>>>"
# The inspection script runs in a fresh subprocess, so it must replicate the
# NVIDIA library-path bootstrap that `ml_dependencies.prepare_ml_runtime` does
# in-process. Without this, torch on Windows can't resolve cuDNN/CUDA DLLs and
# `torch.cuda.is_available()` returns False — which previously pushed every
# fresh invocation into the repair flow, even when the prior repair succeeded.
_TORCH_INSPECTION_SCRIPT_TEMPLATE = """
import json
import os
import platform
import sys
from pathlib import Path


_IS_WINDOWS = platform.system() == "Windows"
_NVIDIA_PACKAGES = (
    "cudnn", "cublas", "cuda_runtime", "cuda_nvrtc", "cuda_cupti",
    "cufft", "curand", "cusolver", "cusparse", "nvjitlink", "nvtx", "cufile",
)


def _site_packages_dirs():
    out = []
    for raw in sys.path:
        if not raw:
            continue
        path = Path(raw)
        if path.name == "site-packages" and path.exists():
            out.append(path)
    return out


def _candidate_lib_dirs():
    subdirs = ("bin", "lib", "lib/x64") if _IS_WINDOWS else ("lib",)
    seen = set()
    found = []
    for site_packages in _site_packages_dirs():
        for pkg in _NVIDIA_PACKAGES:
            for sub in subdirs:
                lib_dir = site_packages / "nvidia" / pkg / sub
                if lib_dir.exists() and lib_dir.is_dir():
                    resolved = lib_dir.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        found.append(resolved)
        if _IS_WINDOWS:
            for rel in ("torch/lib", "torch/bin"):
                t = site_packages / rel
                if t.exists() and t.is_dir():
                    resolved = t.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        found.append(resolved)
    return found


def _prepare_runtime():
    if _IS_WINDOWS:
        env_var = "PATH"
    elif platform.system() == "Darwin":
        env_var = "DYLD_LIBRARY_PATH"
    else:
        env_var = "LD_LIBRARY_PATH"
    existing = [p for p in os.environ.get(env_var, "").split(os.pathsep) if p]
    add_dll = getattr(os, "add_dll_directory", None)
    for lib_dir in _candidate_lib_dirs():
        lib_dir_str = str(lib_dir)
        if lib_dir_str not in existing:
            existing.append(lib_dir_str)
        if _IS_WINDOWS and add_dll is not None:
            try:
                add_dll(lib_dir_str)
            except (OSError, FileNotFoundError):
                pass
    if existing:
        os.environ[env_var] = os.pathsep.join(existing)


_prepare_runtime()

payload = {
    "importable": False,
    "version": None,
    "cuda_build": None,
    "cuda_available": False,
    "device_count": 0,
    "numpy_version": None,
    "error": None,
    "init_error": None,
}
try:
    import torch
    payload["importable"] = True
    payload["version"] = getattr(torch, "__version__", "unknown")
    payload["cuda_build"] = getattr(torch.version, "cuda", None)
    try:
        import numpy
        payload["numpy_version"] = getattr(numpy, "__version__", None)
    except Exception:
        payload["numpy_version"] = None
    try:
        payload["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as init_err:
        payload["cuda_available"] = False
        payload["init_error"] = repr(init_err)
    if payload["cuda_available"]:
        try:
            payload["device_count"] = int(torch.cuda.device_count())
        except Exception as count_err:
            payload["device_count"] = 0
            payload["init_error"] = repr(count_err)
except Exception as error:
    payload["error"] = repr(error)

sys.stdout.write(__YOLOMATIC_BEGIN_LITERAL__ + chr(10))
sys.stdout.write(json.dumps(payload) + chr(10))
sys.stdout.write(__YOLOMATIC_END_LITERAL__ + chr(10))
sys.stdout.flush()
""".strip()


# Substitute marker placeholders without using f-strings or .format(), so the
# script's own braces (dict literals, etc.) don't have to be doubled. This
# avoids a class of formatting bugs that previously broke the inspection
# subprocess.
_TORCH_INSPECTION_SCRIPT = (
    _TORCH_INSPECTION_SCRIPT_TEMPLATE
    .replace("__YOLOMATIC_BEGIN_LITERAL__", repr(_INSPECTION_BEGIN))
    .replace("__YOLOMATIC_END_LITERAL__", repr(_INSPECTION_END))
)


@dataclass(frozen=True)
class TorchInspectionResult:
    importable: bool
    version: str | None
    cuda_build: str | None
    cuda_available: bool
    device_count: int
    numpy_version: str | None
    error: str | None
    init_error: str | None = None

    @property
    def is_cuda_build(self) -> bool:
        return bool(self.cuda_build)


# Caches the most recent inspection result keyed by python executable so
# resolve_training_device doesn't pay the subprocess cost twice in one run.
_INSPECTION_CACHE: dict[str, TorchInspectionResult] = {}


def invalidate_torch_inspection_cache() -> None:
    _INSPECTION_CACHE.clear()


@dataclass(frozen=True)
class TrainingDeviceResolution:
    cancelled: bool
    device: str | None


def _extract_inspection_payload(stdout: str) -> str | None:
    begin = stdout.find(_INSPECTION_BEGIN)
    end = stdout.find(
        _INSPECTION_END, begin + len(_INSPECTION_BEGIN) if begin != -1 else 0
    )
    if begin == -1 or end == -1:
        return None
    return stdout[begin + len(_INSPECTION_BEGIN) : end].strip()


def _failed_inspection(error: str) -> TorchInspectionResult:
    return TorchInspectionResult(
        importable=False,
        version=None,
        cuda_build=None,
        cuda_available=False,
        device_count=0,
        numpy_version=None,
        error=error,
        init_error=None,
    )


def inspect_torch_runtime(
    python_executable: str | None = None,
    use_cache: bool = True,
) -> TorchInspectionResult:
    executable = python_executable or sys.executable
    if use_cache:
        cached = _INSPECTION_CACHE.get(executable)
        if cached is not None:
            return cached
    # Mirror what `import_torch` does before the real import, so the subprocess
    # (which inherits our env) can find bundled cuDNN/CUDA runtime wheels.
    prepare_ml_runtime()
    env = os.environ.copy()
    # Force unbuffered stdout so the marker fence is intact even on abnormal
    # exits, and silence pip/torch ANSI in captured output.
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        process = subprocess.run(
            [executable, "-c", _TORCH_INSPECTION_SCRIPT],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    except OSError as os_error:
        result = _failed_inspection(
            f"Failed to launch inspection subprocess: {os_error}"
        )
        if use_cache:
            _INSPECTION_CACHE[executable] = result
        return result

    payload_text = _extract_inspection_payload(process.stdout or "")
    if payload_text is None:
        # Fall back to legacy behavior (whole stdout is JSON) for robustness if
        # something stripped the markers, but prefer surfacing stderr details.
        payload_text = (process.stdout or "").strip()
    if not payload_text:
        stderr = (process.stderr or "").strip()
        result = _failed_inspection(stderr or "Inspection produced no output")
        if use_cache:
            _INSPECTION_CACHE[executable] = result
        return result

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        stderr = (process.stderr or "").strip()
        snippet = payload_text[-500:] if payload_text else ""
        details = (
            "\n".join(part for part in (snippet, stderr) if part)
            or "Invalid inspection output"
        )
        result = _failed_inspection(details)
        if use_cache:
            _INSPECTION_CACHE[executable] = result
        return result

    result = TorchInspectionResult(
        importable=bool(payload.get("importable", False)),
        version=payload.get("version"),
        cuda_build=payload.get("cuda_build"),
        cuda_available=bool(payload.get("cuda_available", False)),
        device_count=int(payload.get("device_count", 0)),
        numpy_version=payload.get("numpy_version"),
        error=payload.get("error"),
        init_error=payload.get("init_error"),
    )
    if use_cache:
        _INSPECTION_CACHE[executable] = result
    return result


def _normalize_device(requested_device: Any) -> str | None:
    if requested_device is None:
        return None
    normalized = str(requested_device).strip()
    return normalized or None


def _device_requests_cuda(normalized_device: str | None) -> bool:
    if normalized_device is None:
        return False
    lowered = normalized_device.lower()
    return lowered == "cuda" or lowered.replace(",", "").isdigit()


def _detect_nvidia_gpu() -> bool:
    try:
        process = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False

    return process.returncode == 0 and bool(process.stdout.strip())


def _tail_output(text: str, line_count: int = 20) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-line_count:])


def _run_repair_command(
    command: list[str], description: str
) -> subprocess.CompletedProcess[str]:
    console.print(f"[bold cyan]{description}[/bold cyan]")
    with console.status(f"[bold]{description}[/bold]", spinner="dots"):
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )


def _find_uv_project_root(start: Path | None = None) -> Path | None:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "uv.lock").is_file():
            return candidate
    return None


def _resolve_uv_executable() -> str | None:
    return shutil.which("uv")


def _format_inspection_failure(inspection: TorchInspectionResult) -> str:
    return (
        f"Post-repair check: torch={inspection.version}, "
        f"cuda_build={inspection.cuda_build}, "
        f"cuda_available={inspection.cuda_available}, "
        f"device_count={inspection.device_count}, "
        f"numpy={inspection.numpy_version}, "
        f"error={inspection.error}, "
        f"init_error={inspection.init_error}"
    )


def _repair_via_uv_sync(
    executable: str,
    uv_executable: str,
    project_root: Path,
) -> tuple[bool, str, TorchInspectionResult]:
    # `uv sync --reinstall-package` re-fetches the package while keeping the
    # venv tracked by uv.lock. This avoids the loop where `pip install
    # --force-reinstall` leaves a missing-RECORD dist-info that the next
    # `uv run`'s implicit sync tries (and on Windows often fails) to clean up.
    command = [
        uv_executable,
        "sync",
        "--reinstall-package",
        "torch",
        "--reinstall-package",
        "torchvision",
    ]
    description = (
        f"Reinstalling torch / torchvision via `uv sync` in {project_root.name} "
        "(keeps the venv consistent with uv.lock so the next `uv run` won't re-trigger repair)..."
    )
    console.print(f"[bold cyan]{description}[/bold cyan]")
    with console.status(f"[bold]{description}[/bold]", spinner="dots"):
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            cwd=str(project_root),
        )
    command_text = " ".join(command)
    combined_output = "\n".join(
        part.strip()
        for part in (process.stdout, process.stderr)
        if part and part.strip()
    )
    output = f"$ {command_text}\n{combined_output}".strip()

    console.print(
        "[bold cyan]Verifying the repaired PyTorch environment...[/bold cyan]"
    )
    invalidate_torch_inspection_cache()
    inspection = inspect_torch_runtime(executable, use_cache=False)

    if process.returncode != 0 or not inspection.cuda_available:
        output = "\n\n".join([output, _format_inspection_failure(inspection)])
        return False, output, inspection

    return True, output, inspection


def repair_cuda_enabled_torch(
    python_executable: str | None = None,
    index_url: str = DEFAULT_TORCH_CUDA_INDEX_URL,
) -> tuple[bool, str, TorchInspectionResult]:
    executable = python_executable or sys.executable
    uv_project_root = _find_uv_project_root()
    uv_executable = _resolve_uv_executable() if uv_project_root else None
    if uv_project_root and uv_executable:
        return _repair_via_uv_sync(
            executable=executable,
            uv_executable=uv_executable,
            project_root=uv_project_root,
        )
    # Three-phase install: uninstall, install torch from the CUDA index ONLY
    # (using --index-url so pip can't fall back to PyPI's CPU-only wheel — a
    # known footgun on Windows that silently reinstalls CPU torch even after a
    # "successful" repair), then re-pin numpy from PyPI separately.
    commands = [
        [
            executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "torch",
            "torchvision",
            "torchaudio",
        ],
        [
            executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--force-reinstall",
            "--index-url",
            index_url,
            "torch",
            "torchvision",
            "torchaudio",
        ],
        [
            executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--force-reinstall",
            DEFAULT_NUMPY_CONSTRAINT,
        ],
    ]
    descriptions = [
        "Removing existing PyTorch packages from the active environment...",
        f"Installing CUDA-enabled PyTorch packages from {index_url} (this can take several minutes while large wheels download)...",
        f"Re-pinning {DEFAULT_NUMPY_CONSTRAINT} from PyPI for super-gradients compatibility...",
    ]

    outputs: list[str] = []
    for description, command in zip(descriptions, commands):
        process = _run_repair_command(
            command,
            description,
        )
        command_text = " ".join(command)
        combined_output = "\n".join(
            part.strip()
            for part in (process.stdout, process.stderr)
            if part and part.strip()
        )
        outputs.append(f"$ {command_text}\n{combined_output}".strip())
        if process.returncode != 0:
            invalidate_torch_inspection_cache()
            inspection = inspect_torch_runtime(executable, use_cache=False)
            return False, "\n\n".join(outputs), inspection

    console.print(
        "[bold cyan]Verifying the repaired PyTorch environment...[/bold cyan]"
    )
    invalidate_torch_inspection_cache()
    inspection = inspect_torch_runtime(executable, use_cache=False)
    if not inspection.cuda_available:
        outputs.append(
            f"Post-repair check: torch={inspection.version}, cuda_build={inspection.cuda_build}, cuda_available={inspection.cuda_available}, device_count={inspection.device_count}, numpy={inspection.numpy_version}, error={inspection.error}, init_error={inspection.init_error}"
        )
        return False, "\n\n".join(outputs), inspection

    return True, "\n\n".join(outputs), inspection


def resolve_training_device(
    requested_device: Any,
    prefer_gpu: bool = False,
    index_url: str = DEFAULT_TORCH_CUDA_INDEX_URL,
) -> TrainingDeviceResolution:
    normalized_device = _normalize_device(requested_device)
    has_nvidia_gpu = _detect_nvidia_gpu()
    wants_cuda = _device_requests_cuda(normalized_device)
    if normalized_device is None and prefer_gpu and has_nvidia_gpu:
        wants_cuda = True

    if not wants_cuda:
        return TrainingDeviceResolution(cancelled=False, device=normalized_device)

    inspection = inspect_torch_runtime()
    if inspection.cuda_available:
        return TrainingDeviceResolution(cancelled=False, device=normalized_device)

    status_fields: dict[str, str] = {
        "torch": (
            f"{inspection.version} (CUDA: {inspection.cuda_build or 'none'})"
            if inspection.version
            else "not importable"
        ),
        "nvidia-smi": "NVIDIA GPU detected" if has_nvidia_gpu else "no GPU detected",
    }
    if inspection.numpy_version:
        status_fields["numpy"] = inspection.numpy_version
    if inspection.error:
        status_fields["import error"] = inspection.error
    if inspection.init_error:
        status_fields["cuda init error"] = inspection.init_error

    # If torch is already a CUDA build but the runtime can't initialize CUDA,
    # reinstalling torch will not help — the problem is the NVIDIA driver, the
    # OS-level CUDA toolkit, or a missing MSVC redistributable on Windows.
    # Suppress the repair option so the user isn't trapped in a reinstall loop.
    cuda_build_but_runtime_broken = (
        has_nvidia_gpu and inspection.importable and inspection.is_cuda_build
    )

    if cuda_build_but_runtime_broken:
        driver_summary = (
            "[yellow]PyTorch is built with CUDA support but cannot initialize the GPU runtime.[/yellow] "
            f"This installed torch already targets CUDA [cyan]{inspection.cuda_build}[/cyan], "
            "so reinstalling won't change the outcome. The most common causes are an "
            "NVIDIA driver that is too old for this CUDA version, a missing Visual C++ "
            "Redistributable (Windows), or a CUDA / driver version mismatch."
        )
        if _IS_WINDOWS:
            driver_summary += (
                "\n\n[dim]Windows checklist: update the NVIDIA driver to one that supports "
                f"CUDA {inspection.cuda_build}, install the latest VC++ Redistributable, "
                "and confirm `nvidia-smi` reports a CUDA Version >= the torch CUDA build.[/dim]"
            )
        else:
            driver_summary += (
                "\n\n[dim]Linux checklist: update the NVIDIA driver, confirm "
                "`nvidia-smi` reports a CUDA Version >= the torch CUDA build, and "
                "verify cuDNN libraries are reachable on LD_LIBRARY_PATH.[/dim]"
            )
        driver_descriptions = {
            "Continue on CPU": (
                "[bold yellow]Train on CPU for this run.[/bold yellow]\n\n"
                "• Lets you proceed without touching the driver/toolchain.\n"
                "• Dramatically slower than GPU; fine for smoke tests."
            ),
            "Cancel Training": (
                "[bold red]Abort so you can update the driver / toolchain.[/bold red]\n\n"
                "Recommended — reinstalling torch will not fix a driver problem."
            ),
        }
        driver_selection = get_user_choice(
            ["Continue on CPU", "Cancel Training"],
            title="CUDA Runtime Unavailable (driver / toolchain issue)",
            text=driver_summary,
            descriptions=driver_descriptions,
            breadcrumbs=["YOLOmatic", "Training", "GPU Check"],
            status_fields=status_fields,
            tip="`nvidia-smi` shows the maximum CUDA version the driver supports. If that's lower than the torch CUDA build, update the driver.",
        )
        if driver_selection == "Cancel Training":
            return TrainingDeviceResolution(cancelled=True, device=None)
        console.print("[bold yellow]Continuing training on CPU.[/bold yellow]")
        return TrainingDeviceResolution(cancelled=False, device="cpu")

    if has_nvidia_gpu:
        summary = (
            "[yellow]CUDA was requested but PyTorch cannot use it in this environment.[/yellow] "
            "An NVIDIA GPU is present, so this is almost certainly a CUDA-enabled "
            "PyTorch installation issue — a CPU-only torch wheel is installed, or "
            "cuDNN / CUDA runtime libraries are missing. "
            "[dim]`nvidia-smi` working does not guarantee PyTorch can use the GPU.[/dim]"
        )
        descriptions = {
            "Repair CUDA PyTorch": (
                "[bold green]Reinstall torch / torchvision / torchaudio with CUDA 12.8 wheels.[/bold green]\n\n"
                f"• Index: [cyan]{index_url}[/cyan]\n"
                f"• Keeps [cyan]{DEFAULT_NUMPY_CONSTRAINT}[/cyan] for super-gradients compatibility.\n"
                "• Takes a few minutes (large wheels download).\n"
                "• Needs network access and write access to the current Python env."
            ),
            "Continue on CPU": (
                "[bold yellow]Train on CPU instead.[/bold yellow]\n\n"
                "• Works without any reinstall, but training is dramatically slower.\n"
                "• Fine for smoke tests and tiny datasets; impractical for real runs."
            ),
            "Cancel Training": (
                "[bold red]Abort and return to the main menu.[/bold red]\n\n"
                "Pick this if you'd rather fix the environment manually "
                "(e.g. `uv sync`, reinstall CUDA drivers, or swap Python envs)."
            ),
        }
        tip = (
            "If repair has failed before on this machine, skip it and fix the env by hand — "
            "rerunning the same repair rarely succeeds on the second attempt."
        )
        selection = get_user_choice(
            ["Repair CUDA PyTorch", "Continue on CPU", "Cancel Training"],
            title="CUDA Device Unavailable",
            text=summary,
            descriptions=descriptions,
            breadcrumbs=["YOLOmatic", "Training", "GPU Check"],
            status_fields=status_fields,
            tip=tip,
        )
    else:
        summary = (
            "[yellow]CUDA was requested but PyTorch cannot use it in this environment.[/yellow] "
            "No NVIDIA GPU was found via `nvidia-smi`, so automatic CUDA repair "
            "would not help. Either this host has no NVIDIA GPU, or the driver "
            "is not installed / reachable from this shell."
        )
        descriptions = {
            "Continue on CPU": (
                "[bold yellow]Train on CPU.[/bold yellow]\n\n"
                "• Works everywhere, but dramatically slower than GPU.\n"
                "• Fine for a smoke test or tiny dataset."
            ),
            "Cancel Training": (
                "[bold red]Abort and return to the main menu.[/bold red]\n\n"
                "Pick this if you need to install NVIDIA drivers, move to a "
                "GPU host, or switch to an MPS-enabled env first."
            ),
        }
        tip = (
            "On Apple Silicon, set [bold]device: mps[/bold] in your config instead of "
            "[bold]cuda[/bold] — that path is faster than CPU and supported end-to-end."
        )
        selection = get_user_choice(
            ["Continue on CPU", "Cancel Training"],
            title="CUDA Device Unavailable",
            text=summary,
            descriptions=descriptions,
            breadcrumbs=["YOLOmatic", "Training", "GPU Check"],
            status_fields=status_fields,
            tip=tip,
        )

    if selection == "Cancel Training":
        return TrainingDeviceResolution(cancelled=True, device=None)
    if selection == "Continue on CPU":
        console.print("[bold yellow]Continuing training on CPU.[/bold yellow]")
        return TrainingDeviceResolution(cancelled=False, device="cpu")

    success, output, post_repair_inspection = repair_cuda_enabled_torch(
        index_url=index_url
    )
    if success:
        console.print(
            "[bold green]CUDA-enabled PyTorch repair completed successfully.[/bold green]"
        )
        console.print(
            f"[bold green]Updated torch build: {post_repair_inspection.version} (CUDA build: {post_repair_inspection.cuda_build})[/bold green]"
        )
        console.print(
            f"[bold green]Preserved dependency compatibility with {DEFAULT_NUMPY_CONSTRAINT}.[/bold green]"
        )
        return TrainingDeviceResolution(cancelled=False, device=normalized_device)

    tail_output = _tail_output(output) or "(no output captured)"
    failure_summary = (
        "[red]Automatic CUDA-enabled PyTorch repair did not succeed.[/red] "
        "Common causes: no network access, wrong Python env selected, or the "
        "host genuinely lacks a working NVIDIA driver / CUDA toolkit. The last "
        "lines of the pip output are shown in the Context box above."
    )
    failure_status = {
        "post-repair torch": post_repair_inspection.version or "unknown",
        "cuda_available": str(post_repair_inspection.cuda_available),
        "last pip output": tail_output.splitlines()[-1] if tail_output else "—",
    }
    fallback_descriptions = {
        "Continue on CPU": (
            "[bold yellow]Train on CPU for now.[/bold yellow]\n\n"
            "• Lets you move forward with this run without fixing the env.\n"
            "• Expect dramatically slower training than on a working GPU."
        ),
        "Cancel Training": (
            "[bold red]Abort so you can fix the environment manually.[/bold red]\n\n"
            "Recommended if the pip output shows a concrete error "
            "(network failure, permission denied, incompatible CUDA version, …)."
        ),
    }
    fallback_selection = get_user_choice(
        ["Continue on CPU", "Cancel Training"],
        title="CUDA Repair Failed",
        text=failure_summary,
        descriptions=fallback_descriptions,
        breadcrumbs=["YOLOmatic", "Training", "GPU Check", "Repair Failed"],
        status_fields=failure_status,
        tip=(
            "Full pip output was just printed to the scrollback — scroll up after this "
            "menu to see it."
        ),
    )
    if fallback_selection == "Cancel Training":
        return TrainingDeviceResolution(cancelled=True, device=None)

    console.print("[bold yellow]Continuing training on CPU.[/bold yellow]")
    return TrainingDeviceResolution(cancelled=False, device="cpu")


def validate_export_config(
    export_config: dict, console: Console | None = None
) -> tuple[bool, list[str]]:
    """Validate export configuration early to catch issues before training.

    Returns:
        (is_valid, list of error messages)
    """
    if console is None:
        console = Console()

    errors: list[str] = []
    warnings: list[str] = []

    format_type = export_config.get("format", "onnx")
    int8_enabled = export_config.get("int8", False)
    has_data = export_config.get("data") is not None

    # Check INT8 compatibility
    if int8_enabled:
        if format_type == "onnx" and not has_data:
            errors.append(
                "INT8 quantization for ONNX requires calibration data. "
                "Either set 'data' in export config or disable 'int8'. "
                "Recommended: Set int8=False for ONNX, or use TensorRT format for INT8."
            )
        elif format_type in ("torchscript", "onnx"):
            warnings.append(
                f"INT8 quantization for {format_type} may require calibration data. "
                "If export fails, disable int8 or provide data parameter."
            )
        elif format_type == "engine":
            # TensorRT supports INT8 but needs calibration
            if not has_data:
                warnings.append(
                    "INT8 TensorRT export will use default calibration. "
                    "For better accuracy, provide calibration data via 'data' parameter."
                )

    # Check other common issues
    if export_config.get("half", False) and export_config.get("int8", False):
        errors.append(
            "Cannot use both 'half' (FP16) and 'int8' quantization. Choose one."
        )

    if export_config.get("dynamic", False) and format_type not in ("onnx", "engine"):
        warnings.append(
            f"'dynamic' input shapes may not be fully supported for {format_type} format."
        )

    # Print warnings
    for warning in warnings:
        console.print(f"[bold yellow]⚠️ Export Config Warning: {warning}[/bold yellow]")

    # Print errors
    for error in errors:
        console.print(f"[bold red]❌ Export Config Error: {error}[/bold red]")

    return len(errors) == 0, errors
