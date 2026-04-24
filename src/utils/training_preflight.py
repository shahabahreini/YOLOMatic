from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from rich.console import Console

from src.utils.cli import get_user_choice
from src.utils.ml_dependencies import prepare_ml_runtime


console = Console()
DEFAULT_TORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu128"
DEFAULT_NUMPY_CONSTRAINT = "numpy==1.23.0"
# The inspection script runs in a clean subprocess, so it must replicate the
# NVIDIA library-path bootstrap that `ml_dependencies.prepare_ml_runtime` does
# in-process. Without this, torch can't find bundled cuDNN/CUDA runtime wheels
# and `torch.cuda.is_available()` returns False — which previously pushed every
# fresh invocation into the repair flow, even when the prior repair succeeded.
_TORCH_INSPECTION_SCRIPT = """
import json
import os
import platform
import sys
from pathlib import Path


def _candidate_nvidia_lib_dirs():
    relative_dirs = [
        "nvidia/cudnn/lib",
        "nvidia/cublas/lib",
        "nvidia/cuda_runtime/lib",
        "nvidia/cuda_nvrtc/lib",
        "nvidia/cuda_cupti/lib",
        "nvidia/cufft/lib",
        "nvidia/curand/lib",
        "nvidia/cusolver/lib",
        "nvidia/cusparse/lib",
        "nvidia/nvjitlink/lib",
        "nvidia/nvtx/lib",
        "nvidia/cufile/lib",
    ]
    seen = set()
    found = []
    for raw_path in sys.path:
        if not raw_path:
            continue
        site_packages = Path(raw_path)
        if site_packages.name != "site-packages" or not site_packages.exists():
            continue
        for rel in relative_dirs:
            lib_dir = site_packages / rel
            if lib_dir.exists() and lib_dir.is_dir():
                resolved = lib_dir.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    found.append(resolved)
    return found


def _prepare_runtime():
    system = platform.system()
    if system == "Windows":
        env_var = "PATH"
    elif system == "Darwin":
        env_var = "DYLD_LIBRARY_PATH"
    else:
        env_var = "LD_LIBRARY_PATH"
    existing = [p for p in os.environ.get(env_var, "").split(os.pathsep) if p]
    for lib_dir in _candidate_nvidia_lib_dirs():
        lib_dir_str = str(lib_dir)
        if lib_dir_str not in existing:
            existing.append(lib_dir_str)
    if existing:
        os.environ[env_var] = os.pathsep.join(existing)


_prepare_runtime()

payload = {}
try:
    import torch
    try:
        import numpy
        numpy_version = getattr(numpy, "__version__", None)
    except Exception:
        numpy_version = None
    payload = {
        "importable": True,
        "version": getattr(torch, "__version__", "unknown"),
        "cuda_build": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "numpy_version": numpy_version,
        "error": None,
    }
except Exception as error:
    payload = {
        "importable": False,
        "version": None,
        "cuda_build": None,
        "cuda_available": False,
        "device_count": 0,
        "numpy_version": None,
        "error": str(error),
    }
print(json.dumps(payload))
""".strip()


@dataclass(frozen=True)
class TorchInspectionResult:
    importable: bool
    version: str | None
    cuda_build: str | None
    cuda_available: bool
    device_count: int
    numpy_version: str | None
    error: str | None


@dataclass(frozen=True)
class TrainingDeviceResolution:
    cancelled: bool
    device: str | None


def inspect_torch_runtime(
    python_executable: str | None = None,
) -> TorchInspectionResult:
    executable = python_executable or sys.executable
    # Mirror what `import_torch` does before the real import, so the subprocess
    # (which inherits our env) can find bundled cuDNN/CUDA runtime wheels.
    prepare_ml_runtime()
    process = subprocess.run(
        [executable, "-c", _TORCH_INSPECTION_SCRIPT],
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    if process.returncode != 0:
        error = (process.stderr or process.stdout or "Unknown error").strip()
        return TorchInspectionResult(
            importable=False,
            version=None,
            cuda_build=None,
            cuda_available=False,
            device_count=0,
            numpy_version=None,
            error=error,
        )

    try:
        payload = json.loads(process.stdout.strip())
    except json.JSONDecodeError:
        error = (
            process.stdout or process.stderr or "Invalid inspection output"
        ).strip()
        return TorchInspectionResult(
            importable=False,
            version=None,
            cuda_build=None,
            cuda_available=False,
            device_count=0,
            numpy_version=None,
            error=error,
        )

    return TorchInspectionResult(
        importable=bool(payload.get("importable", False)),
        version=payload.get("version"),
        cuda_build=payload.get("cuda_build"),
        cuda_available=bool(payload.get("cuda_available", False)),
        device_count=int(payload.get("device_count", 0)),
        numpy_version=payload.get("numpy_version"),
        error=payload.get("error"),
    )


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


def repair_cuda_enabled_torch(
    python_executable: str | None = None,
    index_url: str = DEFAULT_TORCH_CUDA_INDEX_URL,
) -> tuple[bool, str, TorchInspectionResult]:
    executable = python_executable or sys.executable
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
            "torch",
            "torchvision",
            "torchaudio",
            DEFAULT_NUMPY_CONSTRAINT,
            "--extra-index-url",
            index_url,
        ],
    ]
    descriptions = [
        "Removing existing PyTorch packages from the active environment...",
        "Installing CUDA-enabled PyTorch packages. This can take several minutes while large wheels download...",
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
            inspection = inspect_torch_runtime(executable)
            return False, "\n\n".join(outputs), inspection

    console.print(
        "[bold cyan]Verifying the repaired PyTorch environment...[/bold cyan]"
    )
    inspection = inspect_torch_runtime(executable)
    if not inspection.cuda_available:
        outputs.append(
            f"Post-repair check: torch={inspection.version}, cuda_build={inspection.cuda_build}, cuda_available={inspection.cuda_available}, device_count={inspection.device_count}, numpy={inspection.numpy_version}, error={inspection.error}"
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
