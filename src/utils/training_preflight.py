from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from rich.console import Console

try:
    from src.cli.run import get_user_choice
except ImportError:
    try:
        from cli.run import get_user_choice
    except ImportError:

        def get_user_choice(
            options,
            allow_back=False,
            title="Select an Option",
            text="Use ↑↓ keys to navigate, Enter to select:",
            model_data=None,
        ):
            return options[0]


console = Console()
DEFAULT_TORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu128"
DEFAULT_NUMPY_CONSTRAINT = "numpy==1.23.0"
_TORCH_INSPECTION_SCRIPT = """
import json
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
    process = subprocess.run(
        [executable, "-c", _TORCH_INSPECTION_SCRIPT],
        capture_output=True,
        text=True,
        check=False,
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
            "--index-url",
            index_url,
        ],
    ]

    outputs: list[str] = []
    for command in commands:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
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

    console.print(
        "[bold yellow]CUDA was requested for training, but PyTorch cannot use it in the current environment.[/bold yellow]"
    )
    if inspection.version:
        console.print(
            f"[bold yellow]Detected torch build: {inspection.version} (CUDA build: {inspection.cuda_build})[/bold yellow]"
        )
    if inspection.numpy_version:
        console.print(
            f"[bold yellow]Detected NumPy version: {inspection.numpy_version}[/bold yellow]"
        )
    if inspection.error:
        console.print(
            f"[bold yellow]Torch import error: {inspection.error}[/bold yellow]"
        )
    if has_nvidia_gpu:
        console.print(
            "[bold yellow]An NVIDIA GPU was detected via `nvidia-smi`, so this is likely a CUDA-enabled PyTorch installation issue.[/bold yellow]"
        )
    else:
        console.print(
            "[bold yellow]No NVIDIA GPU was detected via `nvidia-smi`, so automatic CUDA repair is unlikely to help.[/bold yellow]"
        )
    console.print(
        "[bold yellow]`nvidia-smi` working does not guarantee that PyTorch can use CUDA.[/bold yellow]"
    )

    if has_nvidia_gpu:
        selection = get_user_choice(
            ["Fix CUDA-enabled PyTorch now", "Continue on CPU", "Cancel Training"],
            title="CUDA Device Unavailable",
            text="Use ↑↓ keys to repair PyTorch, continue on CPU, or cancel training:",
        )
    else:
        selection = get_user_choice(
            ["Continue on CPU", "Cancel Training"],
            title="CUDA Device Unavailable",
            text="Use ↑↓ keys to continue on CPU or cancel training:",
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

    console.print("[bold red]Automatic CUDA-enabled PyTorch repair failed.[/bold red]")
    tail_output = _tail_output(output)
    if tail_output:
        console.print(tail_output)
    fallback_selection = get_user_choice(
        ["Continue on CPU", "Cancel Training"],
        title="CUDA Repair Failed",
        text="Use ↑↓ keys to continue on CPU or cancel training:",
    )
    if fallback_selection == "Cancel Training":
        return TrainingDeviceResolution(cancelled=True, device=None)

    console.print("[bold yellow]Continuing training on CPU.[/bold yellow]")
    return TrainingDeviceResolution(cancelled=False, device="cpu")
