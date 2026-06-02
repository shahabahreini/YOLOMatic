from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Any

from rich import box
from rich.panel import Panel

from src.utils.cli import (
    console,
    get_user_choice,
    get_user_multi_select,
    ParameterDefinition,
    expected_error_panel,
    NAV_BACK,
)
from src.utils.project import (
    find_available_weights,
    format_weight_label,
    project_root,
)
from src.utils.ml_dependencies import import_ultralytics_yolo
from src.utils.export_config import (
    ExportModelDetails,
    build_export_kwargs,
    extract_model_details,
    filter_export_defaults,
    filter_export_definitions,
    supported_formats_for_model,
)


SUPPORTED_FORMATS = {
    "TensorRT (engine)": "engine",
    "ONNX": "onnx",
    "OpenVINO": "openvino",
    "CoreML": "coreml",
    "NCNN": "ncnn",
    "TorchScript": "torchscript",
    "TF SavedModel": "saved_model",
    "TF Lite": "tflite",
    "TF.js": "tfjs",
    "Paddle": "paddle",
    "Edge TPU": "edgetpu",
    "MNN": "mnn",
    "RKNN": "rknn",
}


def _export_definitions() -> list[ParameterDefinition]:
    return [
        ParameterDefinition(
            name="half",
            category="Optimization",
            default=False,
            value_type="bool",
            description="FP16 Half Precision",
            help_text="Converts model weights from FP32 (32-bit floating point) to FP16 (16-bit floating point). This halves the model file size and speeds up inference on modern GPUs. Not recommended for CPU-only deployments as CPU emulation of FP16 can be slower.",
            affects="Reduces model file size by 50% and dramatically improves inference throughput on GPU-capable hardware.",
            option_descriptions={
                "True": "Export with FP16 weights. High speedup on GPUs, half the storage size.",
                "False": "Export with standard FP32 weights. Safest general compatibility, larger file size.",
            },
            config_section="export",
        ),
        ParameterDefinition(
            name="int8",
            category="Optimization",
            default=False,
            value_type="bool",
            description="INT8 Quantization",
            help_text="Quantizes weights and activations to 8-bit integers for extreme acceleration. Best for edge devices like Raspberry Pi, Google Coral Edge TPU, mobile CPUs, or NPUs. It reduces model size by ~75% and maximizes speed, but can cause minor accuracy drops. Note: Requires calibration data (provided via the 'data' parameter) to preserve accuracy.",
            affects="Reduces file size by 75% and maximizes CPU/NPU inference speed, but requires calibration dataset.",
            option_descriptions={
                "True": "Quantize to 8-bit integer. Maximum speed on edge/CPU hardware, requires calibration dataset.",
                "False": "Disable INT8 quantization. Keeps floating-point weights (FP16/FP32).",
            },
            config_section="export",
        ),
        ParameterDefinition(
            name="dynamic",
            category="Configuration",
            default=False,
            value_type="bool",
            description="Dynamic Batch & Shapes",
            help_text="Enables dynamic input axes for variable batch sizes and image dimensions. Crucial for production deployments (ONNX/TensorRT) where the input batch size or image resolution varies at runtime. If disabled, the exported model will be strictly locked to a fixed batch size and image size (e.g., 1x3x640x640) and will reject other inputs.",
            affects="Allows the exported model to process variable batch sizes and image shapes during inference.",
            option_descriptions={
                "True": "Enable dynamic axes. Flexible batch and image sizes during runtime prediction.",
                "False": "Lock to fixed batch and image size. Slightly faster compilation and minor optimizations.",
            },
            config_section="export",
        ),
        ParameterDefinition(
            name="simplify",
            category="Configuration",
            default=False,
            value_type="bool",
            description="Simplify ONNX Graph",
            help_text="Optimizes the ONNX graph structure by folding constants, removing redundant operations, and fusing constant nodes using 'onnx-simplifier'. Highly recommended for ONNX exports to ensure optimal speed, smaller files, and better compatibility with inference engines.",
            affects="Fuses redundant graph operations, improving ONNX/TensorRT deployment stability and speed.",
            option_descriptions={
                "True": "Simplify the model graph. Recommended for clean and fast ONNX models.",
                "False": "Export the raw, un-optimized computational graph.",
            },
            config_section="export",
        ),
        ParameterDefinition(
            name="workspace",
            category="Configuration",
            default=4.0,
            value_type="float",
            description="TensorRT Workspace Size (GB)",
            help_text="Sets the maximum temporary GPU memory (in GB) allocated for TensorRT engine building. A larger workspace allows the TensorRT compiler to explore more high-performance kernels (tactics), leading to a faster final engine. This is compile-time only and does not increase inference memory usage.",
            min_value=0.5,
            affects="Gives the TensorRT compiler more memory to search for optimal hardware-level kernels.",
            config_section="export",
        ),
        ParameterDefinition(
            name="nms",
            category="Configuration",
            default=False,
            value_type="bool",
            description="Include NMS in Graph",
            help_text="Embeds Non-Maximum Suppression (NMS) directly into the exported model graph (supported by CoreML, TensorRT, etc.). When enabled, the model returns filtered bounding box detections directly. When disabled, raw network outputs are returned, requiring client-side NMS.",
            affects="Moves bounding box filtering from client application code into the model itself.",
            option_descriptions={
                "True": "Embed NMS. Model outputs final, filtered detections directly.",
                "False": "Exclude NMS. Model outputs raw tensors; post-processing must be done in client code.",
            },
            config_section="export",
        ),
        ParameterDefinition(
            name="imgsz",
            category="Configuration",
            default=640,
            value_type="int",
            description="Export Image Resolution",
            help_text="Sets the default input height and width dimensions (e.g., 640 for 640x640) of the model's input layer. Even if dynamic axes/shapes are enabled, this resolution serves as the baseline that the compiler targets for optimization.",
            min_value=32,
            affects="Determines the primary input image size for compilation and static profiling.",
            config_section="export",
        ),
        ParameterDefinition(
            name="batch",
            category="Configuration",
            default=1,
            value_type="int",
            description="Static Batch Size",
            help_text="Sets the fixed batch size (number of images processed concurrently) for the model. For non-dynamic exports, the model is locked to this exact number. If 'Dynamic Batch & Shapes' is enabled, this setting is ignored as the batch size becomes variable.",
            min_value=1,
            affects="Sets the model input tensor batch size when dynamic shapes are disabled.",
            config_section="export",
        ),
        ParameterDefinition(
            name="opset",
            category="Configuration",
            default=17,
            value_type="int",
            description="ONNX Opset Version",
            help_text="Specifies the ONNX operator set version to use. Higher opset versions support more advanced layer translations and optimizations, but might not be compatible with older inference engines. Opset 17 is standard; reduce to 11/12 only for legacy backends.",
            min_value=11,
            affects="Controls compatibility with older/newer inference engine versions (e.g., ONNX Runtime, TensorRT).",
            config_section="export",
        ),
        ParameterDefinition(
            name="device",
            category="Configuration",
            default="0",
            value_type="str",
            description="Export Target Device",
            help_text="Specifies the hardware accelerator (e.g. '0' for CUDA GPU, 'cpu' for CPU) to compile the model on. For TensorRT, the engine MUST be compiled on the exact GPU model (e.g., Jetson vs desktop RTX) intended for deployment, as compilation is device-specific.",
            affects="Determines compilation hardware. Highly critical for hardware-locked formats like TensorRT.",
            config_section="export",
        ),
        ParameterDefinition(
            name="keras",
            category="Configuration",
            default=False,
            value_type="bool",
            description="Use Keras Layers",
            help_text="Forces TensorFlow SavedModel exports to use Keras-compatible layers. Enable this if you plan to import the model back into a Keras training/inference pipeline. Leave disabled for standard deployment in TensorFlow Serving.",
            affects="Changes internal TF SavedModel node structures to match the Keras framework format.",
            option_descriptions={
                "True": "Use Keras wrappers. Best for integration into Keras-based Python pipelines.",
                "False": "Use standard raw TensorFlow graph structures.",
            },
            config_section="export",
        ),
        ParameterDefinition(
            name="optimize",
            category="Optimization",
            default=False,
            value_type="bool",
            description="Runtime Optimizations",
            help_text="Enables target-specific compiler optimization passes. For TorchScript, this applies mobile JIT optimizations for mobile deployment (Lite Interpreter). For other backends, it can enable higher level compiler optimizations.",
            affects="Invokes additional optimization passes depending on the target export format.",
            option_descriptions={
                "True": "Enable backend-specific compilation/mobile optimizations.",
                "False": "Disable extra optimizations, exporting the graph in its standard form.",
            },
            config_section="export",
        ),
        ParameterDefinition(
            name="data",
            category="Configuration",
            default="",
            value_type="str",
            description="Calibration Dataset Config",
            help_text="Specifies the dataset configuration YAML file (e.g., 'data.yaml' or 'coco8.yaml') to load calibration images from. This is strictly required when using 'INT8 Quantization' to feed sample images to the quantizer, letting it calculate activation scales and minimize precision loss.",
            affects="Provides sample images for INT8 calibration to minimize accuracy loss during quantization.",
            config_section="export",
        ),
    ]


def _get_defaults_for_format(fmt: str) -> dict[str, Any]:
    defaults = {
        "half": False,
        "int8": False,
        "dynamic": False,
        "simplify": False,
        "workspace": 4.0,
        "nms": False,
        "imgsz": 640,
        "batch": 1,
        "opset": 17,
        "device": "0",
        "keras": False,
        "optimize": False,
        "data": "",
    }
    if fmt == "engine":
        defaults["half"] = True
        defaults["dynamic"] = True
        defaults["workspace"] = 4.0
        defaults["simplify"] = True
        defaults["opset"] = 11  # Conservative TensorRT default; users can raise it.
    elif fmt == "onnx":
        defaults["dynamic"] = True
        defaults["simplify"] = True
    elif fmt == "openvino":
        defaults["half"] = True
    elif fmt == "coreml":
        defaults["nms"] = True
    return defaults


def load_model_details(weight_path: Path, model: Any | None = None) -> ExportModelDetails:
    if model is None:
        import_ultralytics_yolo()
        from ultralytics import YOLO

        model = YOLO(str(weight_path))

    return extract_model_details(str(weight_path), model)


def print_model_details(details: ExportModelDetails) -> None:
    task = details.task or "unknown"
    class_count = str(details.class_count) if details.class_count is not None else "unknown"
    preview = ", ".join(details.class_names[:8])
    if details.class_names and len(details.class_names) > 8:
        preview += ", ..."

    lines = [
        f"[dim]Weight:[/dim] {Path(details.path).name}",
        f"[dim]Task:[/dim] {task}",
        f"[dim]Classes:[/dim] {class_count}",
    ]
    if details.model_name:
        lines.append(f"[dim]Model:[/dim] {details.model_name}")
    if preview:
        lines.append(f"[dim]Class names:[/dim] {preview}")

    console.print(Panel("\n".join(lines), title="Loaded Model Details", border_style="cyan"))


def select_weight(root: Path, available_weights: Sequence[Path]) -> Path | None:
    options = [format_weight_label(root, path) for path in available_weights]
    options.append("Exit")
    selected = get_user_choice(
        options,
        title="Select Weight",
        text="Pick the trained model weights to export:",
        breadcrumbs=["YOLOmatic", "Export", "Weight Selection"],
    )
    if selected == "Exit" or selected == NAV_BACK:
        return None
    return available_weights[options.index(selected)]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="yolomatic-export")
    parser.add_argument("--weight", help="Path to a .pt weight file.")
    return parser.parse_args(argv)


def get_renamed_path(original_path: Path, format_ext: str, params: dict[str, Any]) -> Path:
    # Build standard naming based on options
    stem = original_path.stem
    parts = [stem]
    
    parts.append(format_ext)
    
    if params.get("int8"):
        parts.append("int8")
    elif params.get("half"):
        parts.append("fp16")
    else:
        parts.append("fp32")
        
    if params.get("dynamic"):
        parts.append("dynamic")
        
    imgsz = params.get("imgsz", 640)
    parts.append(f"img{imgsz}")
    
    batch = params.get("batch", 1)
    if batch > 1 and not params.get("dynamic"):
        parts.append(f"batch{batch}")
    
    new_name = "-".join(parts)
    
    ext_map = {
        "engine": ".engine",
        "onnx": ".onnx",
        "openvino": "_openvino_model",
        "coreml": ".mlpackage",
        "ncnn": "_ncnn_model",
        "torchscript": ".torchscript",
        "saved_model": "_saved_model",
        "pb": ".pb",
        "tflite": ".tflite",
        "edgetpu": "_edgetpu.tflite",
        "tfjs": "_web_model",
        "paddle": "_paddle_model",
        "mnn": ".mnn",
        "rknn": ".rknn",
    }
    return original_path.with_name(new_name + ext_map.get(format_ext, f".{format_ext}"))


def run_post_export_benchmark(original_path: Path, exported_path: Path, imgsz: int, format_label: str) -> None:
    from rich.table import Table
    from rich.rule import Rule
    import time

    choice = get_user_choice(
        ["Yes, run benchmark", "No, skip"],
        title="Benchmark Comparison",
        text="Would you like to run a quick performance comparison between the original and exported model?",
        breadcrumbs=["YOLOmatic", "Export", "Benchmark"]
    )
    if choice != "Yes, run benchmark":
        return

    console.print("\n[bold cyan]Running Realistic Performance Benchmark...[/bold cyan]")
    console.print("[dim]  Using varied synthetic images, CUDA-synchronised per-frame timing, 100 measured runs.[/dim]\n")

    try:
        import numpy as np
    except ImportError:
        console.print("[red]numpy is not installed. Skipping benchmark.[/red]")
        return

    from ultralytics import YOLO

    # ── CUDA sync helper ───────────────────────────────────────────────────────
    try:
        import torch
        _cuda_available = torch.cuda.is_available()
    except ImportError:
        torch = None          # type: ignore[assignment]
        _cuda_available = False

    def _cuda_sync() -> None:
        if _cuda_available:
            torch.cuda.synchronize()

    # ── Realistic image pool ───────────────────────────────────────────────────
    # A blank zeros image short-circuits most activations.  Use a diverse pool
    # of synthetic images that mimic real-world content distribution.
    rng = np.random.default_rng(42)
    POOL_SIZE = 16
    _image_pool: list[np.ndarray] = []
    for _i in range(POOL_SIZE):
        base = rng.integers(40, 200, size=(imgsz, imgsz, 3), dtype=np.uint8)
        # Add structured blobs to trigger non-trivial activations
        for _ in range(rng.integers(3, 10)):
            cx = rng.integers(0, imgsz)
            cy = rng.integers(0, imgsz)
            r  = rng.integers(imgsz // 16, imgsz // 4)
            col = rng.integers(0, 256, size=3)
            y_idx, x_idx = np.ogrid[:imgsz, :imgsz]
            mask = (x_idx - cx) ** 2 + (y_idx - cy) ** 2 <= r ** 2
            base[mask] = col.astype(np.uint8)
        _image_pool.append(base)

    def get_size(p: Path) -> float:
        if not p.exists():
            return 0.0
        if p.is_file():
            return p.stat().st_size / (1024 * 1024)
        total = 0
        import os
        for dirpath, _, filenames in os.walk(p):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
        return total / (1024 * 1024)

    WARMUP_RUNS   = 20   # enough to reach GPU steady-state and fill TRT caches
    MEASURED_RUNS = 100  # statistical sample size

    def profile_model(path: Path, label: str) -> dict:
        sz = get_size(path)
        console.print(f"  [dim]Loading {label}...[/dim]")
        try:
            model = YOLO(str(path))

            # ── warmup: rotate through pool so TRT profiles all shapes ────────
            console.print(f"  [dim]Warming up ({WARMUP_RUNS} runs, varied images)...[/dim]")
            for i in range(WARMUP_RUNS):
                img = _image_pool[i % POOL_SIZE]
                _cuda_sync()
                model(img, verbose=False)
                _cuda_sync()

            # ── measured runs ─────────────────────────────────────────────────
            console.print(f"  [dim]Measuring ({MEASURED_RUNS} runs)...[/dim]")
            frame_times: list[float] = []
            for i in range(MEASURED_RUNS):
                img = _image_pool[i % POOL_SIZE]   # rotate — no caching artefacts
                _cuda_sync()
                t0 = time.perf_counter()
                model(img, verbose=False)
                _cuda_sync()
                t1 = time.perf_counter()
                frame_times.append((t1 - t0) * 1000)   # ms end-to-end (preproc+infer+postproc)

            arr = np.array(frame_times)
            result = {
                "mean_ms":  float(np.mean(arr)),
                "p50_ms":   float(np.percentile(arr, 50)),
                "p95_ms":   float(np.percentile(arr, 95)),
                "p99_ms":   float(np.percentile(arr, 99)),
                "std_ms":   float(np.std(arr)),
                "fps":      float(1000.0 / np.mean(arr)),
                "size_mb":  sz,
            }

            del model
            import gc

            gc.collect()
            if _cuda_available:
                torch.cuda.empty_cache()

            return result

        except Exception as e:
            console.print(f"[red]  Failed to profile {label}: {e}[/red]")
            return {"mean_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0,
                    "std_ms": 0, "fps": 0, "size_mb": sz}

    PT_COLOR  = "steel_blue1"
    EXP_COLOR = "cyan"
    BAR_WIDTH = 22

    pt  = profile_model(original_path, f"PyTorch ({original_path.name})")
    exp = profile_model(exported_path,  f"{format_label} ({exported_path.name})")

    def bar(val: float, ref: float, color: str) -> str:
        if ref <= 0:
            return ""
        filled = max(1, round(val / ref * BAR_WIDTH))
        empty  = BAR_WIDTH - filled
        return f"[{color}]{'█' * filled}[/][dim]{'░' * empty}[/]"

    def delta_tag(exp_val: float, pt_val: float, higher_is_better: bool) -> str:
        if exp_val <= 0 or pt_val <= 0:
            return "[dim]n/a[/dim]"
        pct      = abs(exp_val - pt_val) / pt_val * 100
        ratio    = (exp_val / pt_val) if higher_is_better else (pt_val / exp_val)
        exp_wins = (exp_val > pt_val) if higher_is_better else (exp_val < pt_val)
        c = "green" if exp_wins else "red"
        sym = "▲" if exp_wins else "▼"
        winner = f"[{EXP_COLOR}]{format_label}[/]" if exp_wins else f"[{PT_COLOR}]PyTorch[/]"
        return f"[{c}]{sym} {pct:.1f}%  ({ratio:.2f}×)[/]  → {winner}"

    console.print(Rule(f"  {format_label} vs PyTorch — Realistic Benchmark  ", style="cyan"))
    console.print(
        f"  [{PT_COLOR}]█[/] PyTorch (.pt)   "
        f"[{EXP_COLOR}]█[/] {format_label} ({exported_path.suffix})   "
        f"[dim]· {MEASURED_RUNS} runs, CUDA-synced, {POOL_SIZE} unique images[/dim]\n"
    )

    # ── Latency table (end-to-end per frame) ──────────────────────────────────
    lat = Table(title="[bold]End-to-End Latency  (↓ lower is better)  — includes preprocess + infer + postprocess[/bold]",
                box=box.SIMPLE_HEAD, show_lines=False, padding=(0, 2), title_justify="left")
    lat.add_column("Model",   style="bold", min_width=44, no_wrap=True)
    lat.add_column("Mean ms", justify="right", min_width=9)
    lat.add_column("P50",     justify="right", min_width=8)
    lat.add_column("P95",     justify="right", min_width=8)
    lat.add_column("P99",     justify="right", min_width=8)
    lat.add_column("Std",     justify="right", min_width=7)
    lat.add_column("Bar",     min_width=BAR_WIDTH + 2, no_wrap=True)

    worst_mean = max(pt["mean_ms"], exp["mean_ms"])
    pt_best  = pt["mean_ms"]  < exp["mean_ms"]
    exp_best = not pt_best
    pt_crown  = " [yellow]◀ best[/yellow]" if pt_best  else ""
    exp_crown = " [yellow]◀ best[/yellow]" if exp_best else ""

    lat.add_row(
        f"[{PT_COLOR}]█[/] PyTorch  ({original_path.name}){pt_crown}",
        f"[{PT_COLOR}]{pt['mean_ms']:.2f}[/]",
        f"[dim]{pt['p50_ms']:.2f}[/]",
        f"[dim]{pt['p95_ms']:.2f}[/]",
        f"[dim]{pt['p99_ms']:.2f}[/]",
        f"[dim]±{pt['std_ms']:.2f}[/]",
        bar(pt["mean_ms"], worst_mean, PT_COLOR),
    )
    lat.add_row(
        f"[{EXP_COLOR}]█[/] {format_label}  ({exported_path.name}){exp_crown}",
        f"[{EXP_COLOR}]{exp['mean_ms']:.2f}[/]",
        f"[dim]{exp['p50_ms']:.2f}[/]",
        f"[dim]{exp['p95_ms']:.2f}[/]",
        f"[dim]{exp['p99_ms']:.2f}[/]",
        f"[dim]±{exp['std_ms']:.2f}[/]",
        bar(exp["mean_ms"], worst_mean, EXP_COLOR),
    )
    lat.add_section()
    lat.add_row("[dim]Δ Change[/dim]", delta_tag(exp["mean_ms"], pt["mean_ms"], False),
                "", "", "", "", "")
    console.print(lat)
    console.print()

    # ── Throughput + File Size ─────────────────────────────────────────────────
    def simple_section(title: str, pt_val: float, exp_val: float, unit: str,
                       higher_is_better: bool, fmt: str = ".1f") -> None:
        if pt_val <= 0 and exp_val <= 0:
            return
        ref = max(pt_val, exp_val)
        t = Table(title=f"[bold]{title}[/bold]",
                  box=box.SIMPLE_HEAD, show_lines=False, padding=(0, 2), title_justify="left")
        t.add_column("Model",  style="bold", min_width=44, no_wrap=True)
        t.add_column(unit,     justify="right", min_width=10)
        t.add_column("Bar",    min_width=BAR_WIDTH + 2, no_wrap=True)
        pw = (pt_val < exp_val) if not higher_is_better else (pt_val > exp_val)
        t.add_row(
            f"[{PT_COLOR}]█[/] PyTorch  ({original_path.name})" + (" [yellow]◀ best[/yellow]" if pw else ""),
            f"[{PT_COLOR}]{pt_val:{fmt}}[/]",
            bar(pt_val, ref, PT_COLOR),
        )
        t.add_row(
            f"[{EXP_COLOR}]█[/] {format_label}  ({exported_path.name})" + (" [yellow]◀ best[/yellow]" if not pw else ""),
            f"[{EXP_COLOR}]{exp_val:{fmt}}[/]",
            bar(exp_val, ref, EXP_COLOR),
        )
        t.add_section()
        t.add_row("[dim]Δ Change[/dim]", "", delta_tag(exp_val, pt_val, higher_is_better))
        console.print(t)
        console.print()

    simple_section("Throughput  (↑ higher is better)",  pt["fps"],     exp["fps"],     "FPS", True)
    simple_section("File Size   (↓ lower is better)",   pt["size_mb"], exp["size_mb"], "MB",  False)

    console.print(Rule(style="dim"))
    console.print(
        f"  [dim]Methodology: {POOL_SIZE} unique synthetic images rotated per run · "
        f"{WARMUP_RUNS} warmup runs discarded · CUDA synchronised per frame · "
        f"end-to-end wall-clock (letterbox + infer + NMS)[/dim]\n"
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = project_root()

    try:
        if args.weight:
            weight_path = Path(args.weight).expanduser()
            if not weight_path.is_absolute():
                weight_path = root / weight_path
            if not weight_path.exists():
                raise FileNotFoundError(f"Weight file not found: {weight_path}")
        else:
            available_weights = [w for w in find_available_weights(root) if w.suffix == ".pt"]
            if not available_weights:
                console.print(
                    expected_error_panel(
                        "No .pt weights found in the project. Please train a model first.",
                        title="No Weights Found",
                    )
                )
                input("\nPress Enter to return...")
                return

            weight_path = select_weight(root, available_weights)
            if not weight_path:
                return

        try:
            model_details = load_model_details(weight_path)
        except Exception as error:
            console.print(
                expected_error_panel(
                    f"Could not load model details from {weight_path}: {error}",
                    title="Model Details Error",
                )
            )
            return

        print_model_details(model_details)

        # Select Format
        descriptions = {
            "TensorRT (engine)": "Best for NVIDIA GPUs. Expected Boost: 3x-5x faster inference. Ideal for local workstations and Jetson devices. Requires TRT runtime matching export version.",
            "ONNX": "Universal standard. Expected Boost: 1.5x-2x faster. Great middle-ground for cross-platform deployment and flexible inference engines.",
            "OpenVINO": "Best for Intel CPUs/iGPUs. Expected Boost: 2x-3x faster CPU inference. Highly optimized for systems without discrete GPUs.",
            "CoreML": "Best for Apple Silicon (M1/M2/M3) and iOS. Expected Boost: 2x-4x faster leveraging the dedicated Apple Neural Engine (ANE).",
            "NCNN": "Best for mobile and ARM edge devices. Highly optimized lightweight C++ inference, great for Raspberry Pi and similar edge nodes.",
            "TorchScript": "PyTorch's native JIT format. Modest speedup but offers excellent portability and deployment ease across PyTorch C++ environments.",
            "TF SavedModel": "Standard TensorFlow format. Good for TF Serving or enterprise TensorFlow pipelines.",
            "TF Lite": "Best for Android and embedded devices. Expected Boost: 2x-3x faster, especially when paired with INT8 quantization.",
            "TF.js": "Best for web deployment. Runs directly in the browser via WebGL or WebGPU for zero-install client-side inference.",
            "Paddle": "Export to PaddlePaddle format for deployment specifically inside the Baidu ecosystem.",
            "Edge TPU": "Targeted specifically for Google Coral Edge TPUs. Delivers exceptional speed-per-watt hardware acceleration.",
            "MNN": "Alibaba's Mobile Neural Network. Highly efficient inference framework for iOS and Android.",
            "RKNN": "Rockchip NPU specific format. Essential for running accelerated inference on boards like the RK3588 or RK3568.",
            "Cancel": "Return to the previous menu."
        }
        
        supported_formats = supported_formats_for_model(SUPPORTED_FORMATS, model_details)
        format_label = get_user_choice(
            list(supported_formats.keys()) + ["Cancel"],
            title="Export Format",
            text="Select the target architecture format. (See expected boosting metrics below):",
            descriptions=descriptions,
            breadcrumbs=["YOLOmatic", "Export", "Format Selection"],
        )
        if format_label == "Cancel" or format_label == NAV_BACK:
            return
            
        target_format = supported_formats[format_label]
        
        # Configure Options
        definitions = filter_export_definitions(
            _export_definitions(),
            target_format,
            model_details,
        )
        defaults = filter_export_defaults(
            _get_defaults_for_format(target_format),
            target_format,
            model_details,
        )
        
        # Only select arguments that make sense for the chosen format
        pre_selected = set(defaults.keys())
        
        result = get_user_multi_select(
            parameters=definitions,
            title=f"Configure {format_label} Export Options",
            instruction="[Space] Toggle  [Enter] Edit Value  [F] Start Export  [Q] Cancel",
            pre_selected=pre_selected,
            pre_values=defaults,
        )
        
        if result is None:
            return
            
        selected_params, values = result
        export_kwargs = build_export_kwargs(
            target_format,
            definitions,
            selected_params,
            values,
            warn=console.print,
            model_details=model_details,
        )
        
        console.print(Panel(
            f"[bold cyan]Exporting {weight_path.name}[/bold cyan]\n"
            f"[dim]Format:[/dim] {format_label}\n"
            f"[dim]Options:[/dim] {export_kwargs}",
            border_style="cyan"
        ))

        # Import YOLO and Export
        import_ultralytics_yolo()
        from ultralytics import YOLO
        
        try:
            import shutil

            def _is_trt_tactic_error(exc: Exception) -> bool:
                return "engine build failed" in str(exc).lower()

            def _cleanup_onnx(weight_path: Path) -> None:
                """Remove the intermediate ONNX produced by a failed TRT export."""
                onnx_path = weight_path.with_suffix(".onnx")
                if onnx_path.exists():
                    try:
                        onnx_path.unlink()
                    except Exception:
                        pass

            def _do_export(kwargs: dict) -> str | None:
                m = YOLO(str(weight_path))
                return m.export(**kwargs)

            console.print("[yellow]Starting export process... This may take a while.[/yellow]")

            active_kwargs = export_kwargs.copy()
            exported_path_str = None

            # --- TensorRT-specific retry cascade ---
            if target_format == "engine":
                try:
                    exported_path_str = _do_export(active_kwargs)
                except Exception as e1:
                    if _is_trt_tactic_error(e1) and active_kwargs.get("half"):
                        _cleanup_onnx(weight_path)
                        console.print(
                            "[yellow]⚠ TRT FP16 build failed (ConvTranspose tactic not available on this GPU/TRT version). "
                            "Retrying with FP32...[/yellow]"
                        )
                        active_kwargs["half"] = False
                        try:
                            exported_path_str = _do_export(active_kwargs)
                        except Exception as e2:
                            if _is_trt_tactic_error(e2) and active_kwargs.get("dynamic"):
                                _cleanup_onnx(weight_path)
                                console.print(
                                    "[yellow]⚠ TRT FP32 dynamic build also failed. "
                                    "Retrying with static shapes (batch=1)...[/yellow]"
                                )
                                active_kwargs["dynamic"] = False
                                active_kwargs["batch"] = 1
                                exported_path_str = _do_export(active_kwargs)
                            else:
                                raise e2
                    else:
                        raise e1
            else:
                exported_path_str = _do_export(active_kwargs)

            if exported_path_str:
                exported_path = Path(exported_path_str)
                renamed_path = get_renamed_path(weight_path, target_format, active_kwargs)

                # Move/rename the exported file or directory
                if exported_path.exists():
                    # Handle if target exists
                    if renamed_path.exists():
                        if renamed_path.is_dir():
                            shutil.rmtree(renamed_path)
                        else:
                            renamed_path.unlink()

                    exported_path.rename(renamed_path)
                    console.print("\n[bold green]Success![/bold green] Exported model saved to:")
                    console.print(f"[bold white]{renamed_path}[/bold white]")

                    # Surface any fallback settings that were applied
                    if active_kwargs != export_kwargs:
                        changed = {k: active_kwargs[k] for k in active_kwargs if active_kwargs.get(k) != export_kwargs.get(k)}
                        console.print(f"[dim]Note: export succeeded with adjusted settings: {changed}[/dim]")

                    # Run post export benchmark
                    target_imgsz = active_kwargs.get("imgsz", 640)
                    run_post_export_benchmark(weight_path, renamed_path, target_imgsz, format_label)
                else:
                    console.print(f"\n[bold green]Export finished.[/bold green] But could not locate output at {exported_path}")
            else:
                console.print("\n[bold red]Export failed or returned no path.[/bold red]")

        except Exception as e:
            console.print(
                expected_error_panel(
                    f"Export failed: {str(e)}\n\nMake sure you have the required dependencies installed (e.g. tensorrt, onnx, openvino).",
                    title="Export Error"
                )
            )

    except KeyboardInterrupt:
        pass

    input("\nPress Enter to return...")

if __name__ == "__main__":
    main()
