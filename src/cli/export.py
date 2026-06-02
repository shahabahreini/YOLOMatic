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
    elif fmt == "onnx":
        defaults["dynamic"] = True
        defaults["simplify"] = True
    elif fmt == "openvino":
        defaults["half"] = True
    elif fmt == "coreml":
        defaults["nms"] = True
    return defaults


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
    import time
    
    choice = get_user_choice(
        ["Yes, run benchmark", "No, skip"],
        title="Benchmark Comparison",
        text="Would you like to run a quick performance comparison between the original and exported model?",
        breadcrumbs=["YOLOmatic", "Export", "Benchmark"]
    )
    if choice != "Yes, run benchmark":
        return

    console.print("\n[bold cyan]Running Performance Benchmark...[/bold cyan]")
    
    try:
        import numpy as np
    except ImportError:
        console.print("[red]numpy is not installed. Skipping benchmark.[/red]")
        return
        
    from ultralytics import YOLO
    
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

    dummy_img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    
    def profile_model(path: Path) -> tuple[float, float, float]:
        sz = get_size(path)
        try:
            model = YOLO(str(path))
            # Warmup
            for _ in range(3):
                model(dummy_img, verbose=False)
            
            # Benchmark
            t0 = time.time()
            inf_times = []
            for _ in range(20):
                res = model(dummy_img, verbose=False)
                if res and hasattr(res[0], 'speed') and 'inference' in res[0].speed:
                    inf_times.append(res[0].speed['inference'])
            t1 = time.time()
            
            if inf_times:
                avg_inf = sum(inf_times) / len(inf_times)
            else:
                avg_inf = ((t1 - t0) / 20) * 1000
                
            fps = 20 / (t1 - t0)
            
            del model
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
                
            return avg_inf, fps, sz
        except Exception as e:
            console.print(f"[red]Failed to profile {path.name}: {e}[/red]")
            return 0.0, 0.0, sz

    console.print(f"[dim]Profiling original {original_path.name}...[/dim]")
    pt_inf, pt_fps, pt_sz = profile_model(original_path)
    
    console.print(f"[dim]Profiling exported {exported_path.name}...[/dim]")
    exp_inf, exp_fps, exp_sz = profile_model(exported_path)
    
    speed_diff = (pt_inf / exp_inf) if (pt_inf > 0 and exp_inf > 0) else 1.0

    table = Table(title=f"Performance Comparison: PyTorch vs {format_label}", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Original (.pt)", justify="right")
    table.add_column("Exported", justify="right")
    table.add_column("Change", justify="right")
    table.add_column("Visual", justify="left")

    def draw_bar(val: float, max_val: float, color: str) -> str:
        if max_val <= 0:
            return ""
        width = int((val / max_val) * 20)
        return f"[{color}]" + "█" * width + "[/]"
        
    max_inf = max(pt_inf, exp_inf)
    if max_inf > 0:
        inf_change = f"[green]{speed_diff:.1f}x faster[/green]" if exp_inf < pt_inf else f"[red]{speed_diff:.1f}x slower[/red]"
        table.add_row(
            "Inference (ms)", 
            f"{pt_inf:.2f}", 
            f"{exp_inf:.2f}", 
            inf_change,
            draw_bar(pt_inf, max_inf, "red") + "\n" + draw_bar(exp_inf, max_inf, "green")
        )
        
    max_fps = max(pt_fps, exp_fps)
    if max_fps > 0:
        fps_diff = (exp_fps - pt_fps) / pt_fps * 100 if pt_fps > 0 else 0
        fps_change = f"[green]+{fps_diff:.1f}%[/green]" if fps_diff > 0 else f"[red]{fps_diff:.1f}%[/red]"
        table.add_row(
            "Throughput (FPS)", 
            f"{pt_fps:.1f}", 
            f"{exp_fps:.1f}", 
            fps_change,
            draw_bar(pt_fps, max_fps, "red") + "\n" + draw_bar(exp_fps, max_fps, "green")
        )
        
    max_sz = max(pt_sz, exp_sz)
    if max_sz > 0:
        sz_diff = (exp_sz - pt_sz) / pt_sz * 100 if pt_sz > 0 else 0
        sz_change = f"[red]+{sz_diff:.1f}%[/red]" if sz_diff > 0 else f"[green]{sz_diff:.1f}%[/green]"
        table.add_row(
            "File Size (MB)", 
            f"{pt_sz:.1f}", 
            f"{exp_sz:.1f}", 
            sz_change,
            draw_bar(pt_sz, max_sz, "red") + "\n" + draw_bar(exp_sz, max_sz, "green")
        )
        
    console.print()
    console.print(table)


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
        
        format_label = get_user_choice(
            list(SUPPORTED_FORMATS.keys()) + ["Cancel"],
            title="Export Format",
            text="Select the target architecture format. (See expected boosting metrics below):",
            descriptions=descriptions,
            breadcrumbs=["YOLOmatic", "Export", "Format Selection"],
        )
        if format_label == "Cancel" or format_label == NAV_BACK:
            return
            
        target_format = SUPPORTED_FORMATS[format_label]
        
        # Configure Options
        definitions = _export_definitions()
        defaults = _get_defaults_for_format(target_format)
        
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
        export_kwargs = {"format": target_format}
        
        # Build kwargs from selected options
        for param in definitions:
            if param.name in selected_params:
                val = values.get(param.name, param.default)
                if val == "" and param.value_type == "str":
                    continue  # Ignore empty strings for optional args like data
                export_kwargs[param.name] = val
        
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
            model = YOLO(str(weight_path))
            
            console.print("[yellow]Starting export process... This may take a while.[/yellow]")
            exported_path_str = model.export(**export_kwargs)
            
            if exported_path_str:
                exported_path = Path(exported_path_str)
                renamed_path = get_renamed_path(weight_path, target_format, export_kwargs)
                
                # Move/rename the exported file or directory
                if exported_path.exists():
                    # Handle if target exists
                    if renamed_path.exists():
                        import shutil
                        if renamed_path.is_dir():
                            shutil.rmtree(renamed_path)
                        else:
                            renamed_path.unlink()
                            
                    exported_path.rename(renamed_path)
                    console.print("\n[bold green]Success![/bold green] Exported model saved to:")
                    console.print(f"[bold white]{renamed_path}[/bold white]")
                    
                    # Run post export benchmark
                    target_imgsz = export_kwargs.get("imgsz", 640)
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
