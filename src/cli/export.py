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
    print_stylized_header,
    expected_error_panel,
)
from src.utils.project import (
    find_available_weights,
    format_weight_label,
    project_root,
)
from src.utils.ml_dependencies import import_ultralytics_yolo, MLDependencyError


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
            "half",
            "Optimization",
            False,
            "bool",
            "FP16 Half Precision",
            "Enable FP16 quantization for faster inference.",
            config_section="export",
        ),
        ParameterDefinition(
            "int8",
            "Optimization",
            False,
            "bool",
            "INT8 Precision",
            "Enable INT8 quantization.",
            config_section="export",
        ),
        ParameterDefinition(
            "dynamic",
            "Configuration",
            False,
            "bool",
            "Dynamic Axes",
            "Enable dynamic batching and image sizes (useful for TensorRT/ONNX).",
            config_section="export",
        ),
        ParameterDefinition(
            "simplify",
            "Configuration",
            False,
            "bool",
            "Simplify Graph",
            "Simplify ONNX graph.",
            config_section="export",
        ),
        ParameterDefinition(
            "workspace",
            "Configuration",
            4.0,
            "float",
            "TensorRT Workspace (GB)",
            "Maximum workspace size in GB for TensorRT optimization.",
            min_value=0.5,
            config_section="export",
        ),
        ParameterDefinition(
            "nms",
            "Configuration",
            False,
            "bool",
            "Add NMS",
            "Add Non-Maximum Suppression to CoreML/TensorRT.",
            config_section="export",
        ),
        ParameterDefinition(
            "imgsz",
            "Configuration",
            640,
            "int",
            "Image Size",
            "Input image size for inference.",
            min_value=32,
            config_section="export",
        ),
        ParameterDefinition(
            "batch",
            "Configuration",
            1,
            "int",
            "Batch Size",
            "Static batch size (useful if dynamic batching is disabled).",
            min_value=1,
            config_section="export",
        ),
        ParameterDefinition(
            "opset",
            "Configuration",
            17,
            "int",
            "ONNX Opset Version",
            "ONNX opset version.",
            min_value=11,
            config_section="export",
        ),
        ParameterDefinition(
            "device",
            "Configuration",
            "0",
            "str",
            "Device",
            "Device to run the export on (e.g. '0', 'cpu').",
            config_section="export",
        ),
        ParameterDefinition(
            "keras",
            "Configuration",
            False,
            "bool",
            "Use Keras",
            "Enable Keras layers during export (TF SavedModel only).",
            config_section="export",
        ),
        ParameterDefinition(
            "optimize",
            "Optimization",
            False,
            "bool",
            "Optimize",
            "TorchScript mobile optimizations or DEEPX higher compiler optimization.",
            config_section="export",
        ),
        ParameterDefinition(
            "data",
            "Configuration",
            "",
            "str",
            "Dataset Config (data.yaml)",
            "Required for INT8 calibration. E.g. 'coco8.yaml'.",
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
    if selected == "Exit":
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
        format_label = get_user_choice(
            list(SUPPORTED_FORMATS.keys()) + ["Cancel"],
            title="Export Format",
            text="Select the target architecture format:",
            breadcrumbs=["YOLOmatic", "Export", "Format Selection"],
        )
        if format_label == "Cancel":
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
                    console.print(f"\n[bold green]Success![/bold green] Exported model saved to:")
                    console.print(f"[bold white]{renamed_path}[/bold white]")
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
