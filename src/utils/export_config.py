from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


DETECTION_LIKE_TASKS = frozenset({"detect", "segment", "pose", "obb"})
ONNX_BACKED_FORMATS = frozenset({"onnx", "engine"})
INT8_CAPABLE_FORMATS = frozenset({"engine", "tflite", "edgetpu"})
TASK_LIMITED_FORMATS = {
    "edgetpu": DETECTION_LIKE_TASKS | frozenset({"classify"}),
}
FORMAT_PARAM_NAMES = {
    "engine": {
        "half",
        "int8",
        "dynamic",
        "simplify",
        "workspace",
        "nms",
        "imgsz",
        "batch",
        "opset",
        "device",
        "data",
    },
    "onnx": {
        "half",
        "int8",
        "dynamic",
        "simplify",
        "nms",
        "imgsz",
        "batch",
        "opset",
        "device",
        "data",
    },
    "openvino": {"half", "int8", "dynamic", "imgsz", "batch", "device", "data"},
    "coreml": {"half", "int8", "nms", "imgsz", "batch", "device", "data"},
    "ncnn": {"half", "imgsz", "batch", "device"},
    "torchscript": {"half", "optimize", "imgsz", "batch", "device"},
    "saved_model": {"int8", "keras", "imgsz", "batch", "device", "data"},
    "tflite": {"int8", "imgsz", "batch", "device", "data"},
    "tfjs": {"int8", "imgsz", "batch", "device", "data"},
    "paddle": {"half", "imgsz", "batch", "device"},
    "edgetpu": {"imgsz", "batch", "device"},
    "mnn": {"half", "int8", "imgsz", "batch", "device", "data"},
    "rknn": {"half", "int8", "imgsz", "batch", "device", "data"},
}


@dataclass(frozen=True)
class ExportModelDetails:
    path: str
    task: str | None = None
    class_count: int | None = None
    class_names: tuple[str, ...] = ()
    model_name: str | None = None

    @property
    def normalized_task(self) -> str | None:
        if not self.task:
            return None
        return self.task.strip().lower() or None


def normalize_class_names(raw_names: Any) -> tuple[str, ...]:
    if isinstance(raw_names, dict):
        return tuple(str(raw_names[key]) for key in sorted(raw_names))
    if isinstance(raw_names, (list, tuple)):
        return tuple(str(name) for name in raw_names)
    return ()


def extract_model_details(path: str, model: Any) -> ExportModelDetails:
    raw_names = getattr(model, "names", None)
    if raw_names is None:
        raw_names = getattr(getattr(model, "model", None), "names", None)

    model_name = getattr(model, "model_name", None)
    if model_name is None:
        model_name = getattr(getattr(model, "model", None), "yaml_file", None)
    if model_name is not None:
        model_name = str(model_name)

    class_names = normalize_class_names(raw_names)
    return ExportModelDetails(
        path=path,
        task=getattr(model, "task", None),
        class_count=len(class_names) or None,
        class_names=class_names,
        model_name=model_name,
    )


class ExportParameter(Protocol):
    name: str
    default: Any
    value_type: str


def supported_formats_for_model(
    format_map: dict[str, str],
    model_details: ExportModelDetails,
) -> dict[str, str]:
    task = model_details.normalized_task
    if task is None:
        return dict(format_map)

    return {
        label: fmt
        for label, fmt in format_map.items()
        if task in TASK_LIMITED_FORMATS.get(fmt, {task})
    }


def applicable_export_param_names(
    target_format: str,
    model_details: ExportModelDetails | None = None,
) -> set[str]:
    names = set(FORMAT_PARAM_NAMES.get(target_format, ()))
    task = model_details.normalized_task if model_details is not None else None

    if "nms" in names and task is not None and task not in DETECTION_LIKE_TASKS:
        names.remove("nms")

    if target_format not in ONNX_BACKED_FORMATS:
        names.discard("simplify")
        names.discard("opset")

    if target_format not in INT8_CAPABLE_FORMATS:
        names.discard("data")

    return names


def filter_export_definitions(
    definitions: Sequence[ExportParameter],
    target_format: str,
    model_details: ExportModelDetails | None = None,
) -> list[ExportParameter]:
    applicable_names = applicable_export_param_names(target_format, model_details)
    return [param for param in definitions if param.name in applicable_names]


def filter_export_defaults(
    defaults: dict[str, Any],
    target_format: str,
    model_details: ExportModelDetails | None = None,
) -> dict[str, Any]:
    applicable_names = applicable_export_param_names(target_format, model_details)
    return {
        name: value
        for name, value in defaults.items()
        if name in applicable_names
    }


def build_export_kwargs(
    target_format: str,
    definitions: Sequence[ExportParameter],
    selected_params: set[str],
    values: dict[str, Any],
    warn: Callable[[str], None] | None = None,
    model_details: ExportModelDetails | None = None,
) -> dict[str, Any]:
    export_kwargs = {"format": target_format}

    for param in definitions:
        if param.name not in selected_params:
            continue

        val = values.get(param.name, param.default)
        if val == "" and param.value_type == "str":
            continue
        export_kwargs[param.name] = val

    if target_format != "engine":
        return export_kwargs

    if export_kwargs.get("dynamic") and export_kwargs.get("batch", 1) == 1:
        # TensorRT dynamic batching requires max batch size explicitly defined.
        export_kwargs["batch"] = 16

    task = model_details.normalized_task if model_details is not None else None
    should_limit_opset = (
        export_kwargs.get("opset", 17) > 12
        and export_kwargs.get("dynamic", False)
        and export_kwargs.get("half", False)
        and task == "segment"
    )
    if should_limit_opset:
        if warn is not None:
            warn(
                f"[yellow]Warning: Lowering ONNX opset from {export_kwargs['opset']} to 12 "
                "for TensorRT export because TRT 11 can fail to find ConvTranspose "
                "tactics for dynamic FP16 segmentation models.[/yellow]"
            )
        export_kwargs["opset"] = 12

    return export_kwargs
