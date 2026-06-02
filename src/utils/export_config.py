from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol


class ExportParameter(Protocol):
    name: str
    default: Any
    value_type: str


def build_export_kwargs(
    target_format: str,
    definitions: Sequence[ExportParameter],
    selected_params: set[str],
    values: dict[str, Any],
    warn: Callable[[str], None] | None = None,
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

    if export_kwargs.get("opset", 17) > 12:
        if warn is not None:
            warn(
                f"[yellow]Warning: Lowering ONNX opset from {export_kwargs['opset']} to 12 "
                "for TensorRT export. Opset 17 causes ConvTranspose tactic errors in TRT 11 "
                "with dynamic+FP16 segmentation models.[/yellow]"
            )
        export_kwargs["opset"] = 12

    return export_kwargs
