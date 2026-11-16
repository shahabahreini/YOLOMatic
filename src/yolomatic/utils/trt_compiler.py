from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

# Configure logging
logger = logging.getLogger("yolomatic.trt_compiler")


def compile_onnx_to_trt(
    onnx_path: Path,
    engine_path: Path,
    imgsz: int,
    opt_batch: int,
    max_batch: int,
    half: bool,
    workspace_gb: float = 4.0,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Compiles an ONNX model to a TensorRT engine with dynamic batch size but fixed resolution.

    This resolves issues where Ultralytics' dynamic=True exporter configures overly broad
    optimization profiles that can lead to compiler errors, out-of-memory issues, or poor
    inference performance.
    """
    from yolomatic.utils.ml_dependencies import prepare_ml_runtime

    # Ensure CUDA libraries are in path
    prepare_ml_runtime()

    # Initialize PyTorch CUDA to load runtime libraries into process space
    import torch
    if torch.cuda.is_available():
        try:
            torch.cuda.init()
        except Exception:
            pass

    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    config = builder.create_builder_config()

    # Calculate workspace bytes
    workspace_bytes = int(workspace_gb * (1 << 30))
    is_trt10 = int(trt.__version__.split(".", 1)[0]) >= 10
    if is_trt10 and workspace_bytes > 0:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    elif workspace_bytes > 0:
        config.max_workspace_size = workspace_bytes

    # EXPLICIT_BATCH flag is removed in TRT 10; keep it for TRT 7/8/9
    flag = 0 if is_trt10 else (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)

    # Parse ONNX
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = []
        for i in range(parser.num_errors):
            errors.append(str(parser.get_error(i)))
        raise RuntimeError(f"Failed to parse ONNX file: {', '.join(errors)}")

    # Configure optimization profile
    profile = builder.create_optimization_profile()

    # For every input tensor, set the shapes
    inputs_configured = 0
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        # Check if the input tensor has 4 dimensions [batch, channels, height, width]
        if len(inp.shape) == 4:
            channels = inp.shape[1]
            min_shape = (1, channels, imgsz, imgsz)
            opt_shape = (opt_batch, channels, imgsz, imgsz)
            max_shape = (max_batch, channels, imgsz, imgsz)

            profile.set_shape(inp.name, min=min_shape, opt=opt_shape, max=max_shape)
            inputs_configured += 1

    if inputs_configured == 0:
        raise RuntimeError("No dynamic 4D input tensor found in ONNX graph to apply optimization profile.")

    config.add_optimization_profile(profile)

    # Configure FP16
    if half:
        has_fast_fp16 = getattr(builder, "platform_has_fast_fp16", True)
        if has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

    # Build the engine
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build serialized TensorRT engine. Check builder log.")

    write_engine_file(engine_path, bytes(serialized_engine), metadata)


def write_engine_file(engine_path: Path, serialized_engine: bytes, metadata: dict[str, Any] | None) -> None:
    """Write an engine in the layout Ultralytics' AutoBackend reads.

    Ultralytics prefixes ``.engine`` files with a little-endian int32 length and
    a JSON metadata blob (class names, task, stride, imgsz, ...). Without it a
    loaded engine falls back to generic class names and guesses the task from
    the filename, so segmentation/pose engines were treated as detection.
    """
    with open(engine_path, "wb") as handle:
        if metadata is not None:
            blob = json.dumps(metadata).encode("utf-8")
            handle.write(len(blob).to_bytes(4, byteorder="little", signed=True))
            handle.write(blob)
        handle.write(serialized_engine)


def read_engine_metadata(engine_path: Path) -> dict[str, Any] | None:
    """Inverse of :func:`write_engine_file` (mirrors Ultralytics' reader)."""
    with open(engine_path, "rb") as handle:
        try:
            length = int.from_bytes(handle.read(4), byteorder="little", signed=True)
            if length <= 0 or length > 10_000_000:
                return None
            return json.loads(handle.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
