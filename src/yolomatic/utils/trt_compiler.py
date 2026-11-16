from __future__ import annotations

import logging
from pathlib import Path

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
) -> None:
    """Compiles an ONNX model to a TensorRT engine with dynamic batch size but fixed resolution.

    This resolves issues where Ultralytics' dynamic=True exporter configures overly broad
    optimization profiles that can lead to compiler errors, out-of-memory issues, or poor
    inference performance.
    """
    from src.utils.ml_dependencies import prepare_ml_runtime

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

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
