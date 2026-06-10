---
description: Export YOLOmatic trained models to ONNX, TensorRT, CoreML, TFLite, OpenVINO, and other deployment formats.
---

# Export & Deployment

After training a model with YOLOmatic, you can export it to a deployment-ready format suited for your target hardware. YOLOmatic supports all export formats provided by the Ultralytics framework for YOLO-family models.

---

## Supported Export Formats

| Format | Flag | Target Platform | Notes |
|---|---|---|---|
| **ONNX** | `onnx` | Universal / most runtimes | Recommended default for portability |
| **TensorRT** | `engine` | NVIDIA GPU (Linux/Windows) | Highest GPU throughput; requires CUDA |
| **CoreML** | `coreml` | macOS / iOS | `.mlpackage` output for Apple deployment |
| **TensorFlow SavedModel** | `saved_model` | TF Serving, TFLite source | Full TF model directory |
| **TensorFlow Lite** | `tflite` | Android, embedded MCU | Quantization-friendly |
| **TensorFlow Edge TPU** | `edgetpu` | Google Coral | Requires Edge TPU compiler |
| **OpenVINO** | `openvino` | Intel CPU/GPU/VPU | Intel Neural Compute Stick 2 |
| **PaddlePaddle** | `pb` | Baidu inference | PaddleDetection ecosystem |
| **NCNN** | `ncnn` | ARM / mobile CPU | Tencent mobile runtime |
| **RKNN** | `rknn` | Rockchip NPU | Edge AI boards |

---

## Exporting via the TUI

YOLOmatic exposes export options during the **Configure Model** workflow. After you select a model family and variant, look for the **Export** step in the wizard to choose a format and configure advanced options.

For standalone export after training, run:

```sh
uv run yolomatic-export
```

The export wizard loads the selected `.pt` checkpoint first, shows its detected
task and class metadata, then hides export options that are not applicable to
that model and target format. For example, TensorRT-only workspace settings are
shown only for `engine` exports, and NMS options are hidden for classification
models.

---

## Exporting via Python

After training, export using the Ultralytics `YOLO` class:

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

# ONNX — universal, most compatible
model.export(format="onnx")

# TensorRT — fastest on NVIDIA GPU
model.export(format="engine")

# CoreML — Apple devices
model.export(format="coreml")

# TensorFlow Lite — Android / embedded
model.export(format="tflite")

# OpenVINO — Intel hardware
model.export(format="openvino")

# NCNN — ARM / mobile CPU
model.export(format="ncnn")
```

The exported file is saved in the same directory as the source `.pt` weight.

---

## Advanced Export Options

### Half Precision (FP16)

Reduces model size and improves throughput on supported hardware at a small accuracy cost:

```python
model.export(format="engine", half=True)   # TensorRT FP16
model.export(format="onnx", half=True)     # ONNX FP16
```

### INT8 Quantization

Further reduces size and latency for edge deployment:

```python
model.export(format="tflite", int8=True)   # TFLite INT8
model.export(format="engine", int8=True)   # TensorRT INT8
```

INT8 export requires a calibration dataset for best accuracy.

### Dynamic Input Shapes

Allows the exported model to accept variable batch sizes or image dimensions at inference time:

```python
model.export(format="onnx", dynamic=True)
```

### TensorRT Dynamic Batch Mode

For TensorRT (`engine`) exports, using fully dynamic shapes (dynamic batch size AND variable image resolutions) can sometimes cause compilation to fail or lead to "tactic-not-found" builder crashes.

To solve this, YOLOmatic introduces the `trt_dynamic_batch` parameter:
* **Behavior:** When `dynamic=True` and `trt_dynamic_batch=True`, the builder locks image dimensions to the selected fixed resolution (e.g. 640x640) while keeping only the batch dimension dynamic.
* **Benefits:** Bypasses common TensorRT builder issues while retaining variable batch-size runtime flexibility.
* **TUI Configuration:** Enabled via the **TRT Dynamic Batch Only** toggle in the export wizard step.
* **Config YAML:**
  ```yaml
  export:
    format: engine
    dynamic: true
    trt_dynamic_batch: true
  ```

### TensorRT Workspace

For TensorRT exports, `workspace` controls how much temporary GPU memory the
TensorRT builder may use while compiling the engine:

```python
model.export(format="engine", workspace=8.0)
```

YOLOmatic preserves the selected workspace value instead of clamping it to a
specific GPU profile. Tune it for the GPU that builds the engine: larger values
can let TensorRT search more tactics, while smaller values reduce build-time
memory pressure.

### Include NMS in Export

For formats where post-processing is handled externally, you can embed NMS into the export:

```python
model.export(format="onnx", nms=True)
```

---

## Platform / Hardware Target Mapping

| Deployment Target | Recommended Format | Notes |
|---|---|---|
| NVIDIA GPU server | `engine` (TensorRT) | Best throughput; build on the same CUDA version as the target server |
| NVIDIA Jetson | `engine` (TensorRT) | Use Jetson's local TensorRT; do not cross-compile |
| Apple Silicon Mac | `coreml` | `.mlpackage` runs natively on M1/M2/M3 |
| iPhone / iPad | `coreml` | Submit `.mlpackage` directly to Xcode |
| Android | `tflite` | Integrate with TFLite Android SDK |
| Google Coral (Edge TPU) | `edgetpu` | Requires Edge TPU compiler post-export |
| Intel CPU / iGPU / VPU | `openvino` | Use OpenVINO Model Server for serving |
| ARM CPU / Raspberry Pi | `ncnn` | No external runtime needed; ships as `.bin` + `.param` |
| Rockchip board | `rknn` | Use RKNN Toolkit post-export |
| Cloud API (any framework) | `onnx` | Use ONNX Runtime or Triton Inference Server |

---

## ONNX Export Prerequisites

ONNX export requires `onnx` and `onnxslim` in the environment. YOLOmatic includes these in its dependency set. If you see an import error, re-sync the environment:

```sh
uv sync
```

TensorRT export additionally requires:
1. A CUDA-capable GPU with matching CUDA/cuDNN installed
2. TensorRT installed (matches your CUDA version)
3. The export must run on the same machine where inference will happen
4. A workspace value that fits the build GPU and model size

---

## Roboflow Deployment

After exporting, you can upload and deploy your checkpoint to Roboflow:

```sh
uv run yolomatic-upload
```

See [Cloud Upload](cloud-upload.md) for the full workflow. Note that for Roboflow you upload the `.pt` checkpoint directly — Roboflow handles conversion server-side.

Related pages: [Cloud Upload](cloud-upload.md), [Models](models.md), [CLI Commands](../reference/cli-commands.md).
