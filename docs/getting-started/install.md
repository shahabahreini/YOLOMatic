---
description: Install YOLOmatic with uv, choose optional extras, set up a development environment, and configure credentials.
---

# Install

YOLOmatic supports Python 3.12 and 3.13.

## Package Install

```sh
uv tool install --python 3.12 yolomatic
yolomatic
```

Until a release is published to PyPI, install straight from GitHub:

```sh
uv tool install --python 3.12 "yolomatic @ git+https://github.com/shahabahreini/YOLOMatic"
```

## Optional Extras

The base install covers configuring, training (YOLO and RF-DETR), prediction,
dataset preparation, augmentation, benchmarking and the TUI. Heavier stacks are
opt-in:

| Extra | Adds | Needed for |
|---|---|---|
| `export` | ONNX, ONNX Runtime, onnxslim, OpenVINO, NCNN, TensorFlow, CoreML | Exporting to those formats |
| `tensorrt` | TensorRT (CUDA 12) | `.engine` exports on NVIDIA GPUs |
| `clearml` | ClearML | Experiment tracking |
| `viz` | umap-learn | UMAP plots in benchmark reports (PCA is used otherwise) |
| `sam` | transformers, accelerate | SAM 3.1 fine-tuning and segmentation |
| `all` | Everything above | |

```sh
uv tool install --python 3.12 "yolomatic[export,clearml]"
```

A missing extra is reported with the exact command to install it.

## Repository Install

```sh
git clone https://github.com/shahabahreini/YOLOMatic.git
cd YOLOMatic
uv sync                      # base install + dev tools
uv sync --extra export       # add an extra (or --all-extras)
uv run yolomatic
```

## Credentials

```sh
cp .env.example .env
```

Fill in what you use: `ROBOFLOW_API_KEY` / `ROBOFLOW_WORKSPACE` /
`ROBOFLOW_PROJECT_IDS` for Roboflow, `ULTRALYTICS_API_KEY` for the Ultralytics
Platform, `HF_TOKEN` for SAM 3.1. AI-recommendation keys entered in
**Settings > AI** are stored in your operating system's credential store
(Keychain, Credential Locker or Secret Service); set `YOLOMATIC_KEYRING=0` to
keep them in the settings file instead.

## Logs

Each session writes a rotating log (full tracebacks included) to the OS user log
folder, for example `~/Library/Logs/yolomatic/yolomatic.log` on macOS. Error
panels show the path. Override with `YOLOMATIC_LOG_DIR` or `--log-file`, and use
`yolomatic --debug` to see tracebacks on screen.

## Platform Notes

- On Windows, keep the explicit `--python 3.12` flag for tool installs that use
  the `export` extra. Some ONNX dependencies do not publish Python 3.13 Windows
  wheels and will otherwise fall back to a CMake source build.
- Linux and Windows can use CUDA when the installed PyTorch build supports it.
- macOS uses CPU or Apple Silicon MPS. NVIDIA CUDA is not available on macOS.
- Apple Silicon is tested in CI on a native arm64 macOS runner.
- YOLOmatic detects a CPU-only PyTorch build on GPU machines and offers repair
  guidance from the TUI.
- The interactive TUI needs a terminal. Over SSH use `ssh -t`; in scripts and CI
  call a command directly, e.g. `yolomatic predict --weight best.pt --source img.jpg`.

Related pages: [Quickstart](quickstart.md), [CLI commands](../reference/cli-commands.md), [Cloud upload](../guides/cloud-upload.md).
