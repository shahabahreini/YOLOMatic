---
description: Install YOLOmatic with uv, set up a development environment, and prepare optional Roboflow credentials.
---

# Install

YOLOmatic targets Python 3.12.

## Package Install

```sh
uv tool install --python 3.12 yolomatic
yolomatic
```

## Repository Install

```sh
git clone https://github.com/shahabahreini/YOLOMatic.git
cd YOLOMatic
uv sync
uv run yolomatic
```

## Optional Roboflow Credentials

```sh
cp .env.example .env
```

Fill in `ROBOFLOW_API_KEY`, `ROBOFLOW_WORKSPACE`, and
`ROBOFLOW_PROJECT_IDS` when you want upload or deployment workflows.

## Platform Notes

- On Windows, keep the explicit `--python 3.12` flag for tool installs. Some
  ONNX dependencies do not publish Python 3.13 Windows wheels and will otherwise
  fall back to a CMake source build.
- Linux and Windows can use CUDA when the installed PyTorch build supports it.
- macOS uses CPU or Apple Silicon MPS. NVIDIA CUDA is not available on macOS.
- YOLOmatic detects a CPU-only PyTorch build on GPU machines and offers repair
  guidance from the TUI.

Related pages: [Quickstart](quickstart.md), [CLI commands](../reference/cli-commands.md), [Cloud upload](../guides/cloud-upload.md).
