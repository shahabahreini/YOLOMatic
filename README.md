<div align="center">

<img src="docs/assets/logo.png" alt="YOLOmatic logo" width="520">

# YOLOmatic

**Interactive computer-vision training for YOLO, RF-DETR, SAM 3.1, and Detectron2.**

[![PyPI](https://img.shields.io/pypi/v/yolomatic?style=flat-square)](https://pypi.org/project/yolomatic/)
[![Python](https://img.shields.io/badge/python-3.12-blue?style=flat-square)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square)](LICENSE.md)
[![CI](https://img.shields.io/github/actions/workflow/status/shahabahreini/YOLOMatic/ci.yml?branch=main&style=flat-square&label=ci)](https://github.com/shahabahreini/YOLOMatic/actions)
[![Stars](https://img.shields.io/github/stars/shahabahreini/YOLOMatic?style=flat-square)](https://github.com/shahabahreini/YOLOMatic/stargazers)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-teal?style=flat-square)](https://shahabahreini.github.io/YOLOMatic/)
[![Discussions](https://img.shields.io/badge/discussions-open-orange?style=flat-square)](https://github.com/shahabahreini/YOLOMatic/discussions)

![YOLOmatic terminal wizard](docs/assets/demo.gif)

</div>

YOLOmatic is a production-focused Python CLI/TUI for configuring, training,
fine-tuning, predicting, benchmarking, augmenting, converting, monitoring, and
uploading computer-vision models. It covers YOLO26/12/11/10/9/8, YOLOX,
RF-DETR, SAM 3.1, and Detectron2 from one terminal workflow.

## Table of Contents

- [Why YOLOmatic](#why-yolomatic)
- [30-Second Quickstart](#30-second-quickstart)
- [Feature Matrix](#feature-matrix)
- [FAQ](#faq)
- [Documentation](#links)
- [Versioning](#versioning)

## Why YOLOmatic

- **Interactive wizard UX:** configure models, datasets, fine-tuning, prediction,
  benchmarking, augmentation, NDJSON conversion, TensorBoard, and upload flows
  without hand-writing boilerplate.
- **Hardware-aware configs:** CUDA, Apple Silicon MPS, CPU fallback, workers,
  batch sizes, and runtime preflight checks are handled before training starts.
- **10+ model families in one CLI:** Ultralytics YOLO, native RF-DETR, SAM 3.1,
  Detectron2, Roboflow upload, ClearML tracking, Labelbox conversion, and local
  benchmark reports.

## 30-Second Quickstart

```sh
uv tool install --python 3.12 yolomatic
yolomatic
```

Repository development:

```sh
git clone https://github.com/shahabahreini/YOLOMatic.git
cd YOLOMatic
uv sync
uv run yolomatic
```

Common commands:

```sh
uv run yolomatic-train
uv run yolomatic-predict
uv run yolomatic-benchmark
uv run yolomatic-convert
uv run yolomatic-upload
uv run yolomatic-tensorboard
```

## Versioning

Use the built-in semantic version command before creating a release tag:

```sh
uv run bump patch  # or: minor / major
git tag vX.Y.Z
git push origin vX.Y.Z
```

Release tags must point to a commit already on `main`; GitHub Actions verifies
the tag and package version before publishing the wheel and source archive.

## Feature Matrix

| Area           | YOLOmatic support                                                                     |
| -------------- | ------------------------------------------------------------------------------------- |
| Model families | YOLO26, YOLOv12, YOLO11, YOLOv10, YOLOv9, YOLOv8, YOLOX, RF-DETR, SAM 3.1, Detectron2 |
| Tasks          | Detection, segmentation, classification, pose, OBB, open-vocabulary segmentation      |
| Data           | YOLO folders, COCO JSON, Labelbox NDJSON, Ultralytics-platform NDJSON                 |
| Training       | Fresh training, checkpoint fine-tuning, resume, hardware preflight                    |
| Evaluation     | mAP, F1, per-image rankings, UMAP vector analysis, interactive HTML report            |
| Integrations   | Roboflow, ClearML, TensorBoard, HuggingFace, Ultralytics                              |

## Comparison

| Capability                             | YOLOmatic | Ultralytics CLI | Roboflow              | Hosted YOLO tools |
| -------------------------------------- | --------- | --------------- | --------------------- | ----------------- |
| Interactive terminal wizard            | Yes       | Limited         | Web UI                | Web UI            |
| Multiple YOLO generations              | Yes       | Yes             | Export/deploy focused | Varies            |
| Native RF-DETR training                | Yes       | No              | Deploy focused        | Varies            |
| SAM 3.1 workflows                      | Yes       | No              | Limited               | Varies            |
| Detectron2 training                    | Yes       | No              | No                    | Varies            |
| Hardware-aware config generation       | Yes       | Manual          | Abstracted            | Abstracted        |
| Labelbox/Ultralytics NDJSON conversion | Yes       | No              | Import focused        | Varies            |
| Local benchmark HTML reports           | Yes       | Partial metrics | Hosted analytics      | Varies            |
| Fully local workflow                   | Yes       | Yes             | No                    | No                |

## FAQ

### What is YOLOmatic?

YOLOmatic is a Python 3.12 command-line toolkit with an interactive terminal UI
for computer-vision training workflows.

### Which model families are supported?

YOLO26, YOLOv12, YOLO11, YOLOv10, YOLOv9, YOLOv8, YOLOX, RF-DETR, SAM 3.1, and
Detectron2.

### Is YOLOmatic only for YOLO?

No. YOLOmatic also supports native RF-DETR training and fine-tuning, SAM 3.1
segmentation workflows, and Detectron2 training.

### Does it require a GPU?

No. CUDA is recommended for training, but CPU and Apple Silicon MPS fallbacks
are supported. YOLOmatic detects common CUDA/PyTorch mismatches and offers repair
guidance.

### Can YOLOmatic convert Labelbox NDJSON?

Yes. It converts Labelbox and Ultralytics-platform NDJSON exports into YOLO or
COCO datasets with concurrent image downloads. Ultralytics pose exports can be
converted explicitly to YOLO Pose or COCO Pose while preserving keypoints.

### Can I benchmark trained models?

Yes. `uv run yolomatic-benchmark` compares Ultralytics YOLO checkpoints and
exports such as `.pt`, ONNX, TensorRT `.engine`, TorchScript, and OpenVINO,
then verifies model/dataset task compatibility and evaluates train, valid, test,
or all groups. Detection and instance segmentation report mAP/F1, pose reports
box metrics, and semantic segmentation reports mIoU, pixel accuracy, and Dice.

### Can I export models for deployment?

Yes. `uv run yolomatic-export` exports trained YOLO `.pt` weights to TensorRT,
ONNX, OpenVINO, CoreML, TFLite, TorchScript, NCNN, RKNN, and other Ultralytics
formats. TensorRT workspace is user-controlled so it can be tuned for the GPU
that builds the engine. The export wizard reads the selected checkpoint details
first, then hides options that do not apply to that model task or export format.

### Can I upload models to Roboflow?

Yes. YOLOmatic uploads YOLO checkpoints and deploys RF-DETR checkpoints through
the upload TUI or optional post-training upload config.

## Screenshots

| Configure Model                                                               | Smart Split                                                           |
| ----------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| ![Configure model wizard](docs/assets/screenshots/wizard-configure-model.png) | ![Smart split wizard](docs/assets/screenshots/wizard-smart-split.png) |

## Links

- [Full documentation](https://shahabahreini.github.io/YOLOMatic/)
- [Install guide](https://shahabahreini.github.io/YOLOMatic/getting-started/install/)
- [Models guide](https://shahabahreini.github.io/YOLOMatic/guides/models/)
- [YOLO guide](https://shahabahreini.github.io/YOLOMatic/guides/yolo/)
- [Dataset guide](https://shahabahreini.github.io/YOLOMatic/guides/datasets/)
- [Comparison](https://shahabahreini.github.io/YOLOMatic/comparison/)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [Citation](CITATION.bib)

## Citation

```bibtex
@software{bahreini_jangjoo_yolomatic_2026,
  author = {Bahreini Jangjoo, Shahab},
  title = {{YOLOmatic}: automated computer-vision training CLI for YOLO, RF-DETR, SAM, and Detectron2},
  year = {2026},
  version = {4.4.0},
  license = {Apache-2.0},
  url = {https://github.com/shahabahreini/YOLOMatic},
  note = {Python CLI/TUI for configuring, training, fine-tuning, predicting, benchmarking, and uploading computer-vision models}
}
```

## License

YOLOmatic is licensed under [Apache 2.0](LICENSE.md).

## Acknowledgments

YOLOmatic builds on the Python computer-vision ecosystem, including Ultralytics,
RF-DETR, SAM/HuggingFace, Detectron2, PyTorch, Albumentations, ClearML,
TensorBoard, Roboflow, and MkDocs Material.
