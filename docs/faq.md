---
description: Comprehensive YOLOmatic FAQ covering installation, model families, datasets, training, fine-tuning, prediction, benchmarking, integrations, troubleshooting, security, and citation.
---

# FAQ

## Product Fit

### What is YOLOmatic?

YOLOmatic is a Python 3.12 CLI/TUI for configuring, training, fine-tuning,
predicting, benchmarking, augmenting, converting, monitoring, and uploading
computer-vision models from a local terminal workflow.

### Who is YOLOmatic for?

It is for practitioners who want repeatable local training workflows, researchers
who need citable and reproducible experiments, and teams that want one CLI for
dataset preparation, model configuration, training, evaluation, and deployment
handoff.

### Is YOLOmatic a hosted training platform?

No. YOLOmatic runs locally in your Python environment. It can integrate with
hosted services such as Roboflow, ClearML, HuggingFace, and TensorBoard, but the
core workflows are local and scriptable.

### Is YOLOmatic only a wrapper around Ultralytics?

No. YOLOmatic uses Ultralytics for YOLO-family training, but it also includes
native RF-DETR routing, SAM 3.1 inference and fine-tuning workflows, Detectron2
training support, dataset conversion tools, augmentation, benchmarking, and
cloud upload helpers.

### When should I choose YOLOmatic instead of the raw Ultralytics CLI?

Choose YOLOmatic when you want guided configuration, model-family routing,
hardware-aware defaults, dataset conversion, smart splitting, benchmark reports,
Roboflow upload, ClearML/TensorBoard wiring, and a repeatable terminal UX around
the training lifecycle.

## Installation And Environment

### What Python version does YOLOmatic require?

YOLOmatic targets Python `>=3.12,<3.13`.

### What is the fastest install path?

For end users:

```sh
uv tool install yolomatic
yolomatic
```

For repository development:

```sh
git clone https://github.com/shahabahreini/YOLOMatic.git
cd YOLOMatic
uv sync
uv run yolomatic
```

### Can I use pip instead of uv?

The project metadata is standard Python packaging metadata, but the documented
and tested workflow uses `uv`. `uv` is preferred because the dependency set is
large and includes ML packages that benefit from reproducible locking.

### Does YOLOmatic support Windows?

Yes. Windows is supported, including CUDA-capable systems. For manual PyTorch
CUDA repairs on Windows, prefer `.venv\Scripts\python.exe -m pip ...` instead of
`uv run pip ...` so `uv` does not resync the environment back to the locked
state.

### Does YOLOmatic support macOS?

Yes. macOS can use CPU or Apple Silicon MPS when PyTorch supports it. NVIDIA
CUDA is not available on macOS.

### Does YOLOmatic support Linux?

Yes. Linux is the most common target for CUDA training. YOLOmatic includes
preflight handling for CUDA/cuDNN runtime library paths when the active `.venv`
contains the needed NVIDIA runtime packages.

### Does YOLOmatic require a GPU?

No. A CUDA GPU is strongly recommended for training, but CPU and Apple Silicon
MPS fallbacks are supported. YOLOmatic detects common CUDA/PyTorch mismatches and
offers repair guidance before training starts.

### What happens if PyTorch is CPU-only on a GPU machine?

YOLOmatic detects the mismatch when a GPU is visible but `torch.cuda.is_available()`
is false. The TUI can guide you toward a repair path, CPU fallback, or
cancellation.

## Model Families

### Which model families does YOLOmatic support?

YOLO26, YOLOv12, YOLO11, YOLOv10, YOLOv9, YOLOv8, YOLOX, RF-DETR, SAM 3.1, SAM
3, and Detectron2.

### Which tasks are supported?

Supported tasks include object detection, instance segmentation, image
classification, pose estimation, oriented object detection, open-vocabulary
segmentation, and mask fine-tuning where the selected model family supports the
task.

### Does YOLOmatic support YOLO26?

Yes. YOLO26 detection and segmentation variants are exposed through the YOLO
configuration flow, with model-size choices such as nano, small, medium, large,
and xlarge where supported.

### Does YOLOmatic support YOLOv8, YOLOv9, YOLOv10, YOLO11, and YOLOv12?

Yes. YOLOmatic supports those YOLO generations through the Ultralytics training
route, including detection and the additional task heads supported by each
family.

### Does YOLOmatic support YOLOX?

Yes. YOLOX detection variants are included in the supported model metadata and
configuration workflow.

### Does YOLOmatic support RF-DETR?

Yes. RF-DETR detection and segmentation variants route to the native RF-DETR
trainer and use `.pth` checkpoints instead of Ultralytics `.pt` checkpoints.

### Can YOLOmatic fine-tune RF-DETR?

Yes. YOLOmatic discovers RF-DETR `.pth` checkpoints and writes fine-tuning
configs that pass the checkpoint as RF-DETR pretraining or resume input as
appropriate.

### Does YOLOmatic support SAM 3.1?

Yes. YOLOmatic supports SAM 3 and SAM 3.1 inference workflows, including auto,
text-prompted, and box-prompted segmentation. It also supports SAM 3.1
fine-tuning on COCO-format mask datasets.

### Does YOLOmatic support Detectron2?

Yes. YOLOmatic supports Detectron2 training flows for Faster R-CNN, RetinaNet,
and Mask R-CNN style configurations.

### Does YOLOmatic still support YOLO-NAS?

No. YOLO-NAS training is deprecated in this build because SuperGradients
conflicts with the modern RF-DETR dependency stack.

### Which model should I start with?

For a first smoke test, start with a small model such as `YOLO26n`, `YOLO11n`,
or another nano/small variant. For transformer detection experiments, start with
RF-DETR Nano or Small before moving to larger variants.

## Datasets And Conversion

### What dataset formats does YOLOmatic understand?

YOLOmatic works with YOLO folder datasets, COCO JSON annotations, Labelbox
NDJSON exports, and Ultralytics-platform NDJSON exports.

### What YOLO dataset layout is expected?

A standard layout is:

```text
datasets/name/
  data.yaml
  train/images/
  train/labels/
  valid/images/
  valid/labels/
  test/images/
  test/labels/
```

`data.yaml` should define `train`, `val` or `valid`, optional `test`, `nc`, and
`names`.

### Can YOLOmatic handle flat YOLO datasets?

Yes. Recent dataset summary and parsing logic includes a fallback for flat
datasets that have root-level `images/` and `labels/` folders.

### Can YOLOmatic convert Labelbox NDJSON?

Yes. The NDJSON converter supports Labelbox exports and can emit YOLO labels or
COCO JSON annotations with concurrent image downloads.

### Can YOLOmatic convert Ultralytics-platform NDJSON?

Yes. YOLOmatic auto-detects Ultralytics-platform image rows and reads normalized
`annotations.segments` and `annotations.boxes` into YOLO or COCO output.

### Can YOLOmatic convert polygons and boxes?

Yes. The converter handles bounding boxes and polygons where the source export
contains them. Polygon data can be preserved in segmentation-oriented outputs.

### Can YOLOmatic prepare COCO datasets?

Yes. COCO JSON is used for Detectron2, SAM 3.1 fine-tuning, and benchmarking.
The NDJSON converter can emit COCO annotations when that output format is
selected.

### Can YOLOmatic split datasets?

Yes. The dataset preparation wizard supports random, class-balanced, and
smart-balanced splitting strategies.

### What is smart-balanced splitting?

Smart-balanced splitting preserves rare classes by seeding scarce class examples
first, then filling each split with deterministic random tie-breaking and class
balance pressure.

### What happens if every label file is empty?

YOLOmatic warns clearly that class balancing cannot be meaningful when all label
files are empty. It can still split records, but the result cannot preserve class
coverage.

### Can YOLOmatic augment datasets offline?

Yes. YOLOmatic includes Albumentations-powered offline augmentation with reusable
profiles, more than 20 transforms, split redistribution, and YOLO/COCO output
support.

### Can YOLOmatic combine datasets?

Yes. The dataset combiner can merge multiple YOLO-format datasets, preserve or
remap class names, and hard-link images where possible.

## Configuration And Training

### How do I generate a training config?

Run:

```sh
uv run yolomatic
```

Choose **Configure Model**, select a family and variant, select a dataset, review
hardware-aware settings, and save the generated YAML under `configs/`.

### How do I fine-tune an existing checkpoint?

Use **Configure Fine-Tune** from the main TUI. YOLOmatic discovers compatible
checkpoints, lets you bind one to a dataset, and writes a fresh fine-tuning YAML.

### Which checkpoint formats are supported?

YOLO-family workflows use `.pt`, RF-DETR uses `.pth`, Detectron2 commonly uses
`.pth`, and SAM workflows can use HuggingFace model identifiers or local
artifacts.

### How does YOLOmatic choose the trainer?

The smart training router reads the saved config and dispatches to the correct
trainer: Ultralytics YOLO, native RF-DETR, SAM 3.1, or Detectron2.

### Does YOLOmatic download pretrained weights automatically?

Fresh configs use official pretrained weights where the underlying trainer
supports automatic download. Local checkpoints are used for fine-tuning and
resume workflows.

### Can I resume training?

Yes. Resume behavior depends on the selected trainer and checkpoint type.
YOLOmatic distinguishes fresh training, fine-tuning, and resume paths in the
generated configuration.

### Can YOLOmatic train without ClearML?

Yes. ClearML is optional. If ClearML is not configured, training can continue
without it after an interactive prompt.

### Can YOLOmatic launch TensorBoard?

Yes. `uv run yolomatic-tensorboard` scans discovered training runs and starts
TensorBoard without requiring you to manually locate log directories.

### Does YOLOmatic export models?

YOLOmatic supports export-oriented training flows where the underlying trainer
supports export, including ONNX-related preflight handling for Ultralytics
workflows.

## Prediction And Segmentation

### Can YOLOmatic run prediction on one image?

Yes. `uv run yolomatic-predict` supports single-image prediction from discovered
weights or explicit command arguments.

### Can YOLOmatic run batch prediction on a folder?

Yes. Folder prediction is supported, including progress display and worker-based
parallelism where applicable.

### Can YOLOmatic discover trained weights automatically?

Yes. Prediction workflows scan the project tree for compatible weights so you do
not have to paste full checkpoint paths manually.

### Can YOLOmatic run SAM segmentation from YOLO boxes?

Yes. SAM 3.1 box-prompted mode can use YOLO detections as prompts for mask
generation.

### Can YOLOmatic do text-prompted segmentation?

Yes. SAM 3.1 text-prompted segmentation is available through the SAM inference
workflow.

## Benchmarking And Reports

### Can YOLOmatic benchmark trained models?

Yes. `uv run yolomatic-benchmark` evaluates checkpoints on COCO validation data
and generates an interactive HTML report.

### Which benchmark metrics are included?

Reports include mAP, F1, per-image rankings, prediction confidence inspection,
parallel thumbnails, and UMAP vector scatter plots.

### Which checkpoints can be benchmarked?

Benchmarking currently targets Ultralytics `.pt` checkpoints. Detection and
segmentation task type is auto-detected at runtime.

### Can RF-DETR, SAM, or Detectron2 checkpoints be benchmarked?

Not through the current benchmark engine. Use YOLO `.pt` weights for the
benchmark report workflow; RF-DETR, SAM, and Detectron2 have their own training
or inference paths.

### What annotation format does benchmarking require?

Benchmark validation uses COCO JSON annotations, commonly named
`_annotations.coco.json`.

## Integrations

### Can YOLOmatic upload trained models to Roboflow?

Yes. YOLOmatic uploads YOLO checkpoints and deploys RF-DETR checkpoints through
the upload TUI.

### Can YOLOmatic upload automatically after training?

Yes. Add a `roboflow` block to the training YAML with upload enabled and a target
weight such as `best.pt`.

### Which Roboflow credentials are needed?

Use `.env` values such as `ROBOFLOW_API_KEY`, `ROBOFLOW_WORKSPACE`, and
`ROBOFLOW_PROJECT_IDS`. The workspace value should be the workspace slug.

### Does YOLOmatic support ClearML?

Yes. ClearML can track hyperparameters, metrics, and artifacts. It is optional,
and training can continue without ClearML when it is not configured.

### Does YOLOmatic support HuggingFace?

Yes. SAM 3.1 workflows use HuggingFace model access. Some SAM checkpoints may
require HuggingFace authentication.

### Does YOLOmatic support Roboflow deployment for RF-DETR?

Yes. RF-DETR checkpoint deployment is available through the Roboflow upload
workflow.

## Files, Secrets, And Project Hygiene

### Where should secrets go?

Put secrets in `.env` or your shell environment. Do not commit Roboflow API keys,
HuggingFace tokens, ClearML credentials, datasets, trained weights, or generated
run artifacts.

### Does YOLOmatic vendor datasets or model weights?

No. The repository should not contain datasets, trained weights, or run
artifacts. Workflows discover local project files at runtime.

### Where are generated configs stored?

Generated training configs are stored under `configs/` unless you choose a
different path in the workflow.

### Where are docs for model benchmarks and architecture details?

Use the [models guide](guides/models.md) for detailed supported model families,
task support, benchmark tables, and model-selection recommendations.

## Troubleshooting

### Why does `uv run pip install torch ...` not fix my CUDA install?

`uv run pip ...` can trigger environment synchronization and restore the locked
dependency state. On Windows especially, use the environment Python directly for
manual CUDA repairs, such as `.venv\Scripts\python.exe -m pip ...`.

### Why does `nvidia-smi` show a GPU while PyTorch says CUDA is unavailable?

`nvidia-smi` only proves the driver can see the GPU. PyTorch also needs a CUDA
enabled build and compatible runtime libraries. YOLOmatic checks this mismatch
before training.

### Why does my validation key fail when `data.yaml` uses `valid`?

YOLOmatic accepts `val` or `valid` in dataset configs. Recent fixes also avoid
hard failures when one validation key is missing.

### Why are my converted labels empty?

Check whether the source NDJSON uses Labelbox-style rows or
Ultralytics-platform `type: image` rows. YOLOmatic now auto-detects both, but an
export without annotation objects will still produce empty labels.

### Why does class-balanced splitting warn about empty labels?

Class-balanced and smart-balanced splitting need class labels to preserve class
coverage. Empty label files can be split as images, but they cannot contribute to
class-balance decisions.

### Why is training slow on CPU?

Modern detection and segmentation training is compute-heavy. Use a small model
for smoke tests, reduce image size or batch size, and switch to CUDA when
available.

## Community, Citation, And Licensing

### How do I contribute?

Read the [contributing guide](contributing.md), run `uv sync`, add or update
tests for behavior changes, update relevant docs, and open a pull request using
the repository template.

### How do I report a security issue?

Do not open a public issue. Follow the private disclosure process in
[SECURITY.md](https://github.com/shahabahreini/YOLOMatic/blob/main/SECURITY.md).

### What license does YOLOmatic use?

YOLOmatic is licensed under Apache 2.0.

### How should I cite YOLOmatic?

Use [CITATION.cff](https://github.com/shahabahreini/YOLOMatic/blob/main/CITATION.cff)
for GitHub citation metadata or
[CITATION.bib](https://github.com/shahabahreini/YOLOMatic/blob/main/CITATION.bib)
for BibTeX.
