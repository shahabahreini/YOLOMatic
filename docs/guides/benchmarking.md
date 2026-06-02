---
description: Benchmark YOLOmatic checkpoints with mAP, F1, per-image rankings, UMAP vectors, and interactive HTML reports.
---

# Benchmarking

YOLOmatic benchmarks trained checkpoints against COCO-format validation annotations and generates an interactive HTML report with multiple analysis views.

```sh
uv run yolomatic-benchmark
```

---

## Requirements

Before running the benchmark wizard you need:

1. **A trained, downloaded, or exported Ultralytics model artifact** — `best.pt`, `last.pt`, ONNX, TensorRT `.engine`, TorchScript, OpenVINO, or another Ultralytics YOLO export stored under the project root, `runs/`, or `weights/`
2. **A COCO-format validation set** — a folder containing:
   - `images/` with the validation images
   - `_annotations.coco.json` with COCO-format bounding box or mask annotations

COCO JSON validation sets can be exported from Roboflow using the COCO export preset, or generated from Labelbox NDJSON using `yolomatic-convert`.

!!! note "RF-DETR, SAM, and Detectron2 are not supported"
    The benchmark engine currently targets Ultralytics YOLO checkpoints and exports that can be loaded by `ultralytics.YOLO`. RF-DETR, SAM 3.1, and Detectron2 use their own evaluation paths.

---

## Step-by-Step Walkthrough

### 1. Launch the wizard

```sh
uv run yolomatic-benchmark
```

### 2. Select a checkpoint

The wizard scans benchmark-compatible Ultralytics model artifacts in the project root, `runs/`, and `weights/`, including downloaded Platform weights under `weights/ultralytics/` and exported models such as `.onnx`, `.engine`, `.torchscript`, `.tflite`, `.mlpackage`, `.pb`, `.mnn`, `.rknn`, OpenVINO, NCNN, and other Ultralytics export directories. Choose `best.pt` for production-quality checkpoint results, or compare exported formats to measure deployment-runtime performance.

Exported artifacts are benchmarked with a single-image batch by default because many ONNX, TensorRT, OpenVINO, and mobile exports are fixed to batch size 1. Native `.pt` checkpoints keep the benchmark batch-size setting.

### 3. Locate the validation set

Select the validation folder containing `_annotations.coco.json`. The task type (detection or segmentation) is auto-detected from the checkpoint.

### 4. Configure report options

Choose whether to include:

- UMAP vector analysis (slower; reveals embedding structure)
- Thumbnail gallery (shows example detections per class)
- Full per-image ranking (lists every image by score)

### 5. Wait for evaluation

The benchmark engine runs inference on every validation image, collects predictions, computes IoU matches, and calculates per-class and aggregate metrics.

### 6. Open the HTML report

When complete, an interactive HTML report is opened automatically (or its path is printed). The report is self-contained — no server required.

---

## Report Anatomy

### Summary Section

The top of the report shows aggregate metrics:

| Metric | Description |
|---|---|
| **mAP 50** | Mean Average Precision at IoU threshold 0.50 |
| **mAP 50-95** | Mean Average Precision averaged over IoU 0.50–0.95 in 0.05 steps |
| **Precision** | Fraction of detections that are correct |
| **Recall** | Fraction of ground-truth objects that are detected |
| **F1 Score** | Harmonic mean of Precision and Recall |

### Per-Class Breakdown

A table lists mAP, Precision, Recall, and F1 for each class individually — useful for identifying which classes are underperforming.

### Per-Image Rankings

Images are sorted by score (best to worst). This ranking helps you quickly locate the hardest examples in your validation set — images where the model struggles most. These are candidates for review, annotation improvement, or targeted augmentation.

### Confidence Inspection

A chart shows how Precision and Recall vary across confidence thresholds. Use this to select an optimal deployment threshold — the confidence score that balances false positives and missed detections for your application.

### Thumbnail Gallery

Side-by-side thumbnail grids show ground-truth annotations versus model predictions for sampled images per class. A quick visual sanity check for detection quality.

### UMAP Vector Scatter Plot

UMAP embeds the model's feature representations of each image into 2D space. Clusters in the UMAP plot indicate images the model treats as visually similar. Outliers often correspond to unusual or hard examples. This view is most useful for:

- Detecting dataset quality issues (mislabeled images tend to be isolated)
- Understanding which images the model considers similar

---

## Interpreting Results

| Symptom | Likely Cause |
|---|---|
| Low recall on a specific class | Too few training examples for that class; consider augmentation or more data |
| Low precision across all classes | Confidence threshold is too low; check confidence inspection chart |
| High mAP but poor per-image rankings | A few hard images are dragging down easy ones; inspect the bottom of the ranking |
| UMAP has tight clusters but bad mAP | Feature representations are good but calibration is off; tune confidence threshold |

---

## Preparing a COCO Validation Set

If your dataset is in YOLO format, convert a validation split to COCO:

```sh
uv run yolomatic-convert
```

Select the validation folder and choose COCO JSON as the output format. The converter emits `_annotations.coco.json` compatible with the benchmark engine.

Alternatively, export directly from Roboflow using **Format → COCO** and select the validation version.

Related pages: [Models](models.md), [Datasets](datasets.md), [CLI Commands](../reference/cli-commands.md), [NDJSON Conversion](../reference/ndjson-conversion.md).
