---
description: Prepare YOLO, COCO, Labelbox, and Ultralytics-platform datasets for YOLOmatic training and benchmarking — layouts, data.yaml format, splitting, and combining.
---

# Datasets

YOLOmatic works with standard YOLO folders, COCO JSON annotations, Labelbox NDJSON exports, and Ultralytics-platform NDJSON exports.

---

## YOLO Layout

The standard YOLO dataset layout:

```text
datasets/my_dataset/
  data.yaml
  train/
    images/    ← JPG, PNG training images
    labels/    ← .txt files, one per image
  valid/
    images/
    labels/
  test/        ← optional
    images/
    labels/
```

### `data.yaml` Format

```yaml
# data.yaml
path: datasets/my_dataset     # optional absolute path
train: train/images
val: valid/images              # "val" or "valid" both accepted
test: test/images              # optional

nc: 3                          # number of classes
names:
  0: cat
  1: dog
  2: person
```

YOLOmatic accepts both `val` and `valid` as the validation key. If both are absent, a warning is shown but training may still proceed.

### YOLO Label Format

Each `.txt` label file contains one detection per line:

```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized to [0, 1] relative to image dimensions. For segmentation:

```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

For keypoint/pose estimation:

```
<class_id> <x_center> <y_center> <width> <height> <px1> <py1> <pv1> ... <pxn> <pyn> <pvn>
```

Where `<pxn> <pyn>` are keypoint coordinates (normalized to `[0, 1]`) and `<pvn>` is keypoint visibility (typically `0` for unlabeled, `1` for labeled but occluded, or `2` for labeled and visible). The `data.yaml` must also include `kpt_shape` (e.g. `kpt_shape: [17, 3]`).

### Flat Dataset Layout

YOLOmatic also recognizes a flat layout where `images/` and `labels/` are at the root:

```text
datasets/my_dataset/
  images/
  labels/
  data.yaml
```

Flat datasets are recognized for summary and conversion workflows.

---

## COCO Layout

Use COCO JSON for Detectron2 training, SAM 3.1 fine-tuning, and benchmark validation:

```text
datasets/my_dataset/
  train/
    images/
    _annotations.coco.json
  valid/
    images/
    _annotations.coco.json
  test/              ← optional
    images/
    _annotations.coco.json
```

### COCO JSON Structure

A valid `_annotations.coco.json` must include:

```json
{
  "images": [
    {"id": 1, "file_name": "img001.jpg", "width": 640, "height": 480}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "segmentation": [[x1, y1, x2, y2, ...]],
      "area": 1200,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "cat", "supercategory": "animal"}
  ]
}
```

Roboflow's **COCO export** preset generates this structure directly. The benchmark engine (`yolomatic-benchmark`) also requires this format.

---

## NDJSON Conversion

Convert Labelbox or Ultralytics-platform NDJSON exports to YOLO or COCO:

```sh
uv run yolomatic-convert
```

The converter:

- Auto-detects Labelbox-style rows vs Ultralytics-platform `type: image` rows
- Supports bounding boxes, polygon annotations, and Ultralytics pose annotations
- Downloads images concurrently using a thread pool
- Emits YOLO detection/segmentation/pose labels or COCO/COCO Pose JSON

See [NDJSON Conversion](../reference/ndjson-conversion.md) for details.

---

## Dataset Splitting

Split an existing dataset into train/val/test subsets:

```sh
uv run yolomatic-prepare
```

The wizard supports three strategies:

| Strategy | Description |
|---|---|
| **Random** | Uniform random assignment with a configurable seed |
| **Class-balanced** | Balances class distribution across splits |
| **Smart-balanced** | Seeds splits with rare classes first to prevent rare classes from vanishing in validation/test sets |

See [Smart Split](../reference/smart-split.md) for the algorithm details.

---

## Dataset Combining

YOLOmatic can merge multiple YOLO-format datasets into one:

- Class names are preserved or remapped interactively
- Images are hard-linked where possible (no duplication on same filesystem)
- Output follows the standard YOLO layout with a fresh `data.yaml`

Access combining via **Prepare Dataset → Combine Datasets** in the TUI.

---

## Offline Augmentation

After your dataset is prepared, you can expand it offline using Albumentations-powered augmentation profiles:

```sh
uv run yolomatic
```

Choose **Augment Dataset**. See the [Augmentation guide](augmentation.md) for the full workflow including transform categories, multiplier, split redistribution, and output formats.

---

## Downloading Datasets from Ultralytics Platform

```sh
uv run yolomatic-ultralytics
```

This wizard downloads datasets published on the Ultralytics Platform, converts the NDJSON export to YOLO format, and places the result in the configured output directory (default: `datasets/ultralytics/downloads`).

---

## Dataset Summary Caching

To optimize execution speed across repeated training configuration and dataset validation steps, YOLOmatic caching is applied:
- **Location:** Cached summaries are written to `datasets/.yolomatic_cache/summaries/`.
- **Mechanism:** YOLOmatic computes a unique signature based on directory contents, structure, and sizes.
- **Benefits:** Eliminates expensive folder scanning and disk I/O when analyzing or configuring the same dataset multiple times. If any files or contents change, a new signature is generated and the cache updates automatically.

Related pages: [Smart split](../reference/smart-split.md), [NDJSON Conversion](../reference/ndjson-conversion.md), [Augmentation](augmentation.md), [First training run](../getting-started/first-training-run.md).
