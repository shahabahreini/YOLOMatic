---
description: Prepare YOLO, COCO, Labelbox, and Ultralytics-platform datasets for YOLOmatic training and benchmarking.
---

# Datasets

YOLOmatic works with standard YOLO folders, COCO JSON annotations, Labelbox
NDJSON exports, and Ultralytics-platform NDJSON exports.

## YOLO Layout

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

`data.yaml` should define `train`, `val` or `valid`, `test`, `nc`, and `names`.
Flat `images/` plus `labels/` datasets are also recognized for summary and
conversion workflows.

## COCO Layout

Use COCO JSON for Detectron2, SAM 3.1 fine-tuning, and benchmark validation:

```text
datasets/name/
  train/
    images/
    _annotations.coco.json
  valid/
    images/
    _annotations.coco.json
```

## NDJSON Conversion

Run:

```sh
uv run yolomatic-convert
```

The converter can download referenced images concurrently and emit YOLO or COCO
outputs. See [NDJSON conversion](../reference/ndjson-conversion.md).

## Splitting

The dataset preparation wizard supports random, class-balanced, and
smart-balanced splitting. Smart-balanced splitting seeds rare classes first, then
fills remaining images with deterministic random tie-breaking.

Related pages: [Smart split](../reference/smart-split.md), [First training run](../getting-started/first-training-run.md), [Benchmarking](benchmarking.md).
