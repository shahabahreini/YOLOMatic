---
description: Convert Labelbox and Ultralytics-platform NDJSON exports into YOLO, COCO, and pose datasets with YOLOmatic.
---

# NDJSON Conversion

YOLOmatic converts Labelbox and Ultralytics-platform NDJSON exports to YOLO or
COCO datasets, including pose datasets exported by Ultralytics Platform.

```sh
uv run yolomatic-convert
```

The converter supports:

- Bounding boxes, polygons, and Ultralytics pose annotations
- Concurrent image downloads
- YOLO detection, segmentation, and pose `.txt` label output
- COCO instances and COCO pose JSON output
- Auto-detection for Labelbox-style and Ultralytics-platform rows

For Ultralytics-platform NDJSON, YOLOmatic reads normalized
`annotations.segments`, `annotations.boxes`, and `annotations.pose` values from
`type: image` rows. Pose conversion preserves `train`, `val`, and `test` splits,
uses `kpt_shape` and `flip_idx` from the dataset header, and infers the keypoint
shape when possible if it is omitted.

Choose **YOLO Pose** for Ultralytics pose labels and a pose-aware `data.yaml`.
Choose **COCO Pose** for split-specific `annotations/instances_{split}.json`
files containing `keypoints` and `num_keypoints`. Labelbox pose-object grouping
is not currently supported.

Related pages: [Datasets](../guides/datasets.md), [Smart split](smart-split.md), [CLI commands](cli-commands.md).
