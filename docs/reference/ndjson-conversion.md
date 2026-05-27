---
description: Convert Labelbox and Ultralytics-platform NDJSON exports into YOLO or COCO datasets with YOLOmatic.
---

# NDJSON Conversion

YOLOmatic converts Labelbox and Ultralytics-platform NDJSON exports to YOLO or
COCO datasets.

```sh
uv run yolomatic-convert
```

The converter supports:

- Bounding boxes and polygons
- Concurrent image downloads
- YOLO `.txt` label output
- COCO `_annotations.coco.json` output
- Auto-detection for Labelbox-style and Ultralytics-platform rows

For Ultralytics-platform NDJSON, YOLOmatic reads normalized
`annotations.segments` and `annotations.boxes` from `type: image` rows.

Related pages: [Datasets](../guides/datasets.md), [Smart split](smart-split.md), [CLI commands](cli-commands.md).
