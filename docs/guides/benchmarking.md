---
description: Benchmark YOLOmatic checkpoints with mAP, F1, per-image rankings, UMAP vectors, and interactive HTML reports.
---

# Benchmarking

YOLOmatic benchmarks trained checkpoints against COCO validation annotations and
generates an interactive HTML report.

```sh
uv run yolomatic-benchmark
```

Reports include:

- mAP and F1 summaries
- Per-class and per-image rankings
- Prediction confidence inspection
- UMAP vector scatter plots
- Parallel thumbnail generation

Benchmarking currently targets Ultralytics `.pt` checkpoints. Detection and
segmentation tasks are detected at runtime.

Related pages: [Models](models.md), [Datasets](datasets.md), [CLI commands](../reference/cli-commands.md).
