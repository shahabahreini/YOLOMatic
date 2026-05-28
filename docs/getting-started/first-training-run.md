---
description: Walk through the first YOLOmatic training run using the interactive terminal wizard.
---

# First Training Run

1. Put a YOLO-format dataset under `datasets/`.
2. Run `uv run yolomatic`.
3. Choose **Configure Model**.
4. Pick a model family and size. `YOLO26n` or `YOLO11n` are practical first
   choices for smoke tests.
5. Select the dataset and review the generated settings.
6. Save the YAML config under `configs/`.
7. Choose **Train Model** or run `uv run yolomatic-train`.

Training writes runs, checkpoints, metrics, and logs under the trainer's normal
output directory. TensorBoard can be launched with:

```sh
uv run yolomatic-tensorboard
```

If CUDA is requested but unavailable, YOLOmatic prompts for repair, CPU fallback,
or cancellation before the trainer starts.

Related pages: [YOLO guide](../guides/yolo.md), [Smart split](../reference/smart-split.md), [Benchmarking](../guides/benchmarking.md).
