---
description: Start YOLOmatic, generate a training config, train a model, and run prediction from the command line.
---

# Quickstart

## 30-Second Path

```sh
uv tool install --python 3.12 yolomatic
yolomatic
```

Choose **Configure Model**, select a dataset, generate a config, then choose
**Train Model**.

## Repository Path

```sh
uv sync
uv run yolomatic
```

Useful follow-up commands:

```sh
uv run yolomatic-train
uv run yolomatic-predict
uv run yolomatic-benchmark
uv run yolomatic-tensorboard
```

## Expected Dataset Layout

```text
datasets/my_dataset/
  data.yaml
  train/images/
  train/labels/
  valid/images/
  valid/labels/
  test/images/
  test/labels/
```

COCO JSON datasets are used for Detectron2, SAM 3.1 fine-tuning, and benchmark
validation sets.

Related pages: [First training run](first-training-run.md), [Datasets](../guides/datasets.md), [Configuration](../reference/configuration.md).
