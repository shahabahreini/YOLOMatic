---
description: Configure and fine-tune RF-DETR detection and segmentation models with YOLOmatic.
---

# RF-DETR

YOLOmatic routes RF-DETR configs to the native RF-DETR trainer instead of the
Ultralytics trainer.

## Supported Variants

Detection variants include Nano, Small, Medium, Large, XLarge, and 2XLarge.
Segmentation variants are exposed with task-specific RF-DETR classes.

Fresh training lets RF-DETR download official pretrained weights. Fine-tuning
uses the selected `.pth` checkpoint as `pretrain_weights`, while resume workflows
pass the checkpoint as `resume`.

## Workflow

```sh
uv run yolomatic
```

Choose **Configure Model** or **Configure Fine-Tune**, select RF-DETR, bind a
dataset, save the config, then train:

```sh
uv run yolomatic-train
```

## Upload

RF-DETR deployment to Roboflow is available through:

```sh
uv run yolomatic-upload
```

Related pages: [Models](models.md), [Configuration](../reference/configuration.md), [Cloud upload](cloud-upload.md).
