---
description: Use YOLOmatic for SAM 3.1 open-vocabulary segmentation and mask decoder fine-tuning.
---

# SAM 3.1

YOLOmatic supports SAM 3 and SAM 3.1 segmentation workflows through HuggingFace.

## Inference Modes

```sh
uv run yolomatic-sam
```

Available modes:

- Auto segmentation
- Text-prompted segmentation
- Box-prompted segmentation using YOLO detections

## Fine-Tuning

SAM 3.1 fine-tuning expects COCO-format masks. The trainer fine-tunes the mask
decoder with HuggingFace Trainer and saves artifacts under the configured run
directory.

## Notes

SAM model access may require HuggingFace authentication depending on the
checkpoint. Keep tokens in your shell or `.env`; do not commit them.

Related pages: [Datasets](datasets.md), [CLI commands](../reference/cli-commands.md), [Models](models.md).
