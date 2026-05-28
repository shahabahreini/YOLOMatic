---
description: Use YOLOmatic for SAM 3.1 open-vocabulary segmentation — auto, text-prompted, box-prompted modes — and COCO-format mask decoder fine-tuning.
---

# SAM 3.1

YOLOmatic supports SAM 3 and SAM 3.1 (Segment Anything Model) through HuggingFace. SAM enables open-vocabulary segmentation — segment any object by concept, without predefined class lists.

---

## SAM 3 vs SAM 3.1

| Model | HuggingFace ID | Description |
|---|---|---|
| SAM 3 | `facebook/sam3` | Base model |
| SAM 3.1 | `facebook/sam3.1` | Adds **Object Multiplex** — 7× faster throughput when segmenting multiple objects simultaneously |

For most use cases, prefer SAM 3.1.

---

## Authentication

SAM 3 and SAM 3.1 are gated models on HuggingFace. You must:

1. Accept Meta's terms at [huggingface.co/facebook/sam3.1](https://huggingface.co/facebook/sam3.1)
2. Set a HuggingFace token in your environment

### Setting the HuggingFace Token

**Option 1 — environment variable:**

```env
HF_TOKEN=hf_your_token_here
```

Add to `.env` or export in your shell:

```sh
export HF_TOKEN=hf_your_token_here
```

**Option 2 — HuggingFace CLI:**

```sh
uv run huggingface-cli login
```

This writes the token to `~/.cache/huggingface/token` and is picked up automatically.

!!! warning "Keep tokens out of version control"
    Never commit `.env` or credential files. Add `.env` to `.gitignore`.

---

## Running SAM Inference

```sh
uv run yolomatic-sam
```

The wizard walks you through model selection, inference mode, and source selection.

---

## Inference Modes

### Auto Segmentation

Segments everything in the image automatically — no prompts required. Useful for:

- **Exploratory annotation**: quickly produce candidate masks before manual review
- **Pseudo-label generation**: create segmentation labels for unlabeled images
- **Dataset quality inspection**: visually verify coverage and annotation completeness

Auto mode runs SAM's mask generation pipeline, which proposals masks at multiple granularities and filters by predicted IoU.

### Text-Prompted Segmentation

Provide a text description to segment specific object types:

```
Prompt: "cat"
Prompt: "red car in the background"
Prompt: "all people"
```

Useful for:

- **Targeted extraction** of specific classes from unlabeled images
- **Concept-based segmentation** across a batch of images without predefined class lists
- Generating training masks for a new class you want to add to a detection dataset

### Box-Prompted Segmentation

Provide bounding boxes as spatial prompts. YOLOmatic can use **YOLO detections** as box prompts automatically — giving you a pipeline to upgrade a detection dataset to segmentation masks:

1. Run `yolomatic-predict` to generate detection boxes
2. Run `yolomatic-sam` in box-prompted mode, feeding the detection outputs as prompts
3. SAM generates precise polygon masks for each detected box

---

## Outputs

SAM inference produces:

| Output | Format |
|---|---|
| Overlay images | PNG with masks rendered over the input |
| COCO JSON annotations | `_annotations.coco.json` |
| YOLO segmentation labels | `.txt` files with normalized polygon coordinates |

---

## Fine-Tuning SAM 3.1

YOLOmatic supports fine-tuning the SAM 3.1 mask decoder on custom COCO-format mask datasets.

### Why Fine-Tune?

The base SAM model is trained on SA-1B (general objects). Fine-tuning on domain-specific data improves mask quality for specialized objects such as:

- Medical images (cells, organs, lesions)
- Industrial inspection (defect regions)
- Aerial/satellite imagery (buildings, roads)

### What Gets Fine-Tuned

YOLOmatic fine-tunes only the **mask decoder** and **prompt encoder**. The ViT image encoder is frozen to reduce compute and VRAM requirements.

### Dataset Requirements

Fine-tuning requires a COCO-format dataset with **mask annotations** (not just bounding boxes). Minimum recommended: 200–500 annotated images.

```text
datasets/my_dataset/
  train/
    images/
    _annotations.coco.json   ← must include segmentation polygons
  valid/
    images/
    _annotations.coco.json
```

### Fine-Tuning Workflow

1. Run `uv run yolomatic` and select **Configure Model**
2. Choose **SAM 3.1** as the model family
3. Select a base model (`facebook/sam3.1` or `facebook/sam3`)
4. Select your COCO-format dataset
5. Configure fine-tuning strategy (decoder-only recommended)
6. Save the config and run `uv run yolomatic-train`

The trainer uses the HuggingFace Trainer API and saves artifacts under the configured run directory.

### VRAM Requirements

| Configuration | VRAM Needed |
|---|---|
| SAM 3.1 inference (CPU fallback) | No GPU required |
| SAM 3.1 inference (GPU) | ~4 GB |
| SAM 3.1 fine-tuning (decoder only) | ~8 GB |
| SAM 3.1 fine-tuning (full model) | ~24 GB |

If you exceed VRAM, reduce batch size or use CPU fallback for inference.

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `HF token required` or 401 error | Set `HF_TOKEN` env var or run `huggingface-cli login` |
| `Model download fails` | Accept Meta's terms at [huggingface.co/facebook/sam3.1](https://huggingface.co/facebook/sam3.1) |
| `Out of memory` during inference | SAM 3.1 needs ~4 GB VRAM; use CPU fallback |
| `Out of memory` during fine-tuning | Reduce batch size; use decoder-only fine-tuning strategy |
| Text prompts produce empty masks | Rephrase the prompt; try more specific terms |

Related pages: [Datasets](datasets.md), [CLI Commands](../reference/cli-commands.md), [Models](models.md).
