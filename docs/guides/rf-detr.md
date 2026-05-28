---
description: Configure, train, and fine-tune RF-DETR detection and segmentation models with YOLOmatic. Full variant tables, training parameters, and Roboflow deployment.
---

# RF-DETR

YOLOmatic routes RF-DETR configs to the native RF-DETR trainer instead of the Ultralytics trainer. RF-DETR is a real-time transformer-based detector that achieves the highest mAP of any model family supported by YOLOmatic (60.1 mAP with 2XLarge).

---

## Supported Variants

### Detection Variants

| Model | Class | mAP 50-95 | Latency T4 (ms) | Params (M) | Resolution | License |
|---|---|---:|---:|---:|---:|---|
| RF-DETR-Nano | `RFDETRNano` | 48.4 | 2.3 | 30.5 | 384 | Apache-2.0 |
| RF-DETR-Small | `RFDETRSmall` | 53.0 | 3.5 | 32.1 | 512 | Apache-2.0 |
| RF-DETR-Medium | `RFDETRMedium` | 54.7 | 4.4 | 33.7 | 576 | Apache-2.0 |
| RF-DETR-Large | `RFDETRLarge` | 56.5 | 6.8 | 33.9 | 704 | Apache-2.0 |
| RF-DETR-XLarge | `RFDETRXLarge` | 58.6 | 11.5 | 126.4 | 700 | **PML-1.0** |
| RF-DETR-2XLarge | `RFDETR2XLarge` | 60.1 | 17.2 | 126.9 | 880 | **PML-1.0** |

!!! warning "Plus models require additional dependency"
    RF-DETR XLarge and 2XLarge (the "Plus" variants) require `rfdetr[plus]` and use Roboflow's PML-1.0 model license. Check the license terms before using these variants in commercial products.

### Segmentation Variants

| Model | Class | Default Resolution | License |
|---|---|---:|---|
| RF-DETR-Seg-Nano | `RFDETRSegNano` | 312 | Apache-2.0 |
| RF-DETR-Seg-Small | `RFDETRSegSmall` | 384 | Apache-2.0 |
| RF-DETR-Seg-Medium | `RFDETRSegMedium` | 432 | Apache-2.0 |
| RF-DETR-Seg-Large | `RFDETRSegLarge` | 504 | Apache-2.0 |
| RF-DETR-Seg-XLarge | `RFDETRSegXLarge` | 624 | Apache-2.0 |
| RF-DETR-Seg-2XLarge | `RFDETRSeg2XLarge` | 768 | Apache-2.0 |

---

## When to Use RF-DETR

| Scenario | Recommendation |
|---|---|
| Need the highest possible mAP | RF-DETR-Large or 2XLarge |
| Server-side deployment with GPU | RF-DETR-Small or Large |
| Memory-constrained server | RF-DETR-Nano or Small |
| Real-time transformer detection research | RF-DETR-Medium |
| Edge / CPU deployment | Use YOLO26 instead — RF-DETR is transformer-based and slower on CPU |

---

## Training Workflow

### Step 1 — Configure

```sh
uv run yolomatic
```

Choose **Configure Model** → **RF-DETR** → select a variant and dataset → save the config.

### Step 2 — Train

```sh
uv run yolomatic-train
```

The smart training router reads the `family: RF-DETR` field in the config and dispatches to the native RF-DETR trainer. YOLOmatic downloads official pretrained weights automatically for fresh training.

### Step 3 — Monitor

RF-DETR training writes logs compatible with TensorBoard. Launch monitoring with:

```sh
uv run yolomatic-tensorboard
```

### Step 4 — Upload / Deploy

```sh
uv run yolomatic-upload
```

See the [Upload](#upload-and-deployment) section below.

---

## Training Modes

### Fresh Training

Instantiates the selected RF-DETR model class without a local checkpoint. RF-DETR automatically downloads and caches official pretrained backbone weights.

### Fine-Tuning

YOLOmatic discovers RF-DETR `.pth` checkpoints in the project tree and presents a selector. The selected checkpoint is passed as `pretrain_weights` so RF-DETR loads the weights as the starting point.

Use fine-tuning when you have a previously trained RF-DETR checkpoint and want to adapt it to a new or expanded dataset.

### Resume

Resume passes the checkpoint as `resume` in the config, which restores both model weights and optimizer state. Use resume when training was interrupted and you want to continue from where it stopped.

---

## Key Training Parameters

RF-DETR uses different parameter names from Ultralytics YOLO. The most important ones:

| Parameter | Typical Values | Description |
|---|---|---|
| `epochs` | 50–100 | Transformer models converge faster than CNNs and typically need fewer epochs |
| `batch_size` | 4–8 | Transformers use significantly more VRAM per image than CNNs |
| `lr` | `1e-4` | Learning rate; lower than typical YOLO lr0 |
| `grad_accum_steps` | 4 | Gradient accumulation steps for effective larger batch sizes on limited VRAM |
| `resolution` | 384–880 | Input resolution; each variant has a model-specific default |

### Example Config Fragment

```yaml
family: RF-DETR
model: rfdetr_large
task: detect
dataset: datasets/my_dataset/data.yaml

training:
  epochs: 75
  batch_size: 4
  lr: 0.0001
  grad_accum_steps: 4
```

---

## VRAM Requirements

RF-DETR is significantly more VRAM-intensive than CNN-based YOLO models:

| Variant | Recommended VRAM | Notes |
|---|---|---|
| Nano / Small | 8 GB | Entry-level GPU (RTX 3060, T4) |
| Medium / Large | 16 GB | Mid-range GPU (RTX 3090, A10) |
| XLarge / 2XLarge | 24–40 GB | High-end GPU (A100, H100) |

If you hit OOM errors, reduce `batch_size` and increase `grad_accum_steps` to maintain the same effective batch size.

---

## Dataset Format

RF-DETR training uses YOLO-format datasets with `data.yaml` (same as Ultralytics YOLO). Internally, YOLOmatic converts COCO annotations to the format required by the RF-DETR trainer when needed.

---

## Upload and Deployment

RF-DETR deployment to Roboflow is available through the upload wizard:

```sh
uv run yolomatic-upload
```

RF-DETR deploys using `deploy_to_roboflow(...)` from the `rfdetr` package and requires:

- `ROBOFLOW_API_KEY` in `.env`
- Workspace slug (`ROBOFLOW_WORKSPACE`)
- Project ID (`ROBOFLOW_PROJECT_IDS`)
- Project version (defaults to `1`; override with `--version N`)

Alternatively, use direct CLI flags:

```sh
uv run yolomatic-upload \
  --weight runs/rf-detr/train/weights/best.pth \
  --workspace my-workspace \
  --project-ids my-project \
  --version 1
```

Related pages: [Models](models.md), [Configuration](../reference/configuration.md), [Cloud upload](cloud-upload.md).
