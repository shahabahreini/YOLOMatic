---
description: Complete reference for all YOLO training parameters in YOLOmatic — defaults, valid ranges, and practical guidance for every configurable value.
---

# Advanced Training Parameters

YOLOmatic exposes 70+ YOLO training parameters through the **Configure Model → Advanced Settings** wizard step. This reference documents every parameter with its default value, valid range, and practical guidance.

Parameters are organized into eight categories matching the wizard's interface.

---

## Core Parameters

| Parameter | Default | Range | Description |
|---|---|---|---|
| `epochs` | `300` | 1–10000 | Total training iterations (passes over the full dataset). 100 is sufficient for small fine-tuning; 300 is standard; 500+ for complex datasets where accuracy is still improving. |
| `patience` | `50` | 0–500 | Early stopping: halts training if no improvement for this many epochs. Increase to 100 for noisy datasets. Set to 0 to disable early stopping. |
| `batch` | `-1` | -1–1024 | Images per training step. `-1` enables Auto-Batch, which finds the largest size that fits in VRAM. Only set manually if Auto-Batch causes issues. |
| `imgsz` | `640` | 32–2048 | Input resolution in pixels. Use 320–416 for mobile/speed, 640 for standard, 1280 for small-object detection. Larger sizes use significantly more VRAM. |
| `time` | `None` | 0–1000 | Maximum training duration in hours. Useful on cloud platforms billed by the hour. Training saves and stops when the limit is reached. |

---

## Hardware Parameters

| Parameter | Default | Range | Description |
|---|---|---|---|
| `device` | `"0"` | `0`, `0,1`, `cpu`, `mps`, `cuda`, `npu` | Compute device. `0` = first NVIDIA GPU. `cpu` = CPU inference (slow). `mps` = Apple Silicon. `0,1` = multi-GPU. |
| `workers` | `8` | 0–64 | CPU dataloader threads. Increase if GPU utilization is below 80%; decrease if high CPU usage or DataLoader warnings appear. |
| `cache` | `False` | `False`, `True`/`"ram"`, `"disk"` | Dataset caching. `"ram"` caches images in memory (fast, requires RAM ≥ dataset size). `"disk"` caches to a temp file. `False` reads from disk each epoch. |

---

## Optimizer Parameters

| Parameter | Default | Range | Description |
|---|---|---|---|
| `optimizer` | `"auto"` | auto, SGD, Adam, AdamW, Adamax, NAdam, RAdam, RMSProp, MuSGD | Learning algorithm. `auto` lets YOLO pick based on model size. `SGD` is reliable for large datasets. `AdamW` converges faster on smaller datasets. `MuSGD` is the YOLO26-era hybrid optimizer. |
| `lr0` | `0.01` | 0.000001–1.0 | Initial learning rate. 0.01 is the standard starting point. Reduce to 0.001 if loss is erratic; raise to 0.02 if progress is too slow. |
| `lrf` | `0.01` | 0.0001–1.0 | Final learning rate as a fraction of `lr0`. 0.01 means training finishes at 1% of the starting rate. |
| `momentum` | `0.937` | 0.0–1.0 | SGD momentum (gradient inertia). Heavily tuned for YOLO — do not change unless you are doing research. |
| `weight_decay` | `0.0005` | 0.0–0.1 | L2 regularization penalty on large weights. Increase to 0.001 if the model is overfitting on small datasets. |
| `warmup_epochs` | `3.0` | 0.0–50.0 | Number of warmup epochs at the start of training where the learning rate ramps up slowly. Prevents early instability. Increase to 5.0 for very large models. |
| `warmup_momentum` | `0.8` | 0.0–1.0 | Starting momentum value during the warmup phase; grows to `momentum` over `warmup_epochs`. |
| `cos_lr` | `False` | bool | Use a cosine annealing schedule instead of linear decay. Often produces slightly better final accuracy by annealing more smoothly. |

---

## Augmentation Parameters

These parameters control on-the-fly training augmentation applied by the Ultralytics dataloader. For offline augmentation, see [Augmentation guide](../guides/augmentation.md).

| Parameter | Default | Range | Description |
|---|---|---|---|
| `hsv_h` | `0.015` | 0.0–1.0 | Hue shift range. Keep small (0.015). Large values change object color semantics. |
| `hsv_s` | `0.7` | 0.0–1.0 | Saturation shift range. 0.7 simulates varied camera quality and lighting. |
| `hsv_v` | `0.4` | 0.0–1.0 | Brightness/value shift range. 0.4 handles sunny vs. overcast lighting variation. |
| `degrees` | `0.0` | -180–180 | Random rotation in degrees. 0.0 for upright objects (cars, people); 15–30 for tilted cameras; 180 for orientation-independent objects (cells, minerals). |
| `translate` | `0.1` | 0.0–1.0 | Random image translation as a fraction of image size. 0.1 = 10% shift in x/y. |
| `scale` | `0.5` | 0.0–1.0 | Random zoom range. 0.5 means scale between 0.5× and 1.5×. Critical for detecting objects at varied distances. |
| `shear` | `0.0` | -180–180 | Random shear (slant) in degrees. Use for extreme camera angles only. |
| `perspective` | `0.0` | 0.0–1.0 | 3D perspective distortion. Use 0.0001–0.001 for drone or security footage. |
| `flipud` | `0.0` | 0.0–1.0 | Vertical flip probability. 0.0 for normal cameras; 0.5 for satellite or top-down imagery. |
| `fliplr` | `0.5` | 0.0–1.0 | Horizontal flip probability. 0.5 is standard. Set to 0.0 when left/right orientation is semantically meaningful (text, directional signs). |
| `mosaic` | `1.0` | 0.0–1.0 | Probability of stitching four training images into one. A core YOLO augmentation technique — leave at 1.0 for most runs. |
| `mixup` | `0.0` | 0.0–1.0 | Probability of overlaying two images. Use 0.1 for large, complex datasets. Keep at 0.0 for small datasets. |
| `copy_paste` | `0.0` | 0.0–1.0 | Copy-paste segmentation augmentation — pastes object masks from one image into another. Requires segmentation task. |
| `auto_augment` | `"randaugment"` | `""`, `randaugment`, `autoaugment`, `augmix` | Automatic augmentation policy. `randaugment` is the modern default. Set to `""` to disable. |
| `erasing` | `0.4` | 0.0–1.0 | Random erasing probability — blacks out random image regions. Teaches the model to recognize partially occluded objects. |
| `close_mosaic` | `10` | 0–100 | Number of final epochs that run without mosaic augmentation. Allows the model to fine-tune on clean images before the run ends. |

---

## Loss Weight Parameters

| Parameter | Default | Range | Description |
|---|---|---|---|
| `box` | `7.5` | 0.0–20.0 | Weight for bounding box coordinate loss. Increase if boxes are slightly off while classification is correct. |
| `cls` | `0.5` | 0.0–10.0 | Weight for classification loss. Increase if box locations are correct but class labels are wrong. |
| `dfl` | `1.5` | 0.0–10.0 | Distribution Focal Loss weight for bounding-box edge precision. Rarely needs adjustment. |
| `nbs` | `64` | 1–1024 | Nominal batch size for loss scaling normalization. **Do not change** unless you are an ML researcher adjusting loss scaling behavior. |

---

## Advanced Training Control

| Parameter | Default | Range | Description |
|---|---|---|---|
| `amp` | `True` | bool | Automatic Mixed Precision (FP16). Reduces VRAM use and speeds training. Disable only if you see `NaN` loss values. |
| `pretrained` | `True` | bool | Start from COCO-pretrained weights. Disable only when your domain is radically different from natural images (thermal, medical, ultrasound). |
| `deterministic` | `False` | bool | Force deterministic algorithms for full reproducibility. Slows training; use for scientific paper experiments. |
| `seed` | `0` | 0–999999 | Random seed for reproducible augmentation order and initialization. |
| `rect` | `False` | bool | Rectangular training — groups images by aspect ratio to reduce padding. Can speed up training on wide-format datasets. |
| `fraction` | `1.0` | 0.01–1.0 | Fraction of the dataset to use. Set to 0.1 for a quick smoke test before committing to a full run. |
| `multi_scale` | `0.0` | 0.0–1.0 | Change image size slightly each batch to teach scale invariance. Set 0.1 for datasets with objects at highly varied sizes. |
| `resume` | `False` | bool | Resume training from `last.pt`. Use when training was interrupted and you want to continue from the last saved state. |
| `freeze` | `None` | 0–100 | Number of early layers to freeze during fine-tuning. Try `10` for small datasets that are visually similar to the source domain. |
| `single_cls` | `False` | bool | Treat all objects as one class. Use when you only need to locate objects without classifying them. |
| `save_period` | `-1` | -1–1000 | Save a checkpoint every N epochs. `-1` saves only best and last. Set to 50 to enable periodic checkpoints for crash recovery. |
| `profile` | `False` | bool | Profile model speed during training. For deployment research; not needed for normal accuracy tuning. |

---

## Segmentation Parameters

These parameters apply only to segmentation tasks (when the model variant ends in `-seg`).

| Parameter | Default | Range | Description |
|---|---|---|---|
| `overlap_mask` | `True` | bool | Allow object masks to overlap. Should be `True` for natural scenes where objects occlude each other. |
| `mask_ratio` | `4` | 1–16 | Mask resolution relative to image size. Lower values (1–2) produce higher-quality masks but use more VRAM. 4 is the recommended balance. |

---

## Validation Parameters

| Parameter | Default | Range | Description |
|---|---|---|---|
| `val` | `True` | bool | Run validation after every epoch. Leave enabled to track accuracy during training. |
| `conf` | `None` | 0.0–1.0 | Confidence threshold for validation detections. `None` uses Ultralytics defaults. Use `0.001` to measure maximum potential recall; use `0.25+` for real-world deployment thresholds. |
| `max_det` | `300` | 1–10000 | Maximum detections per image during validation. Increase for dense detection tasks (crowd counting, aerial imagery). |
| `plots` | `True` | bool | Generate training curve and confusion matrix plots after training. Leave enabled for post-training analysis. |

---

## Output Parameters

| Parameter | Default | Description |
|---|---|---|
| `save` | `True` | Write checkpoints and run artifacts. Disable only for throwaway diagnostic runs. |
| `project` | `None` | Parent directory for run output. Defaults to the trainer's standard `runs/` folder. |
| `name` | `None` | Name of the run subdirectory. YOLOmatic generates a timestamped name when left blank. |
| `exist_ok` | `False` | Allow reusing an existing `project/name` directory. Keep `False` to avoid mixing run artifacts. |
| `verbose` | `True` | Print detailed trainer logs. Disable for quieter output in scripted environments. |

---

## RF-DETR Training Parameters

RF-DETR uses a separate trainer with different parameter names. The key parameters exposed through YOLOmatic's RF-DETR configuration wizard are:

| Parameter | Typical Values | Description |
|---|---|---|
| `epochs` | 50–100 | Training epochs |
| `batch_size` | 4–8 | Transformer models need more VRAM per image than CNN-based models |
| `lr` | `1e-4` | Learning rate |
| `grad_accum_steps` | 4 | Gradient accumulation steps for effective larger batch sizes |
| `resolution` | 384–880 | Input resolution; RF-DETR models have model-specific defaults |

See [RF-DETR guide](../guides/rf-detr.md) for the full RF-DETR workflow.

Related pages: [Configuration](../reference/configuration.md), [YOLO Guide](../guides/yolo.md), [Augmentation](../guides/augmentation.md).
