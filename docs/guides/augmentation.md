---
description: Offline dataset augmentation with Albumentations profiles, 20+ transforms, split redistribution, and YOLO/COCO output formats.
---

# Augmentation

YOLOmatic includes built-in offline augmentation powered by [Albumentations](https://albumentations.ai/). Rather than relying solely on on-the-fly training augmentation, you can use this tool to generate a permanently expanded dataset before training starts — useful when your source dataset is small or when you need fine-grained control over the augmented copies.

## Accessing Augmentation

Open the main TUI and choose **Augment Dataset**:

```sh
uv run yolomatic
```

Select **Augment Dataset** from the menu. The wizard guides you through profile selection, source dataset, and output configuration.

---

## Augmentation Profiles

Profiles are reusable YAML configurations stored under `configs/augmentation_profiles/`. Each profile captures a complete set of transforms with per-transform probabilities and parameters.

### Profile Operations

| Action | Description |
|---|---|
| **Create** | Start a fresh profile with a name you choose |
| **Edit** | Modify transform settings on an existing profile |
| **Clone** | Copy an existing profile as a new starting point |
| **Delete** | Remove a profile you no longer need |

Profiles persist across sessions, so you can build up a library of augmentation strategies (e.g., `outdoor-weather`, `medical-imaging`, `industrial-inspection`) and reuse them without reconfiguring each run.

---

## Multiplier

The **multiplier** controls how many augmented copies are generated per source image. A multiplier of `3` means every original image produces 3 additional augmented variants, growing your dataset 4×.

Choose your multiplier based on how small your source dataset is:

| Source Size | Suggested Multiplier | Result Size |
|---|---|---|
| < 200 images | 5–10× | 1000–2000 images |
| 200–1000 images | 2–5× | 400–5000 images |
| > 1000 images | 1–2× | 2000–3000+ images |

---

## Transforms

YOLOmatic exposes 20+ Albumentations transforms organized into five categories. Each transform has an independent enable toggle, probability, and configurable parameters.

### Geometric

| Transform | What It Does |
|---|---|
| `HorizontalFlip` | Mirror image left/right |
| `VerticalFlip` | Flip image upside down |
| `RandomRotate90` | Rotate by 0°, 90°, 180°, or 270° |
| `Rotate` | Arbitrary angle rotation |
| `ShiftScaleRotate` | Combined shift, scale, and rotation |
| `RandomCrop` | Crop a random sub-region |
| `PadIfNeeded` | Pad image to a minimum size |
| `Perspective` | 3D perspective warp |
| `Affine` | General affine transform (shear, scale, translate) |

### Color / Photometric

| Transform | What It Does |
|---|---|
| `RandomBrightnessContrast` | Vary brightness and contrast independently |
| `HueSaturationValue` | Shift hue, saturation, value channels |
| `CLAHE` | Adaptive histogram equalization |
| `ToGray` | Convert to grayscale (probability-gated) |
| `ChannelShuffle` | Randomly swap R/G/B channels |
| `ColorJitter` | Combined brightness, contrast, saturation, hue |

### Blur / Noise

| Transform | What It Does |
|---|---|
| `GaussianBlur` | Gaussian kernel blur |
| `MotionBlur` | Directional motion blur |
| `GaussNoise` | Add Gaussian random noise |
| `ISONoise` | Simulate camera sensor noise |

### Weather / Outdoor

| Transform | What It Does |
|---|---|
| `RandomFog` | Overlay simulated fog |
| `RandomRain` | Overlay rain streaks |
| `RandomShadow` | Add random shadow regions |
| `RandomSnow` | Overlay snow specks |
| `RandomSunFlare` | Add sun-flare lens artifact |

### Erasing / Cutout

| Transform | What It Does |
|---|---|
| `CoarseDropout` | Replace random regions with black/noise |
| `GridDropout` | Remove grid-aligned patches |

---

## Split Redistribution

When augmentation is complete, YOLOmatic can **redistribute** the expanded dataset across train/val/test splits. This pools all images (originals + augmented) and re-partitions them at configurable ratios:

```
Pool all images → shuffle → redistribute
  └── train: 70%
  └── valid: 20%
  └── test:  10%
```

This prevents augmented images from being disproportionately concentrated in the training split while the validation set stays unchanged.

!!! tip "When to use redistribution"
    If your original split has < 50 validation images, redistribution is recommended so the expanded dataset gets a proportionally larger validation set.

---

## Output Formats

After augmentation, choose one of the following output formats:

| Format | Use For |
|---|---|
| **YOLO Detection** | `train/images`, `train/labels` with `.txt` bounding-box annotations |
| **YOLO Segmentation** | `train/images`, `train/labels` with `.txt` polygon annotations |
| **COCO JSON** | `train/images`, `_annotations.coco.json` with full COCO structure |

COCO output is required when the augmented dataset will be used for Detectron2 training, SAM 3.1 fine-tuning, or the YOLOmatic benchmark engine.

---

## Tips

- Run augmentation **before** training so the expanded dataset is cached on disk.
- Keep augmented datasets in a separate folder (e.g., `datasets/my_dataset_aug/`) rather than overwriting the originals.
- Use weather transforms only when your deployment environment includes outdoor/variable lighting conditions.
- Geometric transforms (flip, rotate) should reflect the real-world orientation variance of your objects — don't flip if orientation is semantically meaningful.

Related pages: [Datasets](datasets.md), [Training parameters](../advanced/training-parameters.md), [First training run](../getting-started/first-training-run.md).
