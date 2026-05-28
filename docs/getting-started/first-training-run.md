---
description: Walk through the first YOLOmatic training run — dataset prep, config generation, training, monitoring, prediction, and benchmarking.
---

# First Training Run

This walkthrough goes from a raw dataset to a trained, evaluated model.

---

## 1. Prepare Your Dataset

Place a YOLO-format dataset under `datasets/`:

```text
datasets/my_dataset/
  data.yaml
  train/images/
  train/labels/
  valid/images/
  valid/labels/
```

`data.yaml` must define `train`, `val` (or `valid`), `nc`, and `names`. See [Datasets](../guides/datasets.md) for the exact format.

**No dataset yet?** Use the preparation wizard to split a flat image+label folder:

```sh
uv run yolomatic-prepare
```

Or convert a Labelbox/Ultralytics NDJSON export:

```sh
uv run yolomatic-convert
```

---

## 2. Open the Main TUI

```sh
uv run yolomatic
```

---

## 3. Configure the Model

Choose **Configure Model** from the menu.

1. **Select a model family** — for a first run, choose **YOLO26** or **YOLO11**
2. **Select a variant** — `nano` or `small` is recommended for smoke tests (fast, low VRAM)
3. **Select a task** — `detect` for object detection, `segment` for instance segmentation
4. **Select your dataset** — choose the `data.yaml` from step 1
5. **Review hardware settings** — YOLOmatic detects your CUDA/MPS/CPU environment and suggests appropriate defaults
6. **Save the config** — the wizard writes a YAML file to `configs/`

The generated config captures all settings and is the single source of truth for training.

---

## 4. Start Training

Choose **Train Model** from the TUI, or run directly:

```sh
uv run yolomatic-train
```

**What happens at startup:**

- YOLOmatic reads the saved config from `configs/`
- The smart router dispatches to the correct trainer based on the `family` field
- A hardware preflight check verifies CUDA availability
- If ClearML is configured, a tracking task is created
- Training begins; progress prints per epoch

**If CUDA is unavailable** on a GPU machine, YOLOmatic offers:
- Guided repair (reinstall PyTorch with CUDA support)
- CPU fallback (slow but functional)
- Cancel

---

## 5. Monitor Training

### TensorBoard (real-time)

```sh
uv run yolomatic-tensorboard
```

Open [http://localhost:6006](http://localhost:6006) to see loss curves, mAP, precision, and recall updated after each epoch.

### Training outputs

Ultralytics YOLO writes run artifacts under `runs/detect/train*/` (or `runs/segment/train*/` for segmentation):

```text
runs/detect/train/
  weights/
    best.pt    ← best validation checkpoint
    last.pt    ← final epoch checkpoint
  results.csv  ← per-epoch metrics
  results.png  ← training curve plots
  confusion_matrix.png
  val_batch0_pred.jpg
```

---

## 6. Run Prediction

Test your trained model on new images:

```sh
# Interactive wizard
uv run yolomatic-predict

# Direct single-image prediction
uv run yolomatic-predict --mode single \
  --weight runs/detect/train/weights/best.pt \
  --source path/to/image.jpg

# Batch folder prediction
uv run yolomatic-predict --mode folder \
  --weight runs/detect/train/weights/best.pt \
  --source datasets/my_dataset/test/images \
  --workers 4
```

Prediction outputs (annotated images) are saved alongside the source images.

---

## 7. Benchmark the Model

Generate a detailed HTML evaluation report:

```sh
uv run yolomatic-benchmark
```

You need a COCO-format validation set (`_annotations.coco.json`). The report includes mAP, F1, per-image rankings, confidence inspection, thumbnail gallery, and UMAP scatter plots.

See [Benchmarking](../guides/benchmarking.md) for the full report walkthrough.

---

## 8. Upload to Roboflow (Optional)

Deploy the best checkpoint to Roboflow:

```sh
uv run yolomatic-upload
```

Add credentials to `.env` first:

```env
ROBOFLOW_API_KEY=...
ROBOFLOW_WORKSPACE=...
ROBOFLOW_PROJECT_IDS=...
```

---

## Common First-Run Issues

| Issue | Solution |
|---|---|
| CUDA not available on GPU machine | Follow YOLOmatic's repair prompt, or see [FAQ](../faq.md#troubleshooting) |
| `val` key not found in `data.yaml` | Use `val` or `valid` as the key; both are accepted |
| Training is very slow | Use a smaller model variant (`n` or `s`) or reduce `imgsz` to 320 |
| Out of memory (OOM) error | Reduce `batch` size; set `batch: -1` for Auto-Batch |
| Config not found | Ensure `configs/` contains your YAML before running `yolomatic-train` |

Related pages: [YOLO guide](../guides/yolo.md), [Datasets](../guides/datasets.md), [Smart split](../reference/smart-split.md), [Benchmarking](../guides/benchmarking.md).
