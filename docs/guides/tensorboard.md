---
description: Monitor YOLOmatic training runs with TensorBoard — loss curves, mAP, precision, recall, and auto-discovery of run directories.
---

# TensorBoard

YOLOmatic includes a dedicated TensorBoard launcher that automatically discovers training run directories — no need to manually locate log paths.

```sh
uv run yolomatic-tensorboard
```

---

## What TensorBoard Tracks

When the Ultralytics YOLO trainer runs, it writes TensorBoard event files under the run directory (e.g., `runs/detect/train/`). YOLOmatic's launcher scans the project tree for these files and presents a selector when multiple runs exist.

| Metric | Location in TensorBoard |
|---|---|
| Training box/cls/dfl loss | `train/box_loss`, `train/cls_loss`, `train/dfl_loss` |
| Validation box/cls/dfl loss | `val/box_loss`, `val/cls_loss`, `val/dfl_loss` |
| mAP 50 | `metrics/mAP50(B)` |
| mAP 50-95 | `metrics/mAP50-95(B)` |
| Precision | `metrics/precision(B)` |
| Recall | `metrics/recall(B)` |
| Learning rate | `lr/pg0`, `lr/pg1`, `lr/pg2` |

For segmentation models, additional mask-level metrics (`mAP50(M)`, `mAP50-95(M)`) appear alongside detection metrics.

---

## Launching

```sh
uv run yolomatic-tensorboard
```

If only one run directory is found, TensorBoard starts immediately. If multiple runs exist, a TUI selector lets you choose which run to inspect.

TensorBoard starts on port **6006** by default. Open in your browser:

```
http://localhost:6006
```

---

## Monitoring a Live Training Run

You can launch TensorBoard before or during training. Event files are written incrementally, so the Scalars view refreshes as new epochs complete. Use the **Scalars** tab and enable auto-refresh (top-right toggle in the TensorBoard UI) to watch metrics update in real time.

---

## Comparing Multiple Runs

To compare more than one run simultaneously:

1. Move or link multiple run directories under a common parent, e.g.:

```
runs/
  train_yolo26n/
  train_yolo11s/
  train_rfdetr_nano/
```

2. Point TensorBoard at the parent:

```sh
uv run python -m tensorboard.main --logdir runs/
```

All three runs appear as separate series in the Scalars view with legend entries by folder name.

---

## Windows: Locked File Workaround

On Windows, `uv run` can fail with an access-denied error when `torch` DLLs are locked by another process. If this happens, launch TensorBoard using the environment Python directly:

```powershell
.\.venv\Scripts\python.exe -m src.cli.tensorboard_launcher
```

---

## ClearML vs TensorBoard

Both integrations can run simultaneously. Use TensorBoard for local, instant visual feedback. Use ClearML when you want persistent cloud storage of metrics, artifacts, and hyperparameters across multiple machines or team members.

Related pages: [ClearML](clearml.md), [First Training Run](../getting-started/first-training-run.md), [CLI Commands](../reference/cli-commands.md).
