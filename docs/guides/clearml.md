---
description: Track YOLOmatic training experiments with ClearML — hyperparameters, metrics, model artifacts, and dataset summaries.
---

# ClearML Integration

YOLOmatic integrates with [ClearML](https://clear.ml/) for experiment tracking. When ClearML is configured, YOLOmatic automatically logs hyperparameters, training metrics, model artifacts, and dataset summaries to your ClearML workspace — giving you a full audit trail and easy run comparison across experiments.

ClearML is **optional**. If it is not configured or the service is unavailable, training prompts whether to continue without it or cancel.

---

## Setup

### 1. Install ClearML (if not already present)

ClearML is included in YOLOmatic's dependency set. Verify with:

```sh
uv run python -c "import clearml; print(clearml.__version__)"
```

### 2. Connect to a ClearML Server

You can use the hosted [ClearML Community server](https://app.clear.ml/) (free tier) or self-host a [ClearML Server](https://github.com/allegroai/clearml-server).

Initialize credentials by running:

```sh
uv run clearml-init
```

This opens a browser-based flow to generate an API key and writes `~/clearml.conf`. Follow the prompts to complete authentication.

Alternatively, set credentials via environment variables:

```env
CLEARML_API_HOST=https://api.clear.ml
CLEARML_WEB_HOST=https://app.clear.ml
CLEARML_FILES_HOST=https://files.clear.ml
CLEARML_API_ACCESS_KEY=your_access_key
CLEARML_API_SECRET_KEY=your_secret_key
```

### 3. Verify the Connection

```sh
uv run clearml-task --help
```

No error output means the connection is live.

---

## What YOLOmatic Logs

When ClearML is enabled, each training run creates a **Task** in your ClearML project with:

| Category | Content |
|---|---|
| **Hyperparameters** | All YOLO training parameters (epochs, lr0, batch, imgsz, optimizer, augmentation settings, etc.) |
| **Scalars** | Per-epoch train/val loss, mAP, precision, recall — plotted automatically |
| **Artifacts** | Best checkpoint (`best.pt`), last checkpoint (`last.pt`), training config YAML |
| **Dataset summary** | Class distribution, split sizes, dataset path |
| **System info** | GPU model, VRAM, CUDA version, Python version |

---

## Configuring ClearML in YOLOmatic

ClearML behavior is controlled by the `clearml` section in `configs/yolomatic_settings.yaml`:

```yaml
clearml:
  enabled: true
  require_configured: false
  project_name_template: "{family} Training - {model}"
  task_name_format: "%Y-%m-%d-%H-%M"
  upload_final_model: true
  upload_artifacts: true
  log_hyperparameters: true
  log_dataset_summary: true
```

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Master switch; set to `false` to disable ClearML globally |
| `require_configured` | `false` | If `true`, training is blocked when ClearML is not configured — useful for CI workflows |
| `project_name_template` | `"{family} Training - {model}"` | ClearML project name; `{family}` and `{model}` are substituted at runtime |
| `task_name_format` | `"%Y-%m-%d-%H-%M"` | Python `strftime` format for the task name |
| `upload_final_model` | `true` | Upload `best.pt` as a ClearML model artifact |
| `upload_artifacts` | `true` | Upload the training config YAML and other run artifacts |
| `log_hyperparameters` | `true` | Log all training hyperparameters to the Task |
| `log_dataset_summary` | `true` | Log dataset class counts and split sizes |

See [Settings File Reference](../reference/settings.md) for the full description of all keys.

---

## Project Naming

YOLOmatic generates ClearML project names from the template:

```
{family} Training - {model}
```

For example, training `YOLO26` large produces `YOLO26 Training - yolo26l`. Customize `project_name_template` to group experiments differently (e.g., by dataset or team):

```yaml
project_name_template: "MyDataset/{family}/{model}"
```

---

## Viewing Results

After training starts, open the ClearML web UI (https://app.clear.ml/ or your self-hosted instance) and navigate to the project. Each run appears as a **Task** with:

- **Scalars** tab: loss curves, mAP, precision/recall charts
- **Artifacts** tab: trained checkpoints
- **Hyperparameters** tab: all training parameters used
- **Console** tab: captured stdout from the trainer

### Comparing Runs

Select two or more Tasks → **Compare** to overlay scalar plots and diff hyperparameters side-by-side.

---

## Disabling ClearML for a Single Run

You can bypass ClearML for one run without changing the settings file. When the TUI asks whether to proceed without ClearML, select **Continue without ClearML**.

To disable globally, set `enabled: false` in `configs/yolomatic_settings.yaml`:

```yaml
clearml:
  enabled: false
```

Related pages: [Settings File](../reference/settings.md), [Configuration](../reference/configuration.md), [TensorBoard](tensorboard.md).
