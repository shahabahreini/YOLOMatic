---
description: Complete reference for the yolomatic_settings.yaml file — all keys, defaults, types, and descriptions for ClearML, Roboflow, Ultralytics, Narratives, and AI sections.
---

# Settings File Reference

YOLOmatic reads a persistent settings file at `configs/yolomatic_settings.yaml`. This file controls global behaviour across all workflows — experiment tracking, cloud upload defaults, dataset download paths, TUI verbosity, and AI provider credentials.

If the file does not exist, YOLOmatic uses built-in defaults and creates the file on first save.

---

## Complete Example

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

roboflow:
  upload_wizard_enabled: true
  auto_upload_after_training: false
  auto_upload_weight: best.pt
  default_model_name_template: "{run_name}-best"
  require_dataset_metadata: true
  rfdetr_project_version: 1

ultralytics:
  default_dataset_download_dir: datasets/ultralytics/downloads
  default_model_download_dir: weights/ultralytics
  default_output_root: datasets

narratives:
  mode: guided
  show_setup_guidance: true
  show_success_panels: true
  show_skip_reasons: true

ai:
  provider: Gemini
  gemini_api_key: ""
  openai_api_key: ""
  selected_model: gemini-2.5-flash
```

---

## `clearml` Section

Controls ClearML experiment tracking. See [ClearML guide](../guides/clearml.md) for setup instructions.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Master on/off switch for ClearML integration |
| `require_configured` | bool | `false` | When `true`, training is blocked if ClearML credentials are not present; when `false`, training can continue without ClearML after a prompt |
| `project_name_template` | string | `"{family} Training - {model}"` | Python format string for the ClearML project name; `{family}` and `{model}` are substituted at runtime |
| `task_name_format` | string | `"%Y-%m-%d-%H-%M"` | Python `strftime` format string for the ClearML task name |
| `upload_final_model` | bool | `true` | Upload `best.pt` as a registered ClearML model artifact after training |
| `upload_artifacts` | bool | `true` | Upload the generated training YAML and other run files as ClearML artifacts |
| `log_hyperparameters` | bool | `true` | Log all training hyperparameters to the ClearML Task |
| `log_dataset_summary` | bool | `true` | Log dataset class counts and split sizes to the ClearML Task |

---

## `roboflow` Section

Controls Roboflow upload and deployment defaults. See [Cloud Upload guide](../guides/cloud-upload.md) for the full upload workflow.

| Key | Type | Default | Description |
|---|---|---|---|
| `upload_wizard_enabled` | bool | `true` | Show the Roboflow upload wizard step after training completes |
| `auto_upload_after_training` | bool | `false` | Automatically upload to Roboflow at the end of training without a prompt (requires credentials in `.env`) |
| `auto_upload_weight` | string | `"best.pt"` | Which weight file to upload when `auto_upload_after_training` is true |
| `default_model_name_template` | string | `"{run_name}-best"` | Format string for the Roboflow model name; `{run_name}` is substituted from the training run directory name |
| `require_dataset_metadata` | bool | `true` | Require dataset class names to be present before allowing upload |
| `rfdetr_project_version` | int | `1` | Default Roboflow project version to use for RF-DETR deployment |

### Enabling Auto-Upload

To upload automatically after every training run:

```yaml
roboflow:
  auto_upload_after_training: true
  auto_upload_weight: best.pt
```

Also requires `.env` credentials:

```env
ROBOFLOW_API_KEY=your_api_key
ROBOFLOW_WORKSPACE=your_workspace_slug
ROBOFLOW_PROJECT_IDS=your_project_id
```

---

## `ultralytics` Section

Controls paths used by Ultralytics-platform workflows (dataset download, model download, output root).

| Key | Type | Default | Description |
|---|---|---|---|
| `default_dataset_download_dir` | string | `"datasets/ultralytics/downloads"` | Directory where Ultralytics-platform dataset downloads are saved |
| `default_model_download_dir` | string | `"weights/ultralytics"` | Directory where pretrained Ultralytics weights are cached |
| `default_output_root` | string | `"datasets"` | Root directory for converted or prepared datasets |

---

## `narratives` Section

Controls TUI verbosity — how much explanatory text YOLOmatic shows during wizard flows.

| Key | Type | Default | Description |
|---|---|---|---|
| `mode` | string | `"guided"` | Verbosity preset: `guided` (full explanations), `concise` (shorter prompts), or `quiet` (minimal output) |
| `show_setup_guidance` | bool | `true` | Show setup and configuration guidance panels at the start of wizard flows |
| `show_success_panels` | bool | `true` | Show success confirmation panels after completed steps |
| `show_skip_reasons` | bool | `true` | Explain why optional steps are being skipped |

### Narrative Modes

| Mode | Behaviour |
|---|---|
| `guided` | Full explanatory panels, detailed prompts, and contextual help throughout every workflow step |
| `concise` | Shortened prompts; skips tutorial-style panels while retaining key confirmations |
| `quiet` | Minimal output; useful in scripted or automated environments |

---

## `ai` Section

Controls the AI provider used for intelligent suggestions inside the TUI (dataset analysis, config recommendations).

| Key | Type | Default | Description |
|---|---|---|---|
| `provider` | string | `"Gemini"` | AI provider: `"Gemini"` or `"OpenAI"` |
| `gemini_api_key` | string | `""` | Google Gemini API key; leave blank to use the `GEMINI_API_KEY` environment variable |
| `openai_api_key` | string | `""` | OpenAI API key; leave blank to use the `OPENAI_API_KEY` environment variable |
| `selected_model` | string | `"gemini-2.5-flash"` | Model ID to use for AI-assisted features |

!!! tip "Prefer environment variables for API keys"
    Do not store API keys in `yolomatic_settings.yaml` if the file is committed to version control. Set `GEMINI_API_KEY` or `OPENAI_API_KEY` in your shell or `.env` instead.

---

## Resetting to Defaults

Delete `configs/yolomatic_settings.yaml` and restart YOLOmatic. The file is recreated with all defaults on the next settings save.

---

## Relationship to Training YAML

`yolomatic_settings.yaml` controls YOLOmatic's global behaviour. Individual training configs (under `configs/*.yaml`) contain model-specific parameters and override nothing in the settings file — they are separate documents. The `roboflow` block in a training YAML overrides the global `auto_upload_after_training` default for that specific run only.

Related pages: [Configuration](configuration.md), [ClearML](../guides/clearml.md), [Cloud Upload](../guides/cloud-upload.md).
