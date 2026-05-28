---
description: Upload YOLO checkpoints and deploy RF-DETR checkpoints to Roboflow from YOLOmatic — credentials, interactive wizard, CLI flags, and automatic post-training upload.
---

# Cloud Upload

YOLOmatic uploads YOLO checkpoints and deploys RF-DETR checkpoints to [Roboflow](https://roboflow.com/).

```sh
uv run yolomatic-upload
```

---

## Credential Setup

### 1. Create a `.env` file

```sh
cp .env.example .env
```

### 2. Fill in credentials

```env
ROBOFLOW_API_KEY=your_api_key_here
ROBOFLOW_WORKSPACE=your-workspace-slug
ROBOFLOW_PROJECT_IDS=your-project-id
```

| Variable | Where to find it |
|---|---|
| `ROBOFLOW_API_KEY` | Roboflow → Settings → Roboflow API |
| `ROBOFLOW_WORKSPACE` | The URL slug in `app.roboflow.com/<workspace>/` |
| `ROBOFLOW_PROJECT_IDS` | The project slug from `app.roboflow.com/<workspace>/<project>/` |

`ROBOFLOW_PROJECT_IDS` accepts a comma-separated list for uploading to multiple projects simultaneously:

```env
ROBOFLOW_PROJECT_IDS=project-a,project-b
```

### 3. Keep credentials out of version control

Add `.env` to `.gitignore`. Never commit API keys.

---

## Interactive Wizard

```sh
uv run yolomatic-upload
```

The wizard guides you through:

1. **Weight selection** — scans the project tree for `.pt` (YOLO) and `.pth` (RF-DETR) checkpoints
2. **Workspace** — pre-fills from `.env`; you can override
3. **Project** — pre-fills from `.env`; you can select from a list
4. **Model type** — auto-detected from the checkpoint; you can override
5. **Model name** — suggested from the run name; you can customize
6. **Confirmation** — review and confirm before uploading

---

## Direct CLI Upload

Skip the wizard by providing all arguments on the command line:

```sh
uv run yolomatic-upload \
  --weight runs/detect/train/weights/best.pt \
  --workspace my-workspace \
  --project-ids my-project \
  --model-type yolo26l \
  --model-name my-experiment-best
```

All flags are optional — any omitted flag falls back to `.env` or the interactive prompt.

| Flag | Description |
|---|---|
| `--weight` | Path to checkpoint file |
| `--workspace` | Roboflow workspace slug |
| `--project-ids` | Comma-separated project IDs |
| `--model-type` | Roboflow model type identifier |
| `--model-name` | Model name to register in Roboflow |
| `--version` | Project version for RF-DETR deployment (default: 1) |

---

## Which Weight to Upload

Upload a **full checkpoint** such as `best.pt` or `last.pt`. Do not upload intermediate artifacts like `state_dict.pt` — these are not uploadable Roboflow model weights.

| Checkpoint | When to Use |
|---|---|
| `best.pt` | Best validation performance — use for production deployment |
| `last.pt` | Final training epoch — use to inspect the end state |

---

## YOLO26 Model Type

YOLO26 uploads require a **size-specific** model type. Using the wrong type will cause the upload to fail or deploy incorrectly.

| Variant | Model Type Flag |
|---|---|
| YOLO26 Nano | `yolo26n` |
| YOLO26 Small | `yolo26s` |
| YOLO26 Medium | `yolo26m` |
| YOLO26 Large | `yolo26l` |
| YOLO26 XLarge | `yolo26x` |

---

## RF-DETR Deployment

RF-DETR uses Roboflow's deployment API (`deploy_to_roboflow`), not the standard upload path. The wizard handles the routing automatically based on the checkpoint extension (`.pth` triggers RF-DETR deployment).

RF-DETR deployment requires:
- `ROBOFLOW_API_KEY`
- `ROBOFLOW_WORKSPACE`
- `ROBOFLOW_PROJECT_IDS`
- Project version (default `1`; set with `--version N`)

---

## Automatic Post-Training Upload

Add a `roboflow` block to a training config YAML to trigger automatic upload at the end of training:

```yaml
roboflow:
  upload: true
  weight: best.pt
```

When `upload: true` is set, YOLOmatic uploads the specified weight after training completes — no manual `yolomatic-upload` step needed.

### Global Auto-Upload Default

You can also set the default in `configs/yolomatic_settings.yaml`:

```yaml
roboflow:
  auto_upload_after_training: true
  auto_upload_weight: best.pt
```

This applies to all future training runs unless overridden in the individual training YAML. See [Settings File](../reference/settings.md) for the full settings reference.

---

## After Upload

Once uploaded, your model is available in the Roboflow dashboard under the selected project. From there you can:

- Deploy to the Roboflow Hosted API for REST inference
- Download for on-device deployment (TFLite, ONNX, etc.)
- Tag a version and share with your team
- Run inference in the Roboflow web UI to inspect results

Related pages: [Configuration](../reference/configuration.md), [RF-DETR](rf-detr.md), [YOLO guide](yolo.md), [Settings File](../reference/settings.md).
