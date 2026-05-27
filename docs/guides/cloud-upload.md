---
description: Upload YOLO checkpoints and deploy RF-DETR checkpoints to Roboflow from YOLOmatic.
---

# Cloud Upload

YOLOmatic can upload YOLO checkpoints and deploy RF-DETR checkpoints to
Roboflow.

```sh
uv run yolomatic-upload
```

## Credentials

Use `.env`:

```env
ROBOFLOW_API_KEY=
ROBOFLOW_WORKSPACE=
ROBOFLOW_PROJECT_IDS=
```

`ROBOFLOW_WORKSPACE` is the workspace slug. `ROBOFLOW_PROJECT_IDS` can contain
one or more target project IDs.

## Post-Training Upload

Add this block to a training config:

```yaml
roboflow:
  upload: true
  weight: "best.pt"
```

Related pages: [Configuration](../reference/configuration.md), [RF-DETR](rf-detr.md), [YOLO guide](yolo.md).
