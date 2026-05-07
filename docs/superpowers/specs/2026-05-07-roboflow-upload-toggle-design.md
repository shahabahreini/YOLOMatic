# Roboflow Upload Toggle Design

## Purpose
Allow users to optionally upload their trained YOLO model weights (`best.pt` or `last.pt`) automatically to Roboflow after the training process completes.

## Configuration
The training configuration YAML will support a new `roboflow` block:
```yaml
roboflow:
  upload: true       # Set to true to enable automatic upload
  weight: "best.pt"  # The weight file to upload (e.g., "best.pt", "last.pt")
```

## Integration in `yolo_trainer.py`
1. After the validation and export steps are completed successfully, check if `config.get("roboflow", {}).get("upload")` is `True`.
2. Determine the path to the weight file using the `run_dir` and the `weight` parameter specified in the config.
3. Extract `workspace` and `project` IDs from `dataset_config["roboflow"]` (available when the dataset was downloaded via Roboflow).
4. Identify the `model_type` from the YOLO model used.
5. Reuse the upload logic from `src.cli.upload` (e.g. `upload_model`, `build_candidate`, `stage_upload_candidate`) to perform the upload headless/automatically.
6. Provide clear console feedback about the upload status.

## Fallbacks & Error Handling
- If `upload: true` but the dataset config lacks Roboflow `workspace`/`project`, the upload will log a clear error and skip gracefully so as not to crash the end of the training process.
- Ensure any missing API keys or upload errors do not fail the overall training pipeline retroactively.

## Testing
- Add a mock test or verify locally that setting the flag successfully initiates the upload pipeline without prompting for extra input.