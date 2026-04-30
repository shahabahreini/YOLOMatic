# Predict Folder Progress And Workers Design

## Goal

Improve `yolomatic-predict --mode folder` so folder prediction reports total progress clearly and can optionally process images with multiple worker processes.

## Design

Folder mode will discover supported image files before prediction and show a Rich progress display with total count, completed count, elapsed time, estimated remaining time, and the current file/status. Single-image prediction remains unchanged.

Multiprocessing is opt-in through `--workers`. The default is `1`, preserving the current stable path for GPU users. When `--workers > 1`, YOLOmatic processes individual images in a process pool. Each worker loads the YOLO model once and predicts assigned images, while the parent process owns all terminal output.

## Error Handling

Per-image failures are collected and reported without stopping the whole folder run. Process setup failures still abort the command. The final summary reports total images, successful images, failed images, elapsed time, worker count, and output directories. Failed images are listed in a compact Rich table.

## Testing

Add focused unit tests for image discovery, worker count validation, and batch result summarization helpers. Run predictor import tests or equivalent targeted tests after implementation.
