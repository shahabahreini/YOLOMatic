---
description: YOLOmatic release history and notable changes.
---

# Changelog

All notable changes to YOLOmatic are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-06-10

### Added
- Added support for YOLO pose estimation tasks (YOLO26-pose, YOLOv11-pose, YOLOv8-pose model families).
- Implemented authoritative pose detection via `kpt_shape` in dataset yaml.
- Added TensorRT dynamic batch compilation mode (`trt_dynamic_batch` parameter).
- Added caching mechanism for dataset summaries to reduce redundant computations and disk I/O.

### Changed
- Updated `wrapt` dependency to 2.2.0 in `uv.lock`.
- Preserved user-selected TensorRT export workspace values instead of clamping them to a local GPU memory heuristic.
- Made the standalone export wizard load checkpoint details before export and hide options that do not apply to the selected model task or target format.

### Fixed
- Ensured live TUI updates refresh the UI immediately to prevent stale rendering.

## [v5.0.0] - 2026-05-27

### Added
- Implemented interactive dataset conversion wizard for NDJSON formats.
- Added support for Ultralytics-platform NDJSON conversion and header parsing.
- Implemented rare-class seeding for smart balanced dataset splits.
- Added flat-structure fallback for datasets lacking explicit split keys.
- Added navigation support to CLI configuration wizards using step-machine logic.
- Added progress reporting to dataset splitting functions.
- Added multiprocessing support for NDJSON dataset preparation.

### Changed
- Refactored CLI to simplify rich text handling.
- Optimized TUI rendering and layout performance with layout caching.
- Parallelized dataset inspection and directory listing using `ThreadPoolExecutor`.
- Standardized YOLO label directory resolution and parsing logic.
- Updated project branding assets, including favicons and logo formats.
- Restructured documentation into an MkDocs Material site.
- Updated dependencies including `tensorflow`, `softprops/action-gh-release`, `actions/checkout`, `actions/setup-python`, and `astral-sh/setup-uv`.

### Fixed
- Guarded conversion result elapsed display in CLI.
- Corrected demo GIF filenames and image links in README.
- Removed unused imports and constants across core and CLI modules.
- Disabled MLflow callback in Ultralytics runtime to prevent import conflicts.

## [v4.4.0] - 2026-05-20

### Added
- Implemented Labelbox NDJSON to YOLO/COCO dataset converter.
- Added unified augmentation profile editor with integrated guidance.
- Added SAM 3.1 support, benchmarking, and augmentation capabilities.
- Implemented offline Albumentations-based dataset augmentation engine.
- Added max workers option to benchmark CLI for parallel execution.
- Added prediction confidence visualization to benchmark reports.

### Changed
- Upgraded project to Python 3.12.
- Overhauled report UI grouping and TUI layout proportions.
- Replaced object size sensitivity heatmap with grouped horizontal bar charts.
- Updated dependencies including `blessed`, `importlib-metadata`, and `ultralytics`.

### Fixed
- Updated benchmark tests for Plotly JSON encoding compatibility.
- Cleared environment variables when mocking subprocess in update flow tests.
- Removed unused variables and imports in augmentation modules.

## [v4.3.0] - 2026-05-14

### Added
- Implemented benchmark model evaluation and vector analysis feature.
- Added SAM 3.1 segmentation inference and fine-tuning commands.
- Added "Finish" option to TUI menus for streamlined workflows.
- Implemented dynamic color interpolation for metrics in HTML reports.
- Added full-screen transient live progress display for CLI benchmarks.

### Changed
- Refined benchmark table layout and dynamic sizing.
- Migrated dependency upgrade logic to `uv`.
- Updated Python requirement to 3.12.

### Fixed
- Resolved circular dependencies in benchmark and CLI modules.
- Removed unused imports and variables across core modules.

## [v4.2.0 and earlier]

Earlier history is captured in `git log`. The first public release added Labelbox NDJSON → YOLO/COCO conversion, multi-model config generation, ClearML integration, and TensorBoard launcher.
