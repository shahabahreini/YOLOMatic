## [Unreleased] - 2026-05-28

### Changed
- Updated `wrapt` dependency to 2.2.0 in `uv.lock`.
- Preserved user-selected TensorRT export workspace values instead of clamping
  them to a local GPU memory heuristic.

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

## [v4.2.0] - 2026-05-09

### Changed
- Deprecated YOLO-NAS training support due to dependency conflicts.
- Updated project metadata and GitHub topics.

## [v4.1.0] - 2026-05-08

### Added
- Added Detectron2 support for training and prediction.
- Implemented global settings management via `yolomatic_settings.yaml`.
- Added RF-DETR model support for detection and segmentation.
- Added automated Roboflow model upload functionality.
- Added "Clone Config" functionality to CLI.

### Changed
- Consolidated settings customization into a unified UI.
- Updated dependencies including `boto3`, `roboflow`, and `ultralytics`.

## [v4.0.0] - 2026-04-29

### Added
- Added checkpoint fine-tuning workflow to TUI.
- Implemented interactive TUI wizard for Roboflow model uploads.
- Added CUDA environment repair preflight check using `uv sync`.
- Added export parameter validation and warnings.

### Changed
- Updated major dependencies including `boto3`, `ultralytics`, and `roboflow`.
- Redesigned interactive configuration flow with a unified editor.
- Updated PyTorch and CUDA dependencies to version 2.11.0+cu128.

### Fixed
- Disabled Ultralytics ClearML callbacks to prevent runtime failures.
- Fixed project root resolution to walk up the directory tree.

## [v3.1.0] - 2026-04-24

### Added
- Added comprehensive YOLO training parameter definitions.
- Implemented dependency health check for critical packages.
- Added model family performance charts to TUI.

### Changed
- Enhanced project documentation and configuration guidance.
- Updated model descriptions and performance metrics.

### Fixed
- Bootstrapped NVIDIA library path for torch inspection.
- Improved error handling for TUI sub-commands.

## [v3.0.0] - 2026-04-24

### Added
- Added automated TensorBoard dashboard generation.
- Implemented rich-based TUI for consistent CLI menus.
- Added automated CUDA environment repair preflight.

### Changed
- Restructured project into modular packages.
- Overhauled CLI menus with split layout and breadcrumbs.
- Used `pyproject.toml` as the single source of truth for versioning.

### Fixed
- Improved dataset path resolution logic.
- Enforced strict NumPy compatibility for training.

## [v2.1.0] - 2026-04-23

### Added
- Added `yolomatic-upload` CLI tool for Roboflow model registration.
- Added `yolomatic-predict` command for inference.
- Added routing for YOLO-NAS configurations.

## [v2.0.1] - 2026-03-05

### Added
- Added DatasetAnalyzer for YOLO NAS config generation.

### Changed
- Updated `yolomatic` dependency to v2.0.0.
- Updated packaging configuration for explicit package discovery.

## [v2.0.0] - 2026-03-05

### Added
- Added automated GitHub release workflow.
- Added versioning CLI for semantic version updates.
- Added `yolomatic-tensorboard` CLI command.
- Added segmentation model support and dataset type detection.

### Changed
- Rewrote README.md with YOLOmatic branding.
- Pinned Python version to 3.10.
- Restructured project into modular packages.

## [v1.1.0] - 2026-01-18

### Added
- Added YOLO26 model support with edge optimization.
- Added YOLOv12 model support.
- Added YOLO NAS integration.

### Changed
- Enhanced configuration system and UI/UX.
- Integrated `Rich` library for console logging.

## [v1.0.0] - 2024-12-22

### Added
- Implemented interactive arrow-key navigation menu system.
- Added real-time model comparison tables.
- Added YOLOv11 support.

### Changed
- Migrated UI to `rich` and `blessed` libraries.
- Updated performance metrics for YOLOv11.
