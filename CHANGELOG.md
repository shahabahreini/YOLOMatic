## [7.0.0] - 2026-11-15

A hardening release: dozens of correctness and data-safety fixes, honest exit
codes, working Ctrl-C, a reworked TUI, a much smaller default install, and
Detectron2 / SAM 3.1 training that actually trains. It contains breaking
changes; see **Migrating from 6.x** below.

### Migrating from 6.x

- **Import path:** the package is now `yolomatic` (it was literally `src`).
  `from src.cli.run import main` becomes `from yolomatic.cli.run import main`.
  Console commands are unchanged.
- **Optional extras:** the base install no longer includes TensorFlow, ONNX,
  OpenVINO, CoreML, NCNN, TensorRT or umap-learn. Install what you use:
  `yolomatic[export]` (ONNX/TF/CoreML/OpenVINO/NCNN), `yolomatic[tensorrt]`,
  `yolomatic[clearml]`, `yolomatic[viz]` (UMAP), `yolomatic[sam]`, or
  `yolomatic[all]`. A missing extra now prints the exact install command.
- **Exit codes:** every command exits `0` on success, `1` on failure, `2` for
  usage errors or a missing terminal, `130` when cancelled with Ctrl+C. Several
  commands used to exit `0` after failing.
- **`bump` removed:** use `uv version --bump patch|minor|major` (also updates
  `uv.lock`).
- **API keys:** keys saved in Settings > AI move to your OS credential store
  (Keychain, Credential Locker, Secret Service) the next time settings are
  saved; `configs/yolomatic_settings.yaml` keeps a `keyring:` reference.
  Without a credential store the previous format is used.
- **`yolomatic-upload-dataset --type`** no longer defaults to
  instance-segmentation; the task is detected from the dataset.
- **Detectron2 prediction** needs the `config.yaml` / `yolomatic_detectron2.json`
  that training now writes into each run folder; checkpoints from 6.x runs need
  retraining (or a copied `config.yaml`).

### Added
- `yolomatic <command>`: run any tool without the TUI (`train`, `predict`,
  `export`, `benchmark`, `augment`, `prepare`, `convert`, `sam`, `upload`,
  `upload-dataset`, `tensorboard`, `ultralytics`), plus `--version`, `--debug`,
  `--no-color` and `--log-file`. New `yolomatic-export` and `yolomatic-augment`
  scripts.
- Rotating session log in the OS user log folder (`YOLOMATIC_LOG_DIR`
  overrides); error panels show its path, tracebacks only with `--debug`.
- TUI: `?` keyboard help, accurate footer hints, readline editing with Tab
  path completion, plan summaries visible on every confirmation screen, a
  "terminal too small" notice, opt-in mouse capture (`YOLOMATIC_MOUSE=1`).
- TensorBoard launcher picks a free port and accepts `--port` / `--logdir`.
- Custom dynamic-batch TensorRT engines carry Ultralytics metadata.
- Python 3.13 support.

### Fixed
- Automatic Roboflow upload after YOLO training never worked.
- Dataset combiner crashed on every run (after deleting the old output); it
  now writes only inside `datasets/` and swaps in a finished merge.
- Benchmark ground truth for YOLO detection labels was parsed as polygons, so
  detection mAP/precision/recall were wrong; matching is now class-aware and
  COCO category ids are mapped correctly.
- Data-loss bugs: a failed directory swap could delete migrated files,
  `prepare --overwrite` could delete the source dataset, cache cleanup deleted
  user `.npy` files, and convert/export overwrites accepted unsafe paths.
- Ctrl-C now stops augmentation, benchmark, SAM, conversion, preparation and
  prediction work instead of leaving it running in the background.
- Outputs (datasets, settings, profiles, downloads, COCO files) are written
  atomically or staged, so a crash never leaves half-written results.
- A TUI started without a terminal no longer spins at 100% CPU.
- Errors no longer flash past: failures always pause with the message and a hint.
- Detectron2: RetinaNet class count, repeat dataset registration, a real COCO
  evaluator, converted-dataset image paths, and prediction that loads the
  trained architecture and saves results.
- SAM 3.1 fine-tuning had no loss and failed on batches with different object
  counts; SAM prediction wrote invalid COCO and dropped detection classes.
- Many smaller fixes: export device default on Macs/CPU, labels for the
  `train/images` + `train/labels` upload layout, duplicate Roboflow batches
  after server errors, NDJSON class-id order, GPU probe timeouts, Gemini key
  leaking into error text, UTF-8 file I/O on Windows, stale conversion caches.

### Changed
- Upper bounds on the major ML dependencies; `scikit-learn`, `requests`,
  `platformdirs` and `keyring` declared; unused `termcolor`, `tabulate` and
  `terminaltables` removed.
- Settings values are type-checked (`enabled: "no"` is now false).
- CI runs on pull requests with pytest, a Python 3.13 job, lockfile, type-check
  and dependency-audit jobs; `.pre-commit-config.yaml` added.

Releases 6.1.0-6.4.0 are described in the git history.

## [6.0.0] - 2026-06-19

### Added
- RF-DETR keypoint/pose support (preview) via `RFDETRKeypointPreview` (`rfdetr-pose` family, `RF-DETR-Keypoint` variant), with keypoint loss/schema parameters and automatic conversion of YOLO pose datasets to COCO keypoint JSON.
- Task-filtered, per-family interactive configuration pages: RF-DETR and Detectron2 now expose full parameter pages (documented defaults, ranges, choices, and help) in the TUI, like YOLO. Parameters that do not apply to the selected task (e.g. mask params for detection, pose params for non-pose) are hidden.
- Centralized training parameter catalogs in `src/config/parameters.py` with a `parameters_for(family, task)` helper.
- Task-aware dataset preparation: `prepared_format_for_family()` selects COCO for RF-DETR pose and Detectron2, YOLO otherwise, so both RF-DETR and YOLO load COCO and YOLO datasets correctly.
- Additional Detectron2 solver/input controls (warmup iters, gamma, steps, momentum, weight decay, RoI batch size, score threshold, train sizes).

### Changed
- Hardened pretrained-weight auto-download for RF-DETR and Detectron2 with explicit messaging and clear errors when local weights are missing.
- Bumped project version to 6.0.0.

### Removed
- Stray repository clutter (accidental log dumps and downloaded model weights at the project root).

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
