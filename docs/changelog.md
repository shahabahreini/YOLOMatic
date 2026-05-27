---
description: YOLOmatic release history and notable changes.
---

# Changelog

All notable changes to YOLOmatic are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `llms-full.txt` and MkDocs Material documentation site for richer LLM and search-engine discovery.
- Community files — `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CITATION.bib`, issue and PR templates, `FUNDING.yml`, `dependabot.yml`.
- `ai.txt`, `humans.txt` and schema.org JSON-LD on the docs site.

### Changed
- `README.md` rewritten as a scannable landing page; deep technical content moved to the docs site.
- Smart-split (`src/datasets/prepare.py`) algorithm overhauled: rare-class coverage seeding, fixed `image_fill` direction (was inverted), seeded RNG tie-break that actually changes output across seeds, and a clearer warning when every label file is empty.
- Dataset listing parallelized (`src/utils/project.py`, `src/cli/run.py`) — `os.fwalk` for size scans and `ThreadPoolExecutor` for per-dataset summaries.
- TUI render loop (`src/utils/tui.py`) — dirty-flag gating, SIGWINCH-cached terminal size, shared wizard stepper, terminal-state safety guard, module-level imports.

### Fixed
- Ultralytics-platform NDJSON conversion (`src/cli/convert_ndjson.py`) — previously the Labelbox-only extractor produced empty label files for `type: image` rows with normalized `annotations.segments` / `boxes`. New `_extract_ultralytics_objects` plus auto-detected routing recovers all annotations.
- Flat YOLO datasets (`images/` + `labels/` at root, no split keys in `data.yaml`) now report correct image and annotation counts via a flat-structure fallback in `_summarize_yolo` and `_read_yolo_records`.
- `KeyError: 'val'` in `load_dataset_config` when `data.yaml` uses `valid` or omits the validation key.
- YOLO label resolution no longer breaks when an ancestor directory is also named `images` (replaced fragile `str.replace` with path-component-aware `_resolve_label_dir`).
- TUI wizard back-navigation now restores prior selections at each step instead of dropping out to the main menu.

## [4.4.0] — 2026-05-20

### Added
- Ultralytics-platform NDJSON ingest in **Prepare Dataset** (`src/datasets/prepare.py`) with multiprocessing parsing.
- Progress reporting for dataset splitting; multi-worker NDJSON pipeline.
- Interactive wizards for **Configure Model**, **Configure Fine-Tune**, **Clone Config**, and per-model profile selection — each step now supports backward navigation with state preservation.
- Dataset preparation wizard and splitting strategies (`class_balanced`, `smart_balanced`).
- Convert-NDJSON wizard and About screen.

### Changed
- TUI dataset collection improved with wizard progress indicators and richer details panes.
- Augmentation dataset path resolution and label handling.
- Faster dataset listing in the augment command.

## [4.3.0] — 2026-05-14

### Added
- SAM 3.1 (and SAM 3) open-vocabulary segmentation — auto, text-prompted, and box-prompted inference (`uv run yolomatic-sam`).
- SAM 3.1 fine-tuning on COCO-format datasets via the HuggingFace Trainer.
- Benchmark engine with mAP, F1, per-image rankings, UMAP vector scatter, and interactive HTML reports (`uv run yolomatic-benchmark`).
- Albumentations-powered offline augmentation engine with reusable YAML profiles, 20+ transforms, split redistribution, and YOLO/COCO output.
- Detectron2 training support (Faster R-CNN, RetinaNet, Mask R-CNN).
- Roboflow upload/deploy CLI (`uv run yolomatic-upload`) plus opt-in post-training upload.
- AI recommendation flows and unified AI Settings TUI.

### Changed
- Upgraded to Python 3.12.
- Persistent menu-selection memory and refreshed TUI layout proportions.
- Benchmark report UI overhaul — mAP@50:95, modern grouping, prediction confidence inspector, parallel thumbnail generation.

### Fixed
- Disabled MLflow callback in Ultralytics runtime (was hanging some macOS shells).
- Markup parsing for active-model display in the About screen.

## [4.2.0 and earlier]

Earlier history is captured in `git log`. The first public release added Labelbox NDJSON → YOLO/COCO conversion, multi-model config generation, ClearML integration, and TensorBoard launcher.

[Unreleased]: https://github.com/shahabahreini/YOLOMatic/compare/v4.4.0...HEAD
[4.4.0]: https://github.com/shahabahreini/YOLOMatic/releases/tag/v4.4.0
[4.3.0]: https://github.com/shahabahreini/YOLOMatic/releases/tag/v4.3.0
