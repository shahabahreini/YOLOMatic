# Execution Ledger: Backend Edge-Case Fixes

**Plan:** `plans/backend-edge-cases-audit.md`  
**Status:** Completed  
**Started:** 2026-09-25  
**Completed:** 2026-09-25  

## 1. Coverage Ledger

| Task ID | Name | Priority | Status | Acceptance Check / Evidence |
|---|---|---|---|---|
| **AUD-01** | Cross-device export relocation using `shutil.move` in `src/cli/export.py` | must | done (verified) | Verified with `tests/test_export_cli.py::ExportCliTests::test_export_relocation_calls_shutil_move`. Models exported across separate filesystems/mount points relocate cleanly without POSIX `EXDEV` errors. |
| **AUD-02** | Validation split starvation prevention in `src/datasets/prepare.py` | must | done (verified) | Verified with `tests/test_prepare_dataset.py::SplitDiscoveryTest::test_target_counts_prevents_val_starvation_on_small_datasets`. Clamps `val = max(1, val)` when `val_ratio > 0` and `total >= 2`, guaranteeing non-empty validation splits. |
| **AUD-03** | Error propagation on trainer failure in `rfdetr_trainer.py` & `detectron2_trainer.py` | must | done (verified) | Verified with `tests/test_rfdetr_trainer.py::RFDETRTrainerTests::test_main_raises_system_exit_on_failure` and `tests/test_detectron2_trainer.py::Detectron2TrainerTests::test_main_raises_system_exit_on_failure`. Failures raise `SystemExit(1)` causing runner `_safe_subcommand` to catch error code, display error panel, and pause. |
| **AUD-04** | Resilient non-UTF8 label reading in `src/datasets/validate.py` | should | done (verified) | Verified with `tests/test_dataset_validate.py::ValidateDatasetTests::test_handles_non_utf8_label_file`. `read_text(encoding="utf-8", errors="replace")` prevents `UnicodeDecodeError` crash and records validation error cleanly. |
| **AUD-05** | Filter degenerate bounding boxes in `src/datasets/prepare.py` | should | done (verified) | Verified with `tests/test_prepare_dataset.py::SplitDiscoveryTest::test_read_yolo_annotations_filters_degenerate_bboxes` and `test_read_coco_records_filters_degenerate_bboxes`. Degenerate bboxes with `w <= 0` or `h <= 0` are filtered with warnings, preventing `NaN` gradients. |

## 2. Deviations
*None. All tasks implemented exactly as specified in the audit plan.*

## 3. QC Results
- **Unit & Integration Tests:** 109/109 related tests passing in 2.36s.
- **Repository Regression Suite:** 392/395 tests passing across entire repo (remaining 3 are tests requiring optional CV library `albumentations` not installed in local environment).
- **Code Quality:** Fully backwards-compatible; no signature regressions or breaking changes.

## 4. Unfinished Items
*None. All 5 backend edge-case tasks are fully executed and verified.*

## 5. Handover Summary
- **Context:** Executed the 5 backend audit fixes from `plans/backend-edge-cases-audit.md`.
- **Decisions:**
  - Export artifact relocation uses `shutil.move` across mount points.
  - Validation split allocation guarantees at least 1 validation item when `val_ratio > 0` and `total >= 2`.
  - RF-DETR and Detectron2 trainer failures raise `SystemExit(1)` to prevent immediate screen clearing in the TUI runner.
  - Non-UTF8 dataset labels are read resiliently with `errors="replace"`.
  - Non-positive bounding box dimensions (`w <= 0` or `h <= 0`) are filtered during dataset ingestion and export.
- **State:** 5/5 tasks completed and verified with automated test coverage.
