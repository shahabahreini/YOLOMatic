# Plan: Augmentation Output Progress Feedback and Windows Swap Resilience

| Field | Value |
|---|---|
| Objective | Eliminate UI freezing during dataset assembly, optimize output writing I/O, make directory swapping resilient to Windows NTFS file locks (WinError 5), and provide instant recovery for orphaned staged datasets. |
| Status | Ready for review |
| Version | 1.1.0 |
| Created | 2026-09-06 |

## 1. Objective & Definition of Done
- Goal: Provide continuous, transparent progress feedback across all phases of dataset augmentation (generation, split writing, finalizing), eliminate redundant disk I/O when staging files, resolve Windows `[WinError 5] Access is denied` errors during atomic directory replacement, and enable seamless recovery of completed staged runs.
- Done when:
  1. The augmentation TUI displays distinct, accurate progress indicators throughout the output writing phase (split name, item count, percentage, dynamic time remaining) rather than freezing at 100% `Writing output dataset... 0:00:00`.
  2. Dataset file staging uses zero-copy/fast filesystem moving (`os.replace` / `shutil.move`) instead of duplicate disk copying (`shutil.copy2`), reducing disk write volume and I/O time by over 90%.
  3. Directory swapping uses a resilient swapping utility (`safe_directory_swap`) with exponential backoff retries and fallback item-by-item migration to survive Windows Defender, Windows Search Indexer, and shell handle locks without raising `[WinError 5] Access is denied`.
  4. Backup directory cleanup is safely guarded (`ignore_errors=True`) so lingering background scanner locks on old backups never trigger post-swap crashes.
  5. Misleading error suggestions ("Check that albumentations is installed...") are replaced with accurate diagnostics tailored to filesystem/permission errors.
  6. Existing orphaned `.augmenting-*` directories containing complete outputs (such as the user's unfinalized 24,402-image run) can be detected and recovered immediately through the CLI without re-running augmentation.
- Success measures:
  - Zero frozen UI states during 10,000+ item dataset writing.
  - 100% success rate on Windows directory swap under simulated transient file locking and fallback conditions.
  - 0 lost augmentation runs: completed staged data is preserved and recoverable if swap fails.
- Must not happen:
  - Staged augmented images and labels must not be deleted when a destination swap encounters a filesystem lock.
  - Output dataset integrity (split balances, label formatting, `data.yaml` accuracy) must not regress.
  - The CLI must not crash without explaining the true failure cause.

## 2. Context & Constraints
- Background:
  - In `src/augmentation/engine.py`, `run_augmentation` executes worker processes that stage generated images and metadata JSON in `.augmenting-<uuid>/groups/`.
  - Once workers complete, `run_augmentation` calls `progress_callback(total_source, total_source, "Writing output dataset...")` and enters a synchronous loop copying 24,000+ images via `shutil.copy2` and parsing/writing labels.
  - Rich Progress marks the task completed (100%, 0:00:00 remaining), while disk I/O continues silently for 10-20 minutes with zero callback updates, creating the appearance of an application freeze.
  - At the conclusion of output writing, `build_root.rename(out_root)` attempts to rename the staging directory to the target dataset folder.
  - On Windows NTFS, newly written files trigger real-time scanning by Windows Defender (`MsMpEng.exe`) and Windows Search Indexer (`SearchIndexer.exe`), holding temporary read handles without `FILE_SHARE_DELETE`. Calling Win32 `MoveFileEx` during this window immediately raises `PermissionError: [WinError 5] Access is denied`.
  - If a backup directory was created, line 1421 calls `shutil.rmtree(backup_root)` without `ignore_errors=True`, which can trigger a secondary `WinError 5` crash if indexers hold handles in the backup.
  - The CLI catches this exception and displays a hardcoded suggestion to check if Albumentations is installed, which is misleading.
- Constraints:
  - Must remain strictly cross-platform (Windows, macOS, Linux).
  - Must not require administrator privileges.
  - Must preserve backwards compatibility with `SplitConfig`, `AugmentationStats`, and public signatures.
- Stakeholders:
  - End users running large-scale offline dataset augmentation on Windows and Linux workstations.

## 3. Strategy
- Chosen path:
  1. **Two-Stage Progress Reporting**: Decouple the worker augmentation phase from the output assembly phase. Update `progress_callback` to emit structured stage information, and configure Rich Progress in `src/cli/augment.py` to reset and track output file writing dynamically with dynamic task descriptions.
  2. **In-Tree Fast Move**: Replace `shutil.copy2` with `os.replace` within the staging tree. Because `groups/` and `output/` share the same temporary root on the same filesystem, `os.replace` is an instantaneous directory entry update, eliminating tens of gigabytes of redundant disk writes and dramatically reducing Windows Defender scanning load. Handle semantic segmentation decoding before moving the file.
  3. **Resilient Directory Swap Utility (`safe_directory_swap`)**: Implement an atomic swap function with an extended 8-attempt exponential backoff schedule (0.2s to 5.0s, ~17s total window) to allow antivirus and indexing locks to clear. If atomic rename still fails, fall back to recursive item-by-item migration into the target directory. Ensure `backup_root` removal ignores errors.
  4. **Rescue Tool for Orphaned Datasets**: Provide a CLI recovery prompt that detects existing `.augmenting-*` directories with complete outputs and finalizes them directly into `datasets/<target>/`.
- Why it wins:
  - Eliminates the root causes of both the UI freeze and the `WinError 5` crash.
  - Cuts dataset writing time from minutes to seconds.
  - Recovers the user's existing 24,402 augmented images without requiring a 22-minute re-run.
- Alternatives rejected:
  - *Direct writing from worker processes*: Would require complex cross-process synchronization and breaks reproducible split ordering.
  - *Blind delay (`time.sleep(5)`)*: Brittle and unscientific; exponential backoff with retry loop responds immediately when locks release.

## 4. Scope
- New:
  - `src/utils/fs.py`: `safe_directory_swap` utility with retry backoff, rollback protection, recursive migration fallback, and safe backup deletion.
  - `src/augmentation/recovery.py`: Helper to scan and finalize orphaned `.augmenting-*` runs.
- Updates to existing:
  - `src/augmentation/engine.py`:
    - Fast `os.replace` item transfer instead of `shutil.copy2`.
    - Granular progress reporting during split and label creation.
    - Integration with `safe_directory_swap`.
    - Preservation of staging folder on fatal swap errors.
  - `src/cli/augment.py`:
    - Multi-stage Rich Progress bar supporting dynamic phase descriptions.
    - Context-aware error panel distinguishing filesystem lock/permission errors from missing packages.
    - Startup check offering to recover existing orphaned augmentation runs.
- Explicitly out of scope:
  - Changing augmentation transform algorithms or Albumentations pipeline definitions.
  - Modifying the underlying YOLO split logic or group key regex assignment.
- Change policy:
  - All public function signatures remain backwards-compatible. Progress callbacks support optional phase metadata or transparent progress updates.

## 5. Assumptions
| # | Assumption | Validated by |
|---|---|---|
| A1 | `groups_root` and `build_root` reside on the same filesystem volume. | `stage_root` is created in `source_dataset_path.parent`, guaranteeing identical volume. |
| A2 | Transient `[WinError 5]` on directory rename is caused by Windows antivirus/indexer handles or directory pending delete states. | Documented Windows NTFS semantics; standard fix across build tools (git, npm, python installers). |
| A3 | The user's completed output folder `D:\Shahab\YOLOMatic\datasets\.zenlabel-export_augmented.augmenting-63d5fb64ccba436897653d1f6e06c4a6\output` is intact on disk. | In `run_augmentation`, `shutil.rmtree(stage_root)` is skipped when line 1415 raises, leaving the output folder intact. |

## 6. Phases & Tasks

### Phase 1: High-Performance Output Assembly & Progress Granularity
Checkpoint: `run_augmentation` moves files instantaneously and emits continuous progress updates.

**Task 1.1 — Fast In-Place File Transfer**
- What: Replace `shutil.copy2` with `os.replace` when moving staged images into split folders.
- How:
  - In semantic segmentation mode (`write_as_semantic`), decode the image from `staged_image_path` before moving, rasterize masks, and then call `os.replace(staged_image_path, img_path)`.
  - In standard bbox/seg/pose modes, call `os.replace(staged_image_path, img_path)` directly.
  - Read and parse `staged_image_path.with_suffix(".json")` and remove the JSON file.
  - Wrap in a fallback to `shutil.copy2` if `os.replace` raises an `EXDEV` cross-device error.
- Where: `src/augmentation/engine.py` (inside `run_augmentation`).
- Depends on: None.
- Parallel: No.
- Effort: 30 minutes.
- Priority: High.
- Done when: No duplicate images exist on disk during staging; assembly of 24,000 files completes in seconds.

**Task 1.2 — Granular Progress Updates During Dataset Assembly**
- What: Emit frequent, descriptive progress callbacks during split writing, label creation, and metadata generation.
- How:
  - Calculate `total_out = sum(len(items) for items in split_data.values())`.
  - In the items loop, call `progress_callback(written_count, total_out, f"Writing {split_name} ({idx+1}/{len(items)})...")` every 50-100 items or 1%.
  - Emit clear messages for metadata writing ("Writing data.yaml & split manifest...") and directory swap ("Finalizing dataset location...").
- Where: `src/augmentation/engine.py`.
- Depends on: Task 1.1.
- Parallel: No.
- Effort: 30 minutes.
- Priority: High.
- Done when: Progress callback receives monotonic increments and informative status strings throughout the writing phase.

**Task 1.3 — Dynamic Multi-Stage Progress UX in CLI**
- What: Update `_run_with_progress` in `src/cli/augment.py` to seamlessly adapt to generation and assembly phases.
- How:
  - Replace hardcoded `TextColumn("[cyan]Augmenting[/cyan]")` with `TextColumn("[cyan]{task.description}[/cyan]")`.
  - When switching from augmentation to writing output (detected via message or state), reset `task.total` and `task.completed` to the output dataset total and set `task.description = "Writing Dataset"`.
  - Set `task.description = "Finalizing"` during metadata and swap operations.
  - Display current split name and file rate in the progress text.
- Where: `src/cli/augment.py`.
- Depends on: Task 1.2.
- Parallel: No.
- Effort: 45 minutes.
- Priority: High.
- Done when: Rich Progress bar smoothly transitions across phases without displaying 100% completion or 0:00:00 time remaining prematurely.

### Phase 2: Windows Atomic Swap Resilience & Error Diagnostics
Checkpoint: Directory swapping succeeds on Windows even under file indexer/antivirus locking, and errors are cleanly diagnosed.

**Task 2.1 — Implement `safe_directory_swap` Utility**
- What: Create a bulletproof directory swap function designed for Windows NTFS locking quirks.
- How:
  - Create `src/utils/fs.py` with `safe_directory_swap(src: Path, dst: Path, max_retries: int = 8, backoff_base: float = 0.2)`.
  - If `dst` exists: rename `dst` to `.backup-<uuid>` with retry loop.
  - Rename `src` to `dst` with exponential backoff retry loop (0.2s, 0.4s, 0.8s, 1.5s, 2.0s, 3.0s, 4.0s, 5.0s = ~17s total window).
  - If atomic directory rename fails after retries with `PermissionError` / `OSError`:
    - Automatically fall back to item-by-item recursive move from `src` into `dst`.
  - On unrecoverable failure: restore backup directory and preserve `src`.
  - On success: remove backup directory using `shutil.rmtree(backup_root, ignore_errors=True)`.
- Where: `src/utils/fs.py`.
- Depends on: None.
- Parallel: Yes (with Phase 1).
- Effort: 45 minutes.
- Priority: Critical.
- Done when: Directory swap succeeds reliably across all platforms, including under simulated file lock retries and item-by-item fallback.

**Task 2.2 — Integrate `safe_directory_swap` into `run_augmentation`**
- What: Replace naive `rename` calls in `run_augmentation` with `safe_directory_swap`.
- How:
  - Replace lines 1410-1422 in `src/augmentation/engine.py` with `safe_directory_swap(build_root, out_root)`.
  - If swapping raises an exception, log the exact path of `stage_root` and re-raise with a clear message preserving the folder.
- Where: `src/augmentation/engine.py`.
- Depends on: Task 2.1.
- Parallel: No.
- Effort: 20 minutes.
- Priority: Critical.
- Done when: `run_augmentation` uses `safe_directory_swap` and never deletes a completed staging directory on swap failure.

**Task 2.3 — Accurate Error Diagnostics in CLI**
- What: Fix the error reporting in `_run_with_progress` to provide accurate context for filesystem and locking errors.
- How:
  - Check if the exception is a `PermissionError`, `OSError` with `winerror in (5, 32, 183)`, or disk space error (`ENOSPC` / `winerror 112`).
  - Render an informative panel explaining Windows file locking, suggesting closing open Explorer windows or terminals, and showing the location of preserved staged files.
  - Only show Albumentations/OpenCV hints if the exception is genuinely an `ImportError` or OpenCV wheel error.
- Where: `src/cli/augment.py`.
- Depends on: Task 2.2.
- Parallel: No.
- Effort: 30 minutes.
- Priority: High.
- Done when: `[WinError 5]` displays actionable file-locking guidance rather than irrelevant package installation hints.

### Phase 3: Rescue & Recovery for Orphaned Datasets
Checkpoint: Users can recover already-generated augmentations without re-running long workflows.

**Task 3.1 — Staged Dataset Discovery & Recovery Function**
- What: Provide functionality to scan for orphaned `.augmenting-*` folders with valid `output/` directories and finalize them into `datasets/<output_name>`.
- How:
  - Create `find_recoverable_augmentations(datasets_dir: Path) -> list[tuple[Path, str]]` in `src/augmentation/recovery.py`.
  - Create `recover_augmentation(stage_dir: Path, target_name: str) -> bool`.
- Where: `src/augmentation/recovery.py`.
- Depends on: Task 2.1.
- Parallel: Yes.
- Effort: 40 minutes.
- Priority: Medium.
- Done when: Unfinalized runs are detected and can be migrated to their destination folder with verified dataset structure.

**Task 3.2 — CLI Recovery Prompt**
- What: Add a recovery prompt in the CLI augmentation menu if recoverable staging folders are detected.
- How:
  - When entering the Augment Dataset workflow, check for recoverable folders.
  - If found (e.g. `zenlabel-export_augmented.augmenting-63d5fb64ccba436897653d1f6e06c4a6`), prompt: `"Found an unfinalized augmentation run from a previous session (24,402 images). Recover now?"`.
  - Finalize immediately if accepted.
- Where: `src/cli/augment.py`.
- Depends on: Task 3.1.
- Parallel: No.
- Effort: 30 minutes.
- Priority: Medium.
- Done when: The user can finalize their existing 24,402 augmented dataset directly from the menu.

### Phase 4: Automated Verification & Test Coverage
Checkpoint: All new behaviors and failure edge cases are verified with passing automated tests.

**Task 4.1 — Unit Tests for `safe_directory_swap`**
- What: Comprehensive unit tests for atomic swap, retry backoff, fallback item move, and rollback.
- How:
  - Test clean rename.
  - Test target replacement with backup cleanup.
  - Test simulated `PermissionError` with retry success.
  - Test fallback to item-by-item move when rename fails permanently.
  - Test rollback to backup when swap fails mid-operation.
- Where: `tests/test_fs_utils.py`.
- Depends on: Task 2.1.
- Parallel: Yes.
- Effort: 45 minutes.
- Priority: High.
- Done when: All filesystem swap edge cases pass in test suite.

**Task 4.2 — Integration Tests for Progress & Move Optimization**
- What: Verify `run_augmentation` emits multi-stage progress and performs fast moves.
- How:
  - Track calls to `progress_callback` during `run_augmentation`.
  - Assert that progress callback is invoked during the output writing phase with monotonic progress counts and valid split names.
  - Assert that `groups_root` has no lingering duplicate files.
- Where: `tests/test_augmentation_engine.py`.
- Depends on: Tasks 1.1, 1.2.
- Parallel: No.
- Effort: 30 minutes.
- Priority: High.
- Done when: Progress and move tests pass cleanly.

**Task 4.3 — Orphan Recovery Tests**
- What: Test detection and recovery of orphaned staged datasets.
- How:
  - Synthesize an aborted staging directory with valid `output/` contents.
  - Call `recover_augmentation` and assert destination directory is created with all images and labels intact.
- Where: `tests/test_augmentation_recovery.py`.
- Depends on: Task 3.1.
- Parallel: No.
- Effort: 30 minutes.
- Priority: Medium.
- Done when: Recovery tests pass.

## 7. Risks & Countermeasures
| Risk | Impact | Countermeasure |
|---|---|---|
| Target directory is locked by an external process with exclusive lock during swap | Finalization fails | Extended 8-attempt retries with exponential backoff (~17s window) give transient locks time to clear; recursive file-by-file fallback bypasses folder node lock; staged data is preserved and user is guided to manually finalize or recover. |
| Progress callback signature mismatch in custom caller | TypeError during callback invocation | Retain standard `(done: int, total: int, msg: str)` signature while updating `done` and `total` dynamically for each stage. |
| Cross-device move failure if temporary staging is on another mount | `os.replace` raises `EXDEV` | Catch `OSError` (errno 18 / EXDEV) and fall back to `shutil.move` / `shutil.copy2`. |
| Old backup directory deletion blocked by antivirus | Unhandled exception after successful swap | Guard `shutil.rmtree(backup_root, ignore_errors=True)` so non-critical backup cleanup never crashes a successful run. |

## 8. Verification & Replanning
- Per task:
  - Run targeted unit tests after completing each task (`pytest tests/test_fs_utils.py`, `pytest tests/test_augmentation_engine.py`).
- Overall:
  - End-to-end dry run of dataset augmentation with progress verification.
  - Test simulation of Windows file lock during destination swap.
  - Verify recovery tool on a simulated interrupted `.augmenting-*` folder.
- Replan when:
  - If Windows permissions prevent item-by-item fallback move, investigate volume-level temporary folder placement.

## 9. Traceability
| Objective | Covered by tasks |
|---|---|
| Transparent progress feedback during output writing | Task 1.2, Task 1.3, Task 4.2 |
| Eliminate redundant disk I/O during dataset staging | Task 1.1, Task 4.2 |
| Prevent `[WinError 5]` failures on directory swap | Task 2.1, Task 2.2, Task 4.1 |
| Meaningful diagnostics for filesystem/permission errors | Task 2.3 |
| Recover orphaned staged runs (saving 24,402 images) | Task 3.1, Task 3.2, Task 4.3 |

## 10. Handover Summary
- Executor must know:
  - The staged dataset path in the user's report (`D:\Shahab\YOLOMatic\datasets\.zenlabel-export_augmented.augmenting-63d5fb64ccba436897653d1f6e06c4a6\output`) contains the completed augmented dataset from their 21-minute run. Task 3.1/3.2 will allow immediate recovery without re-running.
  - Moving within the same filesystem (`os.replace`) solves the heavy I/O bottleneck that triggers aggressive Windows Defender locks.
  - In semantic segmentation mode, read the staged image before executing `os.replace` to generate dense class masks safely.
- Cut line:
  - Tasks 1.1–2.3 are core must-haves for resolving the freezing and `WinError 5` crash.
  - Tasks 3.1–3.2 provide high-value data rescue for the user's existing run.
- Open questions:
  - N/A (all technical root causes and platform semantics verified).
- Decision points:
  - N/A.

## 11. Review appendix
- Findings:
  - Line 1421 in `engine.py` calls `shutil.rmtree(backup_root)` without `ignore_errors=True`. If Windows Defender locks a file in the backup directory during deletion, the function crashes even if the dataset swap was successful.
  - The output assembly loop in `run_augmentation` performs 24,402 synchronous file copies (`shutil.copy2`) within the exact same filesystem, writing ~20GB of redundant data and triggering aggressive Windows Defender scans that lead to `[WinError 5]` on directory rename.
  - The TUI progress task description was hardcoded to `"Augmenting"`, masking phase transitions and freezing the percentage at 100% and time remaining at 0:00:00 throughout output dataset writing.
  - The user's 24,402 augmented images are currently intact on disk inside `D:\Shahab\YOLOMatic\datasets\.zenlabel-export_augmented.augmenting-63d5fb64ccba436897653d1f6e06c4a6\output` and can be finalized immediately.
- Decisions:
  - Extend exponential backoff retries to 8 attempts (up to 5.0s delay, ~17s window) to ensure Windows Defender real-time scanning has sufficient time to release locks.
  - Add fallback recursive item-by-item migration in `safe_directory_swap` to handle cases where folder node rename is blocked by a persistent handle.
  - Guard backup removal with `ignore_errors=True`.
  - Implement instant CLI recovery for orphaned `.augmenting-*` folders to rescue the user's completed dataset without repeating the 22-minute augmentation run.
- Remaining risks:
  - Permanent external process locks (e.g. user keeping a locked command prompt or file editor open directly inside the target directory): Mitigated by clear, actionable error panels directing the user to close the blocking application and pointing to the preserved staged data.
