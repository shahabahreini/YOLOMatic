---
description: Profile-based regular YOLO config wizard for augmentation and resource utilization
---

# Goal

Add an interactive profile-driven config creation flow for regular YOLO models that lets the user choose:

- augmentation intensity
- compute utilization
- worker intensity

The wizard should analyze the selected dataset and current machine resources, recommend sensible defaults, explain the trade-offs, and still allow manual overrides before saving the generated YAML.

# Scope

In scope:

- Regular YOLO config generation only
- New TUI prompts during config creation in `yolomatic`
- Augmentation profiles: `Minimum`, `Low`, `Medium`
- Compute utilization profiles: `Conservative`, `Balanced`, `Aggressive`
- Worker profiles: `Light`, `Medium`, `Heavy`
- Recommendation heuristics based on dataset size on disk, dataset file counts, available RAM, CPU core count, and available GPU memory when CUDA is selected
- Explanatory text in prompts so users understand the effect of each profile
- Optional manual override flow before the final config is written

Out of scope:

- YOLO-NAS config generation
- Changing the trainer input schema
- New training flags not already supported by Ultralytics for regular YOLO
- Runtime auto-tuning after training starts
- Persistent user preference storage for profile choices

# Architecture

## New responsibilities in config generation

Extend the regular YOLO config generation path so that config creation becomes a two-stage process:

1. build a base config using current project defaults
2. apply user-selected profiles and heuristics before the YAML is saved

The implementation should stay within the existing boundaries:

- `src/cli/run.py`
  - gather user selections through the current TUI selector flow
  - present recommendations and explanations
  - pass selected profiles into the generator
- `src/config/generator.py`
  - analyze dataset/resource inputs
  - compute recommended profile values
  - translate profiles into concrete `training` keys for regular YOLO configs

## Profile model

Introduce a small internal profile model for regular YOLO config generation. This does not need to change the saved YAML schema. It only needs to organize generation-time choices.

Expected generation-time concepts:

- augmentation profile
- compute profile
- worker profile
- optional manual overrides
- recommendation metadata shown to the user

The saved config should remain a normal regular YOLO config with concrete `training` values.

# Data flow

1. User selects a regular YOLO model variant
2. User selects a dataset
3. Config generator inspects the dataset and gathers metadata:
   - dataset size on disk
   - image file count
   - label file count
4. Config generator inspects current machine resources:
   - available RAM
   - CPU core count
   - CUDA availability
   - available GPU memory if CUDA is available
5. TUI presents recommended profiles with short explanations
6. User chooses either:
   - `Use recommended settings`
   - `Customize selected values`
7. If customizing, the user selects:
   - augmentation profile
   - compute utilization profile
   - worker profile
8. Generator applies the chosen profiles to the base YOLO training config
9. Config is saved as YAML with only standard regular YOLO training keys
10. Existing training flows continue unchanged because the YAML schema is unchanged

# Profile definitions

## Augmentation profiles

### Minimum

Purpose:

- keep the generated config close to the smallest practical setup
- match the style of configs where only core values are enabled and most augmentation keys are omitted or disabled

Output intent:

- keep core training keys such as `epochs`, `imgsz`, `batch`, `workers`, `optimizer`, and `device`
- disable optional augmentation knobs by omitting them or setting them to neutral values
- do not enable additional augmentation beyond the base training essentials

### Low

Purpose:

- introduce a small amount of safe augmentation without making training behavior too aggressive

Enabled augmentation keys:

- `flipud`
- `fliplr`
- `mosaic`
- `mixup`

All other augmentation-heavy keys stay disabled or neutral.

### Medium

Purpose:

- provide a more complete augmentation profile for general training without becoming highly aggressive

Enabled augmentation keys:

- everything from `Low`
- `degrees`
- `hsv_h`
- `hsv_s`
- `hsv_v`
- `translate`
- `scale`

Excluded from this phase:

- `shear`
- `perspective`
- `copy_paste`

This keeps the profile understandable and avoids unexpectedly aggressive transforms.

# Compute utilization profiles

## Conservative

Purpose:

- prioritize stability and lower memory pressure

Expected output behavior:

- safer `batch` target
- `cache` disabled by default
- lower worker recommendation emphasis
- appropriate for limited RAM, limited GPU memory, or large datasets on weaker machines

## Balanced

Purpose:

- provide the default recommendation for most users

Expected output behavior:

- moderate `batch` target
- `cache` enabled only when RAM and dataset size make it reasonable
- medium worker recommendation emphasis

## Aggressive

Purpose:

- push throughput harder when the machine can support it

Expected output behavior:

- high `batch` target, especially on CUDA systems with enough free VRAM
- allow `cache` when dataset and RAM conditions are favorable
- heavier worker recommendation emphasis

## Concrete YAML mapping

For regular YOLO configs, compute profiles should map into existing keys only:

- `batch`
- `cache`
- `workers`
- `device`

Suggested batch behavior:

- CPU or MPS: keep conservative values and avoid aggressive auto-batch assumptions
- CUDA:
  - `Conservative` -> `batch: 0.70`
  - `Balanced` -> `batch: 0.85`
  - `Aggressive` -> `batch: 0.95`
- fallback when GPU info is unavailable but CUDA is selected:
  - use `batch: -1`

These values intentionally leave some headroom instead of targeting 100% memory usage.

# Worker profile heuristic

## Inputs

The worker recommendation should use all of the following:

- dataset size on disk
- dataset image count
- dataset label count
- available RAM
- CPU core count
- available GPU memory when CUDA is used

## Heuristic goals

The heuristic should:

- avoid suggesting too many workers on low-RAM machines
- avoid overly low workers on strong systems where the GPU would starve
- cap recommendations by CPU core count
- remain simple and explainable to users

## Recommendation model

The wizard should compute three recommended numeric worker levels and label them:

- `Light`
- `Medium`
- `Heavy`

The labels shown to the user should include the resolved worker count and a short explanation, for example:

- `Light (4 workers) - lower RAM pressure, safer for large datasets or limited memory`
- `Medium (8 workers) - balanced throughput for most systems`
- `Heavy (12 workers) - highest data-loading throughput if RAM and storage can keep up`

## Heuristic rules

Use a simple scoring approach rather than opaque formulas.

Inputs should be normalized into a pressure estimate:

- dataset pressure increases with larger total size and higher file counts
- RAM capacity reduces pressure when more memory is available
- GPU capacity can justify a higher worker target because faster training needs faster data loading

Practical rules:

1. Start from a CPU-based ceiling derived from available cores
2. Reduce the candidate worker counts when:
   - available RAM is low
   - dataset size is large relative to available RAM
   - file count is very high
3. Increase the candidate worker counts moderately when:
   - RAM is plentiful
   - CPU core count is high
   - CUDA is active and available GPU memory is high enough to sustain larger batches
4. Never exceed a safe upper bound relative to CPU cores
5. Ensure the final profile values are ordered and distinct when possible

A reasonable initial shape is:

- `Light` around 25-35% of safe CPU worker capacity
- `Medium` around 50-65%
- `Heavy` around 75-90%

Then clamp those values based on memory pressure.

# UX

## Prompt sequence

For regular YOLO config creation, add prompts after dataset selection and before file save.

Recommended sequence:

1. analyze dataset and system resources
2. show a concise recommendation summary
3. ask whether to:
   - use recommended settings
   - customize settings
4. if customizing, prompt in this order:
   - augmentation profile
   - compute utilization profile
   - worker profile
5. show a final summary before writing the config

## Explanations

Each profile choice should include a short explanation.

Examples:

- `Minimum - essential training values only, minimal augmentation`
- `Low - enables flips, mosaic, and mixup for mild robustness gains`
- `Medium - adds color and geometric augmentation for stronger generalization`
- `Conservative - lower memory pressure and safer defaults`
- `Balanced - recommended for most systems`
- `Aggressive - higher throughput if your RAM and GPU can support it`

Worker options should explain both performance and memory implications.

## Summary display

The final summary should include:

- chosen augmentation profile
- chosen compute profile
- chosen worker profile and resolved numeric worker count
- detected RAM
- detected GPU memory if applicable
- dataset size and file count summary
- final values written to `training.batch`, `training.cache`, `training.workers`, and the enabled augmentation keys

# Error handling

- If dataset size analysis partially fails, continue with whichever metrics are available and fall back to conservative recommendations
- If RAM detection fails, fall back to CPU core count and conservative worker limits
- If GPU memory detection fails, still allow CUDA and fall back to `batch: -1` or balanced safe values
- If the dataset contains unexpected structure, do not block config creation; just reduce confidence in recommendations
- Never write unsupported keys into the saved YAML

# Compatibility

- Trainers should require no schema changes
- Saved YAML remains compatible with `src/trainers/yolo_trainer.py`
- Existing manually edited config files remain valid
- The new prompts affect only newly generated regular YOLO config files

# Testing

- Verify config generation still works for regular YOLO model variants
- Verify YOLO-NAS config creation is unchanged
- Verify `Minimum`, `Low`, and `Medium` produce the intended augmentation key sets
- Verify compute profiles map to safe `batch` and `cache` values
- Verify worker profile recommendations remain within CPU-core limits
- Verify the final saved YAML can still be consumed by the regular YOLO trainer
- Syntax-check modified Python files with `py_compile`

# Risks and mitigations

- Risk: heuristic recommendations feel arbitrary
  - Mitigation: keep rules simple, expose explanations, and allow manual override
- Risk: aggressive worker counts cause host memory pressure
  - Mitigation: bias recommendations conservative when RAM is uncertain or dataset pressure is high
- Risk: aggressive GPU targets lead to OOM
  - Mitigation: cap targets below 100% and fall back to safer auto-batch behavior when GPU telemetry is unavailable
- Risk: too many prompts slow down config creation
  - Mitigation: offer `Use recommended settings` as the fast path
