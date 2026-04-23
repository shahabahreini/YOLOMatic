---
description: Shared CUDA auto-repair preflight for training
---

# Goal

Add a shared preflight flow that detects when training requests CUDA but the active Python environment cannot use the GPU, then prompts the user to either repair the environment automatically, continue on CPU, or cancel training.

# Scope

In scope:
- Shared preflight used by YOLO and YOLO-NAS training
- Detection of CPU-only Torch builds and CUDA-unavailable Torch runtimes
- Interactive TUI prompt using the existing selector style
- Automated repair using the current `.venv` Python executable
- Post-repair verification and graceful fallback options

Out of scope:
- Silent package mutation without user confirmation
- Non-training commands
- Project-wide dependency source refactors in `pyproject.toml`

# Architecture

## New shared helper

Create a dedicated helper module under `src/utils` for training CUDA preflight and repair.

Responsibilities:
- Inspect requested device from training config
- Detect whether the request implies CUDA usage
- Inspect current Torch build/runtime state
- Offer an interactive decision menu
- Run repair commands with the active interpreter from the current virtual environment
- Re-verify Torch after repair
- Return either updated training parameters or a cancellation signal

## Integration points

- `src/trainers/yolo_trainer.py`
  - Replace inline CUDA mismatch handling with the shared helper
- `src/trainers/nas_trainer.py`
  - Add the same shared preflight before trainer startup

# Data flow

1. Trainer loads config and training parameters
2. Shared preflight checks whether the requested device requires CUDA
3. If CUDA is already available, training continues unchanged
4. If CUDA is unavailable, show a TUI with:
   - `Fix CUDA-enabled PyTorch now`
   - `Continue on CPU`
   - `Cancel Training`
5. If the user chooses repair:
   - resolve the current interpreter path
   - run pip uninstall/install commands through that interpreter
   - verify `torch.cuda.is_available()` again in a fresh subprocess
6. If repair succeeds, keep the CUDA device setting
7. If repair fails, show the error and offer:
   - `Continue on CPU`
   - `Cancel Training`
8. If the user chooses CPU fallback, copy the training parameters and set `device: cpu`

# Repair strategy

Use the active `.venv` interpreter directly instead of `uv run`, because `uv run` may sync the environment first and restore CPU-only Torch.

Preferred repair sequence on Windows and Linux:
1. uninstall `torch`, `torchvision`, `torchaudio`
2. install matching CUDA-enabled wheels from the configured PyTorch index
3. verify Torch version, CUDA build, and device availability in a new subprocess

Initial implementation target:
- default repair index: `https://download.pytorch.org/whl/cu128`
- interpreter: `sys.executable`

# Error handling

- If no virtual environment interpreter can be resolved, show a clear error and offer CPU fallback/cancel
- If pip uninstall/install fails, capture stderr/stdout and present a concise failure message
- If post-repair verification still shows CPU-only Torch, treat repair as failed
- Never continue on CUDA unless verification succeeds
- Never crash on cancel paths; return explicit sentinels and guard cleanup

# UX

- Reuse existing TUI selector style via `get_user_choice`
- Keep messages short and actionable
- Explicitly explain that `nvidia-smi` working does not guarantee Torch CUDA availability
- Print detected Torch build before and after repair attempts when relevant

# Testing

- Syntax-check modified files with `py_compile`
- Verify YOLO trainer still launches and prompts correctly
- Verify cancel path exits without traceback
- Verify CPU fallback still updates training params to `cpu`
- Verify repair helper builds subprocess commands using `sys.executable`

# Risks and mitigations

- Risk: package mutations may fail or partially apply
  - Mitigation: verify after install and never assume success
- Risk: version mismatches among Torch packages
  - Mitigation: install all three packages together from the same index
- Risk: subprocess repair logic becomes platform-sensitive
  - Mitigation: use interpreter-based pip invocations and avoid shell-specific commands
