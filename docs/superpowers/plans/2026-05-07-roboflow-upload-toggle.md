# Roboflow Upload Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow users to optionally upload their trained YOLO model weights (`best.pt` or `last.pt`) automatically to Roboflow after the training process completes.

**Architecture:** We will introduce a new helper function in `yolo_trainer.py` that parses the `roboflow` configuration block, determines the weight file, and delegates the upload using existing functions from `src.cli.upload`. This helper will be called right after the model export step in `main()`.

**Tech Stack:** Python, pytest/unittest, PyYAML, Rich.

---

### Task 1: Add automated upload logic in `yolo_trainer.py`

**Files:**
- Modify: `src/trainers/yolo_trainer.py`
- Test: `tests/test_yolo_trainer.py`

- [ ] **Step 1: Write the failing tests**

```python
# In tests/test_yolo_trainer.py
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.trainers.yolo_trainer import upload_to_roboflow_if_configured

class TestRoboflowUpload(unittest.TestCase):
    @patch("src.trainers.yolo_trainer.upload_model")
    @patch("src.trainers.yolo_trainer.build_candidate")
    def test_upload_skipped_if_not_configured(self, mock_build, mock_upload):
        config = {"roboflow": {"upload": False}}
        upload_to_roboflow_if_configured(config, {}, Path("/tmp/run"), "yolo11", MagicMock())
        mock_build.assert_not_called()
        mock_upload.assert_not_called()

    @patch("src.trainers.yolo_trainer.upload_model")
    @patch("src.trainers.yolo_trainer.build_candidate")
    @patch("src.trainers.yolo_trainer.stage_upload_candidate")
    def test_upload_called_if_configured(self, mock_stage, mock_build, mock_upload):
        config = {"roboflow": {"upload": True, "weight": "best.pt"}}
        dataset_config = {"roboflow": {"workspace": "ws", "project": "proj", "version": "1"}}
        run_dir = Path("/tmp/run")
        
        # Setup mock candidate
        mock_candidate = MagicMock()
        mock_build.return_value = mock_candidate
        mock_stage.return_value = mock_candidate
        
        upload_to_roboflow_if_configured(config, dataset_config, run_dir, "yolov11", MagicMock())
        
        # Verify the candidate is built with the right path
        expected_path = run_dir / "weights" / "best.pt"
        mock_build.assert_called_once_with(expected_path)
        
        # Verify candidate overrides are set
        self.assertEqual(mock_candidate.workspace, "ws")
        self.assertEqual(mock_candidate.project_ids, "proj")
        
        mock_stage.assert_called_once_with(mock_candidate, "yolov11")
        mock_upload.assert_called_once_with(mock_candidate, "yolov11")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_yolo_trainer.py -v`
Expected: FAIL due to `ImportError: cannot import name 'upload_to_roboflow_if_configured'`

- [ ] **Step 3: Write minimal implementation**

In `src/trainers/yolo_trainer.py`, add imports for the upload logic at the top:
```python
from src.cli.upload import build_candidate, stage_upload_candidate, upload_model
```

Then, add the new function:
```python
def upload_to_roboflow_if_configured(config, dataset_config, run_dir, model_type, console):
    """Automatically upload trained weights to Roboflow if configured."""
    roboflow_config = config.get("roboflow", {})
    if not roboflow_config.get("upload", False):
        return

    weight_name = roboflow_config.get("weight", "best.pt")
    console.print(f"\\n[bold green]Preparing to upload {weight_name} to Roboflow...[/bold green]")
    
    if run_dir is None:
        console.print("[bold red]Cannot upload: run directory is unknown.[/bold red]")
        return
        
    weight_path = run_dir / "weights" / weight_name
    if not weight_path.exists():
        console.print(f"[bold red]Cannot upload: Weight file {weight_path} does not exist.[/bold red]")
        return

    dataset_rf = dataset_config.get("roboflow", {})
    workspace = dataset_rf.get("workspace")
    project = dataset_rf.get("project")
    
    if not workspace or not project:
        console.print("[bold red]Cannot upload: Dataset configuration lacks Roboflow workspace or project.[/bold red]")
        return

    try:
        candidate = build_candidate(weight_path)
        candidate.workspace = workspace
        candidate.project_ids = project
        
        # Model type from config might be like "yolov8n", but upload logic
        # expects something like "yolov8". We take the first 6 chars usually, or pass it directly.
        # Actually `model_type` passed to upload logic should be e.g., "yolov11"
        upload_type = model_type
        # Strip size suffix if needed, but `upload_model` and `stage_upload_candidate` 
        # usually just expect the base model type string.
        
        staged = stage_upload_candidate(candidate, upload_type)
        console.print(f"[bold]Uploading to workspace '{workspace}', project '{project}'...[/bold]")
        upload_model(staged, upload_type)
        console.print("[bold green]Roboflow upload completed successfully![/bold green]")
    except Exception as error:
        console.print(f"[bold red]Roboflow upload failed: {error}[/bold red]")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_yolo_trainer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trainers/yolo_trainer.py tests/test_yolo_trainer.py
git commit -m "feat(trainer): add logic to automatically upload weights to roboflow"
```

### Task 2: Hook up the logic in the `main` training flow

**Files:**
- Modify: `src/trainers/yolo_trainer.py:341-356` (around the export block)

- [ ] **Step 1: Write the integration code**

In `src/trainers/yolo_trainer.py` inside the `main()` function, find the block where the model is exported:
```python
        # Export the model
        console.print("\\n[bold green]Exporting model...[/bold green]")
        model.export(**export_params)

        # ADD THIS: Check and perform Roboflow upload
        upload_to_roboflow_if_configured(config, dataset_config, run_dir, model_name, console)

        if run_dir is not None:
```

- [ ] **Step 2: Run all tests to verify no regressions**

Run: `pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/trainers/yolo_trainer.py
git commit -m "feat(trainer): invoke automated roboflow upload at the end of yolo training"
```