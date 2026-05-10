from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console

from src.config.settings import deep_merge, load_settings


def effective_clearml_settings(config: dict[str, Any]) -> dict[str, Any]:
    return deep_merge(load_settings().get("clearml", {}), config.get("clearml", {}))


def maybe_upload_clearml_checkpoint(
    task: Any,
    checkpoint: Path | None,
    config: dict[str, Any],
    console: Console,
) -> None:
    if task is None or checkpoint is None:
        return
    if not effective_clearml_settings(config).get("upload_final_model", True):
        return
    try:
        task.upload_artifact(name="final_model", artifact_object=str(checkpoint))
    except Exception as error:
        console.print(f"[bold yellow]ClearML final model upload skipped: {error}[/bold yellow]")
