from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console

from yolomatic.config.settings import deep_merge, load_settings


def effective_clearml_settings(config_or_clearml: dict[str, Any] | None) -> dict[str, Any]:
    global_clearml = load_settings().get("clearml", {})
    if config_or_clearml is None:
        return global_clearml
    clearml_override = config_or_clearml.get("clearml", config_or_clearml) if "clearml" in config_or_clearml else config_or_clearml
    return deep_merge(global_clearml, clearml_override or {})


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


def unique_run_dir(parent: Path, base_name: str) -> Path:
    """Return ``parent/base_name`` or the first free ``base_name-N`` variant.

    Run folders used to be named to the minute, so two runs started in the
    same minute silently shared (and overwrote) one folder.
    """
    candidate = parent / base_name
    suffix = 2
    while candidate.exists():
        candidate = parent / f"{base_name}-{suffix}"
        suffix += 1
    return candidate


def timestamp_for_run() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


class PostTrainingSteps:
    """Run follow-up steps independently and remember how each one went."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self.results: list[tuple[str, str, str]] = []

    def run(self, name: str, action: Any) -> Any:
        from rich.markup import escape

        from yolomatic.utils.logging_setup import log_exception

        self.console.print(f"\n[bold green]{name}...[/bold green]")
        try:
            value = action()
        except KeyboardInterrupt:
            self.results.append((name, "cancelled", ""))
            raise
        except Exception as error:  # noqa: BLE001 - reported in the summary
            log_exception(error, f"Post-training step '{name}'")
            self.console.print(f"[bold red]{name} failed:[/bold red] {escape(str(error))}")
            self.results.append((name, "failed", str(error)))
            return None
        self.results.append((name, "ok", ""))
        return value

    @property
    def failed(self) -> list[str]:
        return [name for name, status, _ in self.results if status != "ok"]
