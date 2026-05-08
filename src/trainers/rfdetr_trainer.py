from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.cli.upload import build_candidate, deploy_rfdetr_model, load_env_config
from src.config.settings import deep_merge, load_settings
from src.models.rfdetr import get_rfdetr_variant
from src.utils.ml_dependencies import MLDependencyError, import_rfdetr_model_class

console = Console()


def instantiate_rfdetr_model(config: dict[str, Any]) -> object:
    settings = config["settings"]
    variant = get_rfdetr_variant(settings["model_type"])
    training = config.get("training", {})
    constructor_kwargs: dict[str, Any] = {}
    pretrain_weights = training.get("pretrain_weights")
    if pretrain_weights:
        constructor_kwargs["pretrain_weights"] = pretrain_weights

    model_class = import_rfdetr_model_class(variant.class_name)
    return model_class(**constructor_kwargs)


def prepare_training_params(config: dict[str, Any]) -> dict[str, Any]:
    training = dict(config.get("training", {}))
    for key in ("pretrain_weights", "output_dir"):
        training.pop(key, None)
    return {key: value for key, value in training.items() if value is not None}


def prepare_export_params(config: dict[str, Any]) -> dict[str, Any]:
    export = dict(config.get("export", {}))
    export.pop("enabled", None)
    return {key: value for key, value in export.items() if value is not None}


def find_rfdetr_checkpoint(run_dir: Path | None) -> Path | None:
    if run_dir is None or not run_dir.exists():
        return None
    preferred_names = (
        "checkpoint_best_total.pth",
        "checkpoint_best_ema.pth",
        "checkpoint.pth",
    )
    for name in preferred_names:
        matches = sorted(run_dir.rglob(name), key=lambda path: path.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    checkpoints = sorted(run_dir.rglob("*.pth"), key=lambda path: path.stat().st_mtime, reverse=True)
    return checkpoints[0] if checkpoints else None


def effective_clearml_settings(config: dict[str, Any]) -> dict[str, Any]:
    return deep_merge(load_settings().get("clearml", {}), config.get("clearml", {}))


def initialize_clearml_task(config: dict[str, Any], run_name: str):
    clearml = effective_clearml_settings(config)
    if not clearml.get("enabled", True):
        return None
    try:
        from clearml import Task

        return Task.init(
            project_name=clearml.get("project_name", "RF-DETR Training"),
            task_name=run_name,
            tags=["RF-DETR"],
        )
    except Exception as error:
        if clearml.get("require_configured", False):
            raise RuntimeError(f"ClearML is required but not configured: {error}") from error
        console.print(f"[bold yellow]ClearML is not configured: {error}[/bold yellow]")
        return None


def maybe_upload_clearml_checkpoint(task, checkpoint: Path | None, config: dict[str, Any]) -> None:
    if task is None or checkpoint is None:
        return
    if not effective_clearml_settings(config).get("upload_final_model", True):
        return
    try:
        task.upload_artifact(name="final_model", artifact_object=str(checkpoint))
    except Exception as error:
        console.print(f"[bold yellow]ClearML final model upload skipped: {error}[/bold yellow]")


def deploy_to_roboflow_if_configured(config: dict[str, Any], checkpoint: Path | None) -> None:
    roboflow = config.get("roboflow", {})
    if not roboflow.get("upload", False):
        return
    if checkpoint is None:
        console.print("[bold yellow]Skipping Roboflow deploy: no RF-DETR checkpoint was found.[/bold yellow]")
        return
    dataset_rf = config.get("dataset", {}).get("roboflow", {})
    workspace = roboflow.get("workspace") or dataset_rf.get("workspace")
    project = roboflow.get("project_id") or roboflow.get("project") or dataset_rf.get("project")
    version = roboflow.get("version") or dataset_rf.get("version") or roboflow.get("rfdetr_project_version", 1)
    if not workspace or not project:
        console.print("[bold yellow]Skipping Roboflow deploy: workspace/project metadata is missing.[/bold yellow]")
        return
    env_config = load_env_config(Path.cwd())
    candidate = build_candidate(checkpoint)
    deploy_rfdetr_model(env_config.api_key, candidate, workspace, project, int(version), "rfdetr")
    console.print("[bold green]RF-DETR Roboflow deploy completed successfully.[/bold green]")


def print_config_summary(config: dict[str, Any]) -> None:
    settings = config["settings"]
    training = config.get("training", {})
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Category", style="dim", width=22)
    table.add_column("Details")
    table.add_row("Model Type", "RF-DETR")
    table.add_row("Model", str(settings["model_type"]))
    table.add_row("Task", str(settings.get("task", "detection")))
    table.add_row("Dataset", str(settings["dataset"]))
    table.add_row("Auto Download", str(settings.get("auto_download_pretrained", True)))
    table.add_row("Resolution", str(training.get("resolution", "N/A")))
    table.add_row("Batch Size", str(training.get("batch_size", "N/A")))
    table.add_row("Grad Accum Steps", str(training.get("grad_accum_steps", "N/A")))
    table.add_row("Epochs", str(training.get("epochs", "N/A")))
    table.add_row("Device", str(training.get("device", "N/A")))
    console.print(Panel.fit("[bold]RF-DETR Configuration Summary[/bold]", style="bold blue"))
    console.print(table)


def train_from_config(config_file: str | Path) -> Path | None:
    with open(config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    settings = config["settings"]
    dataset_dir = Path(config.get("dataset", {}).get("base_dir") or Path("datasets") / settings["dataset"])
    training = dict(config.get("training", {}))
    output_dir = Path(training.get("output_dir") or "runs/rfdetr")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"{settings['model_type']}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    run_dir = output_dir / run_name

    print_config_summary(config)
    if settings.get("auto_download_pretrained", True):
        console.print(
            "[bold green]RF-DETR will auto-download/cache the selected pretrained weights if they are not already available.[/bold green]"
        )

    model = instantiate_rfdetr_model(config)
    training_params = prepare_training_params(config)
    training_params.setdefault("output_dir", str(run_dir))
    task = initialize_clearml_task(config, run_name)
    if task is not None and effective_clearml_settings(config).get("log_hyperparameters", True):
        task.connect({"settings": settings, "dataset": config.get("dataset", {}), "training": training_params})

    try:
        console.print("\n[bold green]Starting RF-DETR training...[/bold green]")
        model.train(dataset_dir=str(dataset_dir), **training_params)

        if config.get("export", {}).get("enabled", True):
            export_params = prepare_export_params(config)
            export_output_dir = export_params.get("output_dir")
            if export_output_dir:
                export_params["output_dir"] = str(Path(export_output_dir))
            console.print("\n[bold green]Exporting RF-DETR model...[/bold green]")
            model.export(**export_params)
        checkpoint = find_rfdetr_checkpoint(run_dir)
        if checkpoint is None:
            console.print(
                "[bold yellow]RF-DETR training completed, but no .pth checkpoint was found under the expected run directory.[/bold yellow]"
            )
        else:
            console.print(
                f"[bold green]RF-DETR training completed. Best checkpoint: {checkpoint}[/bold green]"
            )
        maybe_upload_clearml_checkpoint(task, checkpoint, config)
        deploy_to_roboflow_if_configured(config, checkpoint)
        return checkpoint
    finally:
        if task is not None:
            task.close()


def main(config_file: str | Path) -> None:
    try:
        train_from_config(config_file)
    except MLDependencyError as error:
        console.print(f"[bold red]{error}[/bold red]")
    except Exception as error:
        console.print(f"[bold red]RF-DETR training failed: {error}[/bold red]")
        import traceback

        console.print(traceback.format_exc())
