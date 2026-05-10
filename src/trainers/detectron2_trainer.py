from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.trainers.common import effective_clearml_settings, maybe_upload_clearml_checkpoint
from src.utils.ml_dependencies import MLDependencyError, import_detectron2

console = Console()


def _import_detectron2_training_symbols() -> dict[str, Any]:
    import_detectron2()
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import DefaultTrainer

    return {
        "model_zoo": model_zoo,
        "get_cfg": get_cfg,
        "register_coco_instances": register_coco_instances,
        "DefaultTrainer": DefaultTrainer,
    }


def find_detectron2_checkpoint(run_dir: Path | None) -> Path | None:
    if run_dir is None or not run_dir.exists():
        return None
    preferred = run_dir / "model_final.pth"
    if preferred.exists():
        return preferred
    checkpoints = sorted(run_dir.rglob("*.pth"), key=lambda path: path.stat().st_mtime, reverse=True)
    return checkpoints[0] if checkpoints else None


def initialize_clearml_task(config: dict[str, Any], run_name: str):
    clearml = effective_clearml_settings(config)
    if not clearml.get("enabled", True):
        return None
    try:
        from clearml import Task

        return Task.init(
            project_name=clearml.get("project_name", "Detectron2 Training"),
            task_name=run_name,
            tags=["Detectron2"],
        )
    except Exception as error:
        if clearml.get("require_configured", False):
            raise RuntimeError(f"ClearML is required but not configured: {error}") from error
        console.print(f"[bold yellow]ClearML is not configured: {error}[/bold yellow]")
        return None

def report_roboflow_skip_if_configured(config: dict[str, Any]) -> None:
    if config.get("roboflow", {}).get("upload", False):
        console.print(
            "[bold yellow]Skipping Roboflow upload: Detectron2 automatic Roboflow deployment is not supported yet.[/bold yellow]"
        )


def _register_datasets(config: dict[str, Any], symbols: dict[str, Any]) -> tuple[str, str | None]:
    dataset = config.get("dataset", {})
    prepared_dir = Path(dataset.get("prepared_dir") or dataset.get("base_dir") or dataset.get("source_dir"))
    split_names = dataset.get("splits", {})
    registered: dict[str, str] = {}
    for split in ("train", "val", "test"):
        split_info = split_names.get(split) or {}
        ann = split_info.get("annotations_path") or split_info.get("annotations")
        if not ann:
            candidate = prepared_dir / "annotations" / f"instances_{split}.json"
            ann = str(candidate) if candidate.exists() else None
        if not ann:
            continue
        image_root = split_info.get("images_path") or split_info.get("images")
        if not image_root:
            image_root = str(prepared_dir / split / "images")
        name = f"yolomatic_{dataset.get('name', prepared_dir.name)}_{split}_{abs(hash(str(prepared_dir))) % 100000}"
        symbols["register_coco_instances"](name, {}, str(ann), str(image_root))
        registered[split] = name
    if "train" not in registered:
        raise FileNotFoundError("Detectron2 training requires a COCO train annotation file.")
    return registered["train"], registered.get("val")


def build_detectron2_cfg(config: dict[str, Any]) -> Any:
    symbols = _import_detectron2_training_symbols()
    model_zoo = symbols["model_zoo"]
    cfg = symbols["get_cfg"]()
    d2_config = config.get("detectron2", {})
    cfg.merge_from_file(model_zoo.get_config_file(d2_config["config_path"]))
    train_name, val_name = _register_datasets(config, symbols)
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,) if val_name else ()
    training = config.get("training", {})
    cfg.DATALOADER.NUM_WORKERS = int(training.get("num_workers", 2))
    cfg.SOLVER.IMS_PER_BATCH = int(training.get("ims_per_batch", 2))
    cfg.SOLVER.BASE_LR = float(training.get("base_lr", 0.00025))
    cfg.SOLVER.MAX_ITER = int(training.get("max_iter", 3000))
    cfg.TEST.EVAL_PERIOD = int(training.get("eval_period", 500))
    cfg.SOLVER.CHECKPOINT_PERIOD = int(training.get("checkpoint_period", 1000))
    cfg.MODEL.DEVICE = str(training.get("device", "cpu"))
    cfg.OUTPUT_DIR = str(training.get("output_dir") or "runs/detectron2")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(config.get("dataset", {}).get("classes", []))
    weights = training.get("weights")
    if weights:
        cfg.MODEL.WEIGHTS = str(weights)
    elif config.get("settings", {}).get("auto_download_pretrained", True):
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(d2_config["weights_url"])
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    return cfg


def print_config_summary(config: dict[str, Any]) -> None:
    settings = config["settings"]
    training = config.get("training", {})
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Category", style="dim", width=22)
    table.add_column("Details")
    table.add_row("Model Type", "Detectron2")
    table.add_row("Model", str(settings["model_type"]))
    table.add_row("Task", str(settings.get("task", "detection")))
    table.add_row("Dataset", str(settings["dataset"]))
    table.add_row("Max Iter", str(training.get("max_iter", "N/A")))
    table.add_row("Batch Size", str(training.get("ims_per_batch", "N/A")))
    table.add_row("Device", str(training.get("device", "N/A")))
    console.print(Panel.fit("[bold]Detectron2 Configuration Summary[/bold]", style="bold blue"))
    console.print(table)


def train_from_config(config_file: str | Path) -> Path | None:
    with open(config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    print_config_summary(config)
    cfg = build_detectron2_cfg(config)
    run_name = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_dir = Path(cfg.OUTPUT_DIR) / run_name
    cfg.OUTPUT_DIR = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    task = initialize_clearml_task(config, run_name)
    if task is not None and effective_clearml_settings(config).get("log_hyperparameters", True):
        task.connect({"settings": config.get("settings", {}), "dataset": config.get("dataset", {}), "training": config.get("training", {})})
    symbols = _import_detectron2_training_symbols()
    try:
        trainer = symbols["DefaultTrainer"](cfg)
        trainer.resume_or_load(resume=False)
        console.print("\n[bold green]Starting Detectron2 training...[/bold green]")
        trainer.train()
        checkpoint = find_detectron2_checkpoint(run_dir)
        if checkpoint is None:
            console.print("[bold yellow]Detectron2 training completed, but no .pth checkpoint was found.[/bold yellow]")
        else:
            console.print(f"[bold green]Detectron2 training completed. Best checkpoint: {checkpoint}[/bold green]")
        maybe_upload_clearml_checkpoint(task, checkpoint, config, console)
        report_roboflow_skip_if_configured(config)
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
        console.print(f"[bold red]Detectron2 training failed: {error}[/bold red]")
