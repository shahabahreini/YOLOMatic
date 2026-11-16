from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from rich.panel import Panel
from rich.table import Table
from yolomatic.ui.theme import CONSOLE

from yolomatic.cli.result import CommandResult
from yolomatic.utils.logging_setup import log_exception
from yolomatic.trainers.common import timestamp_for_run, unique_run_dir
from yolomatic.trainers.common import effective_clearml_settings, maybe_upload_clearml_checkpoint
from yolomatic.utils.ml_dependencies import MLDependencyError, import_detectron2
from yolomatic.utils.training_preflight import resolve_training_device

console = CONSOLE


def _import_detectron2_training_symbols() -> dict[str, Any]:
    import_detectron2()
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import DefaultTrainer

    try:
        from detectron2.evaluation import COCOEvaluator
    except ImportError:  # pragma: no cover - very old detectron2
        COCOEvaluator = None

    return {
        "model_zoo": model_zoo,
        "get_cfg": get_cfg,
        "register_coco_instances": register_coco_instances,
        "DefaultTrainer": DefaultTrainer,
        "DatasetCatalog": DatasetCatalog,
        "MetadataCatalog": MetadataCatalog,
        "COCOEvaluator": COCOEvaluator,
    }


RUN_METADATA_FILE = "yolomatic_detectron2.json"


def make_trainer_class(symbols: dict[str, Any]) -> Any:
    """DefaultTrainer with a COCO evaluator.

    Plain DefaultTrainer has no ``build_evaluator``, so ``TEST.EVAL_PERIOD``
    only logged a warning and never evaluated anything.
    """
    base = symbols["DefaultTrainer"]
    evaluator_cls = symbols.get("COCOEvaluator")

    class YolomaticTrainer(base):  # type: ignore[misc, valid-type]
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            if evaluator_cls is None:
                raise NotImplementedError("detectron2.evaluation.COCOEvaluator is unavailable")
            folder = output_folder or str(Path(cfg.OUTPUT_DIR) / "inference")
            return evaluator_cls(dataset_name, output_dir=folder)

    return YolomaticTrainer


def write_run_metadata(run_dir: Path, cfg: Any, config: dict[str, Any]) -> None:
    """Save what prediction needs to rebuild this exact model.

    ``config.yaml`` is detectron2's own full dump (architecture, class count,
    input sizes); the JSON adds class names and the task. Prediction used to
    start from a bare ``get_cfg()`` and load trained weights into the wrong
    architecture.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")
    dataset = config.get("dataset", {})
    payload = {
        "classes": list(dataset.get("classes", [])),
        "task": config.get("settings", {}).get("task", "detection"),
        "config_path": config.get("detectron2", {}).get("config_path"),
        "meta_architecture": str(getattr(getattr(cfg, "MODEL", None), "META_ARCHITECTURE", "")),
    }
    (run_dir / RUN_METADATA_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
        # COCO file_name values are relative to image_root (for converted datasets
        # they include "<split>/images/"), so prefer it over the images folder.
        image_root = split_info.get("image_root") or split_info.get("images_path") or split_info.get("images")
        if not image_root:
            image_root = str(prepared_dir / split / "images")
        # Stable name (hash() is salted per process) and idempotent registration:
        # training the same dataset twice in one TUI session used to raise
        # "dataset ... is already registered".
        digest = hashlib.sha1(str(prepared_dir.resolve()).encode("utf-8")).hexdigest()[:8]
        name = f"yolomatic_{dataset.get('name', prepared_dir.name)}_{split}_{digest}"
        catalog = symbols.get("DatasetCatalog")
        if catalog is not None and name in catalog.list():
            catalog.remove(name)
            metadata_catalog = symbols.get("MetadataCatalog")
            if metadata_catalog is not None and name in metadata_catalog.list():
                metadata_catalog.remove(name)
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
    # Evaluation needs a validation split; without one EVAL_PERIOD is disabled.
    cfg.TEST.EVAL_PERIOD = int(training.get("eval_period", 500)) if val_name else 0
    cfg.SOLVER.CHECKPOINT_PERIOD = int(training.get("checkpoint_period", 1000))
    cfg.MODEL.DEVICE = str(training.get("device", "cpu"))
    cfg.OUTPUT_DIR = str(training.get("output_dir") or "runs/detectron2")
    num_classes = len(config.get("dataset", {}).get("classes", []))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    if str(cfg.MODEL.META_ARCHITECTURE) == "RetinaNet":
        # RetinaNet reads its own head config (default 80 COCO classes).
        cfg.MODEL.RETINANET.NUM_CLASSES = num_classes

    # Optional solver/input overrides exposed through the interactive configurator.
    if "momentum" in training:
        cfg.SOLVER.MOMENTUM = float(training["momentum"])
    if "weight_decay" in training:
        cfg.SOLVER.WEIGHT_DECAY = float(training["weight_decay"])
    if "warmup_iters" in training:
        cfg.SOLVER.WARMUP_ITERS = int(training["warmup_iters"])
    if "gamma" in training:
        cfg.SOLVER.GAMMA = float(training["gamma"])
    steps = training.get("steps")
    if steps:
        cfg.SOLVER.STEPS = tuple(int(s) for s in steps)
    elif steps == [] or steps == ():
        cfg.SOLVER.STEPS = ()
    if "roi_batch_size_per_image" in training:
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = int(training["roi_batch_size_per_image"])
    if "score_thresh_test" in training:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(training["score_thresh_test"])
        if str(cfg.MODEL.META_ARCHITECTURE) == "RetinaNet":
            cfg.MODEL.RETINANET.SCORE_THRESH_TEST = float(training["score_thresh_test"])
    if "min_size_train" in training:
        cfg.INPUT.MIN_SIZE_TRAIN = (int(training["min_size_train"]),)
    if "max_size_train" in training:
        cfg.INPUT.MAX_SIZE_TRAIN = int(training["max_size_train"])

    weights = training.get("weights")
    if weights:
        weights_path = Path(str(weights))
        looks_like_local = weights_path.suffix in {".pth", ".pkl"} and "://" not in str(weights)
        if looks_like_local and not weights_path.exists():
            raise FileNotFoundError(
                f"Detectron2 weights file not found: {weights}. Provide a valid "
                "checkpoint path, or remove 'training.weights' to auto-download "
                "the pretrained backbone from the model zoo."
            )
        cfg.MODEL.WEIGHTS = str(weights)
    elif config.get("settings", {}).get("auto_download_pretrained", True):
        try:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(d2_config["weights_url"])
        except Exception as error:
            raise RuntimeError(
                "Failed to resolve Detectron2 pretrained weights for "
                f"'{d2_config['weights_url']}': {error}. Check the model config "
                "path or disable auto-download and supply local weights."
            ) from error
        console.print(
            "[bold green]Detectron2 will download/cache the pretrained model "
            f"weights ({d2_config['weights_url']}) on first use.[/bold green]"
        )
    else:
        console.print(
            "[bold yellow]auto_download_pretrained is disabled and no local "
            "weights were provided; training will start from random "
            "initialization.[/bold yellow]"
        )
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
    device = resolve_training_device(config.get("training", {}).get("device", "cpu"))
    if device.cancelled:
        raise KeyboardInterrupt
    if device.device:
        config.setdefault("training", {})["device"] = device.device
    cfg = build_detectron2_cfg(config)
    run_dir = unique_run_dir(Path(cfg.OUTPUT_DIR), timestamp_for_run())
    run_name = run_dir.name
    cfg.OUTPUT_DIR = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(run_dir, cfg, config)
    task = initialize_clearml_task(config, run_name)
    if task is not None and effective_clearml_settings(config).get("log_hyperparameters", True):
        task.connect({"settings": config.get("settings", {}), "dataset": config.get("dataset", {}), "training": config.get("training", {})})
    symbols = _import_detectron2_training_symbols()
    try:
        trainer = make_trainer_class(symbols)(cfg)
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


def main(config_file: str | Path) -> CommandResult:
    try:
        result = train_from_config(config_file)
    except KeyboardInterrupt:
        return CommandResult.cancelled("Detectron2 training interrupted.")
    except MLDependencyError as error:
        return CommandResult.failed(str(error))
    except Exception as error:  # noqa: BLE001
        log_path = log_exception(error, "Detectron2 training")
        return CommandResult.failed(
            f"Detectron2 training failed: {error}",
            hint=f"Full traceback in {log_path}" if log_path else None,
        )
    outputs = {"Checkpoint": result} if isinstance(result, (str, Path)) and result else {}
    return CommandResult.ok("Detectron2 training completed.", outputs=outputs, review=True)
