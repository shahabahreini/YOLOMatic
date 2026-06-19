import argparse
import os
from datetime import datetime
from pathlib import Path

# Ultralytics snapshots this environment variable when its utils module is first
# imported. Keep package installation under YOLOmatic's dependency manager
# instead of letting post-training export mutate the active environment.
os.environ["YOLO_AUTOINSTALL"] = "false"

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.cli.upload import build_candidate, stage_upload_candidate, upload_model
from src.config.settings import DEFAULT_SETTINGS, deep_merge, load_settings

from src.utils.cli import get_user_choice
from src.utils.ml_dependencies import (
    MLDependencyError,
    import_ultralytics_settings,
    import_ultralytics_yolo,
)
from src.utils.project import (
    format_size,
    list_config_files,
    resolve_config_path,
    verify_dataset_directories,
)
from src.datasets.cache import clean_dataset_image_cache, normalize_yolo_cache_setting
from src.utils.project import (
    load_dataset_config as load_project_dataset_config,
)
from src.utils.tensorboard import (
    backfill_ultralytics_tensorboard,
    build_tensorboard_metadata,
    emit_tensorboard_report,
    validate_tensorboard_run,
)
from src.utils.training_preflight import resolve_training_device, validate_export_config
from src.models.rfdetr import is_rfdetr_model

# Initialize Rich console
console = Console()


def configure_ultralytics_runtime():
    os.environ.setdefault("MPLBACKEND", "Agg")

    ultra_settings = import_ultralytics_settings()
    ultra_settings.update(
        {
            "tensorboard": True,
            # YOLOmatic owns the ClearML task lifecycle. Ultralytics' built-in
            # ClearML callback logs plots through matplotlib/Tk at train end,
            # and those optional plots must never abort checkpoint/export flow.
            "clearml": False,
            # MLflow is also optional here. Importing Ultralytics' MLflow
            # callback can fail before training starts when MLflow's protobuf
            # stack is incompatible with the active protobuf runtime.
            "mlflow": False,
        }
    )


def disable_ultralytics_clearml_callbacks(model):
    callbacks = getattr(model, "callbacks", None)
    if not isinstance(callbacks, dict):
        return

    removed_callbacks = 0
    for event_name, event_callbacks in callbacks.items():
        if not isinstance(event_callbacks, list):
            continue
        retained_callbacks = [
            callback
            for callback in event_callbacks
            if not getattr(callback, "__module__", "").endswith(".clearml")
        ]
        removed_callbacks += len(event_callbacks) - len(retained_callbacks)
        callbacks[event_name] = retained_callbacks

    if removed_callbacks:
        console.print(
            f"[bold yellow]Disabled {removed_callbacks} optional Ultralytics ClearML plot callbacks so checkpoint/export steps remain non-fatal.[/bold yellow]"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLO or RF-DETR model from a YAML configuration file."
    )
    parser.add_argument(
        "--config",
        help="Path to a training config YAML file, or a filename inside the configs directory.",
    )
    return parser.parse_args()


def load_dataset_config(dataset_name):
    """Load dataset configuration from data.yaml file."""
    try:
        return load_project_dataset_config(dataset_name, datasets_root="datasets")
    except FileNotFoundError as error:
        console.print(f"[bold red]Error: {error}[/bold red]")
        raise


def verify_directories(dataset_config):
    """Verify that all required directories exist."""
    missing_dirs = verify_dataset_directories(dataset_config)
    if missing_dirs:
        console.print(
            "[bold red]Error: The following directories are missing:[/bold red]"
        )
        for dir in missing_dirs:
            console.print(f" - {dir}")
        raise FileNotFoundError(
            "The following directories are missing:\n" + "\n".join(missing_dirs)
        )


def select_config(config_path):
    config_folder = "configs"
    resolved_path = resolve_config_path(config_path, config_folder)
    if resolved_path is not None:
        console.print(
            f"\n[bold green]Selected configuration: {Path(resolved_path).name}[/bold green]"
        )
        return resolved_path

    yaml_files = list_config_files(config_folder)
    selection = get_user_choice(
        yaml_files + ["Exit"],
        title="Select Configuration",
        text="Use ↑↓ keys to navigate, Enter to select, 'q' to exit:",
    )
    if selection == "Exit":
        return None

    selected_file = os.path.abspath(os.path.join(config_folder, selection))
    console.print(f"\n[bold green]Selected configuration: {selection}[/bold green]")
    return selected_file


def verify_model_file(model_name):
    """Verify if the model file exists and load it, or download if it's a YOLO model."""
    try:
        yolo_class = import_ultralytics_yolo()

        if "nas" in model_name.lower():
            raise MLDependencyError(
                "YOLO-NAS support is deprecated in this build because SuperGradients "
                "conflicts with RF-DETR training dependencies."
            )

        if model_name.lower().startswith("yolo"):
            # Extract version and size from model name (e.g., YOLO26n -> version="26", size="n")
            # Handle both 1-digit (v8, v9) and 2-digit (v10, v11, v12, v26) versions
            remaining = model_name[4:]  # Remove "YOLO" prefix

            # Extract version (can be 1 or 2 digits followed by optional letters)
            version = ""
            size = ""

            for i, char in enumerate(remaining):
                if char.isalpha():
                    # Found the first letter - this is where size starts
                    version = remaining[:i]
                    size = remaining[i:].lower()
                    break

            if not version:
                # If no letters found, use the last character as size
                version = remaining[:-1]
                size = remaining[-1].lower()

            standard_name = f"yolo{version}{size}"
            console.print(
                f"\n[bold]Converting model name {model_name} to {standard_name}[/bold]"
            )
            model = yolo_class(standard_name + ".pt")
            console.print(
                f"\n[bold green]Successfully loaded/downloaded model: {standard_name}[/bold green]"
            )
            return model
        else:
            if not os.path.exists(model_name):
                console.print(
                    f"[bold red]Error: Model file {model_name} not found[/bold red]"
                )
                raise FileNotFoundError(f"Model file {model_name} not found")
            model = yolo_class(model_name)
            console.print(
                f"\n[bold green]Successfully loaded model: {model_name}[/bold green]"
            )
            return model
    except MLDependencyError as e:
        console.print(f"[bold red]{str(e)}[/bold red]")
        return None
    except Exception as e:
        console.print(
            f"[bold red]Error loading model {model_name}: {str(e)}[/bold red]"
        )
        return None


def effective_clearml_settings(config_clearml: dict | None) -> dict:
    global_clearml = load_settings().get("clearml", {})
    return deep_merge(global_clearml, config_clearml or {})


def initialize_clearml_task(project_name, task_name, tags, clearml_settings=None):
    clearml_settings = clearml_settings or DEFAULT_SETTINGS["clearml"]
    if not clearml_settings.get("enabled", True):
        return None
    try:
        from clearml import Task

        return Task.init(
            project_name=project_name,
            task_name=task_name,
            tags=tags,
        )
    except Exception as error:
        if clearml_settings.get("require_configured", False):
            console.print(f"[bold red]ClearML is required but not configured: {error}[/bold red]")
            return False
        console.print(f"[bold yellow]ClearML is not configured: {error}[/bold yellow]")
        selection = get_user_choice(
            ["Continue Without ClearML", "Cancel Training"],
            title="ClearML Setup Required",
            text=(
                "[yellow]ClearML experiment tracking could not be initialized.[/yellow] "
                "Training can still run, but per-epoch metrics, artifacts, and "
                "parameters will not be logged to the ClearML server."
            ),
            descriptions={
                "Continue Without ClearML": (
                    "[bold yellow]Train without remote experiment tracking.[/bold yellow]\n\n"
                    "• Local TensorBoard logs in [cyan]runs/[/cyan] still work.\n"
                    "• Useful if you're iterating locally and don't need a shared dashboard.\n"
                    "• You can re-run `clearml-init` later and the next run will log normally."
                ),
                "Cancel Training": (
                    "[bold red]Abort so you can configure ClearML first.[/bold red]\n\n"
                    "• Run [cyan]clearml-init[/cyan] in another terminal to set up credentials.\n"
                    "• Or edit [cyan]~/clearml.conf[/cyan] directly."
                ),
            },
            status_fields={"error": str(error)[:120]},
            tip=(
                "ClearML is optional — YOLOmatic runs fine without it. Only block "
                "training here if you specifically need the ClearML dashboard for this run."
            ),
        )
        if selection == "Cancel Training":
            return False
        return None


def upload_clearml_final_model(task, run_dir, clearml_settings):
    if task is None or not clearml_settings.get("upload_final_model", True) or run_dir is None:
        return
    for candidate in (run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"):
        if candidate.exists():
            try:
                task.upload_artifact(name="final_model", artifact_object=str(candidate))
            except Exception as error:
                console.print(f"[bold yellow]ClearML final model upload skipped: {error}[/bold yellow]")
            return


def normalize_class_names(names):
    """Return YOLO class names as display-safe strings for list or dict data.yaml forms."""
    if isinstance(names, dict):
        def _sort_key(value):
            text = str(value)
            return (0, int(text)) if text.isdigit() else (1, text)

        return [str(names[key]) for key in sorted(names, key=_sort_key)]
    if isinstance(names, list):
        return [str(name) for name in names]
    return []


def print_config_summary(config, dataset_config):
    """Print a summary of the loaded configurations."""
    console.print(Panel.fit("[bold]Configuration Summary[/bold]", style="bold blue"))
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Category", style="dim", width=20)
    table.add_column("Details")
    class_names = normalize_class_names(dataset_config.get("names", []))

    # Check if it's a YOLO NAS config
    if "experiment" in config:
        # YOLO NAS configuration
        table.add_row("Model Type", "YOLO NAS")
        table.add_row("Model", str(config["model"]["name"]))
        table.add_row("Dataset", str(config["dataset"]["name"]))
        table.add_row("Project Name", str(config["clearml"]["project_name"]))
        table.add_row("Batch Size", str(config["training"]["batch_size"]))
        table.add_row("Max Epochs", str(config["training"]["max_epochs"]))
        table.add_row("Pretrained Weights", str(config["model"]["pretrained_weights"]))
        table.add_row("Number of Classes", str(len(class_names)))
        table.add_row("Classes", ", ".join(class_names))

        # Add experiment info
        table.add_row("Experiment Name", str(config["experiment"]["name_prefix"]))
        table.add_row("Checkpoint Dir", str(config["experiment"]["checkpoint_dir"]))
    else:
        # Regular YOLO configuration
        table.add_row("Model Type", "YOLO")
        table.add_row("Model", str(config["settings"]["model_type"]))
        table.add_row("Dataset", str(config["settings"]["dataset"]))
        table.add_row("Project Name", str(config["clearml"]["project_name"]))
        table.add_row("Batch Size", str(config["training"]["batch"]))
        table.add_row("Epochs", str(config["training"]["epochs"]))
        table.add_row("Image Size", str(config["training"]["imgsz"]))
        table.add_row("Number of Classes", str(len(class_names)))
        table.add_row("Classes", ", ".join(class_names))

        # Add device info if available
        if "device" in config["training"]:
            table.add_row("Device", str(config["training"]["device"]))

    # Add Roboflow info if available
    if "roboflow" in dataset_config:
        table.add_row("Workspace", str(dataset_config["roboflow"]["workspace"]))
        table.add_row("Project", str(dataset_config["roboflow"]["project"]))
        table.add_row("Version", str(dataset_config["roboflow"]["version"]))

    console.print(table)


def upload_to_roboflow_if_configured(config, dataset_config, run_dir, model_type, console):
    """Automatically upload trained weights to Roboflow if configured."""
    roboflow_config = config.get("roboflow", {})
    if not roboflow_config.get("upload", False):
        return

    weight_name = roboflow_config.get("weight") or roboflow_config.get("auto_upload_weight", "best.pt")
    console.print(f"\n[bold green]Preparing to upload {weight_name} to Roboflow...[/bold green]")
    
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
        if roboflow_config.get("require_dataset_metadata", True):
            console.print("[bold red]Cannot upload: Dataset configuration lacks Roboflow workspace or project.[/bold red]")
        else:
            console.print("[bold yellow]Skipping Roboflow upload: dataset Roboflow metadata is missing.[/bold yellow]")
        return

    try:
        candidate = build_candidate(weight_path)
        candidate.workspace = workspace
        candidate.project_ids = project
        
        upload_type = model_type
        
        staged = stage_upload_candidate(candidate, upload_type)
        console.print(f"[bold]Uploading to workspace '{workspace}', project '{project}'...[/bold]")
        upload_model(staged, upload_type)
        console.print("[bold green]Roboflow upload completed successfully![/bold green]")
    except Exception as error:
        console.print(f"[bold red]Roboflow upload failed: {error}[/bold red]")


def main():
    task = None
    try:
        args = parse_args()
        config_file = select_config(args.config)
        if config_file is None:
            console.print(
                "[bold yellow]No configuration selected. Exiting.[/bold yellow]"
            )
            return

        # Load configuration from selected YAML file
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        if "experiment" in config:
            console.print(
                "[bold yellow]YOLO-NAS support is deprecated in this build.[/bold yellow]\n\n"
                "SuperGradients conflicts with RF-DETR's training dependency stack, "
                "so YOLOmatic no longer installs or routes YOLO-NAS training. "
                "Create a YOLO or RF-DETR config instead."
            )
            return

        # Extract parameters from config based on type
        if config.get("settings", {}).get("model_family") == "rfdetr":
            console.print(
                "\n[bold green]Routing RF-DETR configuration to RF-DETR trainer...[/bold green]"
            )
            from src.trainers.rfdetr_trainer import main as rfdetr_main

            rfdetr_main(config_file)
            return
        if config.get("settings", {}).get("model_family") == "detectron2":
            console.print(
                "\n[bold green]Routing Detectron2 configuration to Detectron2 trainer...[/bold green]"
            )
            from src.trainers.detectron2_trainer import main as detectron2_main

            detectron2_main(config_file)
            return
        if config.get("settings", {}).get("model_family") == "sam3.1":
            console.print(
                "\n[bold green]Routing SAM 3.1 configuration to SAM trainer...[/bold green]"
            )
            from src.trainers.sam_trainer import main as sam_main

            sam_main(config_file)
            return
        settings = config["settings"]
        clearml_settings = effective_clearml_settings(config.get("clearml", {}))
        training_params = dict(config["training"])
        export_params = config["export"]

        # Load dataset configuration
        model_name = settings["model_type"]
        dataset_section = config.get("dataset", {})
        dataset_name_or_path = dataset_section.get("prepared_dir") or (
            settings["dataset"] if "dataset" in settings else settings["model_type"]
        )
        dataset_config, data_yaml_path, dataset_path = load_dataset_config(dataset_name_or_path)
        verify_directories(dataset_config)

        normalized_cache, disk_cache_disabled = normalize_yolo_cache_setting(
            training_params.get("cache", False)
        )
        training_params["cache"] = normalized_cache
        config["training"] = training_params
        if disk_cache_disabled:
            console.print(
                "[bold yellow]Persistent dataset cache 'disk' is disabled to prevent "
                "large .npy files; continuing with cache=False.[/bold yellow]"
            )

        cache_cleanup = clean_dataset_image_cache(dataset_path)
        if cache_cleanup.removed_files:
            console.print(
                "[bold green]Removed "
                f"{cache_cleanup.removed_files:,} dataset image-cache files and reclaimed "
                f"{format_size(cache_cleanup.reclaimed_bytes)}.[/bold green]"
            )
        if cache_cleanup.errors:
            console.print(
                f"[bold yellow]Warning: {len(cache_cleanup.errors)} dataset cache files "
                "could not be removed.[/bold yellow]"
            )
        print_config_summary(config, dataset_config)

        # Get the correct model name based on config type
        if is_rfdetr_model(model_name):
            console.print(
                "\n[bold green]Routing RF-DETR configuration to RF-DETR trainer...[/bold green]"
            )
            from src.trainers.rfdetr_trainer import main as rfdetr_main

            rfdetr_main(config_file)
            return
        current_time = datetime.now().strftime(clearml_settings["task_name_format"])
        task_name = f"{model_name}-{current_time}"

        device_resolution = resolve_training_device(training_params.get("device"))
        if device_resolution.cancelled:
            console.print("[bold yellow]Training cancelled.[/bold yellow]")
            return
        if device_resolution.device != training_params.get("device"):
            training_params = dict(training_params)
            training_params["device"] = device_resolution.device
        if "project" not in training_params:
            training_params["project"] = "runs"
        if "name" not in training_params:
            training_params["name"] = task_name

        configure_ultralytics_runtime()

        model = verify_model_file(model_name)
        if model is None:
            console.print("[bold red]Model verification failed. Exiting.[/bold red]")
            return
        disable_ultralytics_clearml_callbacks(model)

        # Initialize ClearML Task
        task = initialize_clearml_task(
            project_name=clearml_settings["project_name"],
            task_name=task_name,
            tags=[model_name[:4].upper() + model_name[4:]],
            clearml_settings=clearml_settings,
        )
        if task is False:
            console.print("[bold yellow]Training cancelled.[/bold yellow]")
            return

        if task is not None and clearml_settings.get("log_hyperparameters", True):
            task.connect(
                {
                    "dataset_config": dataset_config,
                    "training_params": training_params,
                    "export_params": export_params,
                }
            )

        # Validate export configuration early to catch issues before training
        is_valid, export_errors = validate_export_config(export_params, console)
        if not is_valid:
            console.print(
                "\n[bold red]Export configuration errors detected. Please fix your config file.[/bold red]"
            )
            raise ValueError(
                f"Export config validation failed: {'; '.join(export_errors)}"
            )

        # Start training
        console.print("\n[bold green]Starting training...[/bold green]")
        model.train(data=data_yaml_path, **training_params)
        save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
        run_dir = Path(save_dir) if save_dir else None

        # Start validation
        console.print("\n[bold green]Starting validation...[/bold green]")
        model.val(data=data_yaml_path)

        # Export the model
        console.print("\n[bold green]Exporting model...[/bold green]")
        model.export(**export_params)

        upload_clearml_final_model(task, run_dir, clearml_settings)

        upload_to_roboflow_if_configured(config, dataset_config, run_dir, model_name, console)

        if run_dir is not None:
            metadata = build_tensorboard_metadata(
                model_name=model_name,
                dataset_name=settings["dataset"],
                config_path=config_file,
                run_name=task_name,
                device=training_params.get("device"),
                training_params=training_params,
            )
            backfill_ultralytics_tensorboard(
                run_dir=run_dir,
                metadata=metadata,
                device=training_params.get("device"),
            )
            emit_tensorboard_report(console, validate_tensorboard_run(run_dir))

        console.print("\n[bold green]Training completed successfully![/bold green]")

    except FileNotFoundError as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
    except yaml.YAMLError as e:
        console.print(f"[bold red]Invalid YAML configuration: {str(e)}[/bold red]")
    except MLDependencyError as e:
        console.print(f"[bold red]{str(e)}[/bold red]")
    except ValueError as e:
        if "Invalid CUDA 'device" in str(e):
            console.print(f"[bold red]{str(e)}[/bold red]")
            console.print(
                "[bold yellow]PyTorch cannot see a CUDA device in this environment. If `nvidia-smi` works, reinstall a CUDA-enabled torch build in this `.venv`.[/bold yellow]"
            )
        else:
            console.print(f"[bold red]Error: {str(e)}[/bold red]")
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {str(e)}[/bold red]")
        import traceback

        console.print(traceback.format_exc())
    finally:
        if task not in (None, False):
            task.close()


if __name__ == "__main__":
    main()
