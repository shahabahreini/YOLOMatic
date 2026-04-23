import argparse
import os
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.utils.cli import get_user_choice
from src.utils.ml_dependencies import (
    MLDependencyError,
    import_ultralytics_settings,
    import_ultralytics_yolo,
)
from src.utils.project import (
    list_config_files,
    load_dataset_config as load_project_dataset_config,
    resolve_config_path,
    verify_dataset_directories,
)
from src.utils.tensorboard import (
    backfill_ultralytics_tensorboard,
    build_tensorboard_metadata,
    emit_tensorboard_report,
    validate_tensorboard_run,
)
from src.utils.training_preflight import resolve_training_device


# Initialize Rich console
console = Console()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLO or YOLO-NAS model from a YAML configuration file."
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
            console.print(
                f"\n[bold]Detected YOLO-NAS model {model_name}; using SuperGradients trainer[/bold]"
            )
            return model_name

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


def initialize_clearml_task(project_name, task_name, tags):
    try:
        from clearml import Task

        return Task.init(
            project_name=project_name,
            task_name=task_name,
            tags=tags,
        )
    except Exception as error:
        console.print(f"[bold yellow]ClearML is not configured: {error}[/bold yellow]")
        selection = get_user_choice(
            ["Continue Without ClearML", "Cancel Training"],
            title="ClearML Setup Required",
            text="Use ↑↓ keys to choose whether to continue without ClearML or cancel training:",
        )
        if selection == "Cancel Training":
            return False
        return None


def print_config_summary(config, dataset_config):
    """Print a summary of the loaded configurations."""
    console.print(Panel.fit("[bold]Configuration Summary[/bold]", style="bold blue"))
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Category", style="dim", width=20)
    table.add_column("Details")

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
        table.add_row("Number of Classes", str(len(dataset_config["names"])))
        table.add_row("Classes", ", ".join(dataset_config["names"]))

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
        table.add_row("Number of Classes", str(len(dataset_config["names"])))
        table.add_row("Classes", ", ".join(dataset_config["names"]))

        # Add device info if available
        if "device" in config["training"]:
            table.add_row("Device", str(config["training"]["device"]))

    # Add Roboflow info if available
    if "roboflow" in dataset_config:
        table.add_row("Workspace", str(dataset_config["roboflow"]["workspace"]))
        table.add_row("Project", str(dataset_config["roboflow"]["project"]))
        table.add_row("Version", str(dataset_config["roboflow"]["version"]))

    console.print(table)


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

        # Extract parameters from config based on type
        if "experiment" in config:
            # YOLO NAS config
            settings = {
                "model": config["model"]["name"],
                "dataset": config["dataset"]["name"],
            }
            clearml_settings = config["clearml"]
            training_params = config["training"]
            export_params = config["export"]
        else:
            # Regular YOLO config
            settings = config["settings"]
            clearml_settings = config["clearml"]
            training_params = config["training"]
            export_params = config["export"]

        # Load dataset configuration
        dataset_config, data_yaml_path, dataset_path = load_dataset_config(
            settings["dataset"] if "dataset" in settings else settings["model_type"]
        )
        verify_directories(dataset_config)
        print_config_summary(config, dataset_config)

        # Get the correct model name based on config type
        if "experiment" in config:
            model_name = settings["model"]
        else:
            # For regular YOLO config, use model_type
            model_name = settings["model_type"]

        if "experiment" in config:
            console.print(
                "\n[bold green]Routing YOLO-NAS configuration to NAS trainer...[/bold green]"
            )
            from src.trainers.nas_trainer import main as nas_main
            nas_main(config_file)
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

        model = verify_model_file(model_name)
        if model is None:
            console.print("[bold red]Model verification failed. Exiting.[/bold red]")
            return

        # Initialize ClearML Task
        task = initialize_clearml_task(
            project_name=clearml_settings["project_name"],
            task_name=task_name,
            tags=[model_name[:4].upper() + model_name[4:]],
        )
        if task is False:
            console.print("[bold yellow]Training cancelled.[/bold yellow]")
            return

        if task is not None:
            task.connect(
                {
                    "dataset_config": dataset_config,
                    "training_params": training_params,
                    "export_params": export_params,
                }
            )

        # Enable TensorBoard explicitly
        ultra_settings = import_ultralytics_settings()
        ultra_settings.update({"tensorboard": True})

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
