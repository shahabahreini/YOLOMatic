import argparse
import os
from datetime import datetime

import yaml
from clearml import Task
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    from src.cli.run import get_user_choice
    from src.utils.ml_dependencies import (
        MLDependencyError,
        import_torch,
        import_ultralytics_settings,
        import_ultralytics_yolo,
    )
except ImportError:
    from utils.ml_dependencies import (
        MLDependencyError,
        import_torch,
        import_ultralytics_settings,
        import_ultralytics_yolo,
    )

    try:
        from cli.run import get_user_choice
    except ImportError:

        def get_user_choice(
            options,
            allow_back=False,
            title="Select an Option",
            text="Use ↑↓ keys to navigate, Enter to select:",
            model_data=None,
        ):
            return options[0]


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
    dataset_path = os.path.abspath(os.path.join("datasets", dataset_name))
    data_yaml_path = os.path.abspath(os.path.join(dataset_path, "data.yaml"))

    if not os.path.exists(data_yaml_path):
        console.print(
            f"[bold red]Error: data.yaml not found in {dataset_path}[/bold red]"
        )
        raise FileNotFoundError(f"data.yaml not found in {dataset_path}")

    with open(data_yaml_path, "r") as file:
        dataset_config = yaml.safe_load(file)

    base_path = os.path.dirname(data_yaml_path)
    for key in ["train", "val", "test"]:
        relative_path = dataset_config[key].lstrip("/")
        dataset_config[key] = os.path.join(base_path, relative_path)

    return dataset_config, data_yaml_path, dataset_path


def verify_directories(dataset_config):
    """Verify that all required directories exist."""
    missing_dirs = []
    for dir_type, dir_path in [
        ("Training", dataset_config["train"]),
        ("Validation", dataset_config["val"]),
        ("Test", dataset_config["test"]),
    ]:
        if not os.path.exists(dir_path):
            missing_dirs.append(f"{dir_type} directory: {dir_path}")

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
    if config_path:
        candidate_paths = [config_path, os.path.join(config_folder, config_path)]
        for candidate_path in candidate_paths:
            if os.path.isfile(candidate_path):
                selected_file = os.path.abspath(candidate_path)
                console.print(
                    f"\n[bold green]Selected configuration: {os.path.basename(selected_file)}[/bold green]"
                )
                return selected_file

    if not os.path.exists(config_folder):
        missing_name = config_path or "<auto-select>"
        raise FileNotFoundError(
            f"Config file '{missing_name}' not found and config folder '{config_folder}' does not exist."
        )

    yaml_files = sorted(f for f in os.listdir(config_folder) if f.endswith(".yaml"))
    if not yaml_files:
        missing_name = config_path or "<auto-select>"
        raise FileNotFoundError(
            f"Config file '{missing_name}' not found and no YAML files exist in '{config_folder}'."
        )

    if config_path:
        available_configs = ", ".join(yaml_files)
        raise FileNotFoundError(
            f"Config file '{config_path}' not found. Available configs: {available_configs}"
        )

    if len(yaml_files) == 1:
        selected_file = os.path.abspath(os.path.join(config_folder, yaml_files[0]))
        console.print(
            f"\n[bold green]Selected configuration: {yaml_files[0]}[/bold green]"
        )
        return selected_file

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


def resolve_training_device(training_params):
    requested_device = training_params.get("device")
    if requested_device is None:
        return training_params

    try:
        torch = import_torch()
    except MLDependencyError:
        return training_params

    normalized_device = str(requested_device).strip().lower()
    wants_cuda = (
        normalized_device == "cuda" or normalized_device.replace(",", "").isdigit()
    )

    if wants_cuda and not torch.cuda.is_available():
        torch_version = getattr(torch, "__version__", "unknown")
        build_hint = (
            "Your current PyTorch build is CPU-only."
            if "+cpu" in torch_version
            else "PyTorch cannot access CUDA in this environment."
        )
        console.print(
            "[bold yellow]CUDA was requested in the training config, but PyTorch cannot use it.[/bold yellow]"
        )
        console.print(
            f"[bold yellow]Detected torch build: {torch_version}[/bold yellow]"
        )
        console.print(f"[bold yellow]{build_hint}[/bold yellow]")
        console.print(
            "[bold yellow]If `nvidia-smi` works but torch reports no CUDA devices, reinstall a CUDA-enabled PyTorch build for this environment.[/bold yellow]"
        )
        selection = get_user_choice(
            ["Continue on CPU", "Cancel Training"],
            title="CUDA Device Unavailable",
            text="Use ↑↓ keys to continue on CPU or cancel training:",
        )
        if selection == "Cancel Training":
            return False

        updated_training_params = dict(training_params)
        updated_training_params["device"] = "cpu"
        console.print("[bold yellow]Continuing training on CPU.[/bold yellow]")
        return updated_training_params

    return training_params


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
            try:
                from src.trainers.nas_trainer import main as nas_main
            except ImportError:
                from .nas_trainer import main as nas_main
            nas_main(config_file)
            return

        # Verify and load the model
        model = verify_model_file(model_name)
        if model is None:
            console.print("[bold red]Model verification failed. Exiting.[/bold red]")
            return

        training_params = resolve_training_device(training_params)
        if training_params is False:
            console.print("[bold yellow]Training cancelled.[/bold yellow]")
            return

        # Initialize ClearML Task
        current_time = datetime.now().strftime(clearml_settings["task_name_format"])
        task_name = f"{model_name}-{current_time}"

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
        os.environ["YOLO_DATASET_DIR"] = os.path.abspath("datasets/Oxford Pets")
        model.train(data=data_yaml_path, **training_params)

        # Start validation
        console.print("\n[bold green]Starting validation...[/bold green]")
        metrics = model.val(data=data_yaml_path)

        # Export the model
        console.print("\n[bold green]Exporting model...[/bold green]")
        model.export(**export_params)

        console.print("\n[bold green]Training completed successfully![/bold green]")

    except FileNotFoundError as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
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
        if "task" in locals() and task not in (None, False):
            task.close()


if __name__ == "__main__":
    main()
