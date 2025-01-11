import os
import yaml
from datetime import datetime
from ultralytics import YOLO
from clearml import Task
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Initialize Rich console
console = Console()


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


def select_config():
    """Select configuration file from configs directory."""
    config_folder = "configs"
    if not os.path.exists(config_folder):
        console.print(
            f"[bold red]Error: Config folder '{config_folder}' not found.[/bold red]"
        )
        raise FileNotFoundError(f"Config folder '{config_folder}' not found.")

    yaml_files = [f for f in os.listdir(config_folder) if f.endswith(".yaml")]

    if not yaml_files:
        console.print(
            "[bold red]Error: No YAML files found in the configs folder.[/bold red]"
        )
        raise FileNotFoundError("No YAML files found in the configs folder.")

    console.print("\n[bold]Available configuration files:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Number", style="dim", width=12)
    table.add_column("File Name")
    for i, file in enumerate(yaml_files, 1):
        table.add_row(str(i), file)
    console.print(table)

    while True:
        try:
            choice = int(input("\nEnter the number of the configuration file to use: "))
            if 1 <= choice <= len(yaml_files):
                selected_file = os.path.join(config_folder, yaml_files[choice - 1])
                console.print(
                    f"\n[bold green]Selected configuration: {yaml_files[choice - 1]}[/bold green]"
                )
                return selected_file
            else:
                console.print("[bold red]Invalid choice. Please try again.[/bold red]")
        except ValueError:
            console.print("[bold red]Invalid input. Please enter a number.[/bold red]")


def verify_model_file(model_name):
    """Verify if the model file exists and load it, or download if it's a YOLO model."""
    try:
        if model_name.lower().startswith("yolo"):
            version = model_name[4:6]
            size = model_name[-1].lower()
            standard_name = f"yolo{version}{size}"
            console.print(
                f"\n[bold]Converting model name {model_name} to {standard_name}[/bold]"
            )
            model = YOLO(standard_name + ".pt")
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
            model = YOLO(model_name)
            console.print(
                f"\n[bold green]Successfully loaded model: {model_name}[/bold green]"
            )
            return model
    except Exception as e:
        console.print(
            f"[bold red]Error loading model {model_name}: {str(e)}[/bold red]"
        )
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
    try:
        # Select configuration file
        config_file = select_config()
        if config_file is None:
            console.print(
                "[bold red]No configuration file selected. Exiting.[/bold red]"
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

        # Verify and load the model
        model = verify_model_file(model_name)
        if model is None:
            console.print("[bold red]Model verification failed. Exiting.[/bold red]")
            return

        # Initialize ClearML Task
        current_time = datetime.now().strftime(clearml_settings["task_name_format"])
        task_name = f"{model_name}-{current_time}"

        task = Task.init(
            project_name=clearml_settings["project_name"],
            task_name=task_name,
            tags=[model_name[:4].upper() + model_name[4:]],
        )

        # Log configurations to ClearML
        task.connect(
            {
                "dataset_config": dataset_config,
                "training_params": training_params,
                "export_params": export_params,
            }
        )

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
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {str(e)}[/bold red]")
        import traceback

        console.print(traceback.format_exc())
    finally:
        # Ensure ClearML task is closed
        if "task" in locals():
            task.close()


if __name__ == "__main__":
    main()
