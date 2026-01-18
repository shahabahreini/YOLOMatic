# Logger setup
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from shutil import copy2

import torch
import yaml
from blessed import Terminal
from rich import box
from rich.align import Align
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from ruamel.yaml import YAML

from config_generator import YOLOConfigGenerator, YOLONASConfigGenerator
from models import model_data_dict

logging.basicConfig(
    level=logging.WARNING,  # Change from INFO to WARNING
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logging.getLogger("config_generator").setLevel(logging.WARNING)
logger = logging.getLogger("config_generator")

# Only enable DEBUG logging if needed
# logger.setLevel(logging.DEBUG)

console = Console()
term = Terminal()


def check_ultralytics_version():
    """Check and update ultralytics package version with professional UX."""
    try:
        import subprocess

        import pkg_resources
        from packaging import version

        # Get versions
        with console.status("[bold]Checking versions...", spinner="dots"):
            current_version = pkg_resources.get_distribution("ultralytics").version

            try:
                process = subprocess.run(
                    ["pip", "index", "versions", "ultralytics"],
                    capture_output=True,
                    text=True,
                )
                latest_version = (
                    process.stdout.split("Available versions: ")[1]
                    .split(",")[0]
                    .strip()
                )
            except:
                import json
                import urllib.request

                url = "https://pypi.org/pypi/ultralytics/json"
                data = json.load(urllib.request.urlopen(url))
                latest_version = data["info"]["version"]

        clear_screen()
        console.print("\n")

        # Version status
        needs_update = version.parse(current_version) < version.parse(latest_version)
        status_text = "Update Available" if needs_update else "Up to Date"
        status_icon = "⚠️" if needs_update else "✅"
        status_style = "yellow" if needs_update else "green"

        # Create status panel
        status_panel = Panel(
            Align.center(
                f"{status_icon} {status_text}\n\n"
                f"Current: [bold]{current_version}[/bold]   →   Latest: [bold]{latest_version}[/bold]",
                vertical="middle",
            ),
            border_style=status_style,
            padding=(2, 1),
        )

        console.print(status_panel)
        console.print("\n")

        # Handle updates if needed
        if needs_update:
            if (
                get_user_choice(
                    ["Update", "Skip"],
                    text="Update ultralytics?",
                )
                == "Update"
            ):
                with console.status("[bold]Updating ultralytics...", spinner="dots"):
                    result = subprocess.run(
                        ["pip", "install", "--upgrade", "ultralytics"],
                        capture_output=True,
                        text=True,
                    )

                console.print("\n")
                if result.returncode == 0:
                    console.print(
                        Panel(
                            Align.center("✅ Updated successfully", vertical="middle"),
                            border_style="green",
                            padding=(1, 1),
                        )
                    )
                else:
                    console.print(
                        Panel(
                            Align.center(f"❌ Update failed", vertical="middle"),
                            border_style="red",
                            padding=(1, 1),
                        )
                    )

    except Exception as e:
        console.print("\n")
        console.print(
            Panel(
                Align.center(f"❌ Error: {str(e)}", vertical="middle"),
                border_style="red",
                padding=(1, 1),
            )
        )

    console.print("\n")
    input("Press Enter to continue...")


def clear_screen():
    print(term.clear)


def backup_config(config_file):
    """
    Create a backup of the existing configuration file if it exists.

    Args:
        config_file (str): Name of the configuration file to backup
    """
    config_path = os.path.join("configs", config_file)
    if os.path.exists(config_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"config_backup_{timestamp}.yaml"
        backup_path = os.path.join("configs", backup_name)

        try:
            with open(config_path, "r") as source, open(backup_path, "w") as target:
                target.write(source.read())
            console.print(f"✅ Created backup: {backup_name}", style="green")
        except Exception as e:
            console.print(f"⚠️ Failed to create backup: {str(e)}", style="yellow")


def display_configuration_summary(
    model_choice, dataset_name, config_file, dataset_info
):
    """Display a clean summary of the configuration"""
    console = Console()

    # Main configuration table
    table = Table(
        title="Configuration Summary",
        title_style="bold green",
        box=box.ROUNDED,
        padding=(0, 2),
        width=80,
    )
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="white")

    # Add training parameters
    config_path = os.path.join("configs", config_file)
    if not os.path.exists(config_path):
        console.print("[red]Error: Config file not found![/red]")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Detect device including MPS
    device = "💻 CPU"
    if torch.cuda.is_available():
        device = "🚀 GPU (CUDA)"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "🚀 GPU (MPS)"

    # Get model info based on config type
    if "nas" in model_choice.lower():
        model_info = config.get("model", {})
        model_name = model_info.get("name", model_choice)
        experiment_info = config.get("experiment", {})

        # Basic information
        table.add_row("Model", model_name)
        table.add_row("Dataset", dataset_name)
        table.add_row("Device", device)
        table.add_row("Config File", config_file)
        table.add_row("Experiment Name", experiment_info.get("name_prefix", "N/A"))

        # Training parameters
        training = config.get("training", {})
        lr_config = training.get("learning_rate", {})
        optimizer_config = training.get("optimizer", {})

        table.add_row("Batch Size", str(training.get("batch_size", "N/A")))
        table.add_row("Max Epochs", str(training.get("max_epochs", "N/A")))
        table.add_row("Initial LR", str(lr_config.get("initial_lr", "N/A")))
        table.add_row(
            "Warmup Initial LR", str(lr_config.get("warmup_initial_lr", "N/A"))
        )
        table.add_row("Warmup Epochs", str(lr_config.get("warmup_epochs", "N/A")))
        table.add_row("Weight Decay", str(optimizer_config.get("weight_decay", "N/A")))
        table.add_row("Workers", str(training.get("num_workers", "N/A")))
        table.add_row(
            "Pretrained Weights", str(model_info.get("pretrained_weights", "N/A"))
        )

    else:
        # Regular YOLO config
        settings = config.get("settings", {})
        model_name = settings.get("model_type", model_choice)

        # Basic information
        table.add_row("Model", model_name)
        table.add_row("Dataset", dataset_name)
        table.add_row("Device", device)
        table.add_row("Config File", config_file)

        # Training parameters
        training = config.get("training", {})
        table.add_row("Batch Size", str(training.get("batch", "N/A")))
        table.add_row("Epochs", str(training.get("epochs", "N/A")))
        table.add_row("Image Size", str(training.get("imgsz", "N/A")))
        table.add_row("Workers", str(training.get("workers", "N/A")))
        table.add_row("Label Smoothing", str(training.get("label_smoothing", "N/A")))
        table.add_row("Cache", str(training.get("cache", "N/A")))
        table.add_row("Close Mosaic", str(training.get("close_mosaic", "N/A")))

    # Add number of classes
    if "nas" in model_choice.lower():
        classes = config.get("dataset", {}).get("classes", [])
    else:
        classes = config.get("model", {}).get("classes", [])

    table.add_row("Number of Classes", str(len(classes)))
    table.add_row("Classes", ", ".join(classes) if classes else "N/A")

    console.print("\n")
    console.print(table)

    # Display dataset paths in a separate table
    path_table = Table(
        title="Dataset Paths",
        title_style="bold blue",
        box=box.ROUNDED,
        padding=(0, 2),
        width=80,
    )
    path_table.add_column("Type", style="cyan")
    path_table.add_column("Path", style="white")

    if "nas" in model_choice.lower():
        structure = config.get("dataset", {}).get("structure", {})
        base_dir = config.get("dataset", {}).get("base_dir", "")

        train_path = os.path.join(
            base_dir, structure.get("train", {}).get("images", "N/A")
        )
        valid_path = os.path.join(
            base_dir, structure.get("valid", {}).get("images", "N/A")
        )
        test_path = os.path.join(
            base_dir, structure.get("test", {}).get("images", "N/A")
        )

        path_table.add_row("Train", train_path)
        path_table.add_row("Validation", valid_path)
        path_table.add_row("Test", test_path)
    else:
        model_config = config.get("model", {})
        data_dir = model_config.get("data_dir", "")

        train_path = os.path.join(data_dir, model_config.get("train_images_dir", "N/A"))
        valid_path = os.path.join(data_dir, model_config.get("val_images_dir", "N/A"))
        test_path = os.path.join(data_dir, model_config.get("test_images_dir", "N/A"))

        path_table.add_row("Train", train_path)
        path_table.add_row("Validation", valid_path)
        path_table.add_row("Test", test_path)

    console.print("\n")
    console.print(path_table)


def display_paths_info(dataset_info):
    """Display dataset paths in a clean format"""
    console = Console()

    paths_table = Table(
        title="Dataset Paths", title_style="bold green", box=box.ROUNDED
    )
    paths_table.add_column("Type", style="cyan")
    paths_table.add_column("Path", style="white")

    paths_table.add_row("Train", dataset_info.get("train_path", "N/A"))
    paths_table.add_row("Validation", dataset_info.get("valid_path", "N/A"))
    paths_table.add_row("Test", dataset_info.get("test_path", "N/A"))

    console.print("\n")
    console.print(paths_table)


def print_stylized_header(text):
    """
    Print a stylized header using rich.
    """
    header = Text(text, style="bold cyan", justify="center")
    console.print(Panel(header, title="", border_style="cyan"))


def list_datasets():
    datasets_folder = "datasets"
    if not os.path.exists(datasets_folder):
        os.makedirs(datasets_folder)
        console.print(
            f"✨ '{datasets_folder}' folder created. Please add COCO or any other compatible dataset into it.",
            style="bold yellow",
        )
        return None

    datasets = []
    for folder in os.listdir(datasets_folder):
        folder_path = os.path.join(datasets_folder, folder)
        if os.path.isdir(folder_path):
            size = get_folder_size(folder_path)
            datasets.append(
                {
                    "name": folder,
                    "path": os.path.abspath(folder_path),  # Store full path
                    "size": format_size(size),
                }
            )

    if not datasets:
        console.print(
            f"❌ No datasets found in '{datasets_folder}' folder.", style="bold red"
        )
        return None

    table = Table(title="Available Datasets", title_style="bold green")
    table.add_column("Dataset Name", justify="center", style="cyan")
    table.add_column("Size", justify="center", style="cyan")

    for dataset in datasets:
        table.add_row(dataset["name"], dataset["size"])

    console.print(table)

    dataset_names = [d["path"] for d in datasets]  # Use full paths
    name_to_path = {os.path.basename(p): p for p in dataset_names}

    choice = get_user_choice(
        list(name_to_path.keys()),  # Show basename in menu
        allow_back=True,
        title="Select Dataset",
        text="Use ↑↓ keys to navigate, Enter to select, 'b' for back:",
    )

    return name_to_path.get(choice) if choice != "Back" else choice


def print_model_info(model_data):
    """
    Display a comparison table for the selected YOLO generation using rich.
    """
    table = Table(
        title=f"Comparison Table for {model_data[0]['Model']}",
        title_style="bold green",
    )

    # Add headers
    for header in model_data[0].keys():
        table.add_column(header, justify="center", style="cyan")

    # Add rows
    for row in model_data:
        table.add_row(*[str(value) for value in row.values()])

    console.print(table)


def get_user_choice(
    options,
    allow_back=False,
    title="Select an Option",
    text="Use ↑↓ keys to navigate, Enter to select:",
    model_data=None,  # Add parameter for model data
):
    """
    Display an interactive menu using blessed for selection.
    Supports arrow keys, Enter for selection, 'b' for back, and 'q' for quit.
    """
    if allow_back:
        options = options + ["Back"]

    current_selection = 0

    def print_menu():
        print(term.clear)
        print_stylized_header(title)

        # Print comparison table if model data is provided
        if model_data:
            print_model_info(model_data)

        console.print(Text(text, style="bold yellow"))

        for i, option in enumerate(options):
            if i == current_selection:
                # Highlight the selected option
                shortcut = ""
                if option == "Back":
                    shortcut = " (or 'b')"
                elif option == "Exit":
                    shortcut = " (or 'q')"
                print(term.black_on_white(f" > {option}{shortcut}"))
            else:
                shortcut = ""
                if option == "Back":
                    shortcut = " (or 'b')"
                elif option == "Exit":
                    shortcut = " (or 'q')"
                print(f"   {option}{shortcut}")

    with term.cbreak(), term.hidden_cursor():
        while True:
            print_menu()

            key = term.inkey()

            if key.name == "KEY_UP":
                current_selection = (current_selection - 1) % len(options)
            elif key.name == "KEY_DOWN":
                current_selection = (current_selection + 1) % len(options)
            elif key.name == "KEY_ENTER":
                return options[current_selection]
            elif key.lower() == "b" and allow_back:
                return "Back"
            elif key.lower() == "q" and "Exit" in options:
                return "Exit"

            # Add a small delay to prevent screen flicker
            time.sleep(0.05)


def format_timestamp():
    """Generate a formatted timestamp for config files."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_summary(model_choice, dataset_choice):
    """
    Display a summary of the selected configuration using rich.
    """
    clear_screen()
    print_stylized_header("Configuration Summary")

    table = Table(title="Selected Configuration", title_style="bold green")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Model", model_choice)
    table.add_row("Dataset", dataset_choice)
    table.add_row("Timestamp", format_timestamp())

    console.print(table)


def detect_device():
    """
    Detect the appropriate device for training based on system capabilities.
    Returns 'cuda' for NVIDIA GPUs, 'mps' for Apple Silicon, or 'cpu' as fallback.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def update_config(model_choice, dataset_choice):
    """Update the configuration file with the selected model and dataset."""
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)

    config_dir = "configs"
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    dataset_path = Path(dataset_choice)
    dataset_name = dataset_path.name

    # Initialize appropriate generator
    if "nas" in model_choice.lower():
        generator = YOLONASConfigGenerator(str(dataset_path))
        config_file = "config_nas.yaml"
    else:
        generator = YOLOConfigGenerator(str(dataset_path))
        config_file = "config.yaml"

    # Generate and save configuration
    config = generator.generate_config(model_choice)

    # Create backup if needed
    backup_config(config_file)

    # Save new config
    config_path = os.path.join(config_dir, config_file)
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    console.print(f"✅ Configuration saved to: {config_file}", style="bold green")

    # Display summary
    display_configuration_summary(
        model_choice, dataset_name, config_file, generator.dataset_info
    )
    display_paths_info(generator.dataset_info)


def get_folder_size(folder_path):
    """Calculate the total size of a folder in bytes."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
    return total_size


def format_size(size_in_bytes):
    """Format size in bytes to human readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.2f} PB"


def get_model_menu():
    """Get the list of available YOLO models."""
    # Add 'yolo_nas' to the existing models list
    models = [
        "yolo26",
        "yolov12",
        "yolov11",
        "yolov10",
        "yolov9",
        "yolov8",
        "yolox",
        "yolo_nas",
    ]
    return models


def main():
    while True:
        clear_screen()
        print_stylized_header("YOLO Model Selector")

        # Add version check option to main menu
        main_menu_options = ["Select Model", "Check Ultralytics Version", "Exit"]

        main_choice = get_user_choice(
            main_menu_options,
            title="Main Menu",
            text="Use ↑↓ keys to navigate, Enter to select, 'q' to exit:",
        )

        if main_choice == "Exit":
            clear_screen()
            console.print("\U0001f44b Goodbye!", style="bold cyan")
            break

        elif main_choice == "Check Ultralytics Version":
            check_ultralytics_version()
            continue

        elif main_choice == "Select Model":
            # Get model choice
            model_types = get_model_menu()
            model_choice = get_user_choice(
                model_types,
                title="YOLO Model Selector",
                text="Use ↑↓ keys to navigate, Enter to select, 'b' for back:",
                allow_back=True,
            )

            if model_choice == "Back":
                continue

            # Rest of your existing model selection code...
            if model_choice == "yolo_nas":
                # Get YOLO NAS specific models
                nas_models = [model["Model"] for model in model_data_dict["yolo_nas"]]
                model_variant = get_user_choice(
                    nas_models,
                    allow_back=True,
                    title=f"Select {model_choice.upper()} Variant",
                    text="Use ↑↓ keys to navigate, Enter to select, 'b' for back:",
                    model_data=model_data_dict["yolo_nas"],
                )

                if model_variant == "Back":
                    continue

                model_choice = model_variant
            else:
                # Show variants for other YOLO models
                variants = [model["Model"] for model in model_data_dict[model_choice]]
                model_variant = get_user_choice(
                    variants,
                    allow_back=True,
                    title=f"Select {model_choice.upper()} Variant",
                    text="Use ↑↓ keys to navigate, Enter to select, 'b' for back:",
                    model_data=model_data_dict[model_choice],
                )

                if model_variant == "Back":
                    continue

                model_choice = model_variant

            # Continue with dataset selection...
            dataset_choice = list_datasets()
            if dataset_choice == "Back":
                continue
            elif dataset_choice is None:
                continue

            # Show summary and update config
            print_summary(model_choice, dataset_choice)
            update_config(model_choice, dataset_choice)

            # Ask if user wants to continue
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
