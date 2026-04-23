import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from rich import box
from rich.align import Align
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from ruamel.yaml import YAML

from src.config.generator import YOLOConfigGenerator, YOLONASConfigGenerator
from src.models.data import model_data_dict
from src.utils.cli import (
    clear_screen,
    console,
    get_user_choice,
    print_stylized_header,
    render_summary_panel,
    render_table,
)
from src.utils.ml_dependencies import MLDependencyError, import_torch
from src.utils.project import format_size, list_dataset_directories

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logging.getLogger("src.config.generator").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def check_ultralytics_version():
    """Check and update ultralytics package version with professional UX."""
    try:
        import subprocess
        from importlib.metadata import version as get_version

        from packaging import version

        pip_command = [sys.executable, "-m", "pip"]
        active_python = sys.executable

        # Get versions
        with console.status("[bold]Checking versions...", spinner="dots"):
            current_version = get_version("ultralytics")

            try:
                process = subprocess.run(
                    [*pip_command, "index", "versions", "ultralytics"],
                    capture_output=True,
                    text=True,
                )
                latest_version = (
                    process.stdout.split("Available versions: ")[1]
                    .split(",")[0]
                    .strip()
                )
            except Exception:
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
                f"Current: [bold]{current_version}[/bold]   →   Latest: [bold]{latest_version}[/bold]\n"
                f"Python: [bold]{active_python}[/bold]",
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
                        [*pip_command, "install", "--upgrade", "ultralytics"],
                        capture_output=True,
                        text=True,
                    )

                console.print("\n")
                if result.returncode == 0:
                    updated_version = get_version("ultralytics")
                    console.print(
                        Panel(
                            Align.center(
                                "✅ Updated successfully\n\n"
                                f"Installed: [bold]{updated_version}[/bold]\n"
                                "Note: a future `uv sync` can restore the version from `uv.lock` unless you update the lockfile too.",
                                vertical="middle",
                            ),
                            border_style="green",
                            padding=(1, 1),
                        )
                    )
                else:
                    console.print(
                        Panel(
                            Align.center(
                                "❌ Update failed\n\n"
                                f"Command: [bold]{' '.join(pip_command)} install --upgrade ultralytics[/bold]\n"
                                f"{(result.stderr or result.stdout or 'No error output').strip()}",
                                vertical="middle",
                            ),
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


# Removed clear_screen, now imported from src.utils.tui


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
    model_choice,
    dataset_name,
    config_file,
    dataset_info,
    profile_selection=None,
    profile_context=None,
):
    """Display a clean summary of the configuration"""
    # Load config
    config_path = os.path.join("configs", config_file)
    if not os.path.exists(config_path):
        console.print("[red]Error: Config file not found![/red]")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Detect device
    device = "💻 CPU"
    try:
        torch = import_torch()
        if torch.cuda.is_available():
            device = "🚀 GPU (CUDA)"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "🚀 GPU (MPS)"
    except Exception:
        pass

    # Use the new summary panel for a cleaner look
    fields = {
        "Model": model_choice,
        "Dataset": dataset_name,
        "Device": device,
        "Config File": config_file,
    }

    if "nas" in model_choice.lower():
        training = config.get("training", {})
        fields.update({
            "Batch Size": training.get("batch_size", "N/A"),
            "Max Epochs": training.get("max_epochs", "N/A"),
            "Workers": training.get("num_workers", "N/A"),
        })
    else:
        training = config.get("training", {})
        fields.update({
            "Batch Size": training.get("batch", "N/A"),
            "Epochs": training.get("epochs", "N/A"),
            "Image Size": training.get("imgsz", "N/A"),
            "Workers": training.get("workers", "N/A"),
        })

    render_summary_panel("Configuration Summary", fields)

    # Simplified dataset paths display
    path_rows = []
    if "nas" in model_choice.lower():
        structure = config.get("dataset", {}).get("structure", {})
        base_dir = config.get("dataset", {}).get("base_dir", "")
        path_rows = [
            ["Train", os.path.join(base_dir, structure.get("train", {}).get("images", "N/A"))],
            ["Validation", os.path.join(base_dir, structure.get("valid", {}).get("images", "N/A"))],
            ["Test", os.path.join(base_dir, structure.get("test", {}).get("images", "N/A"))],
        ]
    else:
        model_config = config.get("model", {})
        data_dir = model_config.get("data_dir", "")
        path_rows = [
            ["Train", os.path.join(data_dir, model_config.get("train_images_dir", "N/A"))],
            ["Validation", os.path.join(data_dir, model_config.get("val_images_dir", "N/A"))],
            ["Test", os.path.join(data_dir, model_config.get("test_images_dir", "N/A"))],
        ]

    render_table("Dataset Paths", ["Type", "Path"], path_rows, title_style="bold blue")


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


def list_datasets():
    datasets_folder = "datasets"
    if not os.path.exists(datasets_folder):
        os.makedirs(datasets_folder)
        console.print(
            f"✨ '{datasets_folder}' folder created. Please add COCO or any other compatible dataset into it.",
            style="bold yellow",
        )
        return None

    datasets = list_dataset_directories(datasets_folder)

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

    dataset_names = [Path(d["path"]) for d in datasets]
    name_to_path = {path.name: str(path) for path in dataset_names}

    dataset_descriptions = {
        d["name"]: f"Select dataset '{d['name']}' ({d['size']}) located at {d['path']}"
        for d in datasets
    }
    dataset_descriptions["Back"] = "Return to the previous menu."

    choice = get_user_choice(
        list(name_to_path.keys()),  # Show basename in menu
        allow_back=True,
        title="Select Dataset",
        text="Use ↑↓ keys to navigate, Enter to select, 'b' for back:",
        descriptions=dataset_descriptions,
    )

    return name_to_path.get(choice) if choice != "Back" else choice


# Removed print_model_info, now handled by src.utils.tui


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
    try:
        torch = import_torch()
    except MLDependencyError:
        return "cpu"

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_dataset_name = dataset_name.replace(" ", "_").lower()

    # Initialize appropriate generator
    if "nas" in model_choice.lower():
        generator = YOLONASConfigGenerator(str(dataset_path))
        config_file = f"{model_choice}_{safe_dataset_name}_{timestamp}.yaml"
        profile_context = None
        profile_selection = None
    else:
        generator = YOLOConfigGenerator(str(dataset_path))
        config_file = f"{model_choice}_{safe_dataset_name}_{timestamp}.yaml"
        profile_context = generator.get_regular_yolo_profile_context(model_choice)
        profile_selection = None

    # Check dataset type compatibility
    dataset_type = generator.dataset_info.get("task_type", "unknown")
    is_seg_model = "-seg" in model_choice.lower()

    if (
        dataset_type == "segmentation"
        and not is_seg_model
        and "nas" not in model_choice.lower()
    ):
        console.print(
            "\n[bold red]⚠️ WARNING: You selected a Bounding Box (Detection) model, but the dataset appears to be for Instance Segmentation![/bold red]"
        )
        if (
            get_user_choice(
                ["Continue Anyway", "Go Back"],
                text="Do you want to continue?",
                title="Dataset Mismatch",
            )
            == "Go Back"
        ):
            return False
    elif dataset_type == "detection" and is_seg_model:
        console.print(
            "\n[bold red]⚠️ WARNING: You selected a Segmentation model, but the dataset appears to be for Bounding Box Detection![/bold red]"
        )
        if (
            get_user_choice(
                ["Continue Anyway", "Go Back"],
                text="Do you want to continue?",
                title="Dataset Mismatch",
            )
            == "Go Back"
        ):
            return False

    if "nas" not in model_choice.lower():
        profile_selection = choose_regular_yolo_profiles(
            dataset_name,
            profile_context,
            model_choice,
        )
        if profile_selection is None:
            return False
        display_regular_yolo_profile_selection_summary(
            dataset_name,
            profile_selection,
            profile_context,
        )
        config = generator.generate_config(
            model_choice,
            profile_selection,
            profile_context,
        )
    else:
        config = generator.generate_config(model_choice)

    # Save new config
    config_path = os.path.join(config_dir, config_file)
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    console.print(f"✅ Configuration saved to: {config_file}", style="bold green")

    # Display summary
    display_configuration_summary(
        model_choice,
        dataset_name,
        config_file,
        generator.dataset_info,
        profile_selection,
        profile_context,
    )
    display_paths_info(generator.dataset_info)

    return True

def format_profile_name(value: str) -> str:
    return value.replace("_", " ").title()


def build_hint_block(title: str, lines: list[str]) -> str:
    if not lines:
        return ""
    formatted_lines = "\n".join(f"- {line}" for line in lines)
    return f"{title}:\n{formatted_lines}"


def build_regular_yolo_profile_summary_text(
    dataset_name: str,
    profile_context: dict[str, Any],
    model_choice: str,
) -> str:
    model_metrics = profile_context["model_metrics"]
    dataset_metrics = profile_context["dataset_metrics"]
    system_metrics = profile_context["system_metrics"]
    recommended_profiles = profile_context["recommended_profiles"]
    recommended_worker = profile_context["worker_profiles"][
        recommended_profiles["worker"]
    ]

    lines = [
        f"Dataset: {dataset_name}",
        (
            "Model scan: "
            f"{model_choice} "
            f"({format_profile_name(model_metrics['heaviness'])} model)"
        ),
        (
            "Dataset scan: "
            f"{format_size(int(dataset_metrics['total_size_bytes']))}, "
            f"{int(dataset_metrics['image_count'])} images, "
            f"{int(dataset_metrics['label_count'])} labels, "
            f"{int(dataset_metrics['total_file_count'])} files"
        ),
        (
            "System scan: "
            f"{format_size(int(system_metrics['available_ram_bytes']))} RAM free, "
            f"{int(system_metrics['cpu_count'])} CPU cores, "
            f"device={system_metrics['device']}"
        ),
    ]

    if system_metrics["available_gpu_memory_bytes"] is not None:
        gpu_line = f"GPU memory free: {format_size(int(system_metrics['available_gpu_memory_bytes']))}"
        if system_metrics["total_gpu_memory_bytes"] is not None:
            gpu_line += f" of {format_size(int(system_metrics['total_gpu_memory_bytes']))} total"
        lines.append(gpu_line)

    if model_metrics["params_millions"] is not None:
        lines.append(f"Model params: {model_metrics['params_millions']:.1f}M")
    if model_metrics["flops_billions"] is not None:
        lines.append(f"Model FLOPs: {model_metrics['flops_billions']:.1f}B")

    lines.extend(
        [
            "",
            "YOLOmatic recommendation factors:",
            "- model heaviness from variant size, params, and FLOPs",
            "- dataset size, image count, label count, and file count",
            "- available RAM, CPU cores, and detected device",
            "- free GPU memory when CUDA is available",
            "",
            "Recommended profiles:",
            (
                f"- Augmentation: {format_profile_name(recommended_profiles['augmentation'])}"
            ),
            f"- Compute: {format_profile_name(recommended_profiles['compute'])}",
            (
                "- Workers: "
                f"{format_profile_name(recommended_profiles['worker'])} "
                f"({int(recommended_worker['workers'])} workers)"
            ),
        ]
    )
    return "\n".join(lines)


def select_profile_option(
    title: str,
    prompt_text: str,
    option_descriptions: dict[str, str],
    recommended_key: str,
    hint_lines: list[str] | None = None,
) -> str | None:
    option_map: dict[str, str] = {}
    option_labels: list[str] = []

    hint_block = ""
    if hint_lines:
        hint_block = f"\n\n{build_hint_block('Hints', hint_lines)}"

    descriptions: dict[str, str] = {}
    for key, description in option_descriptions.items():
        label = format_profile_name(key)
        if key == recommended_key:
            label = f"{label} [recommended]"
        option_map[label] = key
        option_labels.append(label)
        descriptions[label] = description
    
    descriptions["Back"] = "Return to the previous configuration step."

    choice = get_user_choice(
        option_labels,
        allow_back=True,
        title=title,
        text=f"{prompt_text}{hint_block}",
        descriptions=descriptions,
    )
    if choice == "Back":
        return None
    return option_map[choice]


def display_regular_yolo_profile_selection_summary(
    dataset_name: str,
    profile_selection: dict[str, str],
    profile_context: dict[str, Any],
) -> None:
    model_metrics = profile_context["model_metrics"]
    dataset_metrics = profile_context["dataset_metrics"]
    system_metrics = profile_context["system_metrics"]
    worker_profile = profile_context["worker_profiles"][profile_selection["worker"]]

    clear_screen()
    print_stylized_header("Config Profile Summary")

    table = Table(title="Selected Profile Settings", title_style="bold green")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Dataset", dataset_name)
    table.add_row("Model", str(model_metrics["model_choice"]))
    table.add_row(
        "Model Heaviness",
        format_profile_name(str(model_metrics["heaviness"])),
    )
    if model_metrics["params_millions"] is not None:
        table.add_row("Model Params", f"{model_metrics['params_millions']:.1f}M")
    if model_metrics["flops_billions"] is not None:
        table.add_row("Model FLOPs", f"{model_metrics['flops_billions']:.1f}B")
    table.add_row(
        "Dataset Size",
        format_size(int(dataset_metrics["total_size_bytes"])),
    )
    table.add_row("Image Count", str(int(dataset_metrics["image_count"])))
    table.add_row("Label Count", str(int(dataset_metrics["label_count"])))
    table.add_row(
        "Available RAM",
        format_size(int(system_metrics["available_ram_bytes"])),
    )
    table.add_row("Detected Device", str(system_metrics["device"]))
    if system_metrics["available_gpu_memory_bytes"] is not None:
        table.add_row(
            "Available GPU Memory",
            format_size(int(system_metrics["available_gpu_memory_bytes"])),
        )
    table.add_row(
        "Augmentation Profile",
        format_profile_name(profile_selection["augmentation"]),
    )
    table.add_row(
        "Compute Profile",
        format_profile_name(profile_selection["compute"]),
    )
    table.add_row(
        "Worker Profile",
        (
            f"{format_profile_name(profile_selection['worker'])} "
            f"({int(worker_profile['workers'])} workers)"
        ),
    )
    table.add_row("Worker Notes", str(worker_profile["description"]))
    table.add_row(
        "Augmentation Impact",
        "Controls how many augmentation keys YOLOmatic enables in training",
    )
    table.add_row(
        "Compute Impact",
        "Controls batch aggressiveness and whether cache is enabled when safe",
    )
    table.add_row(
        "Workers Impact",
        "Controls dataloader parallelism based on RAM, CPU, GPU, dataset pressure, and model heaviness",
    )

    console.print(table)


def choose_regular_yolo_profiles(
    dataset_name: str,
    profile_context: dict[str, Any],
    model_choice: str,
) -> dict[str, str] | None:
    summary_text = build_regular_yolo_profile_summary_text(
        dataset_name,
        profile_context,
        model_choice,
    )
    recommended_profiles = profile_context["recommended_profiles"]

    start_option_map = {
        "Recommended": "recommended",
        "Customize": "customize",
    }
    start_descriptions = {
        "Recommended": "Fastest path - let YOLOmatic heuristics decide augmentation, compute, and worker settings for you.",
        "Customize": "Manual path - review and choose your own augmentation intensity, compute aggressiveness, and worker counts.",
        "Back": "Return to dataset selection."
    }

    hint_block = build_hint_block(
        "Hints",
        [
            "Use the recommended option unless you already know you need more or less augmentation.",
            "Compute controls how hard YOLOmatic pushes memory and throughput.",
            "Workers control dataloader parallelism and can increase RAM pressure.",
        ],
    )

    initial_choice = get_user_choice(
        list(start_option_map.keys()),
        allow_back=True,
        title="Regular YOLO Config Profiles",
        text=(
            f"{summary_text}\n\n"
            "Pick the fast path if you want the current codebase heuristics to decide for you. "
            f"Pick customize if you want to review each area manually.\n\n{hint_block}"
        ),
        descriptions=start_descriptions,
    )

    if initial_choice == "Back":
        return None
    if start_option_map[initial_choice] == "recommended":
        return dict(recommended_profiles)

    augmentation_options = {
        "minimum": "Essential training values only with almost no extra augmentation",
        "low": "Mild augmentation using flips, mosaic, and mixup",
        "medium": "Stronger generalization with color and geometric augmentation",
    }
    compute_options = {
        "conservative": "Safer memory usage and lower risk of instability",
        "balanced": "Best default for most systems and datasets",
        "aggressive": "Pushes throughput harder when RAM and GPU headroom are strong",
    }
    worker_options = {
        key: f"{int(details['workers'])} workers - {details['description']}"
        for key, details in profile_context["worker_profiles"].items()
    }

    augmentation_choice = select_profile_option(
        "Select Augmentation Profile",
        f"{summary_text}\n\nChoose the augmentation intensity for this dataset:",
        augmentation_options,
        recommended_profiles["augmentation"],
        [
            "Minimum is the easiest to reason about and keeps the config close to core training values.",
            "Low adds only basic robustness improvements.",
            "Medium adds more color and geometric changes, which can improve generalization but also change training behavior more.",
        ],
    )
    if augmentation_choice is None:
        return None

    compute_choice = select_profile_option(
        "Select Compute Profile",
        f"{summary_text}\n\nChoose how strongly YOLOmatic should push system resources:",
        compute_options,
        recommended_profiles["compute"],
        [
            "This profile mainly affects batch aggressiveness and cache behavior.",
            "Conservative is better when GPU memory is tight or the model is heavy.",
            "Aggressive is best only when your RAM, GPU memory, and dataset pressure all look healthy.",
        ],
    )
    if compute_choice is None:
        return None

    worker_choice = select_profile_option(
        "Select Worker Profile",
        f"{summary_text}\n\nChoose the dataloader worker profile:",
        worker_options,
        recommended_profiles["worker"],
        [
            "More workers can improve throughput, but they also use more RAM and can stress slower disks.",
            "If you are unsure, keep the recommended worker profile.",
            "Heavy worker settings make the most sense when RAM is strong and the GPU needs faster data feeding.",
        ],
    )
    if worker_choice is None:
        return None

    return {
        "augmentation": augmentation_choice,
        "compute": compute_choice,
        "worker": worker_choice,
    }


def get_model_menu():
    """Get the list of available YOLO models."""
    # Add 'yolo_nas' to the existing models list
    models = [
        "yolo26",
        "yolo26-seg",
        "yolov12",
        "yolov12-seg",
        "yolov11",
        "yolov11-seg",
        "yolov10",
        "yolov9",
        "yolov9-seg",
        "yolov8",
        "yolov8-seg",
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
            descriptions={
                "Select Model": "Select a YOLO model and dataset to generate a training configuration.",
                "Check Ultralytics Version": "Check if a new version of the ultralytics package is available and update if needed.",
                "Exit": "Exit the YOLOmatic application.",
            },
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
                descriptions={
                    "yolo26": "State-of-the-art YOLOv8-based models with improved architecture and performance.",
                    "yolo26-seg": "Instance segmentation variants of the YOLO26 model family.",
                    "yolov12": "The latest YOLO generation focusing on extreme efficiency and accuracy.",
                    "yolov12-seg": "Instance segmentation variants of the YOLOv12 model family.",
                    "yolov11": "General-purpose YOLO model with balanced performance.",
                    "yolov11-seg": "Instance segmentation variants of the YOLOv11 model family.",
                    "yolov10": "Real-time object detection model with improved head design.",
                    "yolov9": "Programmable Gradient Information (PGI) based YOLO model.",
                    "yolov9-seg": "Instance segmentation variants of the YOLOv9 model family.",
                    "yolov8": "Industry standard YOLO model for reliable detection.",
                    "yolov8-seg": "Instance segmentation variants of the YOLOv8 model family.",
                    "yolox": "Anchor-free YOLO implementation for high performance.",
                    "yolo_nas": "Neural Architecture Search optimized YOLO models from Deci.ai.",
                },
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
            if not update_config(model_choice, dataset_choice):
                continue

            # Ask if user wants to continue
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
