import os
from ruamel.yaml import YAML
from shutil import copy2
from datetime import datetime
from models import model_data_dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from blessed import Terminal
import time

console = Console()
term = Terminal()


def clear_screen():
    print(term.clear)


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
            datasets.append({"name": folder, "size": format_size(size)})

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

    dataset_names = [d["name"] for d in datasets]
    return get_user_choice(
        dataset_names,
        allow_back=True,
        title="Select Dataset",
        text="Use ↑↓ keys to navigate, Enter to select, 'b' for back:",
    )


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


def update_config(model_choice, dataset_choice):
    """
    Update the configuration file with the selected model and dataset.
    Creates a backup of the existing config if present.
    """
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    config_file = "config.yaml"

    # Create config directory if it doesn't exist
    config_dir = "configs"
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    config_path = os.path.join(config_dir, config_file)

    # Backup existing config if it exists
    if os.path.exists(config_path):
        backup_name = f"config_backup_{format_timestamp()}.yaml"
        backup_path = os.path.join(config_dir, backup_name)
        try:
            copy2(config_path, backup_path)
            console.print(f"✅ Created backup: {backup_name}", style="bold green")
        except Exception as e:
            console.print(f"⚠️  Failed to create backup: {str(e)}", style="bold red")

    # Create new config
    config = {
        "model": {"name": model_choice, "timestamp": format_timestamp()},
        "dataset": {"path": f"datasets/{dataset_choice}", "name": dataset_choice},
        "training": {
            "batch_size": 16,
            "epochs": 100,
            "learning_rate": 0.001,
            "device": "cuda",
        },
    }

    try:
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        console.print(f"✅ Configuration saved to: {config_file}", style="bold green")

        # Display confirmation panel
        confirmation = Panel(
            Text(
                "Configuration has been updated successfully!\n"
                f"Model: {model_choice}\n"
                f"Dataset: {dataset_choice}\n"
                f"Config file: {config_file}",
                style="bold green",
            ),
            title="Success",
            border_style="green",
        )
        console.print(confirmation)

    except Exception as e:
        error_panel = Panel(
            Text(f"Failed to save configuration:\n{str(e)}", style="bold red"),
            title="Error",
            border_style="red",
        )
        console.print(error_panel)


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


def main():
    while True:
        clear_screen()
        print_stylized_header(f"YOLOmatic v1.0.0")
        options = ["YOLOv11", "YOLOv10", "YOLOv9", "YOLOv8", "YOLOX", "Exit"]
        choice = get_user_choice(
            options,
            allow_back=False,
            title="YOLO Model Selector",
            text="Use ↑↓ keys to navigate, Enter to select, 'q' to exit:",
        )

        if choice == "Exit":
            console.print("\n👋 Thank you for using YOLOmatic!", style="bold cyan")
            break

        while True:
            clear_screen()
            print_stylized_header(f"{choice} Models")

            # Get model data for the selected YOLO generation
            model_data = model_data_dict[choice.lower()]

            model_choice = get_user_choice(
                [model["Model"] for model in model_data],
                allow_back=True,
                title=f"Select a Model from {choice}",
                text="Use ↑↓ keys to navigate, Enter to select, 'b' to go back:",
                model_data=model_data,  # Pass model data to display comparison table
            )

            if model_choice == "Back":
                break

            clear_screen()
            print_stylized_header(f"Properties of {model_choice}")

            # Display detailed properties of selected model
            model_info = next(
                model for model in model_data if model["Model"] == model_choice
            )
            table = Table(
                title=f"Properties of {model_choice}", title_style="bold green"
            )

            for header in model_info.keys():
                table.add_column(header, justify="center", style="cyan")

            table.add_row(*[str(value) for value in model_info.values()])
            console.print(table)

            dataset_choice = list_datasets()
            if dataset_choice == "Back":
                continue

            if dataset_choice:
                print_summary(model_choice, dataset_choice)
                update_config(model_choice, dataset_choice)

            # Wait for any key press to continue
            console.print("\nPress any key to continue...", style="bold green")
            with term.cbreak():
                term.inkey()
            break


if __name__ == "__main__":
    main()
