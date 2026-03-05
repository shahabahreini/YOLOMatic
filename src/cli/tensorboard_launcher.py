import subprocess
import sys
from pathlib import Path

# Try to import from src.cli.run for consistent UI, fallback if not possible
try:
    from src.cli.run import console, get_user_choice, print_stylized_header, term
except ImportError:
    # Minimal fallback if src.cli.run is not available
    from rich.console import Console

    console = Console()
    term = None

    def print_stylized_header(text):
        console.print(f"[bold cyan]{text}[/bold cyan]")

    def get_user_choice(options, **kwargs):
        for i, opt in enumerate(options):
            console.print(f"{i+1}. {opt}")
        while True:
            try:
                idx = int(input("Select an option: ")) - 1
                if 0 <= idx < len(options):
                    return options[idx]
            except ValueError:
                pass


def find_run_directories(base_dir):
    """
    Find all subdirectories that contain TensorBoard event files.
    """
    event_dirs = []
    base_path = Path(base_dir)

    if not base_path.exists():
        return event_dirs

    # Recursively search for run directories (typically contain args.yaml)
    for run_file in base_path.rglob("args.yaml"):
        parent_dir = run_file.parent
        if parent_dir not in event_dirs:
            event_dirs.append(parent_dir)

    return sorted(event_dirs)


def main():
    """
    Launch TensorBoard with interactive run selection.
    """
    base_log_dir = "runs"
    run_dirs = find_run_directories(base_log_dir)

    print_stylized_header("TensorBoard Launcher")

    if not run_dirs:
        console.print(
            f"[bold red]No training results found in '{base_log_dir}'.[/bold red]"
        )
        console.print(
            "Please make sure you have started training and that it has progressed enough to write logs."
        )
        sys.exit(1)

    # Prepare options for the user
    options = [str(d) for d in run_dirs]
    options.append("Monitor All (logdir=runs)")
    options.append("Exit")

    choice = get_user_choice(
        options,
        title="Select Run to Monitor",
        text="Use ↑↓ keys to navigate, Enter to select, 'q' to exit:",
    )

    if choice == "Exit" or choice == "Back":
        sys.exit(0)

    if choice == "Monitor All (logdir=runs)":
        selected_logdir = base_log_dir
    else:
        selected_logdir = choice

    console.print(
        f"\n[bold green]Launching TensorBoard with logdir: {selected_logdir}[/bold green]"
    )

    try:
        cmd = ["tensorboard", "--logdir", selected_logdir]
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        console.print("[bold red]Error: 'tensorboard' command not found.[/bold red]")
        console.print(
            "Please ensure that 'tensorboard' is installed in your environment."
        )
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]TensorBoard stopped.[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(
            f"[bold red]An error occurred while launching TensorBoard: {e}[/bold red]"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
