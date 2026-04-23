import subprocess
import sys

from src.utils.cli import console, get_user_choice, print_stylized_header
from src.utils.project import find_run_directories


def build_tensorboard_command(selected_logdir):
    return [sys.executable, "-m", "tensorboard.main", "--logdir", selected_logdir]
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

    # Create descriptions for dynamically discovered runs
    launcher_descriptions = {
        str(d): f"Monitor specific training run located at {d}" for d in run_dirs
    }
    launcher_descriptions["Monitor All (logdir=runs)"] = (
        "Monitor all training runs simultaneously in one TensorBoard dashboard."
    )
    launcher_descriptions["Exit"] = "Close the TensorBoard launcher."

    choice = get_user_choice(
        options,
        title="Select Run to Monitor",
        text="Use ↑↓ keys to navigate, Enter to select, 'q' to exit:",
        descriptions=launcher_descriptions,
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
        cmd = build_tensorboard_command(selected_logdir)
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        console.print("[bold red]Error: Python interpreter not found.[/bold red]")
        console.print(
            "Please ensure that YOLOmatic is running from a valid virtual environment."
        )
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        console.print(
            f"[bold red]TensorBoard exited with status code {exc.returncode}.[/bold red]"
        )
        console.print(
            "Please ensure that 'tensorboard' is installed in your current environment."
        )
        sys.exit(exc.returncode)
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
