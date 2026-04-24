from __future__ import annotations

import subprocess
import sys

from rich.panel import Panel

from src.utils.cli import console, get_user_choice, print_stylized_header
from src.utils.project import find_run_directories


DEFAULT_PORT = 6006


def build_tensorboard_command(selected_logdir: str, port: int = DEFAULT_PORT) -> list[str]:
    return [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(selected_logdir),
        "--port",
        str(port),
    ]


def _run_tensorboard(selected_logdir: str, port: int) -> int:
    """Launch TensorBoard as a blocking subprocess.

    Returns the child process exit code. Catches its own KeyboardInterrupt so
    callers (``main`` or the TUI) can resume instead of terminating.
    """
    console.print(
        Panel(
            f"[bold green]TensorBoard starting[/bold green]\n"
            f"URL:    [bold cyan]http://localhost:{port}[/bold cyan]\n"
            f"logdir: [dim]{selected_logdir}[/dim]\n\n"
            "[dim]Press Ctrl+C here to stop TensorBoard and return to YOLOmatic.[/dim]",
            border_style="green",
            padding=(1, 2),
        )
    )

    try:
        completed = subprocess.run(build_tensorboard_command(selected_logdir, port))
    except FileNotFoundError:
        console.print(
            Panel(
                "[bold red]Python interpreter not found.[/bold red]\n"
                "YOLOmatic must run inside its virtual environment "
                "(`.venv`) so the `tensorboard` module is importable.",
                border_style="red",
                padding=(1, 2),
            )
        )
        return 1
    except KeyboardInterrupt:
        console.print("\n[bold yellow]TensorBoard stopped by user.[/bold yellow]")
        return 0
    except Exception as error:
        console.print(
            Panel(
                f"[bold red]TensorBoard failed to launch:[/bold red] {error}",
                border_style="red",
                padding=(1, 2),
            )
        )
        return 1

    if completed.returncode == 0:
        console.print("[bold green]TensorBoard stopped.[/bold green]")
        return 0

    console.print(
        Panel(
            f"[bold red]TensorBoard exited with code {completed.returncode}.[/bold red]\n"
            "Make sure 'tensorboard' is installed in the active environment "
            "(`uv sync` or `pip install tensorboard`).",
            border_style="red",
            padding=(1, 2),
        )
    )
    return completed.returncode


def main(port: int = DEFAULT_PORT) -> int:
    """Launch TensorBoard with interactive run selection.

    Returns an exit code instead of calling ``sys.exit``, so it is safe to
    invoke from inside the YOLOmatic TUI without tearing down the parent.
    """
    base_log_dir = "runs"
    print_stylized_header("TensorBoard Launcher")

    try:
        run_dirs = find_run_directories(base_log_dir)
    except Exception as error:
        console.print(
            Panel(
                f"[bold red]Could not scan '{base_log_dir}' for runs:[/bold red] {error}",
                border_style="red",
                padding=(1, 2),
            )
        )
        return 1

    if not run_dirs:
        console.print(
            Panel(
                f"[bold red]No training runs found in '{base_log_dir}'.[/bold red]\n"
                "Start a training run first, then return here to inspect the logs.",
                border_style="red",
                padding=(1, 2),
            )
        )
        return 1

    options = [str(path) for path in run_dirs]
    options.append("Monitor All (logdir=runs)")

    descriptions = {
        str(path): f"Open TensorBoard against this specific training run: {path}"
        for path in run_dirs
    }
    descriptions["Monitor All (logdir=runs)"] = (
        "Point TensorBoard at the entire runs/ directory so every run shows up "
        "in one dashboard."
    )
    descriptions["Back"] = "Return to the previous menu without launching."

    choice = get_user_choice(
        options,
        allow_back=True,
        title="Select Run to Monitor",
        text="Use ↑↓ keys to navigate, Enter to select, 'b' to go back:",
        descriptions=descriptions,
        breadcrumbs=["YOLOmatic", "TensorBoard"],
    )

    if choice in ("Back", "Exit"):
        return 0

    selected_logdir = base_log_dir if choice == "Monitor All (logdir=runs)" else choice
    return _run_tensorboard(selected_logdir, port)


if __name__ == "__main__":
    sys.exit(main())
