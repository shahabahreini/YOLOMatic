from __future__ import annotations

import argparse
import socket
import subprocess
import sys
from collections.abc import Sequence

from rich.panel import Panel

from yolomatic.utils.cli import NAV_BACK, console, get_user_choice, print_stylized_header
from yolomatic.utils.project import find_run_directories


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


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        return probe.connect_ex((host, port)) == 0


def choose_port(preferred: int, attempts: int = 20) -> int:
    """Return ``preferred`` or the next free port (TensorBoard's own error for a
    busy port used to be reported as "tensorboard is not installed")."""
    for candidate in range(preferred, preferred + attempts):
        if not port_in_use(candidate):
            return candidate
    raise RuntimeError(f"No free port found in {preferred}-{preferred + attempts - 1}.")


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="yolomatic-tensorboard", description="Launch TensorBoard for YOLOmatic runs.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Preferred port (default: {DEFAULT_PORT}).")
    parser.add_argument("--logdir", help="Log directory to open directly (skips the run picker).")
    args, _unknown = parser.parse_known_args(argv)
    return args


def main(port: int | None = None, argv: Sequence[str] | None = None) -> int:
    """Launch TensorBoard with interactive run selection.

    Returns an exit code instead of calling ``sys.exit``, so it is safe to
    invoke from inside the YOLOmatic TUI without tearing down the parent.
    """
    args = parse_args(argv)
    requested_port = port if port is not None else args.port
    try:
        port = choose_port(requested_port)
    except RuntimeError as error:
        console.print(Panel(f"[bold red]{error}[/bold red]", border_style="red", padding=(1, 2)))
        return 1
    if port != requested_port:
        console.print(f"[yellow]Port {requested_port} is busy; using {port} instead.[/yellow]")
    if args.logdir:
        return _run_tensorboard(args.logdir, port)
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

    if choice in ("Back", "Exit", NAV_BACK):
        return 0

    selected_logdir = base_log_dir if choice == "Monitor All (logdir=runs)" else choice
    return _run_tensorboard(selected_logdir, port)


if __name__ == "__main__":
    sys.exit(main())
