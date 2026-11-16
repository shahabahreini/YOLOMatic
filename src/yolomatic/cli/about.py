"""The About page (moved out of cli/run.py's main loop)."""
from __future__ import annotations

from rich import box
from rich.align import Align
from rich.panel import Panel
from rich.table import Table

from yolomatic.config.settings import roboflow_credential_status, ultralytics_credential_status
from yolomatic.ui.theme import safe
from yolomatic.utils.cli import clear_screen, console
from yolomatic.utils.tui import get_gpu_status


def render_about_page() -> None:
    clear_screen()
    import os
    import platform

    from rich.console import Group
    from rich.text import Text

    from yolomatic.__version__ import __version__
    from yolomatic.utils.ml_dependencies import check_hf_auth

    # --- Terminal Width ---
    term_w = console.width

    # --- System Environment Detection ---
    os_system = platform.system()
    os_release = platform.release()
    os_display = f"{os_system} {os_release}"

    py_version = platform.python_version()
    py_arch = platform.architecture()[0]
    py_display = f"{py_version} ({py_arch})"

    venv_path = os.getenv("VIRTUAL_ENV")
    if venv_path:
        venv_display = os.path.basename(venv_path)
    else:
        venv_display = "None (System Python)"

    # Use the background-cached GPU probe: importing torch here froze the
    # screen for seconds with no feedback.
    gpu_status = get_gpu_status()
    if gpu_status == "Detecting...":
        gpu_info = "[dim]Detecting...[/dim]"
    elif gpu_status == "CPU":
        gpu_info = "[nord.yellow]Inactive[/nord.yellow] (CPU only)"
    else:
        gpu_info = f"[nord.green]Active[/nord.green] ({safe(gpu_status)})"

    # --- Integrations Status ---
    try:
        rf_status = roboflow_credential_status()
        rf_info = "[nord.green]Connected[/nord.green]" if rf_status.get("api_key") else "[dim]Not Configured[/dim]"
    except Exception:
        rf_info = "[nord.red]Error[/nord.red]"

    try:
        ultralytics_status = ultralytics_credential_status()
        ult_info = "[nord.green]Connected[/nord.green]" if ultralytics_status.get("api_key") else "[dim]Not Configured[/dim]"
    except Exception:
        ult_info = "[nord.red]Error[/nord.red]"

    try:
        hf_token = check_hf_auth()
        hf_info = "[nord.green]Connected[/nord.green]" if hf_token else "[dim]Not Configured[/dim]"
    except Exception:
        hf_info = "[nord.red]Error[/nord.red]"

    # --- Centered Logo & Subtitle ---
    banner_lines = [
        " __  ______  __    ____  __  ___  ____  __  ____  ",
        " \\ \\/ / __ \\/ /   / __ \\/  |/  / / __ \\/ /_/ ___/  ",
        "  \\  / /_/ / /___/ /_/ / /|_/ / / /_/ / __/ /__    ",
        "  /_/\\____/_____/\\____/_/  /_/  /_/ /_/\\__/\\___/   "
    ]
    logo_width = max(len(line) for line in banner_lines)

    if term_w >= logo_width + 8:
        gradient_styles = [
            "nord.frost0.bold",
            "nord.frost1.bold",
            "nord.frost2.bold",
            "nord.frost3.bold"
        ]
        banner_str = ""
        for line, style in zip(banner_lines, gradient_styles):
            banner_str += f"[{style}]{line}[/{style}]\n"
        header_element = Align.center(banner_str.strip())
    else:
        header_element = Align.center("[nord.frost0.bold]YOLOmatic[/nord.frost0.bold]")

    subtitle_element = Align.center(
        "[dim]Automated computer-vision training, configuration, and dataset management.[/dim]"
    )

    # --- Quiet, Minimalist Table Layout ---
    left_table = Table.grid(padding=(0, 2))
    left_table.add_column(style="nord.frost2.bold", justify="right", width=12)
    left_table.add_column(style="white", width=28)

    left_table.add_row("Product:", "YOLOmatic")
    left_table.add_row("Version:", f"[nord.green.bold]{__version__}[/nord.green.bold]")
    left_table.add_row("Creator:", "Shahab Bahreini Jangjoo")
    left_table.add_row("Contact:", "[nord.frost1]shahabahreini@hotmail.com[/nord.frost1]")
    left_table.add_row("License:", "[nord.yellow]Apache-2.0[/nord.yellow]")

    right_table = Table.grid(padding=(0, 2))
    right_table.add_column(style="nord.frost2.bold", justify="right", width=14)
    right_table.add_column(style="white", width=26)

    right_table.add_row("OS:", os_display)
    right_table.add_row("Python:", py_display)
    right_table.add_row("PyTorch GPU:", gpu_info)
    right_table.add_row("Virtual Env:", f"[nord.frost1]{venv_display}[/nord.frost1]")
    right_table.add_row("Roboflow:", rf_info)
    right_table.add_row("Ultralytics:", ult_info)
    right_table.add_row("Hugging Face:", hf_info)

    # Split layout matching screen width
    if term_w >= 90:
        split_table = Table.grid(padding=(0, 6))
        split_table.add_column(width=42)
        split_table.add_column(width=42)
        split_table.add_row(left_table, right_table)
    else:
        split_table = Table.grid(padding=(1, 0))
        split_table.add_column(width=min(70, term_w - 4))
        split_table.add_row(left_table)
        split_table.add_row(Text(""))
        split_table.add_row(right_table)

    # Master layout
    dashboard_content = Group(
        header_element,
        Text(""),
        subtitle_element,
        Text(""),
        Text(""),
        Align.center(split_table)
    )

    console.print("\n")
    console.print(
        Panel(
            dashboard_content,
            title="[nord.frost0.bold]About[/nord.frost0.bold]",
            border_style="grey37",
            padding=(2, 4),
            box=box.ROUNDED,
            expand=True
        )
    )
    console.print("\n")
    try:
        input("Press Enter to return to Main Menu...")
    except (EOFError, KeyboardInterrupt):
        pass
