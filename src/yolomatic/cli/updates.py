"""Dependency health check and in-place upgrades (moved out of cli/run.py)."""
from __future__ import annotations

import os
import shutil
import sys

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from yolomatic.utils.cli import clear_screen, console, get_user_choice, print_stylized_header


def _render_dependency_table(statuses):
    """Build the themed comparison Table for a dependency-health report."""
    from yolomatic.utils.version_check import IMPORTANCE_STYLE, SEVERITY_META

    table = Table(
        title="Critical Dependencies",
        title_style="bold cyan",
        border_style="dim",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("Package", style="cyan", no_wrap=True)
    table.add_column("Role", style="dim")
    table.add_column("Importance", justify="center", no_wrap=True)
    table.add_column("Installed", justify="center", no_wrap=True)
    table.add_column("Latest", justify="center", no_wrap=True)
    table.add_column("Status", justify="left", no_wrap=True)

    for status in statuses:
        color, glyph, label = SEVERITY_META[status.severity]
        installed_display = status.installed or "—"
        latest_display = status.latest or (
            f"[dim]{status.error}[/dim]" if status.error else "—"
        )
        table.add_row(
            status.package.display_name,
            status.package.description,
            IMPORTANCE_STYLE[status.package.importance],
            installed_display,
            latest_display,
            f"[{color}]{glyph} {label}[/{color}]",
        )
    return table


def _render_dependency_summary(statuses):
    """High-level banner Panel that calls out the most urgent problem."""
    missing = [s for s in statuses if s.severity == "missing"]
    major = [s for s in statuses if s.severity == "major"]
    minor = [s for s in statuses if s.severity == "minor"]
    patch = [s for s in statuses if s.severity == "patch"]
    unknown = [s for s in statuses if s.severity == "unknown"]

    if missing:
        names = ", ".join(s.package.name for s in missing)
        body = (
            f"[bold red]{len(missing)} critical package(s) missing:[/bold red] {names}\n"
            "[dim]Run `uv sync` (or `pip install -r requirements.txt`) to install them.[/dim]"
        )
        border = "red"
    elif major:
        names = ", ".join(s.package.name for s in major)
        body = (
            f"[bold red]{len(major)} major update(s) available — breaking changes possible:[/bold red] {names}\n"
            "[dim]Review release notes before upgrading.[/dim]"
        )
        border = "red"
    elif minor or patch:
        updates = minor + patch
        names = ", ".join(s.package.name for s in updates)
        body = f"[bold yellow]{len(updates)} non-breaking update(s) available:[/bold yellow] {names}"
        border = "yellow"
    elif unknown:
        names = ", ".join(s.package.name for s in unknown)
        body = (
            f"[bold]PyPI unreachable for {len(unknown)} package(s):[/bold] {names}\n"
            "[dim]Check your network connection and try Refresh.[/dim]"
        )
        border = "dim"
    else:
        body = "[bold green]All tracked dependencies are up to date.[/bold green]"
        border = "green"

    return Panel(body, border_style=border, padding=(1, 2), box=box.ROUNDED)


def build_upgrade_command(packages: list[str]) -> list[str]:
    """``uv pip install`` against the running interpreter, or pip when uv is absent.

    uv works even in venvs created without pip; the pip fallback means a
    missing uv binary no longer crashes with FileNotFoundError.
    """
    if shutil.which("uv"):
        return ["uv", "pip", "install", "--python", sys.executable, "--upgrade", *packages]
    return [sys.executable, "-m", "pip", "install", "--upgrade", *packages]


def _run_pip_upgrade(packages):
    """Upgrade packages in the active environment; return True on success.

    Output streams straight to the terminal: a multi-minute torch download used
    to hide behind a spinner with no progress at all.
    """
    import subprocess

    command = build_upgrade_command(list(packages))
    console.print(f"[dim]$ {' '.join(command)}[/dim]\n")
    try:
        result = subprocess.run(command, env=os.environ.copy(), check=False)
    except OSError as error:
        console.print(Panel(f"[bold red]Could not start the installer:[/bold red] {error}",
                            border_style="red", padding=(1, 2), box=box.ROUNDED))
        return False

    if result.returncode == 0:
        console.print(
            Panel(
                f"[bold green]Successfully upgraded:[/bold green] {', '.join(packages)}\n\n"
                "[dim]`uv sync` may re-pin versions from uv.lock unless the lockfile "
                "is updated afterwards.[/dim]",
                border_style="green",
                padding=(1, 2),
                box=box.ROUNDED,
            )
        )
        return True

    console.print(
        Panel(
            f"[bold red]Upgrade failed (exit {result.returncode}).[/bold red] See the output above.",
            border_style="red",
            padding=(1, 2),
            box=box.ROUNDED,
        )
    )
    return False


def check_for_updates():
    """Themed dependency-health view that matches the rest of the TUI.

    Runs in a loop so Refresh / post-upgrade re-checks stay on the same
    screen style; Back returns to the main menu. All failure modes are
    caught so an offline run never kills the TUI.
    """
    from yolomatic.utils.version_check import check_packages

    while True:
        try:
            clear_screen()
            print_stylized_header("Dependency Health Check")

            try:
                with console.status(
                    "[bold]Querying PyPI for the latest versions...",
                    spinner="dots",
                ):
                    statuses = check_packages()
            except Exception as error:
                console.print(
                    Panel(
                        f"[bold red]Dependency check failed:[/bold red] {error}",
                        border_style="red",
                        padding=(1, 2),
                        box=box.ROUNDED,
                    )
                )
                input("\nPress Enter to return to the main menu...")
                return

            # Shown inside the menu's details pane; printed tables were hidden
            # behind the alternate screen while the user chose an upgrade.
            health_view = Group(_render_dependency_table(statuses), Text(""), _render_dependency_summary(statuses))

            updatable = [s for s in statuses if s.needs_update]
            ultralytics_update = next(
                (s for s in updatable if s.package.name == "ultralytics"),
                None,
            )
            critical_updates = [
                s for s in updatable if s.package.importance == "critical"
            ]

            actions: list[str] = []
            descriptions: dict[str, str] = {}

            if ultralytics_update is not None:
                actions.append("Update Ultralytics Only")
                descriptions["Update Ultralytics Only"] = (
                    f"Upgrade ultralytics from {ultralytics_update.installed} to "
                    f"{ultralytics_update.latest} — the safest, most common choice."
                )
            if len(critical_updates) >= 2 or (
                critical_updates and ultralytics_update is None
            ):
                label = f"Update All Critical ({len(critical_updates)})"
                actions.append(label)
                descriptions[label] = (
                    "Upgrade every critical package with an available update "
                    "(ultralytics, torch, torchvision). Major-version bumps may "
                    "introduce breaking changes — review release notes afterwards."
                )
            if len(updatable) >= 1:
                label = f"Update All Tracked ({len(updatable)})"
                actions.append(label)
                descriptions[label] = (
                    "Upgrade every tracked package with an available update, "
                    "including optional and tooling dependencies."
                )

            actions.append("Refresh")
            descriptions["Refresh"] = (
                "Re-query PyPI and re-read installed versions. Useful after an "
                "upgrade or when PyPI was temporarily unreachable."
            )
            descriptions["Back"] = "Return to the main menu without making changes."

            choice = get_user_choice(
                actions,
                allow_back=True,
                title="Update Actions",
                text=(
                    "Pick an action to continue. "
                    "Recommended action is listed first when available."
                ),
                descriptions=descriptions,
                breadcrumbs=["YOLOmatic", "Dependency Health"],
                body=health_view,
            )

            if choice == "Back":
                return
            if choice == "Refresh":
                continue

            if choice == "Update Ultralytics Only":
                targets = ["ultralytics"]
            elif choice.startswith("Update All Critical"):
                targets = [s.package.name for s in critical_updates]
            elif choice.startswith("Update All Tracked"):
                targets = [s.package.name for s in updatable]
            else:
                return

            confirm = get_user_choice(
                ["Cancel", f"Upgrade {len(targets)} package(s)"],
                title="Confirm Upgrade",
                text="These packages will be upgraded in the active environment:",
                body=Text("\n".join(f"  • {name}" for name in targets)),
                breadcrumbs=["YOLOmatic", "Dependency Health", "Confirm"],
                back_option="Cancel",
            )
            if confirm != f"Upgrade {len(targets)} package(s)":
                continue

            clear_screen()
            print_stylized_header("Applying Updates")
            _run_pip_upgrade(targets)
            console.print()
            input("Press Enter to re-check dependencies...")
            # Loop back and re-render the fresh state.

        except KeyboardInterrupt:
            console.print(
                "\n[bold yellow]Dependency check cancelled by user.[/bold yellow]"
            )
            return
