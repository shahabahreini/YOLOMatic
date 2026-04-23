from __future__ import annotations

import time
from typing import Any, Sequence

from blessed import Terminal
from rich import box
from rich.align import Align
from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

# Initialize shared resources
TUI_CONSOLE = Console()
TUI_TERM = Terminal()


def clear_screen() -> None:
    """Clear the terminal screen."""
    print(TUI_TERM.clear)


def print_header(text: str) -> None:
    """Print a stylized header."""
    header = Text(text, style="bold cyan", justify="center")
    TUI_CONSOLE.print(
        Panel(header, border_style="cyan", padding=(0, 2), box=box.ROUNDED)
    )


class MenuRenderer:
    """Handles the rendering of the interactive menu using a modern split layout."""

    def __init__(
        self,
        options: Sequence[str],
        current_selection: int,
        title: str,
        instruction: str,
        descriptions: dict[str, str] | None = None,
        model_data: list[dict[str, Any]] | None = None,
        breadcrumbs: list[str] | None = None,
    ):
        self.options = options
        self.current_selection = current_selection
        self.title = title
        self.instruction = instruction
        self.descriptions = descriptions or {}
        self.model_data = model_data
        self.breadcrumbs = breadcrumbs or []

    def _is_header(self, option: str) -> bool:
        """Check if an option is a header (non-selectable)."""
        return option.startswith("[") and option.endswith("]")

    def _render_sidebar(self) -> Panel:
        """Render the list of options in a sidebar with grouping support."""
        items = []
        for i, option in enumerate(self.options):
            if self._is_header(option):
                # Header item
                header_text = option[1:-1].upper()
                items.append(Text(f"\n {header_text}", style="bold cyan dim"))
            elif i == self.current_selection:
                # Active selection
                prefix = "➤ "
                text = Text(f"{prefix}{option}", style="bold white on blue")
                items.append(text)
            else:
                items.append(Text(f"  {option}", style="dim"))

        return Panel(
            Group(*items),
            title="[bold cyan]Navigation[/bold cyan]",
            border_style="blue",
            padding=(0, 1),
            expand=True,
        )

    def _render_breadcrumbs(self) -> Text:
        """Render breadcrumbs for the current path."""
        if not self.breadcrumbs:
            return Text("")

        parts = []
        for i, crumb in enumerate(self.breadcrumbs):
            style = "bold cyan" if i == len(self.breadcrumbs) - 1 else "dim white"
            parts.append(Text(crumb, style=style))

        separator = Text(" › ", style="dim")
        breadcrumb_text = separator.join(parts)
        return breadcrumb_text

    def _render_model_table(self) -> Table | None:
        """Render a comparison table if model data is available."""
        if not self.model_data:
            return None

        table = Table(
            expand=True,
            border_style="dim",
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold cyan",
        )

        headers = list(self.model_data[0].keys())
        for header in headers:
            table.add_column(header, justify="center")

        for row in self.model_data:
            current_opt = self.options[self.current_selection]
            is_selected = current_opt == row.get("Model")

            style = "bold yellow" if is_selected else "dim"
            table.add_row(*[str(value) for value in row.values()], style=style)

        return table

    def _render_status_bar(self) -> Panel:
        """Render a small status bar with keyboard hints and app version."""
        from src.__version__ import __version__

        hints = [
            ("[bold yellow]↑↓[/bold yellow]", "Move"),
            ("[bold yellow]Enter[/bold yellow]", "Select"),
            ("[bold yellow]B[/bold yellow]", "Back"),
            ("[bold yellow]Q[/bold yellow]", "Quit"),
        ]

        parts = []
        for key, action in hints:
            parts.append(f"{key} {action}")

        hints_text = Text.from_markup("  •  ".join(parts))
        version_text = Text(f"v{__version__}", style="dim cyan")

        status_table = Table.grid(expand=True)
        status_table.add_column(justify="left", ratio=1)
        status_table.add_column(justify="right")
        status_table.add_row(hints_text, version_text)

        return Panel(status_table, border_style="dim", padding=(0, 1))

    def __rich__(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        # Header
        layout["header"].update(
            Panel(
                Text(self.title, style="bold cyan", justify="center"),
                border_style="cyan",
                box=box.ROUNDED,
            )
        )

        # Body - Split into Sidebar and Content
        layout["body"].split_row(
            Layout(name="sidebar", ratio=1),
            Layout(name="content", ratio=3),
        )

        # Sidebar
        layout["body"]["sidebar"].update(self._render_sidebar())

        # Content Dashboard
        current_option = self.options[self.current_selection]
        description = self.descriptions.get(
            current_option, "No additional info available."
        )

        # Split Right Panel into Header, Main, and Tips
        content_layout = Layout()
        content_layout.split_column(
            Layout(name="ctx", size=4),
            Layout(name="main"),
            Layout(name="tips", size=6),
        )

        # Context Section (Breadcrumbs + Current Selection)
        ctx_group = [
            self._render_breadcrumbs(),
            Text(""),
            Text.from_markup(
                f"Current Choice: [bold yellow]{current_option}[/bold yellow]"
            ),
        ]
        content_layout["ctx"].update(
            Panel(
                Group(*ctx_group),
                border_style="dim",
                title="[dim]Context[/dim]",
                title_align="left",
            )
        )

        # Main Content Section (Instruction + Table/Description)
        main_group = [
            Text(self.instruction, style="bold yellow"),
            Text(""),
        ]

        model_table = self._render_model_table()
        if model_table:
            main_group.append(model_table)
        else:
            main_group.append(Text.from_markup(description))

        content_layout["main"].update(
            Panel(
                Group(*main_group),
                border_style="blue",
                title="[bold cyan]Details[/bold cyan]",
            )
        )

        # Tips / Explanation Section
        tips_group = []
        if "Hints:" in self.instruction or "Hint:" in self.instruction:
            # Try to extract hints if they are in the instruction string
            parts = self.instruction.split("Hints:")
            if len(parts) > 1:
                tips_group.append(Text("Pro-Tips:", style="bold green"))
                tips_group.append(Text(parts[1].strip(), style="dim"))
        elif model_table:
            tips_group.append(Text("Tip:", style="bold green"))
            tips_group.append(
                Text(
                    "Compare mAP and FLOPs to find the best accuracy-to-speed ratio for your hardware.",
                    style="dim",
                )
            )
        else:
            tips_group.append(Text("Guidance:", style="bold green"))
            tips_group.append(
                Text(
                    "Choose the option that best matches your project requirements.",
                    style="dim",
                )
            )

        content_layout["tips"].update(
            Panel(
                Group(*tips_group),
                border_style="dim",
                title="[dim]Explanation[/dim]",
                title_align="left",
            )
        )

        layout["body"]["content"].update(content_layout)

        # Footer
        layout["footer"].update(self._render_status_bar())

        return layout


def get_user_choice(
    options: Sequence[str],
    allow_back: bool = False,
    title: str = "Select an Option",
    text: str = "Navigate the menu to continue:",
    model_data: list[dict[str, Any]] | None = None,
    descriptions: dict[str, str] | None = None,
    breadcrumbs: list[str] | None = None,
) -> str:
    """
    Highly refined interactive menu with grouped options and breadcrumbs.
    """
    selectable_options = list(options)
    if allow_back and "Back" not in selectable_options:
        selectable_options.append("Back")

    # Filter out headers for navigation, but keep them for rendering
    # Actually, we need to know which indices are selectable
    navigable_indices = [
        i
        for i, opt in enumerate(selectable_options)
        if not (opt.startswith("[") and opt.endswith("]"))
    ]

    # Map current_selection (index in navigable_indices) to actual index in selectable_options
    current_nav_idx = 0
    current_selection = navigable_indices[current_nav_idx]

    with TUI_TERM.cbreak(), TUI_TERM.hidden_cursor():
        renderer = MenuRenderer(
            selectable_options,
            current_selection,
            title,
            text,
            descriptions,
            model_data,
            breadcrumbs,
        )

        with Live(
            renderer, console=TUI_CONSOLE, refresh_per_second=10, screen=True
        ) as live:
            while True:
                key = TUI_TERM.inkey(timeout=0.1)

                if key.name == "KEY_UP" or key.lower() == "k":
                    current_nav_idx = (current_nav_idx - 1) % len(navigable_indices)
                elif key.name == "KEY_DOWN" or key.lower() == "j":
                    current_nav_idx = (current_nav_idx + 1) % len(navigable_indices)
                elif key.name == "KEY_ENTER":
                    return selectable_options[navigable_indices[current_nav_idx]]
                elif key.lower() == "b" and "Back" in selectable_options:
                    return "Back"
                elif key.lower() == "q" or key.name == "KEY_ESCAPE":
                    if "Exit" in selectable_options:
                        return "Exit"
                    return (
                        "Back"
                        if "Back" in selectable_options
                        else selectable_options[navigable_indices[0]]
                    )

                # Update renderer state
                current_selection = navigable_indices[current_nav_idx]
                renderer.current_selection = current_selection
                live.update(renderer)


def render_table(
    title: str,
    columns: list[str],
    rows: list[list[str]],
    title_style: str = "bold green",
    border_style: str = "dim",
) -> None:
    """Render a standard table."""
    table = Table(
        title=title,
        title_style=title_style,
        border_style=border_style,
        expand=True,
        box=box.ROUNDED,
    )
    for col in columns:
        table.add_column(col, style="cyan")
    for row in rows:
        table.add_row(*row)
    TUI_CONSOLE.print(table)


def render_summary_panel(
    title: str, fields: dict[str, Any], border_style: str = "green"
) -> None:
    """Render a summary information panel."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan bold")
    table.add_column(style="white")

    for key, value in fields.items():
        table.add_row(f"{key}:", str(value))

    TUI_CONSOLE.print(
        Panel(
            table,
            title=f"[bold]{title}[/bold]",
            border_style=border_style,
            padding=(1, 2),
            box=box.ROUNDED,
        )
    )
