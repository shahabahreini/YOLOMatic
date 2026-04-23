from __future__ import annotations

import time
from typing import Any, Sequence

from blessed import Terminal
from rich.console import Console, Group
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
    TUI_CONSOLE.print(Panel(header, border_style="cyan", padding=(0, 2)))


class MenuRenderer:
    """Handles the rendering of the interactive menu using Rich Layout."""

    def __init__(
        self,
        options: Sequence[str],
        current_selection: int,
        title: str,
        instruction: str,
        descriptions: dict[str, str] | None = None,
        model_data: list[dict[str, Any]] | None = None,
    ):
        self.options = options
        self.current_selection = current_selection
        self.title = title
        self.instruction = instruction
        self.descriptions = descriptions or {}
        self.model_data = model_data

    def _render_options(self) -> Group:
        """Render the list of options with the current selection highlighted."""
        items = []
        for i, option in enumerate(self.options):
            if i == self.current_selection:
                shortcut = ""
                if option == "Back":
                    shortcut = " (or 'b')"
                elif option == "Exit":
                    shortcut = " (or 'q')"
                items.append(Text(f" > {option}{shortcut}", style="bold black on white"))
            else:
                items.append(Text(f"   {option}", style="dim"))
        return Group(*items)

    def _render_model_table(self) -> Table | None:
        """Render a comparison table if model data is available."""
        if not self.model_data:
            return None

        table = Table(
            title=f"Comparison Table for {self.model_data[0].get('Model', 'Selected Model')}",
            title_style="bold green",
            expand=True,
            border_style="dim",
        )

        headers = self.model_data[0].keys()
        for header in headers:
            table.add_column(header, justify="center", style="cyan")

        for row in self.model_data:
            table.add_row(*[str(value) for value in row.values()])

        return table

    def __rich__(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=5),
        )

        # Header
        layout["header"].update(
            Panel(
                Text(self.title, style="bold cyan", justify="center"),
                border_style="cyan",
            )
        )

        # Main Content
        main_group = [
            Text(self.instruction, style="bold yellow"),
            Text(""),  # Spacer
            self._render_options(),
        ]

        model_table = self._render_model_table()
        if model_table:
            main_group.insert(0, model_table)
            main_group.insert(1, Text(""))

        layout["main"].update(Panel(Group(*main_group), border_style="dim"))

        # Footer / Info Box
        current_option = self.options[self.current_selection]
        description = self.descriptions.get(current_option, "No additional info available.")
        
        layout["footer"].update(
            Panel(
                Text.from_markup(f"[bold cyan]Info:[/bold cyan] {description}"),
                title="Description",
                title_align="left",
                border_style="blue",
            )
        )

        return layout


def get_user_choice(
    options: Sequence[str],
    allow_back: bool = False,
    title: str = "Select an Option",
    text: str = "Use ↑↓ keys to navigate, Enter to select:",
    model_data: list[dict[str, Any]] | None = None,
    descriptions: dict[str, str] | None = None,
) -> str:
    """
    Highly refined interactive menu using rich.live and blessed.
    """
    selectable_options = list(options)
    if allow_back and "Back" not in selectable_options:
        selectable_options.append("Back")

    current_selection = 0

    with TUI_TERM.cbreak(), TUI_TERM.hidden_cursor():
        renderer = MenuRenderer(
            selectable_options,
            current_selection,
            title,
            text,
            descriptions,
            model_data,
        )
        
        with Live(renderer, console=TUI_CONSOLE, refresh_per_second=10, screen=True) as live:
            while True:
                key = TUI_TERM.inkey(timeout=0.1)

                if key.name == "KEY_UP":
                    current_selection = (current_selection - 1) % len(selectable_options)
                elif key.name == "KEY_DOWN":
                    current_selection = (current_selection + 1) % len(selectable_options)
                elif key.name == "KEY_ENTER":
                    return selectable_options[current_selection]
                elif key.lower() == "b" and "Back" in selectable_options:
                    return "Back"
                elif key.lower() == "q" and "Exit" in selectable_options:
                    return "Exit"

                # Update renderer state
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
    table = Table(title=title, title_style=title_style, border_style=border_style, expand=True)
    for col in columns:
        table.add_column(col, style="cyan")
    for row in rows:
        table.add_row(*row)
    TUI_CONSOLE.print(table)


def render_summary_panel(title: str, fields: dict[str, Any], border_style: str = "green") -> None:
    """Render a summary information panel."""
    table = Table.grid(padding=(0, 1))
    table.add_column(style="cyan bold")
    table.add_column(style="white")
    
    for key, value in fields.items():
        table.add_row(f"{key}:", str(value))
        
    TUI_CONSOLE.print(Panel(table, title=title, border_style=border_style, padding=(1, 2)))
