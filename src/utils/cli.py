from __future__ import annotations

from src.utils.tui import (
    TUI_CONSOLE,
    clear_screen,
    get_user_choice,
    print_header,
    render_summary_panel,
    render_table,
)

console = TUI_CONSOLE
print_stylized_header = print_header

__all__ = [
    "clear_screen",
    "console",
    "get_user_choice",
    "print_stylized_header",
    "render_summary_panel",
    "render_table",
]
