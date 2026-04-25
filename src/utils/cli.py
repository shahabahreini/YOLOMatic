from __future__ import annotations

from src.utils.tui import (
    TUI_CONSOLE,
    ParameterDefinition,
    NAV_BACK,
    NAV_LIST,
    clear_screen,
    get_parameter_value_input,
    get_user_choice,
    get_user_multi_select,
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
    "get_user_multi_select",
    "get_parameter_value_input",
    "ParameterDefinition",
    "print_stylized_header",
    "render_summary_panel",
    "render_table",
    "NAV_BACK",
    "NAV_LIST",
]
