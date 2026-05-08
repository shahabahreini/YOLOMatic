from __future__ import annotations

from src.utils.tui import (
    TUI_CONSOLE,
    ParameterDefinition,
    NAV_BACK,
    NAV_LIST,
    TUIState,
    build_summary_table,
    clear_screen,
    expected_error_panel,
    format_label,
    format_path,
    get_parameter_value_input,
    get_user_choice,
    get_user_multi_select,
    make_panel,
    make_table,
    print_header,
    render_summary_panel,
    render_table,
    warning_panel,
    shorten_middle,
)

console = TUI_CONSOLE
print_stylized_header = print_header

__all__ = [
    "clear_screen",
    "console",
    "get_user_choice",
    "get_user_multi_select",
    "get_parameter_value_input",
    "make_panel",
    "make_table",
    "expected_error_panel",
    "warning_panel",
    "build_summary_table",
    "format_label",
    "format_path",
    "TUIState",
    "ParameterDefinition",
    "print_stylized_header",
    "render_summary_panel",
    "render_table",
    "NAV_BACK",
    "NAV_LIST",
    "shorten_middle",
]
