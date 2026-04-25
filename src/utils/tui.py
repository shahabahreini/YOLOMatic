from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from blessed import Terminal
from rich import box
from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Initialize shared resources
TUI_CONSOLE = Console()
TUI_TERM = Terminal()

# Navigation signals
NAV_BACK = "__BACK__"
NAV_LIST = "__LIST__"


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
        tip: str | None = None,
        status_fields: dict[str, str] | None = None,
    ):
        self.options = options
        self.current_selection = current_selection
        self.title = title
        self.instruction = instruction
        self.descriptions = descriptions or {}
        self.model_data = model_data
        self.breadcrumbs = breadcrumbs or []
        self.tip = tip
        self.status_fields = status_fields or {}

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

    def _family_key_for_option(self, option: str) -> str:
        """Map a menu option string to its model_data_dict family key."""
        o = option.lower()
        mapping = {
            "yolo26-seg": "yolo26-seg",
            "yolo26": "yolo26",
            "yolov12-seg": "yolov12-seg",
            "yolov12": "yolov12",
            "yolov11-seg": "yolov11-seg",
            "yolov11": "yolov11",
            "yolov10": "yolov10",
            "yolov9-seg": "yolov9-seg",
            "yolov9": "yolov9",
            "yolov8-seg": "yolov8-seg",
            "yolov8": "yolov8",
            "yolox": "yolox",
            "yolo_nas": "yolo_nas",
        }
        return mapping.get(o, o)

    def _render_family_charts(self, family_key: str) -> RenderableType | None:
        """Build stacked horizontal bar charts (mAP and Params) for a model family."""
        from src.models.data import model_data_dict

        rows = model_data_dict.get(family_key)
        if not rows:
            return None

        BAR_WIDTH = 22
        BAR_FILL = "█"
        BAR_EMPTY = "░"

        def _parse(v: Any) -> float | None:
            if v is None or str(v).strip() in ("-", ""):
                return None
            try:
                return float(str(v).split()[0].replace("%", ""))
            except (ValueError, AttributeError):
                return None

        def _short_name(full: str) -> str:
            """Extract the size suffix letter(s) from a model name for compact display."""
            suffixes = ["-seg", "-cls", "-pose", "-obb"]
            name = full
            task_tag = ""
            for sfx in suffixes:
                if name.lower().endswith(sfx):
                    name = name[: -len(sfx)]
                    task_tag = sfx
                    break
            # last character(s) are the size identifier
            label = name[-1:].upper() if name else full
            return label + task_tag.replace("-seg", "")

        def _bar(value: float, max_val: float, width: int, fill_style: str) -> Text:
            filled = round((value / max_val) * width) if max_val > 0 else 0
            bar_text = Text()
            bar_text.append(BAR_FILL * filled, style=fill_style)
            bar_text.append(BAR_EMPTY * (width - filled), style="dim")
            return bar_text

        def _build_chart(
            title: str,
            key_candidates: list[str],
            fill_style: str,
            unit: str,
        ) -> Table | None:
            data: list[tuple[str, float]] = []
            for row in rows:
                for key in key_candidates:
                    val = _parse(row.get(key))
                    if val is not None:
                        data.append((_short_name(str(row.get("Model", ""))), val))
                        break
            if not data:
                return None

            max_val = max(v for _, v in data)
            chart = Table.grid(padding=(0, 1))
            chart.add_column(justify="right", style="dim", min_width=4)
            chart.add_column(min_width=BAR_WIDTH)
            chart.add_column(justify="left", style="bold")

            chart_title = Text(f" {title} ", style="bold cyan")
            chart.add_row(Text(""), chart_title, Text(""))
            for label, val in data:
                chart.add_row(
                    Text(label, style="dim cyan"),
                    _bar(val, max_val, BAR_WIDTH, fill_style),
                    Text(f"{val:.1f}{unit}", style="dim"),
                )
            return chart

        mAP_keys = ["mAPval 50-95", "mAPval box 50-95"]
        param_keys = ["params (M)"]

        map_chart = _build_chart("mAP 50-95", mAP_keys, "green", "")
        param_chart = _build_chart("Params (M)", param_keys, "cyan", "M")

        parts: list[RenderableType] = []
        if map_chart:
            parts.append(Panel(map_chart, border_style="dim green", padding=(0, 1)))
        if param_chart:
            parts.append(Panel(param_chart, border_style="dim cyan", padding=(0, 1)))

        if not parts:
            return None

        chart_table = Table.grid(expand=True)
        for _ in parts:
            chart_table.add_column(ratio=1)
        chart_table.add_row(*parts)
        return chart_table

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

    def _render_context_panel(self, current_option: str) -> Panel:
        """Compact top-of-right-panel strip: breadcrumbs + current choice + status fields."""
        lines: list[RenderableType] = []

        breadcrumb_line = self._render_breadcrumbs()
        if breadcrumb_line.plain:
            lines.append(breadcrumb_line)

        lines.append(
            Text.from_markup(
                f"Current Choice: [bold yellow]{current_option}[/bold yellow]"
            )
        )

        if self.status_fields:
            status_table = Table.grid(padding=(0, 2))
            status_table.add_column(style="dim cyan")
            status_table.add_column(style="white")
            for key, value in self.status_fields.items():
                status_table.add_row(f"{key}:", str(value))
            lines.append(status_table)

        return Panel(
            Group(*lines),
            border_style="dim",
            title="[dim]Context[/dim]",
            title_align="left",
            padding=(0, 1),
        )

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

        current_option = self.options[self.current_selection]
        is_header = current_option.startswith("[") and current_option.endswith("]")
        if is_header:
            default_description = (
                f"[bold cyan]{current_option}[/bold cyan]\n\n"
                "This is a category header. Use ↑↓ arrow keys to navigate to "
                "the selectable items below this category."
            )
        else:
            default_description = (
                f"[bold cyan]{current_option}[/bold cyan]\n\n"
                "Press [bold yellow]Enter[/bold yellow] to select this option.\n"
                "Use [bold yellow]↑↓[/bold yellow] to navigate, "
                "[bold yellow]B[/bold yellow] to go back."
            )
        description = self.descriptions.get(current_option, default_description)

        # Compute context panel size: 2 rows of content + 2 rows of chrome
        # (more if status_fields are provided).
        breadcrumb_line = self._render_breadcrumbs()
        ctx_size = 2 + (1 if breadcrumb_line.plain else 0)
        ctx_size += len(self.status_fields)
        ctx_size += 2  # panel borders + padding
        ctx_size = max(4, ctx_size)

        # Right column layout: Context (compact) + Main (flex) + Tip (optional)
        show_tip = self.tip is not None
        content_layout = Layout()
        splits = [
            Layout(name="ctx", size=ctx_size),
            Layout(name="main"),
        ]
        if show_tip:
            # Reserve 3 rows plus however many wrapped lines the tip needs;
            # cap at a third of the screen so the main panel keeps dominance.
            tip_lines = max(1, self.tip.count("\n") + 1)
            splits.append(Layout(name="tip", size=min(tip_lines + 3, 8)))
        content_layout.split_column(*splits)

        # Context
        content_layout["ctx"].update(self._render_context_panel(current_option))

        # Main — instruction rendered with markup + per-option description
        model_table = self._render_model_table()
        family_charts = self._render_family_charts(
            self._family_key_for_option(current_option)
        )

        main_group: list[RenderableType] = []
        if self.instruction:
            main_group.append(Text.from_markup(self.instruction, style="yellow"))
            main_group.append(Text(""))

        if model_table:
            main_group.append(model_table)
        elif family_charts:
            split_table = Table.grid(expand=True)
            split_table.add_column(ratio=5)
            split_table.add_column(ratio=4)
            split_table.add_row(
                Text.from_markup(description),
                family_charts,
            )
            main_group.append(split_table)
        else:
            main_group.append(Text.from_markup(description))

        content_layout["main"].update(
            Panel(
                Group(*main_group),
                border_style="blue",
                title="[bold cyan]Details[/bold cyan]",
                padding=(1, 2),
            )
        )

        # Optional tip panel — only render when the caller passed one, so
        # screens without a specific tip aren't padded with filler text.
        if show_tip:
            content_layout["tip"].update(
                Panel(
                    Text.from_markup(self.tip),
                    border_style="dim green",
                    title="[bold green]Tip[/bold green]",
                    title_align="left",
                    padding=(0, 1),
                )
            )

        layout["body"]["content"].update(content_layout)
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
    tip: str | None = None,
    status_fields: dict[str, str] | None = None,
) -> str:
    """
    Highly refined interactive menu with grouped options and breadcrumbs.

    Optional:
        tip           – a screen-specific hint shown in a small bottom-right panel.
                        When omitted, the panel is hidden (no generic filler).
        status_fields – key/value pairs rendered in the Context strip at the
                        top of the right pane (e.g. detected dataset format).
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
            tip=tip,
            status_fields=status_fields,
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


@dataclass
class ParameterDefinition:
    """Definition of a configurable parameter with metadata."""

    name: str
    category: str
    default: Any
    value_type: str
    description: str
    help_text: str
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[str] | None = None


class MultiSelectRenderer:
    """Handles rendering of a unified selection and editing menu for parameters."""

    def __init__(
        self,
        parameters: list[ParameterDefinition],
        selected: set[str],
        values: dict[str, Any],
        current_index: int,
        title: str,
        instruction: str,
        focus: str = "list",  # "list" or "input"
        input_buffer: str = "",
    ):
        self.parameters = parameters
        self.selected = selected
        self.values = values
        self.current_index = current_index
        self.title = title
        self.instruction = instruction
        self.focus = focus
        self.input_buffer = input_buffer
        self.filtered_params = parameters

    def _render_checkbox(self, param: ParameterDefinition, is_active: bool) -> Text:
        checked = "[x]" if param.name in self.selected else "[ ]"
        
        # Show the current value if it's selected or has a custom value
        current_val = self.values.get(param.name, param.default)
        val_str = f": [yellow]{current_val}[/yellow]" if param.name in self.selected else ""
        
        text = Text()
        text.append(f"{checked} ", style="bold cyan" if param.name in self.selected else "dim")
        text.append(param.name, style="bold" if is_active and self.focus == "list" else "")
        text.append(Text.from_markup(val_str))

        if is_active and self.focus == "list":
            text.stylize("on blue")
            return Text("➤ ", style="bold yellow") + text
        
        return Text("  ") + text

    def _render_sidebar(self) -> Panel:
        VISIBLE_ITEMS = 25
        total_items = len(self.filtered_params)
        
        # Windowing logic
        half_window = VISIBLE_ITEMS // 2
        start_idx = max(0, min(self.current_index - half_window, total_items - VISIBLE_ITEMS))
        end_idx = min(start_idx + VISIBLE_ITEMS, total_items)

        items: list[Text] = []
        if start_idx > 0:
            items.append(Text(f"↑ {start_idx} more above", style="dim cyan"))

        current_category: str | None = None
        for i in range(start_idx, end_idx):
            param = self.filtered_params[i]
            if param.category != current_category:
                current_category = param.category
                if items and not (start_idx > 0 and len(items) == 1):
                    items.append(Text(""))
                items.append(Text(f"[{current_category.upper()}]", style="bold cyan dim"))

            items.append(self._render_checkbox(param, i == self.current_index))

        if end_idx < total_items:
            items.append(Text(f"↓ {total_items - end_idx} more below", style="dim cyan"))

        border_style = "blue" if self.focus == "list" else "dim"
        title_prefix = "➤ " if self.focus == "list" else ""
        
        return Panel(
            Group(*items),
            title=f"[bold cyan]{title_prefix}Parameters ({self.current_index + 1}/{total_items})[/bold cyan]",
            border_style=border_style,
            padding=(0, 1),
            expand=True,
        )

    def _render_content(self) -> Panel:
        if not self.filtered_params or self.current_index >= len(self.filtered_params):
            return Panel(Text("No parameters available"), border_style="dim")

        param = self.filtered_params[self.current_index]
        current_val = self.values.get(param.name, param.default)

        info_lines = [
            Text.from_markup(f"[bold cyan]Parameter:[/bold cyan] {param.name}"),
            Text.from_markup(f"[bold cyan]Category:[/bold cyan] {param.category}"),
            Text.from_markup(f"[bold cyan]Type:[/bold cyan] {param.value_type}"),
            Text.from_markup(f"[bold cyan]Default:[/bold cyan] {param.default}"),
            Text(""),
            Text.from_markup("[bold yellow]Description:[/bold yellow]"),
            Text(param.description),
            Text(""),
            Text.from_markup("[bold green]Help:[/bold green]"),
            Text(param.help_text, style="dim"),
        ]

        if param.min_value is not None or param.max_value is not None:
            info_lines.append(Text(""))
            range_str = f"Range: {param.min_value if param.min_value is not None else '-∞'} to {param.max_value if param.max_value is not None else '+∞'}"
            info_lines.append(Text.from_markup(f"[bold magenta]{range_str}[/bold magenta]"))

        if param.allowed_values:
            info_lines.append(Text(""))
            info_lines.append(Text.from_markup(f"[bold magenta]Allowed values:[/bold magenta] {', '.join(param.allowed_values)}"))

        # Input Area
        info_lines.append(Text(""))
        border_style = "blue" if self.focus == "input" else "dim"
        title_prefix = "➤ " if self.focus == "input" else ""
        
        input_content = []
        if param.value_type == "bool":
            # Simple toggle display
            input_content.append(Text("Value: ", style="bold"))
            input_content.append(Text(str(current_val), style="bold yellow"))
            input_content.append(Text(" (Press Space/Enter to toggle)", style="dim"))
        elif param.allowed_values:
            # Cycle display
            input_content.append(Text("Value: ", style="bold"))
            input_content.append(Text(str(current_val), style="bold yellow"))
            input_content.append(Text(" (Press Up/Down to cycle)", style="dim"))
        else:
            # Text input display
            display_val = self.input_buffer if self.focus == "input" else str(current_val)
            input_content.append(Text("Value: ", style="bold"))
            input_content.append(Text(display_val, style="bold yellow"))
            if self.focus == "input":
                input_content.append(Text("█", style="blink bold white"))
            else:
                input_content.append(Text(" (Press Enter to edit)", style="dim"))

        info_lines.append(
            Panel(
                Text.assemble(*input_content),
                title=f"{title_prefix}Edit Value",
                border_style=border_style,
                padding=(0, 1)
            )
        )

        return Panel(
            Group(*info_lines),
            title="[bold cyan]Details & Configuration[/bold cyan]",
            border_style="blue",
            padding=(1, 2),
            expand=True,
        )

    def _render_status_bar(self) -> Panel:
        if self.focus == "list":
            hints = [
                ("[bold yellow]↑↓[/bold yellow]", "Move"),
                ("[bold yellow]Space[/bold yellow]", "Toggle"),
                ("[bold yellow]Enter/→[/bold yellow]", "Edit Value"),
                ("[bold yellow]A/N[/bold yellow]", "All/None"),
                ("[bold yellow]F[/bold yellow]", "Finish"),
                ("[bold yellow]Q[/bold yellow]", "Back"),
            ]
        else:
            hints = [
                ("[bold yellow]Enter[/bold yellow]", "Save"),
                ("[bold yellow]Esc/←/B[/bold yellow]", "Back to List"),
                ("[bold yellow]Up/Down[/bold yellow]", "Cycle Options"),
            ]

        parts = [f"{key} {action}" for key, action in hints]
        hints_text = Text.from_markup("  •  ".join(parts))

        selected_count = len(self.selected)
        count_text = Text(f"Selected: {selected_count}", style="bold cyan")

        status_table = Table.grid(expand=True)
        status_table.add_column(justify="left", ratio=1)
        status_table.add_column(justify="right")
        status_table.add_row(hints_text, count_text)

        return Panel(status_table, border_style="dim", padding=(0, 1))

    def __rich__(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        layout["header"].update(
            Panel(
                Text(self.title, style="bold cyan", justify="center"),
                border_style="cyan",
                box=box.ROUNDED,
            )
        )

        layout["body"].split_row(
            Layout(name="sidebar", ratio=1),
            Layout(name="content", ratio=2),
        )

        layout["body"]["sidebar"].update(self._render_sidebar())
        layout["body"]["content"].update(self._render_content())
        layout["footer"].update(self._render_status_bar())

        return layout


def get_user_multi_select(
    parameters: list[ParameterDefinition],
    title: str = "Select Parameters",
    instruction: str = "Use Space to toggle parameters, Enter to edit values:",
    pre_selected: set[str] | None = None,
    pre_values: dict[str, Any] | None = None,
) -> tuple[set[str], dict[str, Any]] | None:
    """
    Interactive unified interface for parameter selection and value editing.

    Returns (selected_names, values_dict) or None if cancelled.
    """
    selected = pre_selected.copy() if pre_selected else set()
    values = pre_values.copy() if pre_values else {}
    current_index = 0
    focus = "list"
    input_buffer = ""

    with TUI_TERM.cbreak(), TUI_TERM.hidden_cursor():
        renderer = MultiSelectRenderer(
            parameters=parameters,
            selected=selected,
            values=values,
            current_index=current_index,
            title=title,
            instruction=instruction,
            focus=focus,
            input_buffer=input_buffer,
        )

        with Live(renderer, console=TUI_CONSOLE, refresh_per_second=10, screen=True) as live:
            while True:
                key = TUI_TERM.inkey(timeout=0.1)
                param = parameters[current_index]

                if focus == "list":
                    if key.name == "KEY_UP" or key.lower() == "k":
                        current_index = (current_index - 1) % len(parameters)
                    elif key.name == "KEY_DOWN" or key.lower() == "j":
                        current_index = (current_index + 1) % len(parameters)
                    elif key == " ":
                        if param.name in selected:
                            selected.remove(param.name)
                        else:
                            selected.add(param.name)
                    elif key.name == "KEY_RIGHT" or key.name == "KEY_ENTER":
                        focus = "input"
                        input_buffer = str(values.get(param.name, param.default))
                        # For bool, just toggle immediately if desired, or let the input mode handle it
                    elif key.lower() == "a":
                        selected = {p.name for p in parameters}
                    elif key.lower() == "n":
                        selected = set()
                    elif key.lower() == "f":  # Finish
                        return selected, values
                    elif key.lower() == "q" or key.name == "KEY_ESCAPE":
                        return None
                
                elif focus == "input":
                    if key.name == "KEY_LEFT" or key.name == "KEY_ESCAPE" or key.lower() == "b":
                        focus = "list"
                    elif key.name == "KEY_ENTER":
                        # Validate and save
                        try:
                            if param.value_type == "int":
                                val = int(input_buffer)
                            elif param.value_type == "float":
                                val = float(input_buffer)
                            elif param.value_type == "bool":
                                # Toggle logic for bool handled by space/enter too
                                val = input_buffer.lower() == "true"
                            else:
                                val = input_buffer
                            
                            # Simple validation check
                            valid = True
                            if param.min_value is not None and isinstance(val, (int, float)) and val < param.min_value:
                                valid = False
                            if param.max_value is not None and isinstance(val, (int, float)) and val > param.max_value:
                                valid = False
                            if param.allowed_values and val not in param.allowed_values:
                                valid = False
                            
                            if valid:
                                values[param.name] = val
                                selected.add(param.name)  # Auto-select if value is modified
                                focus = "list"
                        except ValueError:
                            pass # Keep editing
                    
                    elif param.value_type == "bool":
                        if key == " " or key.name == "KEY_ENTER":
                            current_bool = input_buffer.lower() == "true"
                            input_buffer = str(not current_bool)
                    
                    elif param.allowed_values:
                        if key.name == "KEY_UP" or key.name == "KEY_DOWN":
                            try:
                                curr_idx = param.allowed_values.index(input_buffer)
                            except ValueError:
                                curr_idx = 0
                            
                            if key.name == "KEY_UP":
                                next_idx = (curr_idx - 1) % len(param.allowed_values)
                            else:
                                next_idx = (curr_idx + 1) % len(param.allowed_values)
                            input_buffer = param.allowed_values[next_idx]
                    
                    elif key.name == "KEY_BACKSPACE":
                        input_buffer = input_buffer[:-1]
                    elif key and not key.is_sequence:
                        input_buffer += key

                # Sync renderer
                renderer.current_index = current_index
                renderer.selected = selected
                renderer.values = values
                renderer.focus = focus
                renderer.input_buffer = input_buffer
                live.update(renderer)


def get_parameter_value_input(
    param: ParameterDefinition,
    current_value: Any | None = None,
) -> Any | None:
    """
    Interactive input for a parameter value with validation.

    Returns:
        - The entered/converted value
        - NAV_BACK if the user wants to go back to the previous parameter
        - NAV_LIST if the user wants to return to the selection list
    """
    value_to_edit = current_value if current_value is not None else param.default

    # 1. Specialized handling for Boolean values (Fast Path)
    if param.value_type == "bool":
        choice = get_user_choice(
            ["True", "False", "Back to Previous", "Back to List"],
            title=f"Set Value: {param.name}",
            text=(
                f"[bold cyan]Parameter:[/bold cyan] {param.name}\n"
                f"[bold cyan]Description:[/bold cyan] {param.description}\n\n"
                f"[dim]{param.help_text}[/dim]\n\n"
                f"Current value: [bold yellow]{value_to_edit}[/bold yellow]"
            ),
            tip="Boolean parameters are toggled instantly for faster configuration.",
        )
        if choice == "True":
            return True
        if choice == "False":
            return False
        if choice == "Back to Previous":
            return NAV_BACK
        if choice == "Back to List":
            return NAV_LIST
        return value_to_edit

    # 2. Specialized handling for Allowed Values (Fast Path)
    if param.allowed_values:
        options = list(param.allowed_values)
        options.extend(["Back to Previous", "Back to List"])
        choice = get_user_choice(
            options,
            title=f"Set Value: {param.name}",
            text=(
                f"[bold cyan]Parameter:[/bold cyan] {param.name}\n"
                f"[bold cyan]Description:[/bold cyan] {param.description}\n\n"
                f"[dim]{param.help_text}[/dim]\n\n"
                f"Current value: [bold yellow]{value_to_edit}[/bold yellow]"
            ),
            tip=f"Pick from one of the {len(param.allowed_values)} valid options.",
        )
        if choice == "Back to Previous":
            return NAV_BACK
        if choice == "Back to List":
            return NAV_LIST
        return choice

    # 3. Interactive Text Input for Numeric/String values
    clear_screen()
    print_header(f"Set Value: {param.name}")

    console = TUI_CONSOLE
    console.print(
        Panel(
            Group(
                Text.from_markup(f"[bold cyan]Parameter:[/bold cyan] {param.name}"),
                Text.from_markup(f"[bold cyan]Type:[/bold cyan] {param.value_type}"),
                Text.from_markup(f"[bold cyan]Default:[/bold cyan] {param.default}"),
                Text(""),
                Text(param.description),
                Text(""),
                Text.from_markup(f"[dim]{param.help_text}[/dim]"),
            ),
            border_style="blue",
            padding=(1, 2),
        )
    )

    if param.min_value is not None or param.max_value is not None:
        range_str = f"Range: {param.min_value if param.min_value is not None else '-∞'} to {param.max_value if param.max_value is not None else '+∞'}"
        console.print(f"[bold magenta]{range_str}[/bold magenta]")

    console.print(f"\nCurrent value: [bold yellow]{value_to_edit}[/bold yellow]")
    console.print(
        "\n[dim]Commands: [bold yellow]Enter[/bold yellow] (keep current)  •  "
        "[bold yellow]B[/bold yellow] (back)  •  "
        "[bold yellow]L[/bold yellow] (list)  •  "
        "[bold yellow]Esc[/bold yellow] (cancel/back)[/dim]"
    )

    while True:
        try:
            user_input = input("\nNew value: ").strip()
        except (EOFError, KeyboardInterrupt):
            return NAV_BACK

        if not user_input:
            return value_to_edit

        if user_input.lower() == "b":
            return NAV_BACK
        if user_input.lower() == "l":
            return NAV_LIST

        try:
            if param.value_type == "int":
                value = int(user_input)
                if param.min_value is not None and value < param.min_value:
                    console.print(f"[red]Value must be >= {param.min_value}[/red]")
                    continue
                if param.max_value is not None and value > param.max_value:
                    console.print(f"[red]Value must be <= {param.max_value}[/red]")
                    continue
                return value
            elif param.value_type == "float":
                value = float(user_input)
                if param.min_value is not None and value < param.min_value:
                    console.print(f"[red]Value must be >= {param.min_value}[/red]")
                    continue
                if param.max_value is not None and value > param.max_value:
                    console.print(f"[red]Value must be <= {param.max_value}[/red]")
                    continue
                return value
            elif param.value_type == "str":
                return user_input
            else:
                return user_input
        except ValueError as e:
            console.print(f"[red]Invalid value: {e}[/red]")
            continue
