from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from blessed import Terminal
from rich import box
from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Initialize shared resources
TUI_CONSOLE = Console()
TUI_TERM = Terminal()

# Navigation signals
NAV_BACK = "__BACK__"
NAV_LIST = "__LIST__"


class TUIState(str, Enum):
    """Shared visual states for TUI panels and status messages."""

    NORMAL = "normal"
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CONFIRMATION = "confirmation"


PANEL_BORDER_STYLES: dict[TUIState, str] = {
    TUIState.NORMAL: "blue",
    TUIState.INFO: "cyan",
    TUIState.SUCCESS: "green",
    TUIState.WARNING: "yellow",
    TUIState.ERROR: "red",
    TUIState.CONFIRMATION: "yellow",
}

STATUS_HINTS: dict[str, list[tuple[str, str]]] = {
    "menu": [("↑↓", "Move"), ("Enter", "Select"), ("B", "Back"), ("Q", "Quit")],
    "menu_finish": [("↑↓", "Move"), ("Enter", "Select"), ("F", "Continue"), ("B", "Back"), ("Q", "Quit")],
    "confirmation": [("↑↓", "Move"), ("Enter", "Confirm"), ("B", "Back"), ("Q", "Cancel")],
    "parameter_list": [("↑↓", "Move"), ("Space", "Toggle"), ("Enter/→", "Edit"), ("F", "Finish"), ("Q", "Back")],
    "parameter_input": [("Type", "Edit"), ("Enter", "Save"), ("Esc/←/B", "Back")],
    "progress": [("Ctrl+C", "Cancel")],
    "back": [("Enter", "Continue"), ("B", "Back")],
}


def is_enter_key(key: Any) -> bool:
    """Return True for Enter across blessed terminal variants."""
    return key.name == "KEY_ENTER" or str(key) in {"\n", "\r", "\r\n"}


FORWARD_OPTION_PREFIXES = (
    "confirm",
    "continue",
    "done",
    "finish",
    "save",
    "start",
)


def is_forward_option(option: str) -> bool:
    """Return True for explicit forward/finish actions."""
    normalized = option.strip().lower()
    return normalized.startswith(FORWARD_OPTION_PREFIXES)


def resolve_finish_option(
    options: Sequence[str],
    finish_options: set[str] | None = None,
) -> str | None:
    """Find the menu option that should be activated by the F shortcut."""
    if finish_options:
        for option in options:
            if option in finish_options:
                return option
        return None
    matches = [option for option in options if is_forward_option(option)]
    return matches[0] if len(matches) == 1 else None


def shorten_middle(value: str, max_chars: int = 44) -> str:
    """Compact long menu labels while preserving the beginning and file suffix."""
    if len(value) <= max_chars:
        return value
    if max_chars <= 8:
        return value[:max_chars]

    marker = "..."
    remaining = max_chars - len(marker)
    front = max(1, remaining // 2)
    back = max(1, remaining - front)
    return f"{value[:front]}{marker}{value[-back:]}"


def format_label(value: Any, max_chars: int = 64) -> str:
    """Render a compact user-facing label while preserving useful suffixes."""
    return shorten_middle(str(value), max_chars=max_chars)


def format_path(value: Any, max_chars: int = 80) -> str:
    """Render a compact path with the filename or suffix kept visible."""
    text = str(value)
    if len(text) <= max_chars:
        return text
    normalized = text.replace("\\", "/")
    if "/" in normalized:
        tail = normalized.rsplit("/", 1)[-1]
        if len(tail) + 4 <= max_chars:
            return f".../{tail}"
    return shorten_middle(text, max_chars=max_chars)


def make_panel(
    content: RenderableType | str,
    *,
    title: str | None = None,
    state: TUIState = TUIState.NORMAL,
    padding: tuple[int, int] = (1, 2),
    expand: bool = True,
) -> Panel:
    """Create a standard TUI panel for expected states and summaries."""
    renderable: RenderableType = Text.from_markup(content) if isinstance(content, str) else content
    return Panel(
        renderable,
        title=f"[bold]{title}[/bold]" if title else None,
        title_align="left",
        border_style=PANEL_BORDER_STYLES[state],
        padding=padding,
        box=box.ROUNDED,
        expand=expand,
    )


def make_table(
    *,
    title: str | None = None,
    expand: bool = True,
    box_style: box.Box = box.SIMPLE_HEAD,
) -> Table:
    """Create a compact, consistent Rich table for TUI pages."""
    return Table(
        title=title,
        title_style="bold cyan",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        box=box_style,
        expand=expand,
        pad_edge=False,
    )


def render_hints(hint_set: str = "menu") -> Text:
    """Render keyboard hints in the shared footer style."""
    hints = STATUS_HINTS.get(hint_set, STATUS_HINTS["menu"])
    return Text.from_markup(
        "  •  ".join(
            f"[bold yellow]{key}[/bold yellow] {action}" for key, action in hints
        )
    )


def expected_error_panel(message: str, *, title: str = "Cannot Continue", next_step: str | None = None) -> Panel:
    """Render expected validation or environment failures without traceback noise."""
    lines: list[RenderableType] = [Text.from_markup(message)]
    if next_step:
        lines.extend([Text(""), Text.from_markup(f"[bold yellow]Next:[/bold yellow] {next_step}")])
    return make_panel(Group(*lines), title=title, state=TUIState.ERROR)


def warning_panel(message: str, *, title: str = "Review Required", next_step: str | None = None) -> Panel:
    lines: list[RenderableType] = [Text.from_markup(message)]
    if next_step:
        lines.extend([Text(""), Text.from_markup(f"[bold yellow]Next:[/bold yellow] {next_step}")])
    return make_panel(Group(*lines), title=title, state=TUIState.WARNING)


def build_summary_table(fields: dict[str, Any]) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim cyan", no_wrap=True)
    table.add_column(style="white")
    for key, value in fields.items():
        table.add_row(f"{key}:", format_path(value))
    return table


def clear_screen() -> None:
    """Clear the terminal screen."""
    TUI_CONSOLE.clear()


def print_header(text: str) -> None:
    """Print a stylized header."""
    header = Text(text, style="bold cyan", justify="center")
    TUI_CONSOLE.print(make_panel(header, state=TUIState.INFO, padding=(0, 2)))


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
        finish_option: str | None = None,
        wizard_steps: list[str] | None = None,
        wizard_current_step: int | None = None,
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
        self.finish_option = finish_option
        self.wizard_steps = wizard_steps
        self.wizard_current_step = wizard_current_step

    def _is_header(self, option: str) -> bool:
        """Check if an option is a header (non-selectable)."""
        return option.startswith("[") and option.endswith("]")

    def _visible_option_range(self) -> tuple[int, int]:
        """Return the slice of options that fits in the sidebar."""
        total_options = len(self.options)
        if total_options == 0:
            return 0, 0

        # Leave room for the panel border, title, layout chrome, and overflow
        # indicators so Rich does not clip the selected row outside the box.
        visible_items = max(3, min(total_options, TUI_TERM.height - 15))
        half_window = visible_items // 2
        start = max(
            0,
            min(self.current_selection - half_window, total_options - visible_items),
        )
        end = min(start + visible_items, total_options)
        return start, end

    def _sidebar_label_width(self) -> int:
        """Estimate available label width in the left navigation column."""
        # Using ~40% of width (ratio 2:3)
        return max(12, min(64, (TUI_TERM.width * 2 // 5) - 8))

    def _render_sidebar(self) -> Panel:
        """Render the list of options in a sidebar with grouping support."""
        items = []
        start, end = self._visible_option_range()
        label_width = self._sidebar_label_width()
        if start > 0:
            items.append(Text(f"↑ {start} more above", style="dim cyan"))

        for i, option in enumerate(self.options[start:end], start=start):
            if self._is_header(option):
                # Header item
                header_text = option[1:-1].upper()
                items.append(Text(f"\n {header_text}", style="bold cyan dim"))
            elif i == self.current_selection:
                # Active selection
                prefix = "➤ "
                text = Text(
                    f"{prefix}{shorten_middle(option, max_chars=label_width)}",
                    style="bold white on blue",
                )
                items.append(text)
            else:
                style = "green" if option.lstrip().startswith("✓") else "dim"
                items.append(
                    Text(f"  {shorten_middle(option, max_chars=label_width)}", style=style)
                )

        if end < len(self.options):
            items.append(Text(f"↓ {len(self.options) - end} more below", style="dim cyan"))

        return make_panel(
            Group(*items),
            title=f"[bold cyan]Navigation ({self.current_selection + 1}/{len(self.options)})[/bold cyan]",
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
            "detectron2-seg": "detectron2-seg",
            "detectron2": "detectron2",
            "rfdetr-seg": "rfdetr-seg",
            "rfdetr": "rfdetr",
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

        table = make_table(expand=True)

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

        hints_text = render_hints("menu_finish" if self.finish_option else "menu")
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
                f"Current Choice: [bold yellow]{escape(format_label(current_option))}[/bold yellow]"
            )
        )

        if self.status_fields:
            status_table = Table.grid(padding=(0, 2))
            status_table.add_column(style="dim cyan")
            status_table.add_column(style="white")
            for key, value in self.status_fields.items():
                status_table.add_row(f"{key}:", format_path(value, max_chars=32))
            lines.append(status_table)

        return make_panel(
            Group(*lines),
            title="[dim]Context[/dim]",
            state=TUIState.NORMAL,
            padding=(0, 1),
        )

    def __rich__(self) -> Layout:
        layout = Layout()
        show_stepper = self.wizard_steps and self.wizard_current_step is not None
        header_size = 5 if show_stepper else 3
        layout.split_column(
            Layout(name="header", size=header_size),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        # Header
        if show_stepper:
            header_text = Text()
            header_text.append(self.title + "\n", style="bold cyan")
            header_text.append("Wizard Progress:  ", style="dim white")
            for i, step in enumerate(self.wizard_steps):
                is_active = i == self.wizard_current_step
                is_completed = i < self.wizard_current_step
                
                if is_active:
                    bullet = "●"
                    style = "bold yellow"
                elif is_completed:
                    bullet = "✓"
                    style = "bold green"
                else:
                    bullet = "○"
                    style = "dim white"
                
                header_text.append(f"{bullet} {step}", style=style)
                if i < len(self.wizard_steps) - 1:
                    line_style = "green" if is_completed else "dim"
                    header_text.append(" ── ", style=line_style)
            
            layout["header"].update(
                make_panel(
                    header_text,
                    state=TUIState.INFO,
                    padding=(0, 1),
                )
            )
        else:
            layout["header"].update(
                make_panel(
                    Text(self.title, style="bold cyan", justify="center"),
                    state=TUIState.INFO,
                    padding=(0, 1),
                )
            )

        # Body - Split into Sidebar and Content
        layout["body"].split_row(
            Layout(name="sidebar", ratio=2),
            Layout(name="content", ratio=3),
        )

        # Sidebar
        layout["body"]["sidebar"].update(self._render_sidebar())

        current_option = self.options[self.current_selection]
        is_header = current_option.startswith("[") and current_option.endswith("]")
        if is_header:
            default_description = (
                f"[bold cyan]{escape(current_option)}[/bold cyan]\n\n"
                "This is a category header. Use ↑↓ arrow keys to navigate to "
                "the selectable items below this category."
            )
        else:
            default_description = (
                f"[bold cyan]{escape(current_option)}[/bold cyan]\n\n"
                "Press [bold yellow]Enter[/bold yellow] to select this option.\n"
                "Use [bold yellow]↑↓[/bold yellow] to navigate, "
                "[bold yellow]B[/bold yellow] to go back."
            )
            if self.finish_option:
                default_description += (
                    f"\nPress [bold yellow]F[/bold yellow] to continue with "
                    f"[bold green]{escape(format_label(self.finish_option))}[/bold green]."
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
            split_table.add_column(ratio=1)
            split_table.add_column(ratio=1)
            split_table.add_row(
                Text.from_markup(description),
                family_charts,
            )
            main_group.append(split_table)
        else:
            main_group.append(Text.from_markup(description))

        content_layout["main"].update(
            make_panel(
                Group(*main_group),
                title="[bold cyan]Details[/bold cyan]",
                padding=(1, 2),
            )
        )

        # Optional tip panel — only render when the caller passed one, so
        # screens without a specific tip aren't padded with filler text.
        if show_tip:
            content_layout["tip"].update(
                make_panel(
                    Text.from_markup(self.tip),
                    title="[bold green]Tip[/bold green]",
                    state=TUIState.SUCCESS,
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
    finish_options: set[str] | None = None,
    initial_selection: int | str | None = None,
    wizard_steps: list[str] | None = None,
    wizard_current_step: int | None = None,
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
    finish_option = resolve_finish_option(selectable_options, finish_options)

    # Filter out headers for navigation, but keep them for rendering
    # Actually, we need to know which indices are selectable
    navigable_indices = [
        i
        for i, opt in enumerate(selectable_options)
        if not (opt.startswith("[") and opt.endswith("]"))
    ]

    # Map current_selection (index in navigable_indices) to actual index in selectable_options
    current_nav_idx = 0
    if initial_selection is not None:
        if isinstance(initial_selection, int):
            if 0 <= initial_selection < len(navigable_indices):
                current_nav_idx = initial_selection
        elif isinstance(initial_selection, str):
            # Try exact match first
            actual_idx = -1
            if initial_selection in selectable_options:
                actual_idx = selectable_options.index(initial_selection)
            else:
                # Try matching by stripping common TUI marks
                stripped_initial = initial_selection.lstrip("✓ ").strip()
                for i, opt in enumerate(selectable_options):
                    if opt.lstrip("✓ ").strip() == stripped_initial:
                        actual_idx = i
                        break

            if actual_idx != -1:
                # Find the closest navigable index
                closest_nav = 0
                min_dist = float("inf")
                for i, nav_idx in enumerate(navigable_indices):
                    dist = abs(nav_idx - actual_idx)
                    if dist < min_dist:
                        min_dist = dist
                        closest_nav = i
                current_nav_idx = closest_nav

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
            finish_option=finish_option,
            wizard_steps=wizard_steps,
            wizard_current_step=wizard_current_step,
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
                elif key.name == "KEY_PGUP":
                    visible_items = max(3, min(len(selectable_options), TUI_TERM.height - 15))
                    current_nav_idx = max(0, current_nav_idx - visible_items)
                elif key.name == "KEY_PGDN":
                    visible_items = max(3, min(len(selectable_options), TUI_TERM.height - 15))
                    current_nav_idx = min(len(navigable_indices) - 1, current_nav_idx + visible_items)
                elif key.name == "KEY_HOME":
                    current_nav_idx = 0
                elif key.name == "KEY_END":
                    current_nav_idx = len(navigable_indices) - 1
                elif is_enter_key(key):
                    return selectable_options[navigable_indices[current_nav_idx]]
                elif key.lower() == "f" and finish_option is not None:
                    return finish_option
                elif key.lower() == "b" and "Back" in selectable_options:
                    return "Back"
                elif key.lower() == "q" or key.name == "KEY_ESCAPE":
                    if "Exit" in selectable_options:
                        return "Exit"
                    if "Back" in selectable_options:
                        return "Back"
                    return NAV_BACK

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
    table = make_table(title=title, expand=True, box_style=box.ROUNDED)
    table.title_style = title_style
    table.border_style = border_style
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

    state = TUIState.SUCCESS if border_style == "green" else TUIState.INFO
    TUI_CONSOLE.print(make_panel(table, title=title, state=state, padding=(1, 2)))


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
    affects: str | None = None
    option_descriptions: dict[str, str] | None = None
    config_section: str = "training"


TRUE_VALUES = {"true", "1", "yes", "y", "on"}
FALSE_VALUES = {"false", "0", "no", "n", "off"}


def parse_parameter_value(param: ParameterDefinition, raw_value: Any) -> Any:
    """Convert raw TUI input into the parameter's configured Python type."""
    if raw_value is None:
        return None

    raw_text = str(raw_value).strip()
    normalized_text = raw_text.lower()
    if param.value_type.startswith("optional_") and normalized_text in {
        "",
        "none",
        "null",
    }:
        return None

    if param.value_type == "bool":
        if isinstance(raw_value, bool):
            return raw_value
        if normalized_text in TRUE_VALUES:
            return True
        if normalized_text in FALSE_VALUES:
            return False
        raise ValueError("Use True or False.")

    if param.value_type == "optional_bool":
        if isinstance(raw_value, bool):
            return raw_value
        if normalized_text in TRUE_VALUES:
            return True
        if normalized_text in FALSE_VALUES:
            return False
        raise ValueError("Use True, False, or None.")

    if param.value_type == "bool_or_str":
        if isinstance(raw_value, bool):
            return raw_value
        if normalized_text in TRUE_VALUES:
            return True
        if normalized_text in FALSE_VALUES:
            return False
        return raw_text

    if param.value_type in {"int", "optional_int"}:
        return int(raw_text)

    if param.value_type in {"float", "optional_float"}:
        return float(raw_text)

    if param.value_type in {"str", "optional_str"}:
        return raw_text

    if param.value_type == "int_list":
        if normalized_text in {"", "none", "null"}:
            return None
        return [
            int(part.strip())
            for part in raw_text.split(",")
            if part.strip()
        ]

    return raw_value


def validate_parameter_value(param: ParameterDefinition, value: Any) -> str | None:
    """Return a user-facing validation error, or None when the value is valid."""
    if value is None:
        return None

    if param.min_value is not None and isinstance(value, (int, float)):
        if value < param.min_value:
            return f"Value must be >= {param.min_value}."

    if param.max_value is not None and isinstance(value, (int, float)):
        if value > param.max_value:
            return f"Value must be <= {param.max_value}."

    if param.allowed_values is not None and (
        value not in param.allowed_values and str(value) not in param.allowed_values
    ):
        allowed = ", ".join(repr(option) for option in param.allowed_values)
        return f"Choose one of: {allowed}."

    return None


def convert_and_validate_parameter_value(
    param: ParameterDefinition,
    raw_value: Any,
) -> tuple[bool, Any, str | None]:
    """Parse and validate a parameter value for interactive editors and tests."""
    try:
        value = parse_parameter_value(param, raw_value)
    except (TypeError, ValueError) as error:
        return False, None, f"Invalid {param.value_type}: {error}"

    validation_error = validate_parameter_value(param, value)
    if validation_error:
        return False, value, validation_error

    return True, value, None


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
        validation_error: str | None = None,
        wizard_steps: list[str] | None = None,
        wizard_current_step: int | None = None,
    ):
        self.parameters = parameters
        self.selected = selected
        self.values = values
        self.current_index = current_index
        self.title = title
        self.instruction = instruction
        self.focus = focus
        self.input_buffer = input_buffer
        self.validation_error = validation_error
        self.filtered_params = parameters
        self.wizard_steps = wizard_steps
        self.wizard_current_step = wizard_current_step

    def _render_checkbox(self, param: ParameterDefinition, is_active: bool) -> Text:
        checked = "[x]" if param.name in self.selected else "[ ]"
        current_val = self.values.get(param.name, param.default)

        text = Text()
        text.append(f"{checked} ", style="bold cyan" if param.name in self.selected else "dim")
        text.append(format_label(param.name, max_chars=34), style="bold" if is_active and self.focus == "list" else "")
        if param.name in self.selected:
            text.append(": ", style="dim")
            text.append(str(current_val), style="yellow")

        if is_active and self.focus == "list":
            text.stylize("on blue")
            return Text("➤ ", style="bold yellow") + text
        
        return Text("  ") + text

    def _render_sidebar(self) -> Panel:
        total_items = len(self.filtered_params)

        # Windowing logic
        visible_items = max(4, min(total_items, TUI_TERM.height - 12))
        half_window = visible_items // 2
        start_idx = max(0, min(self.current_index - half_window, total_items - visible_items))
        end_idx = min(start_idx + visible_items, total_items)

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
            Text.from_markup(f"[bold cyan]Saved To:[/bold cyan] config.{param.config_section}"),
            Text.from_markup(f"[bold cyan]Type:[/bold cyan] {param.value_type}"),
            Text.from_markup(f"[bold cyan]Default:[/bold cyan] {param.default}"),
            Text.from_markup(f"[bold cyan]Current:[/bold cyan] {current_val}"),
        ]

        if self.validation_error:
            info_lines.extend(
                [
                    Text.from_markup(f"[bold red]Validation:[/bold red] {self.validation_error}"),
                    Text(""),
                ]
            )
        else:
            info_lines.append(Text(""))

        info_lines.extend(
            [
                Text.from_markup("[bold yellow]What It Controls:[/bold yellow]"),
                Text(param.description),
            ]
        )

        if param.affects:
            info_lines.extend(
                [
                    Text(""),
                    Text.from_markup("[bold yellow]Training Impact:[/bold yellow]"),
                    Text.from_markup(param.affects),
                ]
            )

        if param.min_value is not None or param.max_value is not None:
            info_lines.append(Text(""))
            range_str = f"Range: {param.min_value if param.min_value is not None else '-∞'} to {param.max_value if param.max_value is not None else '+∞'}"
            info_lines.append(Text.from_markup(f"[bold magenta]{range_str}[/bold magenta]"))

        if param.allowed_values:
            info_lines.append(Text(""))
            info_lines.append(Text.from_markup(f"[bold magenta]Allowed values:[/bold magenta] {', '.join(repr(value) for value in param.allowed_values)}"))

        info_lines.extend(
            [
                Text(""),
                Text.from_markup("[bold green]How To Choose:[/bold green]"),
                Text.from_markup(param.help_text, style="dim"),
            ]
        )

        option_descriptions = param.option_descriptions
        if option_descriptions is None and param.value_type == "bool":
            option_descriptions = {
                "True": "Enable this behavior. It changes the training area described in Training Impact.",
                "False": "Disable this behavior. YOLO uses the standard/default path for this setting.",
            }

        if option_descriptions:
            option_table = Table(
                show_header=True,
                header_style="bold cyan",
                border_style="dim",
                box=box.SIMPLE,
                expand=True,
            )
            option_table.add_column("Value", style="yellow", no_wrap=True)
            option_table.add_column("Effect")
            for option, description in option_descriptions.items():
                display_option = "<empty>" if option == "" else str(option)
                option_table.add_row(display_option, Text.from_markup(description))
            info_lines.extend(
                [
                    Text(""),
                    Text.from_markup("[bold yellow]Value Meanings:[/bold yellow]"),
                    option_table,
                ]
            )

        # Input Area
        info_lines.append(Text(""))
        title_prefix = "➤ " if self.focus == "input" else ""
        
        input_content = []
        if param.value_type == "bool":
            # Simple toggle display
            input_content.append(Text("Value: ", style="bold"))
            display_val = self.input_buffer if self.focus == "input" else str(current_val)
            input_content.append(Text(str(display_val), style="bold yellow"))
            input_content.append(Text(" (Enter/Space toggles and saves)", style="dim"))
        elif param.allowed_values:
            # Cycle display
            display_val = self.input_buffer if self.focus == "input" else str(current_val)
            input_content.append(Text("Value: ", style="bold"))
            input_content.append(Text(str(display_val), style="bold yellow"))
            input_content.append(Text(" (Up/Down cycles, typing allowed, Enter saves)", style="dim"))
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
            make_panel(
                Text.assemble(*input_content),
                title=f"{title_prefix}Edit Value",
                state=TUIState.INFO if self.focus == "input" else TUIState.NORMAL,
                padding=(0, 1)
            )
        )

        return make_panel(
            Group(*info_lines),
            title="[bold cyan]Details & Configuration[/bold cyan]",
            padding=(1, 2),
            expand=True,
        )

    def _render_status_bar(self) -> Panel:
        if self.focus == "list":
            hints_text = render_hints("parameter_list")
        else:
            param = self.filtered_params[self.current_index]
            if param.value_type == "bool":
                hints_text = Text.from_markup(
                    "[bold yellow]Enter/Space[/bold yellow] Toggle + Save  •  "
                    "[bold yellow]Esc/←/B[/bold yellow] Back"
                )
            elif param.allowed_values:
                hints_text = Text.from_markup(
                    "[bold yellow]↑↓[/bold yellow] Cycle  •  "
                    "[bold yellow]Type[/bold yellow] Adjust  •  "
                    "[bold yellow]Enter[/bold yellow] Save  •  "
                    "[bold yellow]Esc/←/B[/bold yellow] Back"
                )
            else:
                hints_text = render_hints("parameter_input")

        selected_count = len(self.selected)
        count_text = Text(f"Selected: {selected_count}", style="bold cyan")

        status_table = Table.grid(expand=True)
        status_table.add_column(justify="left", ratio=1)
        status_table.add_column(justify="right")
        status_table.add_row(hints_text, count_text)

        return Panel(status_table, border_style="dim", padding=(0, 1))

    def __rich__(self) -> Layout:
        layout = Layout()
        show_stepper = self.wizard_steps and self.wizard_current_step is not None
        header_size = 5 if show_stepper else 3
        layout.split_column(
            Layout(name="header", size=header_size),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        # Header
        if show_stepper:
            header_text = Text()
            header_text.append(self.title + "\n", style="bold cyan")
            header_text.append("Wizard Progress:  ", style="dim white")
            for i, step in enumerate(self.wizard_steps):
                is_active = i == self.wizard_current_step
                is_completed = i < self.wizard_current_step
                
                if is_active:
                    bullet = "●"
                    style = "bold yellow"
                elif is_completed:
                    bullet = "✓"
                    style = "bold green"
                else:
                    bullet = "○"
                    style = "dim white"
                
                header_text.append(f"{bullet} {step}", style=style)
                if i < len(self.wizard_steps) - 1:
                    line_style = "green" if is_completed else "dim"
                    header_text.append(" ── ", style=line_style)
            
            layout["header"].update(
                make_panel(
                    header_text,
                    state=TUIState.INFO,
                    padding=(0, 1),
                )
            )
        else:
            layout["header"].update(
                make_panel(
                    Text(self.title, style="bold cyan", justify="center"),
                    state=TUIState.INFO,
                    padding=(0, 1),
                )
            )

        layout["body"].split_row(
            Layout(name="sidebar", ratio=11),
            Layout(name="content", ratio=9),
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
    wizard_steps: list[str] | None = None,
    wizard_current_step: int | None = None,
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
    typed_input_started = False
    validation_error: str | None = None

    def current_buffer_value(param: ParameterDefinition) -> str:
        current_value = values.get(param.name, param.default)
        if param.value_type == "bool":
            try:
                return str(parse_parameter_value(param, current_value))
            except ValueError:
                return str(bool(current_value))
        if param.allowed_values and current_value not in param.allowed_values:
            return str(param.default if param.default in param.allowed_values else param.allowed_values[0])
        return "" if current_value is None else str(current_value)

    def cycle_allowed_value(param: ParameterDefinition, raw_value: str, step: int) -> str:
        if not param.allowed_values:
            return raw_value
        try:
            current_allowed_index = param.allowed_values.index(raw_value)
        except ValueError:
            current_allowed_index = 0
        next_index = (current_allowed_index + step) % len(param.allowed_values)
        return param.allowed_values[next_index]

    def save_input_value(param: ParameterDefinition) -> bool:
        nonlocal focus, validation_error
        is_valid, value, error = convert_and_validate_parameter_value(
            param,
            input_buffer,
        )
        if is_valid:
            values[param.name] = value
            selected.add(param.name)
            validation_error = None
            focus = "list"
            return True
        validation_error = error
        return False

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
            validation_error=validation_error,
            wizard_steps=wizard_steps,
            wizard_current_step=wizard_current_step,
        )

        with Live(renderer, console=TUI_CONSOLE, refresh_per_second=10, screen=True) as live:
            while True:
                key = TUI_TERM.inkey(timeout=0.1)
                param = parameters[current_index]

                if focus == "list":
                    if key.name == "KEY_UP" or key.lower() == "k":
                        current_index = (current_index - 1) % len(parameters)
                        validation_error = None
                    elif key.name == "KEY_DOWN" or key.lower() == "j":
                        current_index = (current_index + 1) % len(parameters)
                        validation_error = None
                    elif key.name == "KEY_PGUP":
                        visible_items = max(4, min(len(parameters), TUI_TERM.height - 12))
                        current_index = max(0, current_index - visible_items)
                        validation_error = None
                    elif key.name == "KEY_PGDN":
                        visible_items = max(4, min(len(parameters), TUI_TERM.height - 12))
                        current_index = min(len(parameters) - 1, current_index + visible_items)
                        validation_error = None
                    elif key.name == "KEY_HOME":
                        current_index = 0
                        validation_error = None
                    elif key.name == "KEY_END":
                        current_index = len(parameters) - 1
                        validation_error = None
                    elif key == " ":
                        if param.name in selected:
                            selected.remove(param.name)
                        else:
                            selected.add(param.name)
                        validation_error = None
                    elif key.name == "KEY_RIGHT" or is_enter_key(key):
                        focus = "input"
                        input_buffer = current_buffer_value(param)
                        typed_input_started = False
                        validation_error = None
                    elif key.lower() == "a":
                        selected = {p.name for p in parameters}
                        validation_error = None
                    elif key.lower() == "n":
                        selected = set()
                        validation_error = None
                    elif key.lower() == "f":  # Finish
                        return selected, values
                    elif key.lower() == "q" or key.name == "KEY_ESCAPE":
                        return None
                
                elif focus == "input":
                    is_string_param = param.value_type in {"str", "optional_str", "bool_or_str"}
                    should_exit_input = (
                        key.name == "KEY_ESCAPE"
                        or key.name == "KEY_LEFT"
                        or (not is_string_param and key.lower() == "b")
                    )
                    if should_exit_input:
                        focus = "list"
                        validation_error = None
                    elif param.value_type == "bool":
                        if key == " " or is_enter_key(key) or key.name in {
                            "KEY_UP",
                            "KEY_DOWN",
                            "KEY_RIGHT",
                        }:
                            try:
                                current_bool = parse_parameter_value(
                                    param,
                                    input_buffer,
                                )
                            except ValueError:
                                current_bool = bool(param.default)
                            values[param.name] = not current_bool
                            selected.add(param.name)
                            input_buffer = str(values[param.name])
                            validation_error = None
                            focus = "list"
                    elif param.allowed_values:
                        if key.name in {"KEY_UP", "KEY_DOWN", "KEY_RIGHT"}:
                            step = -1 if key.name == "KEY_UP" else 1
                            input_buffer = cycle_allowed_value(
                                param,
                                input_buffer,
                                step,
                            )
                            validation_error = None
                        elif is_enter_key(key):
                            save_input_value(param)
                        elif key.name == "KEY_BACKSPACE":
                            input_buffer = input_buffer[:-1]
                            typed_input_started = True
                            validation_error = None
                        elif key and not key.is_sequence:
                            if typed_input_started:
                                input_buffer += key
                            else:
                                input_buffer = str(key)
                            typed_input_started = True
                            validation_error = None
                    elif is_enter_key(key):
                        save_input_value(param)
                    elif key.name == "KEY_BACKSPACE":
                        input_buffer = input_buffer[:-1]
                        validation_error = None
                    elif key and not key.is_sequence:
                        input_buffer += key
                        validation_error = None

                # Sync renderer
                renderer.current_index = current_index
                renderer.selected = selected
                renderer.values = values
                renderer.focus = focus
                renderer.input_buffer = input_buffer
                renderer.validation_error = validation_error
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
    if param.value_type in {"str", "optional_str", "bool_or_str"}:
        console.print(
            "\n[dim]Commands: [bold yellow]Enter[/bold yellow] (keep current)  •  "
            "[bold yellow]:b[/bold yellow] (back)  •  "
            "[bold yellow]:l[/bold yellow] (list)[/dim]"
        )
    else:
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

        if user_input.lower() == ":b" or user_input.lower() == ":back":
            return NAV_BACK
        if user_input.lower() == ":l" or user_input.lower() == ":list":
            return NAV_LIST

        if param.value_type not in {"str", "optional_str", "bool_or_str"}:
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
