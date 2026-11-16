"""Single source of truth for colours, glyphs, and the shared Rich console.

Every screen should render through :data:`CONSOLE` and take colours from the
named styles below instead of hard-coding ``border_style="red"`` or hex values.
That keeps the look consistent, lets ``NO_COLOR`` / width settings apply
everywhere, and lets live regions (progress bars, menus) route prints correctly.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, TextIO

from rich.console import Console
from rich.markup import escape
from rich.theme import Theme


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

# Nord palette used by the About and goodbye screens.
NORD: dict[str, str] = {
    "nord.frost0": "#8FBCBB",
    "nord.frost1": "#88C0D0",
    "nord.frost2": "#81A1C1",
    "nord.frost3": "#5E81AC",
    "nord.red": "#BF616A",
    "nord.yellow": "#EBCB8B",
    "nord.green": "#A3BE8C",
}

STYLES: dict[str, str] = {
    **NORD,
    # Rich cannot combine a theme name with "bold" inside one style string, so
    # bold variants are explicit theme entries.
    **{f"{name}.bold": f"bold {colour}" for name, colour in NORD.items()},
    "state.normal": PANEL_BORDER_STYLES[TUIState.NORMAL],
    "state.info": PANEL_BORDER_STYLES[TUIState.INFO],
    "state.success": PANEL_BORDER_STYLES[TUIState.SUCCESS],
    "state.warning": PANEL_BORDER_STYLES[TUIState.WARNING],
    "state.error": PANEL_BORDER_STYLES[TUIState.ERROR],
    "key": "bold yellow",
    "muted": "dim",
    "accent": "bold cyan",
}

THEME = Theme(STYLES)


@dataclass(frozen=True)
class Glyphs:
    pointer: str
    check: str
    cross: str
    warning: str
    bullet: str
    ellipsis: str
    search: str


UNICODE_GLYPHS = Glyphs(pointer="➤", check="✓", cross="✗", warning="⚠", bullet="•", ellipsis="…", search="🔍")
ASCII_GLYPHS = Glyphs(pointer=">", check="+", cross="x", warning="!", bullet="*", ellipsis="...", search="?")


def supports_unicode(stream: TextIO | None = None) -> bool:
    """Return False for legacy code pages (e.g. cp1252 consoles) or when forced off."""
    if os.environ.get("YOLOMATIC_ASCII", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    encoding = getattr(stream if stream is not None else sys.stdout, "encoding", None) or ""
    return "utf" in encoding.lower()


GLYPHS: Glyphs = UNICODE_GLYPHS if supports_unicode() else ASCII_GLYPHS

# The one console every module prints through. Rich already honours NO_COLOR,
# FORCE_COLOR, COLUMNS and TERM; the theme adds YOLOmatic's named styles.
CONSOLE = Console(theme=THEME)


def safe(value: Any) -> str:
    """Escape user-provided text (paths, class names, errors) before embedding it in markup.

    Without escaping, a class named ``[car]`` disappears and an unmatched
    ``[/x]`` raises ``MarkupError`` in the middle of a render.
    """
    return escape(str(value))


def gpu_badge_markup(status: str) -> str:
    """Footer badge for the cached GPU status (shared by every renderer)."""
    if status == "Detecting...":
        return f"[dim]GPU: {GLYPHS.search}[/dim]"
    if status == "CPU":
        return "[dim]GPU: CPU[/dim]"
    return f"[bold green]GPU: {safe(status)}[/bold green]"
