"""Regression tests for the Phase-5 TUI fixes."""
from __future__ import annotations

import io
import subprocess
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

from rich.console import Console
from rich.text import Text

from yolomatic.utils import tui


class NonInteractiveTest(unittest.TestCase):
    def test_yolomatic_without_tty_exits_fast_with_code_2(self) -> None:
        # Regression: with stdin/stdout not a TTY the menu loop spun at 100% CPU forever.
        started = time.monotonic()
        result = subprocess.run(
            [sys.executable, "-m", "yolomatic"],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("terminal", (result.stdout + result.stderr).lower())
        self.assertLess(time.monotonic() - started, 30)

    def test_menus_refuse_to_run_without_a_terminal(self) -> None:
        fake_term = MagicMock(is_a_tty=False, _keyboard_fd=None)
        with patch.object(tui, "TUI_TERM", fake_term):
            with self.assertRaises(tui.NonInteractiveTerminalError):
                tui.get_user_choice(["A", "B"])
            with self.assertRaises(tui.NonInteractiveTerminalError):
                tui.get_user_multi_select([tui.ParameterDefinition("x", "c", 1, "int", "d", "h")])

    def test_menu_without_selectable_options_is_a_clear_error(self) -> None:
        with self.assertRaises(ValueError):
            tui.get_user_choice(["[Header Only]"])

    def test_cli_version_and_help(self) -> None:
        version = subprocess.run([sys.executable, "-m", "yolomatic", "--version"],
                                 capture_output=True, text=True, timeout=60)
        self.assertEqual(version.returncode, 0)
        self.assertIn("yolomatic", version.stdout)
        helptext = subprocess.run([sys.executable, "-m", "yolomatic", "--help"],
                                  capture_output=True, text=True, timeout=60)
        self.assertEqual(helptext.returncode, 0)
        for command in ("train", "predict", "export", "benchmark", "tensorboard"):
            self.assertIn(command, helptext.stdout)


def _fake_key(text: str, name: str | None = None):
    key = MagicMock()
    key.__str__ = lambda self: text
    key.__bool__ = lambda self: True
    key.name = name
    key.lower = lambda: text.lower()
    return key


class _InteractiveTerm:
    """Minimal blessed stand-in that replays scripted keys."""

    is_a_tty = True
    _keyboard_fd = 0
    width, height = 120, 40

    def __init__(self, keys):
        self._keys = list(keys)

    def cbreak(self):
        import contextlib
        return contextlib.nullcontext()

    hidden_cursor = cbreak
    mouse_enabled = cbreak

    def inkey(self, timeout=None):
        if not self._keys:
            raise KeyboardInterrupt
        item = self._keys.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class MenuKeysTest(unittest.TestCase):
    def _choose(self, options, keys, **kwargs):
        quiet = Console(file=io.StringIO(), force_terminal=False, width=120, height=40)
        with patch.object(tui, "TUI_TERM", _InteractiveTerm(keys)), patch.object(tui, "TUI_CONSOLE", quiet):
            return tui.get_user_choice(options, **kwargs)

    def test_ctrl_c_on_main_menu_quits_and_elsewhere_goes_back(self) -> None:
        self.assertEqual(self._choose(["Train", "Exit"], [KeyboardInterrupt()]), "Exit")
        self.assertEqual(self._choose(["Train", "Other"], [KeyboardInterrupt()], allow_back=True), "Back")

    def test_help_overlay_toggles_without_selecting(self) -> None:
        enter = _fake_key("\n", "KEY_ENTER")
        self.assertEqual(self._choose(["Alpha", "Beta"], [_fake_key("?"), _fake_key("?"), enter]), "Alpha")

    def test_boolean_prompt_back_means_back(self) -> None:
        # Regression: Esc on a True/False prompt returned the current value and moved on.
        param = tui.ParameterDefinition("flag", "c", True, "bool", "d", "h")
        with patch.object(tui, "get_user_choice", return_value="Back to Previous"):
            self.assertEqual(tui.get_parameter_value_input(param, True), tui.NAV_BACK)
        with patch.object(tui, "get_user_choice", return_value=tui.NAV_BACK):
            self.assertEqual(tui.get_parameter_value_input(param, True), tui.NAV_BACK)

    def test_list_navigation_is_opt_in(self) -> None:
        # Regression: typing ":l" returned "__LIST__" to callers that then did float("__LIST__").
        param = tui.ParameterDefinition("conf", "c", 0.25, "float", "d", "h", min_value=0.0, max_value=1.0)
        with patch("builtins.input", side_effect=[":l", "0.5"]), patch.object(tui, "clear_screen"), \
             patch.object(tui, "TUI_CONSOLE", Console(file=io.StringIO())):
            self.assertEqual(tui.get_parameter_value_input(param, 0.25), 0.5)
        with patch("builtins.input", side_effect=[":l"]), patch.object(tui, "clear_screen"), \
             patch.object(tui, "TUI_CONSOLE", Console(file=io.StringIO())):
            self.assertEqual(tui.get_parameter_value_input(param, 0.25, allow_list=True), tui.NAV_LIST)

    def test_typed_values_use_the_shared_validator(self) -> None:
        param = tui.ParameterDefinition("conf", "c", 0.25, "float", "d", "h", min_value=0.0, max_value=1.0)
        with patch("builtins.input", side_effect=["7", "abc", "0.4"]), patch.object(tui, "clear_screen"), \
             patch.object(tui, "TUI_CONSOLE", Console(file=io.StringIO())):
            self.assertEqual(tui.get_parameter_value_input(param, 0.25), 0.4)


class RenderingTest(unittest.TestCase):
    def test_malformed_markup_in_descriptions_does_not_crash(self) -> None:
        rendered = tui.markup_or_text("class [/x] name")
        self.assertIsInstance(rendered, Text)
        self.assertIn("[/x]", rendered.plain)

    def test_confirmation_body_is_rendered_in_details_pane(self) -> None:
        summary = tui.build_summary_panel("Plan", {"Source": "datasets/leaves", "Images": 42})
        renderer = tui.MenuRenderer(["Start", "Back"], 0, "Confirm", "", {"Start": "Begin now."})
        renderer.body = summary
        out = Console(file=io.StringIO(), width=140, height=40, force_terminal=False)
        with patch.object(tui, "_term_w", 140), patch.object(tui, "_term_h", 40):
            out.print(renderer)
        text = out.file.getvalue()
        self.assertIn("datasets/leaves", text)
        self.assertIn("Begin now.", text)

    def test_tiny_terminal_shows_notice(self) -> None:
        renderer = tui.MenuRenderer(["A"], 0, "T", "")
        out = Console(file=io.StringIO(), width=40, height=10, force_terminal=False)
        out.print(renderer)
        self.assertIn("Terminal too small", out.file.getvalue())

    def test_footer_hints_match_the_screen(self) -> None:
        main_menu = tui.MenuRenderer(["Train", "Exit"], 0, "Main", "")
        sub_menu = tui.MenuRenderer(["Do", "Back"], 0, "Sub", "", back_option="Back")
        out = Console(file=io.StringIO(), width=140, force_terminal=False)
        out.print(main_menu._render_status_bar())
        main_text = out.file.getvalue()
        self.assertIn("Quit", main_text)
        self.assertIn("Help", main_text)
        out = Console(file=io.StringIO(), width=140, force_terminal=False)
        out.print(sub_menu._render_status_bar())
        self.assertNotIn("Quit", out.file.getvalue())


if __name__ == "__main__":
    unittest.main()
