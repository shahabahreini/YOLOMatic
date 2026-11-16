"""Tests for the shared Phase-3 foundations: fs staging, jobs, results, logging, theme."""
from __future__ import annotations

import _thread
import io
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from yolomatic.cli.result import (
    EXIT_CANCELLED,
    EXIT_FAILED,
    CommandResult,
    CommandStatus,
    coerce_result,
    run_standalone,
)
from yolomatic.ui import theme
from yolomatic.utils import logging_setup
from yolomatic.utils.fs import (
    COMPLETION_MARKER,
    atomic_write_text,
    is_staging_name,
    staging_dir,
)
from yolomatic.utils.jobs import (
    CancelToken,
    JobStatus,
    cancellable_executor,
    run_with_progress,
)


def _quiet_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=80)


class AtomicWriteTest(unittest.TestCase):
    def test_replaces_content_and_leaves_no_temp_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "settings.yaml"
            target.write_text("old", encoding="utf-8")
            os.chmod(target, 0o640)
            atomic_write_text(target, "new ✓")
            self.assertEqual(target.read_text(encoding="utf-8"), "new ✓")
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["settings.yaml"])
            if os.name == "posix":
                self.assertEqual(target.stat().st_mode & 0o777, 0o640)

    def test_failed_write_keeps_old_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "data.yaml"
            target.write_text("original", encoding="utf-8")
            with patch("yolomatic.utils.fs.os.replace", side_effect=OSError(28, "No space left")):
                with self.assertRaises(OSError):
                    atomic_write_text(target, "partial")
            self.assertEqual(target.read_text(encoding="utf-8"), "original")
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["data.yaml"])


class StagingDirTest(unittest.TestCase):
    def test_success_swaps_in_and_marks_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            final = Path(tmp) / "out"
            with staging_dir(final, mark_complete=True) as staging:
                self.assertTrue(is_staging_name(staging.name))
                (staging / "a.txt").write_text("a", encoding="utf-8")
                self.assertFalse(final.exists())
            self.assertEqual((final / "a.txt").read_text(encoding="utf-8"), "a")
            self.assertTrue((final / COMPLETION_MARKER).exists())
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["out"])

    def test_failure_or_interrupt_keeps_existing_output(self) -> None:
        for exc in (RuntimeError("boom"), KeyboardInterrupt()):
            with self.subTest(exc=type(exc).__name__), tempfile.TemporaryDirectory() as tmp:
                final = Path(tmp) / "out"
                final.mkdir()
                (final / "keep.txt").write_text("keep", encoding="utf-8")
                with self.assertRaises(type(exc)):
                    with staging_dir(final, overwrite=True) as staging:
                        (staging / "new.txt").write_text("new", encoding="utf-8")
                        raise exc
                self.assertEqual(sorted(p.name for p in final.iterdir()), ["keep.txt"])
                self.assertEqual([p.name for p in Path(tmp).iterdir()], ["out"])

    def test_refuses_existing_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            final = Path(tmp) / "out"
            final.mkdir()
            with self.assertRaises(FileExistsError):
                with staging_dir(final):
                    pass


class JobsTest(unittest.TestCase):
    def test_completed_failed_and_cooperative_cancel(self) -> None:
        console = _quiet_console()
        done = run_with_progress(lambda token: 42, console=console)
        self.assertEqual((done.status, done.value), (JobStatus.COMPLETED, 42))

        def boom(token: CancelToken) -> None:
            raise ValueError("bad input")

        failed = run_with_progress(boom, console=console)
        self.assertEqual(failed.status, JobStatus.FAILED)
        self.assertIsInstance(failed.error, ValueError)

        token = CancelToken()
        token.cancel()

        def loop(token: CancelToken) -> None:
            while True:
                token.raise_if_cancelled()
                time.sleep(0.01)

        cancelled = run_with_progress(loop, token=token, console=console)
        self.assertEqual(cancelled.status, JobStatus.CANCELLED)

    def test_ctrl_c_cancels_worker_and_waits_for_it(self) -> None:
        # Regression: Ctrl-C used to return to the menu while the worker kept running.
        finished_cleanly = threading.Event()

        def job(token: CancelToken) -> str:
            time.sleep(0.1)
            _thread.interrupt_main()  # simulate the user pressing Ctrl-C
            deadline = time.monotonic() + 5
            while not token.cancelled and time.monotonic() < deadline:
                time.sleep(0.01)
            time.sleep(0.05)  # finish the current unit of work
            finished_cleanly.set()
            token.raise_if_cancelled()
            return "unreachable"

        outcome = run_with_progress(job, console=_quiet_console(), refresh_per_second=50)
        self.assertEqual(outcome.status, JobStatus.CANCELLED)
        self.assertTrue(finished_cleanly.is_set(), "runner returned before the worker stopped")

    def test_cancellable_executor_drops_queued_work(self) -> None:
        token = CancelToken()
        started: list[int] = []
        gate = threading.Event()

        def task(i: int) -> None:
            started.append(i)
            gate.wait(1)

        pool = ThreadPoolExecutor(max_workers=1)
        with cancellable_executor(pool, token) as executor:
            for i in range(20):
                executor.submit(task, i)
            token.cancel()
            gate.set()
        self.assertLess(len(started), 20)


class CommandResultTest(unittest.TestCase):
    def test_coercion_and_exit_codes(self) -> None:
        self.assertEqual(coerce_result(None).status, CommandStatus.OK)
        self.assertEqual(coerce_result(0).exit_code, 0)
        self.assertEqual(coerce_result(3).exit_code, EXIT_FAILED)
        self.assertEqual(coerce_result(EXIT_CANCELLED).status, CommandStatus.CANCELLED)
        self.assertEqual(CommandResult.failed("x", hint="y").exit_code, EXIT_FAILED)
        result = CommandResult.ok(outputs={"weights": "runs/best.pt"})
        self.assertEqual(result.outputs["weights"], Path("runs/best.pt"))

    def test_run_standalone_maps_results_and_ctrl_c(self) -> None:
        cases = [
            (lambda: CommandResult.failed("nope"), EXIT_FAILED),
            (lambda: None, 0),
            (lambda: (_ for _ in ()).throw(KeyboardInterrupt()), EXIT_CANCELLED),
        ]
        for func, code in cases:
            with self.subTest(code=code), self.assertRaises(SystemExit) as caught:
                run_standalone(func)
            self.assertEqual(caught.exception.code, code)


class LoggingSetupTest(unittest.TestCase):
    def tearDown(self) -> None:
        for handler in list(logging.getLogger().handlers):
            if getattr(handler, "_yolomatic_managed", False):
                logging.getLogger().removeHandler(handler)
                handler.close()

    def test_importing_modules_does_not_configure_logging(self) -> None:
        # Regression: generator/analyzer called basicConfig at import time, which
        # made the entry point's own logging setup a silent no-op.
        code = (
            "import logging, yolomatic.config.generator, yolomatic.datasets.analyzer; "
            "print(len(logging.getLogger().handlers))"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "0")

    def test_traceback_goes_to_file_not_console(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            buffer = io.StringIO()
            console = Console(file=buffer, force_terminal=False, width=120)
            with patch.dict(os.environ, {"YOLOMATIC_LOG_DIR": tmp}):
                path = logging_setup.configure_logging(verbose=False, console=console)
            self.assertEqual(path, Path(tmp) / "yolomatic.log")
            try:
                raise RuntimeError("secret detail")
            except RuntimeError as error:
                self.assertEqual(logging_setup.log_exception(error, "export failed"), path)
            for handler in logging.getLogger().handlers:
                handler.flush()
            text = path.read_text(encoding="utf-8")
            self.assertIn("Traceback", text)
            self.assertIn("secret detail", text)
            self.assertNotIn("secret detail", buffer.getvalue())

            logging.getLogger("yolomatic.test").warning("visible warning")
            self.assertIn("visible warning", buffer.getvalue())

    def test_run_log_mirrors_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with logging_setup.RunLog(tmp) as log_path:
                logging.getLogger("yolomatic.trainer").info("epoch 1 done")
            self.assertIn("epoch 1 done", log_path.read_text(encoding="utf-8"))


class ThemeTest(unittest.TestCase):
    def test_safe_escapes_markup_from_user_data(self) -> None:
        console = Console(file=io.StringIO(), force_terminal=False, width=80)
        console.print(f"class: {theme.safe('[car]')} path: {theme.safe('a/[/x]b')}")
        rendered = console.file.getvalue()
        self.assertIn("[car]", rendered)
        self.assertIn("[/x]", rendered)

    def test_gpu_badge_and_ascii_fallback(self) -> None:
        self.assertIn("CPU", theme.gpu_badge_markup("CPU"))
        console = Console(file=io.StringIO(), force_terminal=False, width=80)
        console.print(theme.gpu_badge_markup("RTX [bold]4090"))
        self.assertIn("RTX [bold]4090", console.file.getvalue())
        legacy = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        self.assertFalse(theme.supports_unicode(legacy))
        with patch.dict(os.environ, {"YOLOMATIC_ASCII": "1"}):
            self.assertFalse(theme.supports_unicode(sys.stdout))

    def test_named_styles_render_with_shared_console(self) -> None:
        console = Console(file=io.StringIO(), theme=theme.THEME, force_terminal=True, width=40)
        console.print("[nord.green]ok[/nord.green] [nord.frost0.bold]bold[/nord.frost0.bold]")
        self.assertIn("ok", console.file.getvalue())


if __name__ == "__main__":
    unittest.main()
