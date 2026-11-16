"""Central logging configuration for every YOLOmatic entry point.

Library modules only ever do ``logger = logging.getLogger(__name__)``; they must
not call ``logging.basicConfig`` at import time (that silently disabled the
entry-point setup before). Entry points call :func:`configure_logging` once.

Console output stays quiet (warnings and errors, no tracebacks) so the TUI is
not flooded, while a rotating session log keeps full detail -- including
tracebacks -- so a friendly error panel can point at the file.
"""
from __future__ import annotations

import logging
import os
import tempfile
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from yolomatic.ui.theme import CONSOLE

_FILE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s [%(threadName)s] %(message)s"
_NOISY_LOGGERS = ("matplotlib", "PIL", "urllib3", "h5py", "numba", "fsspec")
_MANAGED_ATTR = "_yolomatic_managed"

_session_log_path: Path | None = None


def default_log_dir() -> Path:
    """Per-user log folder (``YOLOMATIC_LOG_DIR`` overrides it)."""
    override = os.environ.get("YOLOMATIC_LOG_DIR")
    if override:
        return Path(override).expanduser()
    try:
        from platformdirs import user_log_path

        return user_log_path("yolomatic", appauthor=False)
    except Exception:  # noqa: BLE001 - platformdirs missing or unusable
        return Path(tempfile.gettempdir()) / "yolomatic-logs"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def configure_logging(
    *,
    verbose: bool | None = None,
    log_file: Path | str | None = None,
    console: Console | None = None,
    session_log: bool = True,
) -> Path | None:
    """Configure the root logger; safe to call more than once.

    ``verbose`` (or ``YOLOMATIC_DEBUG=1``) lowers the console threshold to DEBUG and
    shows rich tracebacks. ``log_file`` overrides the default rotating session log.
    Returns the active log file path, if any.
    """
    global _session_log_path
    if verbose is None:
        verbose = _env_flag("YOLOMATIC_DEBUG")

    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _MANAGED_ATTR, False):
            root.removeHandler(handler)
            handler.close()

    console_handler = RichHandler(
        console=console or CONSOLE,
        level=logging.DEBUG if verbose else logging.WARNING,
        show_time=False,
        show_path=verbose,
        markup=False,
        rich_tracebacks=verbose,
    )
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    setattr(console_handler, _MANAGED_ATTR, True)
    handlers: list[logging.Handler] = [console_handler]

    _session_log_path = None
    target = Path(log_file).expanduser() if log_file else None
    if target is None and session_log:
        target = default_log_dir() / "yolomatic.log"
    open_error: OSError | None = None
    if target is not None:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                target, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
            setattr(file_handler, _MANAGED_ATTR, True)
            handlers.append(file_handler)
            _session_log_path = target
        except OSError as error:  # read-only home etc. must never stop the app
            open_error = error

    for handler in handlers:
        root.addHandler(handler)
    # Third-party libraries log at INFO+ (DEBUG with --debug); YOLOmatic's own
    # loggers always reach the file at DEBUG.
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    logging.getLogger("yolomatic").setLevel(logging.DEBUG)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    if open_error is not None:
        logging.getLogger(__name__).warning("Could not open log file %s: %s", target, open_error)
    return _session_log_path


def session_log_path() -> Path | None:
    """Path of the active session log, or ``None`` when file logging is off."""
    return _session_log_path


def log_exception(error: BaseException, context: str) -> Path | None:
    """Record ``error`` with its traceback in the log file only; return the log path.

    Use it next to a friendly error panel: the user sees the short message, and
    the panel can say where the full traceback is.
    """
    logging.getLogger("yolomatic").debug(
        "%s: %s", context, error, exc_info=(type(error), error, error.__traceback__)
    )
    return _session_log_path


class RunLog:
    """Context manager that mirrors all logging into ``<run_dir>/yolomatic.log``."""

    def __init__(self, run_dir: Path | str) -> None:
        self.path = Path(run_dir) / "yolomatic.log"
        self._handler: logging.Handler | None = None
        self._previous_level: int | None = None

    def __enter__(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(self.path, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        logging.getLogger().addHandler(handler)
        self._handler = handler
        app_logger = logging.getLogger("yolomatic")
        self._previous_level = app_logger.level
        app_logger.setLevel(logging.DEBUG)
        return self.path

    def __exit__(self, *_exc: object) -> None:
        if self._handler is not None:
            logging.getLogger().removeHandler(self._handler)
            self._handler.close()
            self._handler = None
        if self._previous_level is not None:
            logging.getLogger("yolomatic").setLevel(self._previous_level)
            self._previous_level = None
