"""The contract between subcommands, the TUI runner, and the shell.

Every subcommand ``main()`` returns a :class:`CommandResult`. The interactive
runner uses it to decide whether to pause (always on failure, never on cancel),
and standalone entry points turn it into an exit code, so scripts and CI can
finally tell a failed export from a successful one (many used to exit 0).

Legacy subcommands that still return ``None`` are treated as success.
"""
from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, NoReturn

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_CANCELLED = 130  # conventional 128 + SIGINT


class CommandStatus(str, Enum):
    OK = "ok"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class CommandResult:
    status: CommandStatus
    message: str = ""
    hint: str | None = None
    outputs: Mapping[str, Path] = field(default_factory=dict)
    # True when the command printed a summary the user should read before the
    # menu redraws (the TUI runner pauses for it).
    review: bool = False

    @classmethod
    def ok(
        cls,
        message: str = "",
        *,
        outputs: Mapping[str, Path | str] | None = None,
        review: bool = False,
    ) -> CommandResult:
        return cls(
            CommandStatus.OK,
            message,
            outputs={k: Path(v) for k, v in (outputs or {}).items()},
            review=review,
        )

    @classmethod
    def cancelled(cls, message: str = "Cancelled.") -> CommandResult:
        return cls(CommandStatus.CANCELLED, message)

    @classmethod
    def failed(cls, message: str, *, hint: str | None = None) -> CommandResult:
        return cls(CommandStatus.FAILED, message, hint=hint)

    @property
    def exit_code(self) -> int:
        return {
            CommandStatus.OK: EXIT_OK,
            CommandStatus.CANCELLED: EXIT_CANCELLED,
            CommandStatus.FAILED: EXIT_FAILED,
        }[self.status]


def coerce_result(value: Any) -> CommandResult:
    """Normalise whatever a subcommand returned (``None``, an int, a result)."""
    if isinstance(value, CommandResult):
        return value
    if value is None or value is True:
        return CommandResult.ok()
    if value is False:
        return CommandResult.failed("Command reported failure.")
    if isinstance(value, int):
        if value == EXIT_OK:
            return CommandResult.ok()
        if value == EXIT_CANCELLED:
            return CommandResult.cancelled()
        return CommandResult.failed(f"Command exited with status {value}.")
    return CommandResult.ok()


def exit_code_for(value: Any) -> int:
    return coerce_result(value).exit_code


def run_standalone(func: Callable[..., Any], *args: Any, **kwargs: Any) -> NoReturn:
    """Run a subcommand as a console script and exit with its status code."""
    try:
        value = func(*args, **kwargs)
    except KeyboardInterrupt:
        sys.exit(EXIT_CANCELLED)
    sys.exit(exit_code_for(value))


def standalone_entrypoint(func: Callable[..., Any]) -> Callable[..., NoReturn]:
    """Decorator form of :func:`run_standalone` for ``[project.scripts]`` targets."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> NoReturn:
        from yolomatic.utils.logging_setup import configure_logging

        configure_logging()
        run_standalone(func, *args, **kwargs)

    return wrapper
