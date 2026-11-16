"""Cancellable background jobs with a live progress view.

Long operations (augmentation, benchmarks, SAM inference, conversions) run on a
worker thread while the UI thread renders progress. Before this module each
command rolled its own thread + ``Live`` loop, and Ctrl-C only stopped the UI
loop: the worker kept writing files after the user was back at the menu.

The contract here:

* the job receives a :class:`CancelToken` and calls ``token.raise_if_cancelled()``
  (or checks ``token.cancelled``) between units of work;
* Ctrl-C sets the token, the view switches to "Stopping...", and the runner
  waits for the job to wind down before returning;
* the caller gets a :class:`JobOutcome` that says whether the job finished,
  was cancelled, or failed -- it never has to guess from side effects.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Executor
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

from rich.console import Console, RenderableType
from rich.live import Live
from rich.text import Text

from yolomatic.ui.theme import CONSOLE

T = TypeVar("T")


class JobCancelled(Exception):
    """Raised inside a job when its :class:`CancelToken` has been set."""


class CancelToken:
    """Thread-safe cancellation flag shared between the UI and a worker."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise JobCancelled()

    def wait(self, timeout: float | None = None) -> bool:
        """Sleep up to ``timeout`` seconds; return True early if cancelled."""
        return self._event.wait(timeout)


class JobStatus(str, Enum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class JobOutcome(Generic[T]):
    status: JobStatus
    value: T | None = None
    error: BaseException | None = None

    @property
    def ok(self) -> bool:
        return self.status is JobStatus.COMPLETED


def _default_stopping_view() -> RenderableType:
    return Text("Stopping... waiting for the current step to finish (Ctrl+C again to force).", style="yellow")


def run_with_progress(
    job: Callable[[CancelToken], T],
    *,
    render: Callable[[], RenderableType] | None = None,
    token: CancelToken | None = None,
    console: Console | None = None,
    refresh_per_second: float = 8.0,
    stopping_view: Callable[[], RenderableType] = _default_stopping_view,
    transient: bool = False,
) -> JobOutcome[T]:
    """Run ``job(token)`` on a worker thread while rendering ``render()`` live.

    A first Ctrl-C requests cancellation and keeps waiting so partial output is
    left in a known state; a second Ctrl-C stops waiting and re-raises
    ``KeyboardInterrupt`` (the worker is a daemon thread and dies with the app).
    """
    token = token or CancelToken()
    result: dict[str, object] = {}
    done = threading.Event()

    def worker() -> None:
        try:
            result["value"] = job(token)
        except JobCancelled:
            result["cancelled"] = True
        except BaseException as error:  # noqa: BLE001 - reported via JobOutcome
            result["error"] = error
        finally:
            done.set()

    thread = threading.Thread(target=worker, name="yolomatic-job", daemon=True)
    interval = 1.0 / max(refresh_per_second, 1.0)
    view = render or (lambda: Text("Working...", style="cyan"))

    with Live(view(), console=console or CONSOLE, refresh_per_second=refresh_per_second,
              transient=transient) as live:
        stopping = False
        # Start inside the Live block so a Ctrl-C can only arrive once the
        # interrupt handler below is in place.
        thread.start()
        while not done.is_set():
            try:
                done.wait(interval)
                live.update(stopping_view() if stopping else view())
            except KeyboardInterrupt:
                if stopping:
                    raise
                stopping = True
                token.cancel()
                live.update(stopping_view())
        if not stopping:
            live.update(view())

    thread.join()
    if "error" in result:
        error = result["error"]
        assert isinstance(error, BaseException)
        return JobOutcome(JobStatus.FAILED, error=error)
    if result.get("cancelled") or token.cancelled:
        return JobOutcome(JobStatus.CANCELLED, value=result.get("value"))  # type: ignore[arg-type]
    return JobOutcome(JobStatus.COMPLETED, value=result.get("value"))  # type: ignore[arg-type]


@contextmanager
def cancellable_executor(executor: Executor, token: CancelToken | None = None) -> Iterator[Executor]:
    """Use an executor so that Ctrl-C/cancel drops queued work instead of draining it.

    ``with ThreadPoolExecutor() as pool`` waits for *every* submitted task on
    exit, so an interrupted run kept processing the whole queue. This wrapper
    cancels pending futures when the block exits early.
    """
    clean_exit = False
    try:
        yield executor
        clean_exit = token is None or not token.cancelled
    finally:
        executor.shutdown(wait=True, cancel_futures=not clean_exit)


def sleep_unless_cancelled(token: CancelToken | None, seconds: float) -> None:
    """Sleep that wakes up immediately when ``token`` is cancelled."""
    if token is None:
        time.sleep(seconds)
    elif token.wait(seconds):
        raise JobCancelled()
