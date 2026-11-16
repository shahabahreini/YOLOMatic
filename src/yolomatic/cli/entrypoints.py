"""Console-script targets for ``[project.scripts]``.

Generated console scripts run ``sys.exit(target())``. Subcommand ``main()``
functions return a :class:`~yolomatic.cli.result.CommandResult` (so the TUI can
decide when to pause), which must be translated into a process exit code here:
0 on success, 1 on failure, 130 when cancelled with Ctrl-C.
"""
from __future__ import annotations

import importlib
from typing import NoReturn

from yolomatic.cli.result import run_standalone


def _run(module: str, func: str = "main") -> NoReturn:
    from yolomatic.utils.logging_setup import configure_logging

    configure_logging()
    target = getattr(importlib.import_module(module), func)
    run_standalone(target)


def predict() -> NoReturn:
    _run("yolomatic.cli.predict")


def tensorboard() -> NoReturn:
    _run("yolomatic.cli.tensorboard_launcher")


def train() -> NoReturn:
    _run("yolomatic.trainers.yolo_trainer")


def upload() -> NoReturn:
    _run("yolomatic.cli.upload")


def upload_dataset() -> NoReturn:
    _run("yolomatic.cli.upload_dataset")


def benchmark() -> NoReturn:
    _run("yolomatic.cli.benchmark")


def sam() -> NoReturn:
    _run("yolomatic.cli.sam_predict")


def convert() -> NoReturn:
    _run("yolomatic.cli.convert_ndjson")


def prepare() -> NoReturn:
    _run("yolomatic.cli.prepare_dataset")


def ultralytics() -> NoReturn:
    _run("yolomatic.cli.ultralytics_platform")


def export() -> NoReturn:
    _run("yolomatic.cli.export")


def augment() -> NoReturn:
    _run("yolomatic.cli.augment")
