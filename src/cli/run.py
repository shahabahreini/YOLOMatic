import logging
import os
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import yaml
from rich import box
from rich.align import Align
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from ruamel.yaml import YAML

from src.config.generator import YOLOConfigGenerator, YOLONASConfigGenerator
from src.models.data import model_data_dict
from src.utils.cli import (
    ParameterDefinition,
    NAV_BACK,
    NAV_LIST,
    clear_screen,
    console,
    get_parameter_value_input,
    get_user_choice,
    get_user_multi_select,
    print_stylized_header,
    render_summary_panel,
    render_table,
)
from src.utils.ml_dependencies import MLDependencyError, import_torch
from src.utils.project import format_size, list_dataset_directories

# Comprehensive YOLO training parameter definitions for fully customized config
YOLO_TRAINING_PARAMETERS: list[ParameterDefinition] = [
    # Core Training Parameters
    ParameterDefinition(
        name="epochs",
        category="core",
        default=300,
        value_type="int",
        description="Number of training epochs",
        help_text="Total number of complete passes through the training dataset. More epochs can improve accuracy but may cause overfitting. For most datasets, 100-300 epochs are sufficient.",
        min_value=1,
        max_value=10000,
    ),
    ParameterDefinition(
        name="patience",
        category="core",
        default=50,
        value_type="int",
        description="Early stopping patience",
        help_text="Number of epochs to wait without improvement in validation metrics before stopping training early. Set to 0 to disable early stopping. Prevents overfitting and saves time.",
        min_value=0,
        max_value=500,
    ),
    ParameterDefinition(
        name="batch",
        category="core",
        default=-1,
        value_type="int",
        description="Batch size for training",
        help_text="Number of images processed together in one forward/backward pass. -1 enables auto-batch which automatically finds the largest batch size that fits in GPU memory. Larger batches provide more stable gradients but require more VRAM.",
        min_value=-1,
        max_value=1024,
    ),
    ParameterDefinition(
        name="imgsz",
        category="core",
        default=640,
        value_type="int",
        description="Input image size",
        help_text="Size to which input images are resized (square). Larger sizes can detect smaller objects but require more memory and computation. Common values: 320 (fast), 640 (standard), 1280 (high quality).",
        min_value=32,
        max_value=2048,
    ),
    ParameterDefinition(
        name="device",
        category="hardware",
        default="0",
        value_type="str",
        description="Device for training",
        help_text="Device to run training on. '0' for first GPU, '0,1' for multi-GPU, 'cpu' for CPU training, or 'mps' for Apple Silicon. Auto-detects CUDA if available.",
        allowed_values=[
            "0",
            "0,1",
            "0,1,2,3",
            "cpu",
            "mps",
            "cuda",
            "npu",
            "npu:0",
            "-1",
        ],
    ),
    ParameterDefinition(
        name="workers",
        category="hardware",
        default=8,
        value_type="int",
        description="DataLoader workers",
        help_text="Number of parallel processes for data loading. More workers can speed up data loading but use more CPU and RAM. Set to 0 for main process loading (slower but less resource intensive).",
        min_value=0,
        max_value=64,
    ),
    ParameterDefinition(
        name="cache",
        category="hardware",
        default="False",
        value_type="str",
        description="Cache dataset in memory",
        help_text="Whether to cache the entire dataset in RAM for faster access. 'True' or 'ram' loads all images into memory. 'disk' caches to disk. 'False' disables caching.",
        allowed_values=["True", "False", "ram", "disk"],
    ),
    ParameterDefinition(
        name="optimizer",
        category="optimizer",
        default="auto",
        value_type="str",
        description="Optimizer type",
        help_text="Optimization algorithm for weight updates. 'auto' lets YOLO choose (usually SGD for large batches, AdamW for small). SGD is more stable for large datasets, AdamW adapts learning rates per parameter.",
        allowed_values=[
            "auto",
            "SGD",
            "Adam",
            "AdamW",
            "Adamax",
            "NAdam",
            "RAdam",
            "RMSProp",
            "MuSGD",
        ],
    ),
    ParameterDefinition(
        name="lr0",
        category="optimizer",
        default=0.01,
        value_type="float",
        description="Initial learning rate",
        help_text="Starting learning rate for the optimizer. Higher values train faster but may diverge. Lower values are more stable but slower. Typical values: 0.01 (SGD), 0.001 (Adam).",
        min_value=0.000001,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="lrf",
        category="optimizer",
        default=0.01,
        value_type="float",
        description="Final learning rate factor",
        help_text="Final learning rate = lr0 * lrf. Determines how much the learning rate decreases by the end of training. Smaller values mean more aggressive decay.",
        min_value=0.0001,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="momentum",
        category="optimizer",
        default=0.937,
        value_type="float",
        description="SGD momentum / Adam beta1",
        help_text="Momentum factor for SGD (accelerates convergence) or beta1 for Adam optimizers. Higher values smooth out updates using past gradients. Typical: 0.9-0.95 for SGD, 0.9 for Adam.",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="weight_decay",
        category="optimizer",
        default=0.0005,
        value_type="float",
        description="Weight decay (L2 regularization)",
        help_text="L2 regularization coefficient that penalizes large weights to prevent overfitting. Applied to all trainable parameters. Higher values = stronger regularization.",
        min_value=0.0,
        max_value=0.1,
    ),
    ParameterDefinition(
        name="warmup_epochs",
        category="optimizer",
        default=3.0,
        value_type="float",
        description="Warmup epochs",
        help_text="Number of epochs to gradually ramp up learning rate from a small value to lr0. Helps stabilize early training. Can be fractional (e.g., 3.5 epochs).",
        min_value=0.0,
        max_value=50.0,
    ),
    ParameterDefinition(
        name="warmup_momentum",
        category="optimizer",
        default=0.8,
        value_type="float",
        description="Warmup initial momentum",
        help_text="Starting momentum during warmup period, gradually increasing to the full momentum value. Lower initial momentum reduces instability in early training.",
        min_value=0.0,
        max_value=1.0,
    ),
    # Augmentation Parameters
    ParameterDefinition(
        name="hsv_h",
        category="augmentation",
        default=0.015,
        value_type="float",
        description="HSV hue augmentation",
        help_text="Maximum fraction of hue shift in HSV color space. Adds random hue variations to images. Helps model generalize to different lighting conditions. 0 = disabled.",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="hsv_s",
        category="augmentation",
        default=0.7,
        value_type="float",
        description="HSV saturation augmentation",
        help_text="Maximum fraction of saturation shift. Makes colors more or less vivid randomly. Important for generalizing to different camera/color settings.",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="hsv_v",
        category="augmentation",
        default=0.4,
        value_type="float",
        description="HSV value (brightness) augmentation",
        help_text="Maximum fraction of brightness shift. Simulates different lighting conditions. Critical for robust detection in varying light environments.",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="degrees",
        category="augmentation",
        default=0.0,
        value_type="float",
        description="Rotation degrees",
        help_text="Maximum degrees to rotate images randomly. Helps the model detect objects at different orientations. Start with small values (5-15) unless objects can be upside-down.",
        min_value=-180.0,
        max_value=180.0,
    ),
    ParameterDefinition(
        name="translate",
        category="augmentation",
        default=0.1,
        value_type="float",
        description="Translation fraction",
        help_text="Maximum fraction of image size to translate (shift) images randomly. Helps with detecting objects that may be near image edges. Fraction of image dimension.",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="scale",
        category="augmentation",
        default=0.5,
        value_type="float",
        description="Scale augmentation",
        help_text="Gain factor for scaling (1 +/- scale). 0.5 means images scaled 0.5x to 1.5x. Essential for detecting objects at different distances/sizes.",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="shear",
        category="augmentation",
        default=0.0,
        value_type="float",
        description="Shear degrees",
        help_text="Maximum degrees for shear transformation (slanting). Distorts images along an axis. Usually not needed unless objects can appear sheared (e.g., perspective views).",
        min_value=-180.0,
        max_value=180.0,
    ),
    ParameterDefinition(
        name="perspective",
        category="augmentation",
        default=0.0,
        value_type="float",
        description="Perspective distortion",
        help_text="Strength of random perspective transformation. Simulates viewing objects from different angles. Useful for applications with varying camera viewpoints.",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="flipud",
        category="augmentation",
        default=0.0,
        value_type="float",
        description="Vertical flip probability",
        help_text="Probability of flipping images vertically (upside-down). Only enable if objects can naturally appear upside-down in your use case.",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="fliplr",
        category="augmentation",
        default=0.5,
        value_type="float",
        description="Horizontal flip probability",
        help_text="Probability of flipping images horizontally (mirror). Almost always useful (0.5 is standard) unless objects have left/right asymmetry that matters.",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="mosaic",
        category="augmentation",
        default=1.0,
        value_type="float",
        description="Mosaic augmentation probability",
        help_text="Probability of combining 4 images into a mosaic. Excellent for teaching models about objects near edges and small object detection. Strongly recommended (1.0).",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="mixup",
        category="augmentation",
        default=0.0,
        value_type="float",
        description="MixUp augmentation probability",
        help_text="Probability of blending two images together. Helps with model regularization and handling occlusions. Can hurt performance if dataset is small.",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="copy_paste",
        category="augmentation",
        default=0.0,
        value_type="float",
        description="Copy-paste augmentation probability",
        help_text="Probability of copying objects from one image and pasting onto another. Great for instance segmentation and increasing object density in training images.",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="auto_augment",
        category="augmentation",
        default="randaugment",
        value_type="str",
        description="Auto augmentation policy",
        help_text="Automatic augmentation strategy to apply. 'randaugment' applies random compositions of basic augmentations. Set to empty string to disable.",
        allowed_values=["", "randaugment", "autoaugment", "augmix"],
    ),
    ParameterDefinition(
        name="erasing",
        category="augmentation",
        default=0.4,
        value_type="float",
        description="Random erasing probability",
        help_text="Probability of randomly erasing (covering) rectangular regions of images. Forces model to learn from partial object views. Common in classification.",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="close_mosaic",
        category="augmentation",
        default=10,
        value_type="int",
        description="Disable mosaic last N epochs",
        help_text="Number of final epochs to disable mosaic augmentation. Disabling before end helps stabilize training and can improve final metrics. Set to 0 to keep mosaic until end.",
        min_value=0,
        max_value=100,
    ),
    # Loss Parameters
    ParameterDefinition(
        name="box",
        category="loss",
        default=7.5,
        value_type="float",
        description="Box loss gain",
        help_text="Weight for bounding box regression loss. Higher values prioritize accurate localization. Standard values are 7.5 for YOLOv8/11/12 and 7.5 for YOLO26.",
        min_value=0.0,
        max_value=20.0,
    ),
    ParameterDefinition(
        name="cls",
        category="loss",
        default=0.5,
        value_type="float",
        description="Classification loss gain",
        help_text="Weight for classification loss. Higher values prioritize correct class prediction over localization. Usually kept lower than box loss.",
        min_value=0.0,
        max_value=10.0,
    ),
    ParameterDefinition(
        name="dfl",
        category="loss",
        default=1.5,
        value_type="float",
        description="Distribution Focal Loss gain",
        help_text="Weight for Distribution Focal Loss used in bounding box regression. Only applies to models using DFL. Helps with precise bounding box edges.",
        min_value=0.0,
        max_value=10.0,
    ),
    # Advanced Parameters
    ParameterDefinition(
        name="amp",
        category="advanced",
        default=True,
        value_type="bool",
        description="Automatic Mixed Precision",
        help_text="Enable Automatic Mixed Precision (AMP) training. Uses FP16 for some operations to speed up training and reduce memory usage. Usually safe to leave enabled.",
    ),
    ParameterDefinition(
        name="pretrained",
        category="advanced",
        default=True,
        value_type="bool",
        description="Use pretrained weights",
        help_text="Whether to start from pretrained COCO weights or train from scratch. Pretrained weights significantly speed up convergence. Disable only for very different domains.",
    ),
    ParameterDefinition(
        name="deterministic",
        category="advanced",
        default=False,
        value_type="bool",
        description="Deterministic training",
        help_text="Force deterministic algorithms for reproducibility. May be slower but ensures same results across runs. Useful for debugging and research.",
    ),
    ParameterDefinition(
        name="seed",
        category="advanced",
        default=0,
        value_type="int",
        description="Random seed",
        help_text="Seed for random number generators. Set to any positive integer for reproducible results. 0 means random seed each run.",
        min_value=0,
        max_value=999999,
    ),
    ParameterDefinition(
        name="rect",
        category="advanced",
        default=False,
        value_type="bool",
        description="Rectangular training",
        help_text="Use rectangular inference/training instead of square. Can improve speed when images have extreme aspect ratios. May affect accuracy slightly.",
    ),
    ParameterDefinition(
        name="save_period",
        category="advanced",
        default=-1,
        value_type="int",
        description="Save checkpoint every N epochs",
        help_text="Frequency of saving model checkpoints during training. -1 means only save final model. Positive values save intermediate models (useful for resuming or ensemble).",
        min_value=-1,
        max_value=1000,
    ),
    ParameterDefinition(
        name="fraction",
        category="advanced",
        default=1.0,
        value_type="float",
        description="Dataset fraction to use",
        help_text="Fraction of training dataset to use (0.0 to 1.0). Useful for quick experiments or when using a subset of a large dataset. 1.0 uses all data.",
        min_value=0.01,
        max_value=1.0,
    ),
    # Segmentation-specific Parameters
    ParameterDefinition(
        name="overlap_mask",
        category="segmentation",
        default=True,
        value_type="bool",
        description="Keep overlapping masks",
        help_text="For segmentation: whether to allow overlapping object masks (True) or merge them (False). True preserves individual object boundaries.",
    ),
    ParameterDefinition(
        name="mask_ratio",
        category="segmentation",
        default=4,
        value_type="int",
        description="Mask downsampling ratio",
        help_text="Downsampling factor for segmentation masks relative to image size. Higher values = smaller masks = less memory but coarser masks. 4 is standard (640px image -> 160px mask).",
        min_value=1,
        max_value=16,
    ),
    # Validation Parameters
    ParameterDefinition(
        name="val",
        category="validation",
        default=True,
        value_type="bool",
        description="Run validation during training",
        help_text="Whether to run validation after each epoch to monitor performance on the validation set. Disable to speed up training when you don't need intermediate metrics.",
    ),
    ParameterDefinition(
        name="conf",
        category="validation",
        default=0.001,
        value_type="float",
        description="Confidence threshold for validation",
        help_text="Minimum confidence score for detections during validation. Lower values (0.001) include more predictions for mAP calculation. Higher values filter low-confidence detections.",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="max_det",
        category="validation",
        default=300,
        value_type="int",
        description="Maximum detections per image",
        help_text="Maximum number of detections to consider per image during validation. Prevents memory issues in scenes with many objects.",
        min_value=1,
        max_value=10000,
    ),
    ParameterDefinition(
        name="plots",
        category="validation",
        default=True,
        value_type="bool",
        description="Generate validation plots",
        help_text="Whether to generate and save validation plots (PR curves, confusion matrix, etc.) after training. Requires matplotlib.",
    ),
    # Additional Advanced Parameters (from latest Ultralytics docs)
    ParameterDefinition(
        name="time",
        category="core",
        default=None,
        value_type="float",
        description="Maximum training time",
        help_text="Maximum training time in hours. If set, overrides epochs when time limit is reached first. Useful for time-constrained training jobs.",
        min_value=0.0,
        max_value=1000.0,
    ),
    ParameterDefinition(
        name="cos_lr",
        category="optimizer",
        default=False,
        value_type="bool",
        description="Cosine learning rate scheduler",
        help_text="Use cosine annealing learning rate scheduler. Smoothly decreases LR following cosine curve instead of linear decay. Often improves final results.",
    ),
    ParameterDefinition(
        name="nbs",
        category="loss",
        default=64,
        value_type="int",
        description="Nominal batch size",
        help_text="Nominal batch size for loss normalization. Used to normalize loss when using gradient accumulation or varying batch sizes. Standard is 64.",
        min_value=1,
        max_value=1024,
    ),
    ParameterDefinition(
        name="multi_scale",
        category="advanced",
        default=0.0,
        value_type="float",
        description="Multi-scale training",
        help_text="Enable multi-scale training by varying image size +/- this fraction during training. Helps model generalize to different object sizes. 0.0 disables, 0.1 varies by +/-10%.",
        min_value=0.0,
        max_value=1.0,
    ),
    ParameterDefinition(
        name="resume",
        category="advanced",
        default=False,
        value_type="bool",
        description="Resume training",
        help_text="Resume training from last checkpoint. Automatically finds the most recent run and continues from there. Useful after interruptions.",
    ),
    ParameterDefinition(
        name="single_cls",
        category="advanced",
        default=False,
        value_type="bool",
        description="Single class mode",
        help_text="Treat all classes as a single class. Useful when you want to detect objects without distinguishing between classes (object vs no-object).",
    ),
]

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logging.getLogger("src.config.generator").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@contextmanager
def _scoped_argv(prog: str) -> Iterator[None]:
    """Temporarily replace ``sys.argv`` so sub-commands' argparse calls see a
    clean argument vector instead of inheriting the TUI's own args.
    """
    saved_argv = sys.argv
    sys.argv = [prog]
    try:
        yield
    finally:
        sys.argv = saved_argv


def _safe_subcommand(
    label: str,
    target: Callable[..., Any],
    *,
    prog: str | None = None,
) -> None:
    """Run a submodule's ``main()`` from the TUI with unified error handling.

    The TUI invokes each command-line tool (training, prediction, TensorBoard,
    upload, dataset tools) by calling its ``main`` directly rather than
    shelling out. That keeps the session in one process but means any
    ``sys.exit``/``KeyboardInterrupt``/unexpected exception inside the
    sub-command would otherwise kill the TUI. This wrapper neutralises all of
    those, shows a panelled error, and always pauses for Enter so the user
    can read the output before the menu redraws.
    """
    clear_screen()
    entrypoint_name = prog or label.lower().replace(" ", "-")
    try:
        with _scoped_argv(entrypoint_name):
            target()
    except SystemExit as error:
        code = error.code
        if code not in (None, 0):
            console.print(
                Panel(
                    f"[bold yellow]{label} exited with status {code}.[/bold yellow]",
                    border_style="yellow",
                    padding=(1, 2),
                )
            )
    except KeyboardInterrupt:
        console.print(f"\n[bold yellow]{label} cancelled by user.[/bold yellow]")
    except MLDependencyError as error:
        console.print(
            Panel(
                f"[bold red]{label} cannot run — missing dependency:[/bold red]\n{error}\n\n"
                "[dim]Run `uv sync` (or re-install requirements) and try again.[/dim]",
                border_style="red",
                padding=(1, 2),
            )
        )
    except FileNotFoundError as error:
        console.print(
            Panel(
                f"[bold red]{label} failed — file not found:[/bold red] {error}",
                border_style="red",
                padding=(1, 2),
            )
        )
    except Exception as error:
        console.print(
            Panel(
                f"[bold red]{label} failed unexpectedly:[/bold red] {error}",
                border_style="red",
                padding=(1, 2),
            )
        )
        console.print(traceback.format_exc(), style="dim")
    finally:
        console.print()
        try:
            input("Press Enter to return to the main menu...")
        except (EOFError, KeyboardInterrupt):
            # User hit Ctrl+D/Ctrl+C at the pause prompt — just return.
            console.print()


def _render_dependency_table(statuses):
    """Build the themed comparison Table for a dependency-health report."""
    from src.utils.version_check import IMPORTANCE_STYLE, SEVERITY_META

    table = Table(
        title="Critical Dependencies",
        title_style="bold cyan",
        border_style="dim",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("Package", style="cyan", no_wrap=True)
    table.add_column("Role", style="dim")
    table.add_column("Importance", justify="center", no_wrap=True)
    table.add_column("Installed", justify="center", no_wrap=True)
    table.add_column("Latest", justify="center", no_wrap=True)
    table.add_column("Status", justify="left", no_wrap=True)

    for status in statuses:
        color, glyph, label = SEVERITY_META[status.severity]
        installed_display = status.installed or "—"
        latest_display = status.latest or (
            f"[dim]{status.error}[/dim]" if status.error else "—"
        )
        table.add_row(
            status.package.display_name,
            status.package.description,
            IMPORTANCE_STYLE[status.package.importance],
            installed_display,
            latest_display,
            f"[{color}]{glyph} {label}[/{color}]",
        )
    return table


def _render_dependency_summary(statuses):
    """High-level banner Panel that calls out the most urgent problem."""
    missing = [s for s in statuses if s.severity == "missing"]
    major = [s for s in statuses if s.severity == "major"]
    minor = [s for s in statuses if s.severity == "minor"]
    patch = [s for s in statuses if s.severity == "patch"]
    unknown = [s for s in statuses if s.severity == "unknown"]

    if missing:
        names = ", ".join(s.package.name for s in missing)
        body = (
            f"[bold red]{len(missing)} critical package(s) missing:[/bold red] {names}\n"
            "[dim]Run `uv sync` (or `pip install -r requirements.txt`) to install them.[/dim]"
        )
        border = "red"
    elif major:
        names = ", ".join(s.package.name for s in major)
        body = (
            f"[bold red]{len(major)} major update(s) available — breaking changes possible:[/bold red] {names}\n"
            "[dim]Review release notes before upgrading.[/dim]"
        )
        border = "red"
    elif minor or patch:
        updates = minor + patch
        names = ", ".join(s.package.name for s in updates)
        body = f"[bold yellow]{len(updates)} non-breaking update(s) available:[/bold yellow] {names}"
        border = "yellow"
    elif unknown:
        names = ", ".join(s.package.name for s in unknown)
        body = (
            f"[bold]PyPI unreachable for {len(unknown)} package(s):[/bold] {names}\n"
            "[dim]Check your network connection and try Refresh.[/dim]"
        )
        border = "dim"
    else:
        body = "[bold green]All tracked dependencies are up to date.[/bold green]"
        border = "green"

    return Panel(body, border_style=border, padding=(1, 2), box=box.ROUNDED)


def _run_pip_upgrade(packages):
    """Install --upgrade for the given packages; return True on success."""
    import subprocess

    pip_command = [sys.executable, "-m", "pip", "install", "--upgrade", *packages]
    with console.status(
        f"[bold]Upgrading {len(packages)} package(s)...", spinner="dots"
    ):
        result = subprocess.run(pip_command, capture_output=True, text=True)

    if result.returncode == 0:
        console.print(
            Panel(
                f"[bold green]Successfully upgraded:[/bold green] "
                f"{', '.join(packages)}\n\n"
                "[dim]`uv sync` may re-pin the previous versions from uv.lock — "
                "regenerate the lockfile to make these upgrades permanent.[/dim]",
                border_style="green",
                padding=(1, 2),
                box=box.ROUNDED,
            )
        )
        return True

    error_output = (result.stderr or result.stdout or "No error output").strip()
    console.print(
        Panel(
            f"[bold red]Upgrade failed (exit {result.returncode}).[/bold red]\n\n"
            f"[dim]{error_output}[/dim]",
            border_style="red",
            padding=(1, 2),
            box=box.ROUNDED,
        )
    )
    return False


def check_for_updates():
    """Themed dependency-health view that matches the rest of the TUI.

    Runs in a loop so Refresh / post-upgrade re-checks stay on the same
    screen style; Back returns to the main menu. All failure modes are
    caught so an offline run never kills the TUI.
    """
    from src.utils.version_check import check_packages

    while True:
        try:
            clear_screen()
            print_stylized_header("Dependency Health Check")

            try:
                with console.status(
                    "[bold]Querying PyPI for the latest versions...",
                    spinner="dots",
                ):
                    statuses = check_packages()
            except Exception as error:
                console.print(
                    Panel(
                        f"[bold red]Dependency check failed:[/bold red] {error}",
                        border_style="red",
                        padding=(1, 2),
                        box=box.ROUNDED,
                    )
                )
                input("\nPress Enter to return to the main menu...")
                return

            console.print(_render_dependency_table(statuses))
            console.print()
            console.print(_render_dependency_summary(statuses))

            updatable = [s for s in statuses if s.needs_update]
            ultralytics_update = next(
                (s for s in updatable if s.package.name == "ultralytics"),
                None,
            )
            critical_updates = [
                s for s in updatable if s.package.importance == "critical"
            ]

            actions: list[str] = []
            descriptions: dict[str, str] = {}

            if ultralytics_update is not None:
                actions.append("Update Ultralytics Only")
                descriptions["Update Ultralytics Only"] = (
                    f"Upgrade ultralytics from {ultralytics_update.installed} to "
                    f"{ultralytics_update.latest} — the safest, most common choice."
                )
            if len(critical_updates) >= 2 or (
                critical_updates and ultralytics_update is None
            ):
                label = f"Update All Critical ({len(critical_updates)})"
                actions.append(label)
                descriptions[label] = (
                    "Upgrade every critical package with an available update "
                    "(ultralytics, torch, torchvision). Major-version bumps may "
                    "introduce breaking changes — review release notes afterwards."
                )
            if len(updatable) >= 1:
                label = f"Update All Tracked ({len(updatable)})"
                actions.append(label)
                descriptions[label] = (
                    "Upgrade every tracked package with an available update, "
                    "including optional and tooling dependencies."
                )

            actions.append("Refresh")
            descriptions["Refresh"] = (
                "Re-query PyPI and re-read installed versions. Useful after an "
                "upgrade or when PyPI was temporarily unreachable."
            )
            descriptions["Back"] = "Return to the main menu without making changes."

            choice = get_user_choice(
                actions,
                allow_back=True,
                title="Update Actions",
                text=(
                    "Pick an action to continue. "
                    "Recommended action is listed first when available."
                ),
                descriptions=descriptions,
                breadcrumbs=["YOLOmatic", "Dependency Health"],
            )

            if choice == "Back":
                return
            if choice == "Refresh":
                continue

            if choice == "Update Ultralytics Only":
                targets = ["ultralytics"]
            elif choice.startswith("Update All Critical"):
                targets = [s.package.name for s in critical_updates]
            elif choice.startswith("Update All Tracked"):
                targets = [s.package.name for s in updatable]
            else:
                return

            clear_screen()
            print_stylized_header("Applying Updates")
            _run_pip_upgrade(targets)
            console.print()
            input("Press Enter to re-check dependencies...")
            # Loop back and re-render the fresh state.

        except KeyboardInterrupt:
            console.print(
                "\n[bold yellow]Dependency check cancelled by user.[/bold yellow]"
            )
            return


# Removed clear_screen, now imported from src.utils.tui


def backup_config(config_file):
    """
    Create a backup of the existing configuration file if it exists.

    Args:
        config_file (str): Name of the configuration file to backup
    """
    config_path = os.path.join("configs", config_file)
    if os.path.exists(config_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"config_backup_{timestamp}.yaml"
        backup_path = os.path.join("configs", backup_name)

        try:
            with open(config_path, "r") as source, open(backup_path, "w") as target:
                target.write(source.read())
            console.print(f"✅ Created backup: {backup_name}", style="green")
        except Exception as e:
            console.print(f"⚠️ Failed to create backup: {str(e)}", style="yellow")


def display_configuration_summary(
    model_choice,
    dataset_name,
    config_file,
    dataset_info,
    profile_selection=None,
    profile_context=None,
):
    """Display a clean summary of the configuration"""
    # Load config
    config_path = os.path.join("configs", config_file)
    if not os.path.exists(config_path):
        console.print("[red]Error: Config file not found![/red]")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Detect device
    device = "💻 CPU"
    try:
        torch = import_torch()
        if torch.cuda.is_available():
            device = "🚀 GPU (CUDA)"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "🚀 GPU (MPS)"
    except Exception:
        pass

    # Use the new summary panel for a cleaner look
    fields = {
        "Model": model_choice,
        "Dataset": dataset_name,
        "Device": device,
        "Config File": config_file,
    }

    if "nas" in model_choice.lower():
        training = config.get("training", {})
        fields.update(
            {
                "Batch Size": training.get("batch_size", "N/A"),
                "Max Epochs": training.get("max_epochs", "N/A"),
                "Workers": training.get("num_workers", "N/A"),
            }
        )
    else:
        training = config.get("training", {})
        fields.update(
            {
                "Batch Size": training.get("batch", "N/A"),
                "Epochs": training.get("epochs", "N/A"),
                "Image Size": training.get("imgsz", "N/A"),
                "Workers": training.get("workers", "N/A"),
            }
        )

    # Indicate if this is a fully customized config
    if profile_selection and profile_selection.get("mode") == "fully_customized":
        fields["Config Mode"] = "Fully Customized"

    render_summary_panel("Configuration Summary", fields)

    # Simplified dataset paths display
    path_rows = []
    if "nas" in model_choice.lower():
        structure = config.get("dataset", {}).get("structure", {})
        base_dir = config.get("dataset", {}).get("base_dir", "")
        path_rows = [
            [
                "Train",
                os.path.join(base_dir, structure.get("train", {}).get("images", "N/A")),
            ],
            [
                "Validation",
                os.path.join(base_dir, structure.get("valid", {}).get("images", "N/A")),
            ],
            [
                "Test",
                os.path.join(base_dir, structure.get("test", {}).get("images", "N/A")),
            ],
        ]
    else:
        model_config = config.get("model", {})
        data_dir = model_config.get("data_dir", "")
        path_rows = [
            [
                "Train",
                os.path.join(data_dir, model_config.get("train_images_dir", "N/A")),
            ],
            [
                "Validation",
                os.path.join(data_dir, model_config.get("val_images_dir", "N/A")),
            ],
            [
                "Test",
                os.path.join(data_dir, model_config.get("test_images_dir", "N/A")),
            ],
        ]

    render_table("Dataset Paths", ["Type", "Path"], path_rows, title_style="bold blue")


def display_paths_info(dataset_info):
    """Display dataset paths in a clean format"""
    console = Console()

    paths_table = Table(
        title="Dataset Paths", title_style="bold green", box=box.ROUNDED
    )
    paths_table.add_column("Type", style="cyan")
    paths_table.add_column("Path", style="white")

    paths_table.add_row("Train", dataset_info.get("train_path", "N/A"))
    paths_table.add_row("Validation", dataset_info.get("valid_path", "N/A"))
    paths_table.add_row("Test", dataset_info.get("test_path", "N/A"))

    console.print("\n")
    console.print(paths_table)


def list_datasets():
    datasets_folder = "datasets"
    if not os.path.exists(datasets_folder):
        os.makedirs(datasets_folder)
        console.print(
            f"✨ '{datasets_folder}' folder created. Please add COCO or any other compatible dataset into it.",
            style="bold yellow",
        )
        return None

    datasets = list_dataset_directories(datasets_folder)

    if not datasets:
        console.print(
            f"❌ No datasets found in '{datasets_folder}' folder.", style="bold red"
        )
        return None

    table = Table(title="Available Datasets", title_style="bold green")
    table.add_column("Dataset Name", justify="center", style="cyan")
    table.add_column("Size", justify="center", style="cyan")

    for dataset in datasets:
        table.add_row(dataset["name"], dataset["size"])

    console.print(table)

    dataset_names = [Path(d["path"]) for d in datasets]
    name_to_path = {path.name: str(path) for path in dataset_names}

    dataset_descriptions = {
        d["name"]: f"Select dataset '{d['name']}' ({d['size']}) located at {d['path']}"
        for d in datasets
    }
    dataset_descriptions["Back"] = "Return to the previous menu."

    choice = get_user_choice(
        list(name_to_path.keys()),  # Show basename in menu
        allow_back=True,
        title="Select Dataset",
        text="Use ↑↓ keys to navigate, Enter to select, 'b' for back:",
        descriptions=dataset_descriptions,
    )

    return name_to_path.get(choice) if choice != "Back" else choice


# Removed print_model_info, now handled by src.utils.tui


def format_timestamp():
    """Generate a formatted timestamp for config files."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_summary(model_choice, dataset_choice):
    """
    Display a summary of the selected configuration using rich.
    """
    clear_screen()
    print_stylized_header("Configuration Summary")

    table = Table(title="Selected Configuration", title_style="bold green")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Model", model_choice)
    table.add_row("Dataset", dataset_choice)
    table.add_row("Timestamp", format_timestamp())

    console.print(table)


def detect_device():
    """
    Detect the appropriate device for training based on system capabilities.
    Returns 'cuda' for NVIDIA GPUs, 'mps' for Apple Silicon, or 'cpu' as fallback.
    """
    try:
        torch = import_torch()
    except MLDependencyError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def update_config(model_choice, dataset_choice):
    """Update the configuration file with the selected model and dataset."""
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)

    config_dir = "configs"
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    dataset_path = Path(dataset_choice)
    dataset_name = dataset_path.name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_dataset_name = dataset_name.replace(" ", "_").lower()

    # Initialize appropriate generator
    if "nas" in model_choice.lower():
        generator = YOLONASConfigGenerator(str(dataset_path))
        config_file = f"{model_choice}_{safe_dataset_name}_{timestamp}.yaml"
        profile_context = None
        profile_selection = None
    else:
        generator = YOLOConfigGenerator(str(dataset_path))
        config_file = f"{model_choice}_{safe_dataset_name}_{timestamp}.yaml"
        profile_context = generator.get_regular_yolo_profile_context(model_choice)
        profile_selection = None

    # Check dataset type compatibility
    dataset_type = generator.dataset_info.get("task_type", "unknown")
    is_seg_model = "-seg" in model_choice.lower()

    # Determine if there's a mismatch
    mismatch_type = None
    if (
        dataset_type == "segmentation"
        and not is_seg_model
        and "nas" not in model_choice.lower()
    ):
        mismatch_type = "seg_model_needed"
    elif dataset_type == "detection" and is_seg_model:
        mismatch_type = "det_model_needed"
    elif dataset_type == "unknown":
        mismatch_type = "unknown"

    if mismatch_type:
        classes = generator.dataset_info.get("classes", []) or []
        num_classes = len(classes)
        dataset_path_display = str(generator.dataset_path)

        if mismatch_type == "seg_model_needed":
            title = "Dataset / Model Mismatch"
            model_kind = "detection"
            detected_label_format = "Segmentation polygons (7+ odd values per line)"
            expected_by_model = "Bounding boxes (5 values per line)"
            recommended_model = f"{model_choice}-seg"
            summary = (
                f"[yellow]Your model expects boxes, but your labels are polygons.[/yellow] "
                f"[bold]{model_choice}[/bold] is a detection model — it only predicts "
                f"[bold](x, y, w, h)[/bold] — while the label files under "
                f"[cyan]{dataset_name}[/cyan] contain segmentation polygons. "
                f"Training will discard the mask data and, in most cases, train "
                f"a detector that is weaker than one started from box labels directly."
            )
            continue_detail = (
                "[bold yellow]Train the detection model on polygon labels anyway.[/bold yellow]\n\n"
                f"• [cyan]{model_choice}[/cyan] will collapse each polygon to its "
                "bounding box at load time.\n"
                "• Mask granularity and per-pixel precision are [red]lost[/red] — "
                "the trained model predicts boxes, not masks.\n"
                "• Pick this only if you actually want box outputs from polygon "
                "labels and accept the signal loss."
            )
            change_model_detail = (
                "[bold green]Go back and pick a segmentation model.[/bold green]  [dim](recommended)[/dim]\n\n"
                f"• Recommended: [green]{recommended_model}[/green] — same backbone, "
                "adds a mask head that consumes polygons directly.\n"
                "• Any [green]-seg[/green] variant (yolo26-seg, yolov11-seg, "
                "yolov9-seg, yolov8-seg) works on your current labels without "
                "any dataset changes.\n"
                "• Fastest path when the dataset is already annotated as polygons."
            )
            fix_dataset_detail = (
                "[bold cyan]Keep the detection model and flatten polygons to boxes first.[/bold cyan]\n\n"
                "• Convert each polygon to its tight bounding box (ultralytics "
                "CLI, or a short script that reads the label .txt files).\n"
                "• Update [dim]data.yaml[/dim] so labels are 5 values per line.\n"
                "• Re-run YOLOmatic once the labels are in detection format."
            )
            tip = (
                "If you're unsure which way to go, [bold]Choose a Different Model[/bold] "
                "is safest — it keeps your polygon data intact and just swaps the "
                "model head."
            )
        elif mismatch_type == "det_model_needed":
            title = "Dataset / Model Mismatch"
            model_kind = "segmentation"
            detected_label_format = "Bounding boxes (5 values per line)"
            expected_by_model = "Segmentation polygons (7+ odd values per line)"
            recommended_model = model_choice.replace("-seg", "")
            summary = (
                f"[yellow]Your model expects polygons, but your labels are boxes.[/yellow] "
                f"[bold]{model_choice}[/bold] is a segmentation model — its mask head "
                "needs polygon vertices to learn boundaries — while the label files "
                f"under [cyan]{dataset_name}[/cyan] contain only bounding boxes. "
                "Segmentation training on box-only data typically fails outright or "
                "produces invalid masks."
            )
            continue_detail = (
                "[bold red]Train the segmentation model on box-only labels anyway.[/bold red]\n\n"
                "• The mask head has no polygon data to learn from — training "
                "usually [red]crashes at loss computation[/red] or yields "
                "unusable mask predictions.\n"
                "• Only pick this if you are deliberately debugging "
                "Ultralytics' segmentation pipeline."
            )
            change_model_detail = (
                "[bold green]Go back and pick the detection variant.[/bold green]  [dim](recommended)[/dim]\n\n"
                f"• Recommended: [green]{recommended_model}[/green] — same family, "
                "drops the mask head so it works on box labels as-is.\n"
                "• Any non-seg variant works on your current labels without "
                "any dataset changes.\n"
                "• Fastest path when your labels are already boxes."
            )
            fix_dataset_detail = (
                "[bold cyan]Keep the segmentation model and re-annotate with polygons.[/bold cyan]\n\n"
                "• Boxes cannot be 'upgraded' to masks — you need real per-pixel "
                "annotation.\n"
                "• Use Roboflow, CVAT, or Label Studio to trace polygons over "
                "each instance.\n"
                "• Re-run YOLOmatic once labels have 7+ odd values per line "
                "(class id + polygon points)."
            )
            tip = (
                "If you just need a working training run, [bold]Choose a Different Model[/bold] "
                "is the cheapest fix — re-annotating a dataset can take hours to days."
            )
        else:  # unknown
            title = "Dataset Format Not Recognized"
            model_kind = "unknown"
            detected_label_format = "Could not determine (labels missing or empty?)"
            expected_by_model = "YOLO .txt labels (boxes or polygons)"
            recommended_model = "—"
            summary = (
                "[yellow]YOLOmatic scanned the dataset but could not classify the "
                "label format.[/yellow] Either the label files are missing, empty, "
                "or in a format YOLO does not recognize (for example, still in "
                "COCO JSON form).\n\n"
                f"Scanned path: [cyan]{dataset_path_display}[/cyan]\n"
                "Expected layout: one [bold].txt[/bold] per image with [bold]5 values[/bold] "
                "per line (boxes) or [bold]7+ odd values[/bold] per line (polygons)."
            )
            continue_detail = (
                "[bold red]Proceed without a confirmed label format.[/bold red]\n\n"
                "• Ultralytics will attempt to train, but the run will "
                "[red]fail at data loading[/red] if no labels are actually present.\n"
                "• Only pick this if you know the labels exist somewhere "
                "YOLOmatic's detector didn't check."
            )
            change_model_detail = (
                "[bold cyan]Return to the model picker.[/bold cyan]\n\n"
                "• Useful if you want to try a different model family while you "
                "investigate the dataset in parallel.\n"
                "• Note: the mismatch will re-appear on the next run until "
                "the labels are fixed."
            )
            fix_dataset_detail = (
                "[bold green]Stop and fix the dataset first.[/bold green]  [dim](recommended)[/dim]\n\n"
                "• Confirm [dim]train/labels/*.txt[/dim] exist and are non-empty.\n"
                "• Check [dim]data.yaml[/dim] `train:` / `val:` paths resolve to "
                "the right folders.\n"
                "• If labels are COCO JSON, convert them to YOLO .txt first.\n"
                "• Re-run YOLOmatic once the label format is in place."
            )
            tip = (
                "Most common cause: [bold]data.yaml[/bold] paths point to a folder that "
                "exists but contains no .txt files, or images live in one folder and "
                "labels in another with mismatched names."
            )

        status_fields: dict[str, str] = {
            "Selected Model": f"{model_choice} ({model_kind})",
            "Dataset": f"{dataset_name} ({num_classes} classes)",
            "Labels Found": detected_label_format,
            "Model Expects": expected_by_model,
        }
        if recommended_model != "—":
            status_fields["Suggested Model"] = recommended_model

        options = [
            "Continue Anyway",
            "Choose a Different Model",
            "Fix Dataset First",
        ]
        descriptions = {
            "Continue Anyway": continue_detail,
            "Choose a Different Model": change_model_detail,
            "Fix Dataset First": fix_dataset_detail,
        }

        choice = get_user_choice(
            options,
            title=title,
            text=summary,
            descriptions=descriptions,
            breadcrumbs=["YOLOmatic", "Configure Model", "Dataset Type Check"],
            status_fields=status_fields,
            tip=tip,
        )

        if choice != "Continue Anyway":
            return False
        # Continue with "Continue Anyway"

    custom_params = None
    if "nas" not in model_choice.lower():
        profile_selection = choose_regular_yolo_profiles(
            dataset_name,
            profile_context,
            model_choice,
        )
        if profile_selection is None:
            return False

        # Handle fully customized mode
        if profile_selection.get("mode") == "fully_customized":
            result = run_fully_customized_config_flow(
                dataset_name, model_choice, profile_context
            )
            if result is None:
                return False
            custom_params = result.get("params", {})
            # Generate base config and apply custom params directly
            config = generator.generate_config(
                model_choice,
                dict(profile_context["recommended_profiles"]),
                profile_context,
            )
            # Override with custom parameters
            config["training"].update(custom_params)
        else:
            display_regular_yolo_profile_selection_summary(
                dataset_name,
                profile_selection,
                profile_context,
            )
            config = generator.generate_config(
                model_choice,
                profile_selection,
                profile_context,
            )
    else:
        config = generator.generate_config(model_choice)

    # Save new config
    config_path = os.path.join(config_dir, config_file)
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    console.print(f"✅ Configuration saved to: {config_file}", style="bold green")

    # Display summary
    display_configuration_summary(
        model_choice,
        dataset_name,
        config_file,
        generator.dataset_info,
        profile_selection,
        profile_context,
    )
    display_paths_info(generator.dataset_info)

    return True


def format_profile_name(value: str) -> str:
    return value.replace("_", " ").title()


def build_hint_block(title: str, lines: list[str]) -> str:
    if not lines:
        return ""
    formatted_lines = "\n".join(f"- {line}" for line in lines)
    return f"{title}:\n{formatted_lines}"


def build_regular_yolo_profile_summary_text(
    dataset_name: str,
    profile_context: dict[str, Any],
    model_choice: str,
) -> str:
    model_metrics = profile_context["model_metrics"]
    dataset_metrics = profile_context["dataset_metrics"]
    system_metrics = profile_context["system_metrics"]
    recommended_profiles = profile_context["recommended_profiles"]
    recommended_worker = profile_context["worker_profiles"][
        recommended_profiles["worker"]
    ]
    worker_reason = profile_context.get("worker_recommendation_reason", "")

    lines = [
        f"Dataset: {dataset_name}",
        (
            "Model scan: "
            f"{model_choice} "
            f"({format_profile_name(model_metrics['heaviness'])} model)"
        ),
        (
            "Dataset scan: "
            f"{format_size(int(dataset_metrics['total_size_bytes']))}, "
            f"{int(dataset_metrics['image_count'])} images, "
            f"{int(dataset_metrics['label_count'])} labels, "
            f"{int(dataset_metrics['total_file_count'])} files"
        ),
        (
            "System scan: "
            f"{format_size(int(system_metrics['available_ram_bytes']))} RAM free, "
            f"{int(system_metrics['cpu_count'])} CPU cores, "
            f"device={system_metrics['device']}"
        ),
    ]

    if system_metrics["available_gpu_memory_bytes"] is not None:
        gpu_line = f"GPU memory free: {format_size(int(system_metrics['available_gpu_memory_bytes']))}"
        if system_metrics["total_gpu_memory_bytes"] is not None:
            gpu_line += f" of {format_size(int(system_metrics['total_gpu_memory_bytes']))} total"
        lines.append(gpu_line)

    if model_metrics["params_millions"] is not None:
        lines.append(f"Model params: {model_metrics['params_millions']:.1f}M")
    if model_metrics["flops_billions"] is not None:
        lines.append(f"Model FLOPs: {model_metrics['flops_billions']:.1f}B")

    lines.extend(
        [
            "",
            "YOLOmatic recommendation factors:",
            "- model heaviness from variant size, params, and FLOPs",
            "- dataset size, image count, label count, and file count",
            "- available RAM, CPU cores, and detected device",
            "- free GPU memory when CUDA is available",
            "",
            "Recommended profiles:",
            (
                f"- Augmentation: {format_profile_name(recommended_profiles['augmentation'])}"
            ),
            f"- Compute: {format_profile_name(recommended_profiles['compute'])}",
            (
                "- Workers: "
                f"{format_profile_name(recommended_profiles['worker'])} "
                f"({int(recommended_worker['workers'])} workers)"
            ),
        ]
    )
    if worker_reason:
        lines.append(f"- Worker rationale: {worker_reason}")
    lines.extend(
        [
            "",
            "Worker guidance:",
            "- More workers are a throughput setting, not a quality setting.",
            "- Too many workers can reduce stability through CPU contention, RAM pressure, storage thrashing, and noisier batch timing.",
            "- If validation metrics drop after raising workers, move back toward Light or Medium.",
        ]
    )
    return "\n".join(lines)


def select_profile_option(
    title: str,
    prompt_text: str,
    option_descriptions: dict[str, str],
    recommended_key: str,
    hint_lines: list[str] | None = None,
) -> str | None:
    option_map: dict[str, str] = {}
    option_labels: list[str] = []

    hint_block = ""
    if hint_lines:
        hint_block = f"\n\n{build_hint_block('Hints', hint_lines)}"

    descriptions: dict[str, str] = {}
    for key, description in option_descriptions.items():
        label = format_profile_name(key)
        if key == recommended_key:
            label = f"{label} [recommended]"
        option_map[label] = key
        option_labels.append(label)
        descriptions[label] = description

    descriptions["Back"] = "Return to the previous configuration step."

    choice = get_user_choice(
        option_labels,
        allow_back=True,
        title=title,
        text=f"{prompt_text}{hint_block}",
        descriptions=descriptions,
    )
    if choice == "Back":
        return None
    return option_map[choice]


def display_regular_yolo_profile_selection_summary(
    dataset_name: str,
    profile_selection: dict[str, str],
    profile_context: dict[str, Any],
) -> None:
    model_metrics = profile_context["model_metrics"]
    dataset_metrics = profile_context["dataset_metrics"]
    system_metrics = profile_context["system_metrics"]
    worker_profile = profile_context["worker_profiles"][profile_selection["worker"]]

    clear_screen()
    print_stylized_header("Config Profile Summary")

    table = Table(title="Selected Profile Settings", title_style="bold green")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Dataset", dataset_name)
    table.add_row("Model", str(model_metrics["model_choice"]))
    table.add_row(
        "Model Heaviness",
        format_profile_name(str(model_metrics["heaviness"])),
    )
    if model_metrics["params_millions"] is not None:
        table.add_row("Model Params", f"{model_metrics['params_millions']:.1f}M")
    if model_metrics["flops_billions"] is not None:
        table.add_row("Model FLOPs", f"{model_metrics['flops_billions']:.1f}B")
    table.add_row(
        "Dataset Size",
        format_size(int(dataset_metrics["total_size_bytes"])),
    )
    table.add_row("Image Count", str(int(dataset_metrics["image_count"])))
    table.add_row("Label Count", str(int(dataset_metrics["label_count"])))
    table.add_row(
        "Available RAM",
        format_size(int(system_metrics["available_ram_bytes"])),
    )
    table.add_row("Detected Device", str(system_metrics["device"]))
    if system_metrics["available_gpu_memory_bytes"] is not None:
        table.add_row(
            "Available GPU Memory",
            format_size(int(system_metrics["available_gpu_memory_bytes"])),
        )
    table.add_row(
        "Augmentation Profile",
        format_profile_name(profile_selection["augmentation"]),
    )
    table.add_row(
        "Compute Profile",
        format_profile_name(profile_selection["compute"]),
    )
    table.add_row(
        "Worker Profile",
        (
            f"{format_profile_name(profile_selection['worker'])} "
            f"({int(worker_profile['workers'])} workers)"
        ),
    )
    table.add_row("Worker Notes", str(worker_profile["description"]))
    if profile_context.get("worker_recommendation_reason"):
        table.add_row(
            "Worker Rationale",
            str(profile_context["worker_recommendation_reason"]),
        )
    table.add_row(
        "Augmentation Impact",
        "Controls how many augmentation keys YOLOmatic enables in training",
    )
    table.add_row(
        "Compute Impact",
        "Controls batch aggressiveness and whether cache is enabled when safe",
    )
    table.add_row(
        "Workers Impact",
        "Controls dataloader parallelism; more workers can raise throughput, but they can also hurt stability if RAM, CPU, or storage become bottlenecks",
    )

    console.print(table)


def choose_regular_yolo_profiles(
    dataset_name: str,
    profile_context: dict[str, Any],
    model_choice: str,
) -> dict[str, str] | None:
    summary_text = build_regular_yolo_profile_summary_text(
        dataset_name,
        profile_context,
        model_choice,
    )
    recommended_profiles = profile_context["recommended_profiles"]

    start_option_map = {
        "Recommended": "recommended",
        "Customize": "customize",
        "Fully Customized": "fully_customized",
    }
    start_descriptions = {
        "Recommended": "Fastest path - let YOLOmatic heuristics decide augmentation, compute, and worker settings for you.",
        "Customize": "Manual path - review and choose your own augmentation intensity, compute aggressiveness, and worker counts.",
        "Fully Customized": "Expert path - individually select and configure every training parameter with detailed explanations.",
        "Back": "Return to dataset selection.",
    }

    hint_block = build_hint_block(
        "Hints",
        [
            "Use the recommended option unless you already know you need more or less augmentation.",
            "Compute controls how hard YOLOmatic pushes memory and throughput.",
            "Workers control dataloader parallelism and can increase RAM pressure.",
        ],
    )

    initial_choice = get_user_choice(
        list(start_option_map.keys()),
        allow_back=True,
        title="Regular YOLO Config Profiles",
        text=(
            f"{summary_text}\n\n"
            "Pick the fast path if you want the current codebase heuristics to decide for you. "
            f"Pick customize if you want to review each area manually.\n\n{hint_block}"
        ),
        descriptions=start_descriptions,
    )

    if initial_choice == "Back":
        return None
    if start_option_map[initial_choice] == "recommended":
        return dict(recommended_profiles)
    if start_option_map[initial_choice] == "fully_customized":
        return {"mode": "fully_customized"}

    augmentation_options = {
        "minimum": "Essential training values only with almost no extra augmentation",
        "low": "Mild augmentation using flips, mosaic, and mixup",
        "medium": "Stronger generalization with color and geometric augmentation",
    }
    compute_options = {
        "conservative": "Safer memory usage and lower risk of instability",
        "balanced": "Best default for most systems and datasets",
        "aggressive": "Pushes throughput harder when RAM and GPU headroom are strong",
    }
    worker_options = {
        key: f"{int(details['workers'])} workers - {details['description']}"
        for key, details in profile_context["worker_profiles"].items()
    }

    augmentation_choice = select_profile_option(
        "Select Augmentation Profile",
        f"{summary_text}\n\nChoose the augmentation intensity for this dataset:",
        augmentation_options,
        recommended_profiles["augmentation"],
        [
            "Minimum is the easiest to reason about and keeps the config close to core training values.",
            "Low adds only basic robustness improvements.",
            "Medium adds more color and geometric changes, which can improve generalization but also change training behavior more.",
        ],
    )
    if augmentation_choice is None:
        return None

    compute_choice = select_profile_option(
        "Select Compute Profile",
        f"{summary_text}\n\nChoose how strongly YOLOmatic should push system resources:",
        compute_options,
        recommended_profiles["compute"],
        [
            "This profile mainly affects batch aggressiveness and cache behavior.",
            "Conservative is better when GPU memory is tight or the model is heavy.",
            "Aggressive is best only when your RAM, GPU memory, and dataset pressure all look healthy.",
        ],
    )
    if compute_choice is None:
        return None

    worker_choice = select_profile_option(
        "Select Worker Profile",
        f"{summary_text}\n\nChoose the dataloader worker profile:",
        worker_options,
        recommended_profiles["worker"],
        [
            "Workers change throughput, not the optimization target, so higher values are not automatically better.",
            "Too many workers can reduce training quality indirectly by causing CPU contention, RAM pressure, disk thrashing, and less stable batch preparation.",
            "If you are unsure, keep the recommended worker profile and only raise it when the GPU is starved and the machine still has clear headroom.",
        ],
    )
    if worker_choice is None:
        return None

    return {
        "augmentation": augmentation_choice,
        "compute": compute_choice,
        "worker": worker_choice,
    }


def run_fully_customized_config_flow(
    dataset_name: str,
    model_choice: str,
    profile_context: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Interactive flow for fully customized parameter selection.

    Allows users to check/uncheck individual parameters and set custom values
    with detailed explanations for each parameter. Supports bi-directional
    navigation and quick-select for boolean/enum types.
    """
    from rich import box
    from rich.panel import Panel
    from rich.table import Table

    # State
    current_selected_names = {
        "epochs",
        "patience",
        "batch",
        "imgsz",
        "device",
        "workers",
        "optimizer",
    }
    custom_values: dict[str, Any] = {}
    param_lookup = {p.name: p for p in YOLO_TRAINING_PARAMETERS}

    while True:
        clear_screen()
        print_stylized_header("Fully Customized Configuration")
        console.print(
            Panel(
                "[bold yellow]Welcome to the Unified Configurator![/bold yellow]\n\n"
                "• [cyan]Left Pane[/cyan]: Select parameters with [bold yellow]Space[/bold yellow].\n"
                "• [cyan]Right Pane[/cyan]: Edit values with [bold yellow]Enter[/bold yellow] or [bold yellow]Right Arrow[/bold yellow].\n"
                "• [cyan]Navigation[/cyan]: Use [bold yellow]B[/bold yellow] or [bold yellow]Left Arrow[/bold yellow] to return to the list.\n"
                "• [cyan]Finish/Back[/cyan]: Press [bold yellow]F[/bold yellow] to finish or [bold yellow]Q[/bold yellow] to go back to the menu.",
                border_style="cyan",
                padding=(1, 2),
            )
        )

        result = get_user_multi_select(
            parameters=YOLO_TRAINING_PARAMETERS,
            title="Fully Customized Configuration",
            instruction="[Space] Toggle  [Enter/→] Edit Value  [F] Finish",
            pre_selected=current_selected_names,
            pre_values=custom_values,
        )

        if result is None:
            return None

        selected_names, updated_values = result
        current_selected_names = selected_names
        custom_values = updated_values

        if not selected_names:
            console.print(
                "[yellow]No parameters selected. Using defaults.[/yellow]"
            )
            return {"mode": "fully_customized", "params": {}}

        # Display summary
        clear_screen()
        print_stylized_header("Configuration Summary")

        table = Table(
            title=f"Selected Parameters ({len(selected_names)} configured)",
            title_style="bold green",
            border_style="dim",
            box=box.ROUNDED,
        )
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="yellow")
        table.add_column("Category", style="dim")

        # Only show parameters that are selected
        for name in sorted(selected_names):
            param = param_lookup.get(name)
            if param:
                val = custom_values.get(name, param.default)
                table.add_row(name, str(val), param.category)

        console.print(table)

        choice = get_user_choice(
            [
                "Confirm and Continue",
                "Go Back and Modify",
                "Back to Mode Selection",
            ],
            title="Confirm Configuration",
            text=(
                "Review the custom parameter table above. "
                "These values will be written into your training YAML."
            ),
            descriptions={
                "Confirm and Continue": (
                    "[bold green]Accept these values and write the training YAML.[/bold green]\n\n"
                    "• The generated config is saved to [cyan]configs/[/cyan] with a timestamp.\n"
                    "• You can still edit the YAML by hand before launching training."
                ),
                "Go Back and Modify": (
                    "[bold yellow]Return to the unified configurator.[/bold yellow]\n\n"
                    "• Your current selections and values are preserved."
                ),
                "Back to Mode Selection": (
                    "[bold cyan]Return to the configuration mode selection screen.[/bold cyan]\n\n"
                    "• All current custom configuration will be discarded."
                ),
            },
            tip="Anything you change later in the YAML will override what you picked here.",
        )

        if choice == "Confirm and Continue":
            # Filter custom_values to only include selected names
            final_params = {name: custom_values.get(name, param_lookup[name].default) 
                          for name in selected_names}
            return {"mode": "fully_customized", "params": final_params}
        elif choice == "Back to Mode Selection":
            return None
        # Else choice == "Go Back and Modify", loop back to the unified configurator

    return {"mode": "fully_customized", "params": custom_values}


def get_model_menu():
    """Get the list of available YOLO models grouped by category."""
    models = [
        "[Detection]",
        "yolo26",
        "yolov12",
        "yolov11",
        "yolov10",
        "yolov9",
        "yolov8",
        "yolox",
        "[Segmentation]",
        "yolo26-seg",
        "yolov12-seg",
        "yolov11-seg",
        "yolov9-seg",
        "yolov8-seg",
        "[Specialized]",
        "yolo_nas",
    ]
    return models


def main():
    while True:
        try:
            _main_loop_iteration()
        except KeyboardInterrupt:
            # Ctrl+C at the main menu exits cleanly instead of dumping a trace.
            clear_screen()
            console.print("\n[bold cyan]\U0001f44b Goodbye![/bold cyan]")
            return
        except _ExitTUI:
            clear_screen()
            console.print("\U0001f44b Goodbye!", style="bold cyan")
            return
        except Exception as error:
            # Last-resort safety net — report and re-enter the menu rather than
            # crashing the whole TUI.
            console.print(
                Panel(
                    f"[bold red]An unexpected error occurred:[/bold red] {error}",
                    border_style="red",
                    padding=(1, 2),
                )
            )
            console.print(traceback.format_exc(), style="dim")
            try:
                input("\nPress Enter to return to the main menu...")
            except (EOFError, KeyboardInterrupt):
                return


class _ExitTUI(Exception):
    """Raised internally when the user chooses Exit from the main menu."""


def _main_loop_iteration():
    while True:
        clear_screen()
        print_stylized_header("YOLO Model Selector")

        # Full workflow surface: configure, train, predict, monitor, publish,
        # and curate datasets — all routed through the same TUI.
        main_menu_options = [
            "[Configure & Train]",
            "Configure Model",
            "Train Model",
            "[Evaluate & Monitor]",
            "Run Prediction",
            "Launch TensorBoard",
            "[Datasets & Deployment]",
            "Combine Datasets",
            "Upload to Roboflow",
            "[Maintenance]",
            "Check for Updates",
            "About YOLOmatic",
            "Exit",
        ]

        main_choice = get_user_choice(
            main_menu_options,
            title="Main Menu",
            text="Pick a task to begin:",
            descriptions={
                "Configure Model": (
                    "Walk through the YOLOmatic wizard to pick a model family, choose a "
                    "variant that fits your hardware, and auto-generate a training YAML "
                    "tailored to your dataset and system resources."
                ),
                "Train Model": (
                    "Train (and validate + export) a YOLO or YOLO-NAS model using one of "
                    "the saved configs under ./configs. Routes YOLO-NAS to SuperGradients "
                    "and everything else to Ultralytics automatically."
                ),
                "Run Prediction": (
                    "Run inference on a single image or a folder of images using trained "
                    "weights discovered in the project root or runs/ directory."
                ),
                "Launch TensorBoard": (
                    "Open a TensorBoard dashboard against a specific run or the entire "
                    "runs/ directory. YOLOmatic back-fills metrics, artifacts, and sample "
                    "images automatically."
                ),
                "Combine Datasets": (
                    "Merge several YOLO datasets into a unified one — class names are "
                    "deduplicated, labels are remapped, and images are hard-linked where "
                    "possible for near-zero cost."
                ),
                "Upload to Roboflow": (
                    "Publish a trained checkpoint to Roboflow. Reads ROBOFLOW_API_KEY / "
                    "WORKSPACE / PROJECT_IDS from .env and stages the weight correctly for "
                    "Roboflow's deploy API."
                ),
                "Check for Updates": (
                    "Run a dependency health check across every critical package — "
                    "ultralytics, torch, torchvision, super-gradients, tensorboard, "
                    "roboflow, onnx, onnxruntime. Each is classified by severity "
                    "(patch / minor / major / missing), with one-click upgrades."
                ),
                "About YOLOmatic": "Technical details, creator info, and version history.",
                "Exit": "Safely exit the application.",
            },
            breadcrumbs=["YOLOmatic"],
        )

        if main_choice == "Exit":
            raise _ExitTUI()

        elif main_choice == "Check for Updates":
            check_for_updates()
            continue

        elif main_choice == "Train Model":
            from src.trainers.yolo_trainer import main as trainer_main

            _safe_subcommand("Training", trainer_main, prog="yolomatic-train")
            continue

        elif main_choice == "Run Prediction":
            from src.cli.predict import main as predict_main

            _safe_subcommand("Prediction", predict_main, prog="yolomatic-predict")
            continue

        elif main_choice == "Launch TensorBoard":
            from src.cli.tensorboard_launcher import main as tensorboard_main

            _safe_subcommand(
                "TensorBoard", tensorboard_main, prog="yolomatic-tensorboard"
            )
            continue

        elif main_choice == "Upload to Roboflow":
            from src.cli.upload import main as upload_main

            _safe_subcommand("Roboflow Upload", upload_main, prog="yolomatic-upload")
            continue

        elif main_choice == "Combine Datasets":
            from src.utils.combine_datasets import main as combine_main

            _safe_subcommand("Dataset Combiner", combine_main, prog="yolomatic-combine")
            continue

        elif main_choice == "About YOLOmatic":
            clear_screen()
            from src.__version__ import __version__

            # Use a more structured layout for the About screen
            about_table = Table.grid(padding=(0, 2))
            about_table.add_column(style="bold cyan", justify="right")
            about_table.add_column(style="white")

            about_table.add_row("Product:", "YOLOmatic")
            about_table.add_row("Version:", f"{__version__}")
            about_table.add_row("Creator:", "Shahab Bahreini Jangjoo")
            about_table.add_row("Contact:", "shahabahreini@hotmail.com")
            about_table.add_row("", "")
            about_table.add_row(
                "Description:", "A powerful CLI tool for automated YOLO training,"
            )
            about_table.add_row("", "configuration, and dataset management.")

            console.print("\n" * 2)
            console.print(
                Panel(
                    Align.center(about_table),
                    title="[bold cyan]About YOLOmatic[/bold cyan]",
                    border_style="cyan",
                    padding=(2, 4),
                    box=box.ROUNDED,
                )
            )
            console.print("\n")
            input("Press Enter to return to Main Menu...")
            continue

        elif main_choice == "Configure Model":
            # Get model choice
            model_types = get_model_menu()
            model_choice = get_user_choice(
                model_types,
                title="YOLO Model Selector",
                text="Choose a model family for your project:",
                allow_back=True,
                descriptions={
                    "yolo26": (
                        "[bold cyan]YOLO26[/bold cyan]  [green]● Latest — 2026[/green]\n\n"
                        "[bold]Architecture[/bold]\n"
                        "  • End-to-end NMS-free inference (no post-processing step)\n"
                        "  • DFL removed — simpler export, wider edge compatibility\n"
                        "  • MuSGD optimizer (hybrid SGD + Muon, inspired by LLM training)\n"
                        "  • ProgLoss + STAL loss for better small-object accuracy\n\n"
                        "[bold]Benchmarks[/bold]  [dim](COCO val2017, detection)[/dim]\n"
                        "  • mAP:    40.9 (nano)  →  57.5 (xlarge)\n"
                        "  • Params: 2.4M (nano)  →  55.7M (xlarge)\n"
                        "  • Speed:  1.7 ms T4 TensorRT (nano)  |  CPU ONNX not published\n\n"
                        "[bold]Best for[/bold]\n"
                        "  Edge devices, IoT, robotics, CPU-only and mobile deployments"
                    ),
                    "yolo26-seg": (
                        "[bold cyan]YOLO26-Seg[/bold cyan]  [green]● Latest — 2026[/green]\n\n"
                        "[bold]Architecture[/bold]\n"
                        "  • Same NMS-free, DFL-removed, MuSGD base as YOLO26\n"
                        "  • Instance segmentation head for pixel-level boundary detection\n"
                        "  • Edge-optimized — fast CPU ONNX inference among seg models\n\n"
                        "[bold]Benchmarks[/bold]  [dim](COCO val2017, segmentation)[/dim]\n"
                        "  • mAP box: 33.9 (nano)  →  47.0 (xlarge)\n"
                        "  • Params:  2.7M (nano)  →  62.8M (xlarge)\n"
                        "  • Speed:   53 ms CPU ONNX (nano)  |  2.1 ms T4 TensorRT (nano)\n\n"
                        "[bold]Best for[/bold]\n"
                        "  Pixel-level detection on resource-constrained or edge hardware"
                    ),
                    "yolov12": (
                        "[bold cyan]YOLOv12[/bold cyan]  [yellow]● Research — 2025[/yellow]\n\n"
                        "[bold]Architecture[/bold]\n"
                        "  • Area Attention mechanism — large receptive field, attention-based\n"
                        "  • R-ELAN (Residual Efficient Layer Aggregation Networks)\n"
                        "  • Optional FlashAttention for memory-efficient training\n"
                        "  • Higher peak accuracy than YOLO11 at cost of stability\n\n"
                        "[bold]Benchmarks[/bold]  [dim](COCO val2017, detection)[/dim]\n"
                        "  • mAP:    40.6 (nano)  →  55.2 (xlarge)\n"
                        "  • Params: 2.6M (nano)  →  59.1M (xlarge)\n"
                        "  • Speed:  1.64 ms T4 TensorRT (nano)  |  CPU ONNX not published\n\n"
                        "[bold]Recommendation[/bold]\n"
                        "  [bold red]Not recommended for production[/bold red] — training instability and "
                        "high GPU memory consumption. Use YOLO11 or YOLO26 for production."
                    ),
                    "yolov12-seg": (
                        "[bold cyan]YOLOv12-Seg[/bold cyan]  [yellow]● Research — 2025[/yellow]\n\n"
                        "[bold]Architecture[/bold]\n"
                        "  • Attention-centric YOLO12 base with segmentation head\n"
                        "  • Area Attention + R-ELAN backbone\n\n"
                        "[bold]Benchmarks[/bold]  [dim](COCO val2017)[/dim]\n"
                        "  • Mask mAP: not yet officially published by Ultralytics\n"
                        "  • Speed:    not yet officially published by Ultralytics\n\n"
                        "[bold]Recommendation[/bold]\n"
                        "  [bold red]Not recommended for production[/bold red] — inherits YOLO12 "
                        "instability. Use YOLO11-seg or YOLO26-seg instead."
                    ),
                    "yolov11": (
                        "[bold cyan]YOLOv11[/bold cyan]  [green]● Stable — 2024[/green]\n\n"
                        "[bold]Architecture[/bold]\n"
                        "  • Improved backbone and neck over YOLOv8\n"
                        "  • 22% fewer parameters than YOLOv8m with higher mAP\n"
                        "  • Supports all tasks: Detect, Segment, Classify, Pose, OBB\n\n"
                        "[bold]Benchmarks[/bold]  [dim](COCO val2017, detection)[/dim]\n"
                        "  • mAP:    39.5 (nano)  →  54.7 (xlarge)\n"
                        "  • Params: 2.6M (nano)  →  56.9M (xlarge)\n"
                        "  • Speed:  56 ms CPU ONNX (nano)  |  1.5 ms T4 TensorRT (nano)\n\n"
                        "[bold]Best for[/bold]\n"
                        "  Production, enterprise, mission-critical applications"
                    ),
                    "yolov11-seg": (
                        "[bold cyan]YOLOv11-Seg[/bold cyan]  [green]● Stable — 2024[/green]\n\n"
                        "[bold]Architecture[/bold]\n"
                        "  • YOLO11 backbone with segmentation head\n"
                        "  • Proven training stability across diverse datasets\n\n"
                        "[bold]Benchmarks[/bold]  [dim](COCO val2017, segmentation)[/dim]\n"
                        "  • mAP box:  38.9 (nano)  →  54.7 (xlarge)\n"
                        "  • mAP mask: 32.0 (nano)  →  43.8 (xlarge)\n"
                        "  • Params:   2.9M (nano)  →  62.1M (xlarge)\n"
                        "  • Speed:    66 ms CPU ONNX (nano)  |  2.9 ms T4 TensorRT (nano)\n\n"
                        "[bold]Best for[/bold]\n"
                        "  Production segmentation requiring reliability and full benchmark data"
                    ),
                    "yolov10": (
                        "[bold cyan]YOLOv10[/bold cyan]  [dim]● Mature[/dim]\n\n"
                        "[bold]Architecture[/bold]\n"
                        "  • Anchor-free, NMS-free inference (pre-YOLO26 pioneer)\n"
                        "  • 6 size variants: N / S / M / B / L / X\n"
                        "  • Dual-head design: one for training, one for inference\n\n"
                        "[bold]Benchmarks[/bold]  [dim](COCO val2017, detection)[/dim]\n"
                        "  • mAP:     38.5 (N)  →  54.4 (X)\n"
                        "  • Latency: 1.84 ms (N)  →  10.70 ms (X) T4 TensorRT\n\n"
                        "[bold]Recommendation[/bold]\n"
                        "  Prefer YOLO11 or YOLO26 for new projects. Use when an existing "
                        "pipeline is already built on YOLOv10."
                    ),
                    "yolov9": (
                        "[bold cyan]YOLOv9[/bold cyan]  [dim]● Mature[/dim]\n\n"
                        "[bold]Architecture[/bold]\n"
                        "  • Programmable Gradient Information (PGI) — preserves full\n"
                        "    information through deep network layers\n"
                        "  • Generalised Efficient Layer Aggregation Network (GELAN)\n"
                        "  • 5 variants: t / s / m / c / e\n\n"
                        "[bold]Benchmarks[/bold]  [dim](COCO val2017, detection)[/dim]\n"
                        "  • mAP:    38.3 (tiny)  →  55.6 (extra-large)\n"
                        "  • Params: 2.0M (tiny)  →  58.1M (extra-large)\n\n"
                        "[bold]Best for[/bold]\n"
                        "  When PGI gradient stability is required or a YOLOv9 checkpoint "
                        "is already available"
                    ),
                    "yolov9-seg": (
                        "[bold cyan]YOLOv9-Seg[/bold cyan]  [dim]● Mature[/dim]\n\n"
                        "[bold]Architecture[/bold]\n"
                        "  • PGI-based backbone with segmentation head\n"
                        "  • Same GELAN feature aggregation as YOLOv9 detection\n"
                        "  • 5 variants: t / s / m / c / e\n\n"
                        "[bold]Benchmarks[/bold]  [dim](COCO val2017)[/dim]\n"
                        "  • Mask mAP: not officially published by Ultralytics\n\n"
                        "[bold]Best for[/bold]\n"
                        "  Segmentation when PGI gradient properties are desired "
                        "or an existing YOLOv9-seg checkpoint is in use"
                    ),
                    "yolov8": (
                        "[bold cyan]YOLOv8[/bold cyan]  [dim]● Mature — 2023[/dim]\n\n"
                        "[bold]Architecture[/bold]\n"
                        "  • Anchor-free, decoupled detection head\n"
                        "  • Industry-standard baseline — extensively documented\n"
                        "  • Broadest third-party tool and framework support\n"
                        "  • 5 variants: n / s / m / l / x\n\n"
                        "[bold]Benchmarks[/bold]  [dim](COCO val2017, detection)[/dim]\n"
                        "  • mAP:    37.3 (nano)  →  53.9 (xlarge)\n"
                        "  • Params: 3.2M (nano)  →  68.2M (xlarge)\n"
                        "  • Speed:  80 ms CPU ONNX (nano)  |  0.99 ms A100 TensorRT (nano)\n\n"
                        "[bold]Best for[/bold]\n"
                        "  Legacy compatibility, existing YOLOv8 pipelines, or when "
                        "maximum ecosystem support is required"
                    ),
                    "yolov8-seg": (
                        "[bold cyan]YOLOv8-Seg[/bold cyan]  [dim]● Mature — 2023[/dim]\n\n"
                        "[bold]Architecture[/bold]\n"
                        "  • YOLOv8 backbone with segmentation head\n"
                        "  • Most widely supported segmentation baseline\n"
                        "  • 5 variants: n / s / m / l / x\n\n"
                        "[bold]Benchmarks[/bold]  [dim](COCO val2017, segmentation)[/dim]\n"
                        "  • mAP box:  36.7 (nano)  →  53.4 (xlarge)\n"
                        "  • mAP mask: 30.5 (nano)  →  43.4 (xlarge)\n"
                        "  • Params:   3.4M (nano)  →  71.8M (xlarge)\n"
                        "  • Speed:    96 ms CPU ONNX (nano)  |  1.21 ms A100 TensorRT (nano)\n\n"
                        "[bold]Best for[/bold]\n"
                        "  Production segmentation requiring maximum ecosystem compatibility"
                    ),
                    "yolox": (
                        "[bold cyan]YOLOX[/bold cyan]  [dim]● Mature[/dim]\n\n"
                        "[bold]Architecture[/bold]\n"
                        "  • Anchor-free with decoupled classification/regression head\n"
                        "  • Simpler training setup than anchor-based predecessors\n"
                        "  • 4 variants: S / M / L / X\n\n"
                        "[bold]Benchmarks[/bold]  [dim](COCO val2017, detection)[/dim]\n"
                        "  • mAP:    40.5 (S)  →  51.1 (X)\n"
                        "  • Params: 9.0M (S)  →  99.1M (X)\n"
                        "  • FPS:    102 (S)  →  58 (X)  [dim](V100 GPU)[/dim]\n\n"
                        "[bold]Best for[/bold]\n"
                        "  When a clean anchor-free baseline and training stability "
                        "are the primary requirements"
                    ),
                    "yolo_nas": (
                        "[bold cyan]YOLO-NAS[/bold cyan]  [dim]● Specialized[/dim]\n\n"
                        "[bold]Architecture[/bold]\n"
                        "  • Designed via Neural Architecture Search (SuperGradients)\n"
                        "  • QA-RepVGG blocks with hardware-aware quantization\n"
                        "  • 3 variants: S / M / L\n"
                        "  • Trained with SuperGradients, not Ultralytics\n\n"
                        "[bold]Benchmarks[/bold]  [dim](COCO val2017, detection)[/dim]\n"
                        "  • mAP:    47.5 (S)  →  53.2 (L)\n"
                        "  • Params: 9.3M (S)  →  25.2M (L)\n"
                        "  • Speed:  2.4 ms T4 TensorRT (S)  |  90 ms CPU ONNX (S)\n\n"
                        "[bold]Best for[/bold]\n"
                        "  Hardware-targeted deployment on NVIDIA T4 or Jetson where "
                        "NAS-optimized latency is the priority"
                    ),
                },
                breadcrumbs=["YOLOmatic", "Model Selection"],
            )

            if model_choice == "Back":
                continue

            # Rest of your existing model selection code...
            if model_choice == "yolo_nas":
                # Get YOLO NAS specific models
                nas_models = [model["Model"] for model in model_data_dict["yolo_nas"]]
                model_variant = get_user_choice(
                    nas_models,
                    allow_back=True,
                    title=f"Select {model_choice.upper()} Variant",
                    text="Choose the model size that fits your hardware:",
                    model_data=model_data_dict["yolo_nas"],
                    breadcrumbs=["YOLOmatic", "Model Selection", model_choice],
                )

                if model_variant == "Back":
                    continue

                model_choice = model_variant
            else:
                # Show variants for other YOLO models
                variants = [model["Model"] for model in model_data_dict[model_choice]]
                model_variant = get_user_choice(
                    variants,
                    allow_back=True,
                    title=f"Select {model_choice.upper()} Variant",
                    text="Choose the model size that fits your hardware:",
                    model_data=model_data_dict[model_choice],
                    breadcrumbs=["YOLOmatic", "Model Selection", model_choice],
                )

                if model_variant == "Back":
                    continue

                model_choice = model_variant

            # Continue with dataset selection...
            try:
                dataset_choice = list_datasets()
            except Exception as error:
                console.print(
                    Panel(
                        f"[bold red]Failed to list datasets:[/bold red] {error}",
                        border_style="red",
                        padding=(1, 2),
                    )
                )
                input("\nPress Enter to return to the main menu...")
                continue
            if dataset_choice == "Back":
                continue
            elif dataset_choice is None:
                continue

            # Show summary and update config. Any failure during config
            # generation must not tear down the TUI — report and return.
            print_summary(model_choice, dataset_choice)
            try:
                if not update_config(model_choice, dataset_choice):
                    continue
            except KeyboardInterrupt:
                console.print(
                    "\n[bold yellow]Configuration cancelled by user.[/bold yellow]"
                )
                input("\nPress Enter to return to the main menu...")
                continue
            except Exception as error:
                console.print(
                    Panel(
                        f"[bold red]Configuration failed:[/bold red] {error}",
                        border_style="red",
                        padding=(1, 2),
                    )
                )
                console.print(traceback.format_exc(), style="dim")
                input("\nPress Enter to return to the main menu...")
                continue

            # Ask if user wants to continue
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
