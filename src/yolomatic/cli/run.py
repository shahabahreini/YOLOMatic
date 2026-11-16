import argparse
import os
import sys
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from rich import box
from rich.align import Align
from rich.panel import Panel
from rich.table import Table
from ruamel.yaml import YAML

from yolomatic.cli.result import EXIT_CANCELLED, CommandResult, CommandStatus, coerce_result
from yolomatic.config.generator import (
    Detectron2ConfigGenerator,
    RFDETRConfigGenerator,
    SAMConfigGenerator,
    YOLOConfigGenerator,
)

# Training parameter catalogs live in src/config/parameters.py
from yolomatic.config.parameters import (  # noqa: E402
    YOLO_TRAINING_PARAMETERS,
    parameters_for,
)
from yolomatic.config.settings import (
    load_settings,
    reset_settings,
    roboflow_credential_status,
    save_settings,
    ultralytics_credential_status,
)
from yolomatic.datasets import summarize_dataset
from yolomatic.models.data import model_data_dict

# Model family descriptions — defined at module level to avoid rebuilding on every wizard entry
from yolomatic.models.descriptions import MODEL_DESCRIPTIONS as _MODEL_DESCRIPTIONS  # noqa: E402
from yolomatic.models.detectron2 import get_detectron2_variant, is_detectron2_model
from yolomatic.models.rfdetr import get_rfdetr_variant, is_rfdetr_model
from yolomatic.models.sam import is_sam_model
from yolomatic.ui.theme import CONSOLE
from yolomatic.utils.cli import (
    ParameterDefinition,
    clear_screen,
    console,
    get_user_choice,
    get_user_multi_select,
    print_stylized_header,
    render_summary_panel,
    render_table,
    shorten_middle,
)
from yolomatic.utils.ml_dependencies import MLDependencyError, import_torch
from yolomatic.utils.project import (
    FineTuneCandidate,
    find_finetune_candidates,
    format_size,
    infer_ultralytics_task_from_name,
    is_detectron2_source,
    is_rfdetr_source,
    list_config_files,
    list_dataset_directories,
    project_root,
)

_CACHED_DATASETS = None
_CACHED_DATASET_DESCRIPTIONS = None


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


def _pause_for_user(prompt: str = "Press Enter to return to the main menu...") -> None:
    console.print()
    try:
        input(prompt)
    except (EOFError, KeyboardInterrupt):
        console.print()


def _failure_panel(label: str, message: str, hint: str | None, log_path: Any = None) -> Panel:
    from yolomatic.ui.theme import safe

    body = f"[bold red]{safe(label)} failed:[/bold red] {safe(message)}"
    if hint:
        body += f"\n\n[bold yellow]Next:[/bold yellow] {safe(hint)}"
    if log_path:
        body += f"\n\n[dim]Details: {safe(log_path)}[/dim]"
    return Panel(body, border_style="red", padding=(1, 2))


def _report_command_result(label: str, result: CommandResult) -> None:
    """Show the outcome of a subcommand and pause only when there is something to read."""
    from yolomatic.ui.theme import safe

    if result.status is CommandStatus.FAILED:
        console.print(_failure_panel(label, result.message or "see the messages above", result.hint))
        _pause_for_user()
    elif result.status is CommandStatus.OK and result.outputs:
        lines = [f"[bold green]{safe(label)} finished.[/bold green]"]
        if result.message:
            lines.append(safe(result.message))
        lines.append("")
        lines.extend(f"[cyan]{safe(name)}:[/cyan] {safe(path)}" for name, path in result.outputs.items())
        console.print(Panel("\n".join(lines), title="Outputs", border_style="green", padding=(1, 2)))
        _pause_for_user()
    elif result.status is CommandStatus.OK and result.review:
        _pause_for_user()
    elif result.status is CommandStatus.CANCELLED and result.message:
        console.print(f"[yellow]{safe(result.message)}[/yellow]")


def _safe_subcommand(
    label: str,
    target: Callable[..., Any],
    *,
    prog: str | None = None,
) -> CommandResult:
    """Run a submodule's ``main()`` from the TUI with unified error handling.

    The TUI invokes each command-line tool (training, prediction, TensorBoard,
    upload, dataset tools) by calling its ``main`` directly rather than
    shelling out. That keeps the session in one process but means any
    ``sys.exit``/``KeyboardInterrupt``/unexpected exception inside the
    sub-command would otherwise kill the TUI. This wrapper neutralises all of
    those and turns every outcome into a :class:`CommandResult`:

    * failures always pause, so the message cannot flash past;
    * successes that wrote files list them, then pause;
    * cancellations return straight to the menu.
    """
    from yolomatic.utils.logging_setup import log_exception

    clear_screen()
    entrypoint_name = prog or label.lower().replace(" ", "-")
    result: CommandResult
    reported = False
    try:
        with _scoped_argv(entrypoint_name):
            result = coerce_result(target())
    except SystemExit as error:
        code = error.code
        if code in (None, 0):
            result = CommandResult.ok()
        elif code == EXIT_CANCELLED:
            result = CommandResult.cancelled(f"{label} cancelled.")
        else:
            result = CommandResult.failed(f"exited with status {code}.")
    except KeyboardInterrupt:
        result = CommandResult.cancelled(f"{label} cancelled by user.")
    except MLDependencyError as error:
        result = CommandResult.failed(
            f"missing dependency: {error}",
            hint="Install the missing package (see the message above) and try again.",
        )
    except FileNotFoundError as error:
        log_exception(error, f"{label} failed")
        result = CommandResult.failed(f"file not found: {error}")
    except Exception as error:  # noqa: BLE001 - last line of defence for the TUI
        log_path = log_exception(error, f"{label} failed unexpectedly")
        if os.environ.get("YOLOMATIC_DEBUG"):
            console.print(traceback.format_exc(), style="dim")
        console.print(_failure_panel(label, f"{type(error).__name__}: {error}", None, log_path))
        _pause_for_user()
        result = CommandResult.failed(str(error))
        reported = True
    finally:
        global _CACHED_DATASETS, _CACHED_DATASET_DESCRIPTIONS
        _CACHED_DATASETS = None
        _CACHED_DATASET_DESCRIPTIONS = None
    if not reported:
        _report_command_result(label, result)
    return result


from yolomatic.cli.about import render_about_page  # noqa: E402
from yolomatic.cli.updates import (  # noqa: E402,F401 - re-exported for callers/tests
    _render_dependency_summary,
    _render_dependency_table,
    _run_pip_upgrade,
    check_for_updates,
)


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

    with open(config_path, encoding="utf-8") as f:
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

    if is_detectron2_model(str(model_choice)):
        training = config.get("training", {})
        dataset_config = config.get("dataset", {})
        fields.update(
            {
                "Max Iter": training.get("max_iter", "N/A"),
                "Images/Batch": training.get("ims_per_batch", "N/A"),
                "Workers": training.get("num_workers", "N/A"),
                "Dataset Format": dataset_config.get("prepared_format", "N/A"),
                "Task": config.get("settings", {}).get("task", "N/A"),
            }
        )
    elif is_rfdetr_model(str(model_choice)):
        training = config.get("training", {})
        fields.update(
            {
                "Batch Size": training.get("batch_size", "N/A"),
                "Grad Accum": training.get("grad_accum_steps", "N/A"),
                "Epochs": training.get("epochs", "N/A"),
                "Resolution": training.get("resolution", "N/A"),
                "Auto Download": config.get("settings", {}).get(
                    "auto_download_pretrained", "N/A"
                ),
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

    # Indicate if this is a fully customized or AI-recommended config
    if profile_selection and profile_selection.get("mode") == "fully_customized":
        fields["Config Mode"] = "Fully Customized"
    elif profile_selection and profile_selection.get("mode") == "ai_recommendation":
        fields["Config Mode"] = "AI Recommendation"

    render_summary_panel("Configuration Summary", fields)

    render_table(
        "Dataset Paths",
        ["Type", "Path"],
        dataset_path_rows_for_config(model_choice, config),
        title_style="bold blue",
    )


def dataset_path_rows_for_config(
    model_choice: str, config: dict[str, Any]
) -> list[list[str]]:
    """Return summary path rows for each supported config schema."""
    if is_detectron2_model(str(model_choice)):
        splits = config.get("dataset", {}).get("splits", {})

        def _split_path(split_name: str, key: str) -> str:
            split = splits.get(split_name, {})
            value = split.get(key) if isinstance(split, dict) else None
            return value or "N/A"

        return [
            ["Train Images", _split_path("train", "images_path")],
            ["Train Annotations", _split_path("train", "annotations_path")],
            ["Validation Images", _split_path("val", "images_path")],
            ["Validation Annotations", _split_path("val", "annotations_path")],
            ["Test Images", _split_path("test", "images_path")],
            ["Test Annotations", _split_path("test", "annotations_path")],
        ]

    if is_rfdetr_model(str(model_choice)):
        dataset_config = config.get("dataset", {})
        base_dir = dataset_config.get("base_dir", "")
        return [
            ["Train", os.path.join(base_dir, "train/images")],
            ["Validation", os.path.join(base_dir, "valid/images")],
            ["Test", os.path.join(base_dir, "test/images")],
        ]

    if "nas" in model_choice.lower():
        structure = config.get("dataset", {}).get("structure", {})
        base_dir = config.get("dataset", {}).get("base_dir", "")
        return [
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

    model_config = config.get("model", {})
    data_dir = model_config.get("data_dir", "")
    return [
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


def display_paths_info(dataset_info):
    """Display dataset paths in a clean format"""
    console = CONSOLE

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


def list_datasets(wizard_steps: list[str] | None = None, wizard_current_step: int | None = None):
    global _CACHED_DATASETS, _CACHED_DATASET_DESCRIPTIONS

    datasets_folder = "datasets"
    if not os.path.exists(datasets_folder):
        os.makedirs(datasets_folder)
        console.print(
            f"✨ '{datasets_folder}' folder created. Please add COCO or any other compatible dataset into it.",
            style="bold yellow",
        )
        input("\nPress Enter to return...")
        return None

    def _build_description(d: dict[str, Any]) -> tuple[str, str]:
        try:
            summary = d.get("summary") or summarize_dataset(d["path"])
            classes = ", ".join(summary.classes[:8]) or "No classes found"
            if len(summary.classes) > 8:
                classes += ", ..."
            split_lines = []
            for split_name, split in summary.splits.items():
                split_lines.append(
                    f"  • {split_name}: {split.image_count} images, "
                    f"{split.annotation_count} annotations, {split.missing_file_count} missing ({split.status})"
                )
            health = "Valid" if not summary.errors else "Blocking errors"
            if summary.warnings and not summary.errors:
                health = "Warnings"
            from yolomatic.ui.theme import safe

            desc = (
                f"[bold cyan]{safe(d['name'])}[/bold cyan]\n\n"
                f"[bold]Format:[/bold] {summary.format.upper()}    "
                f"[bold]Task:[/bold] {summary.task.title()}\n"
                f"[bold]Size:[/bold] {d['size']}    "
                f"[bold]Images:[/bold] {summary.image_count}    "
                f"[bold]Annotations:[/bold] {summary.annotation_count}\n\n"
                f"[bold]Splits[/bold]\n"
                + ("\n".join(split_lines) or "  No splits found")
                + "\n\n"
                f"[bold]Classes:[/bold] {len(summary.classes)} total — {safe(classes)}\n"
                f"[bold]Health:[/bold] {health}\n"
                f"[bold]Compatibility:[/bold] YOLO/RF-DETR: {summary.compatibility.get('yolo', 'unknown')}; "
                f"Detectron2: {summary.compatibility.get('detectron2', 'unknown')}\n"
                f"[dim]Conversions, when needed, are written under datasets/.yolomatic_cache.[/dim]\n"
                f"[dim]{safe(d['path'])}[/dim]"
            )
        except Exception as error:  # noqa: BLE001 - shown in the details pane
            from yolomatic.ui.theme import safe

            desc = (
                f"[bold yellow]{safe(d['name'])}[/bold yellow]\n\n"
                f"YOLOmatic could not inspect this dataset cleanly: {safe(error)}\n"
                f"[dim]{safe(d['path'])}[/dim]"
            )
        return d["name"], desc

    while True:
        if _CACHED_DATASETS is None:
            with console.status("[bold cyan]Scanning and analyzing available datasets...", spinner="dots"):
                datasets = list_dataset_directories(datasets_folder)
                dataset_descriptions: dict[str, str] = {}
                if datasets:
                    for d in datasets:
                        name, desc = _build_description(d)
                        dataset_descriptions[name] = desc
            _CACHED_DATASETS = datasets
            _CACHED_DATASET_DESCRIPTIONS = dataset_descriptions
        else:
            datasets = _CACHED_DATASETS
            dataset_descriptions = dict(_CACHED_DATASET_DESCRIPTIONS)

        if not datasets:
            console.print(
                f"❌ No datasets found in '{datasets_folder}' folder.", style="bold red"
            )
            input("\nPress Enter to return...")
            return None

        dataset_names = [Path(d["path"]) for d in datasets]
        name_to_path = {path.name: str(path) for path in dataset_names}
        dataset_descriptions["Back"] = "Return to the previous menu."

        choice = get_user_choice(
            list(name_to_path.keys()),  # Show basename in menu
            allow_back=True,
            title="Select Dataset",
            text="Use ↑↓ keys to navigate, Enter to select, 'b' for back, 'r' to refresh:",
            descriptions=dataset_descriptions,
            wizard_steps=wizard_steps,
            wizard_current_step=wizard_current_step,
            allow_refresh=True,
        )

        if choice == "__refresh__":
            _CACHED_DATASETS = None
            _CACHED_DATASET_DESCRIPTIONS = None
            clear_screen()
            continue

        return name_to_path.get(choice) if choice != "Back" else choice


def select_saved_config_file(
    title: str = "Clone Saved Config",
    wizard_steps: list[str] | None = None,
    wizard_current_step: int | None = None,
) -> Path | None:
    config_dir = Path("configs")
    with console.status("[bold cyan]Scanning saved configurations...", spinner="dots"):
        yaml_files = list_config_files(config_dir)
    if not yaml_files:
        console.print(
            Panel(
                "[bold yellow]No saved YAML configs were found in ./configs.[/bold yellow]\n\n"
                "Create a config first, then return here to clone it.",
                border_style="yellow",
                padding=(1, 2),
            )
        )
        input("\nPress Enter to return to the main menu...")
        return None

    descriptions: dict[str, str] = {}
    for filename in yaml_files:
        path = config_dir / filename
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime).strftime(
                "%Y-%m-%d %H:%M"
            )
            size = format_size(path.stat().st_size)
        except OSError:
            modified = "unknown"
            size = "unknown"
        descriptions[filename] = (
            f"[bold cyan]{filename}[/bold cyan]\n\n"
            f"Path: [yellow]{path}[/yellow]\n"
            f"Modified: [yellow]{modified}[/yellow]\n"
            f"Size: [yellow]{size}[/yellow]\n\n"
            f"Menu label: [dim]{shorten_middle(filename)}[/dim]"
        )
    descriptions["Back"] = "Return to the main menu."

    choice = get_user_choice(
        yaml_files,
        allow_back=True,
        title=title,
        text=(
            "Choose the saved YAML to clone. Long filenames are shortened in the "
            "left menu; the full filename is shown here in the details pane."
        ),
        descriptions=descriptions,
        breadcrumbs=["YOLOmatic", "Clone Config"],
        wizard_steps=wizard_steps,
        wizard_current_step=wizard_current_step,
    )
    if choice == "Back":
        return None
    return config_dir / choice


def extract_regular_yolo_model_choice(config: dict[str, Any]) -> str | None:
    settings = config.get("settings")
    if not isinstance(settings, dict):
        return None
    for key in ("base_model_type", "model_type"):
        value = settings.get(key)
        if value:
            return str(value)
    return None


def collect_known_config_sections(
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    sections: dict[str, dict[str, Any]] = {}
    for param in YOLO_TRAINING_PARAMETERS:
        section = config.get(param.config_section)
        if isinstance(section, dict) and param.name in section:
            sections.setdefault(param.config_section, {})[param.name] = section[
                param.name
            ]
    return sections


def apply_config_sections(
    config: dict[str, Any],
    sections: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    for section_name, section_values in sections.items():
        if not section_values:
            continue
        config.setdefault(section_name, {}).update(section_values)
    return config


def clone_config_filename(source_path: Path, dataset_name: str) -> str:
    def slug(value: str, max_len: int) -> str:
        safe = "".join(char.lower() if char.isalnum() else "_" for char in value).strip(
            "_"
        )
        safe = "_".join(part for part in safe.split("_") if part)
        return (safe or "config")[:max_len]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_slug = slug(source_path.stem, 32)
    dataset_slug = slug(dataset_name, 28)
    return f"clone_{source_slug}_{dataset_slug}_{timestamp}.yaml"


def clone_saved_config_flow() -> bool:
    steps = ["Select Source YAML", "Select Target Dataset"]
    step = 0
    source_path: Path | None = None
    source_config: dict = {}
    model_choice: str | None = None
    dataset_choice: str | None = None

    while 0 <= step < 2:
        if step == 0:
            source_path = select_saved_config_file(wizard_steps=steps, wizard_current_step=0)
            if source_path is None:
                return False  # user backed out of first step → exit wizard

            try:
                with open(source_path, encoding="utf-8") as file:
                    source_config = yaml.safe_load(file) or {}
            except yaml.YAMLError as error:
                console.print(
                    Panel(
                        f"[bold red]Invalid YAML in {source_path.name}:[/bold red] {error}",
                        border_style="red",
                        padding=(1, 2),
                    )
                )
                input("\nPress Enter to select a different file...")
                continue  # stay on step 0

            if "experiment" in source_config:
                console.print(
                    Panel(
                        "[bold yellow]YOLO-NAS config cloning is not supported yet.[/bold yellow]\n\n"
                        "Clone regular Ultralytics YOLO configs from this flow. YOLO-NAS "
                        "configs use a different nested training schema.",
                        border_style="yellow",
                        padding=(1, 2),
                    )
                )
                input("\nPress Enter to select a different file...")
                continue  # stay on step 0

            model_choice = extract_regular_yolo_model_choice(source_config)
            if not model_choice:
                console.print(
                    Panel(
                        "[bold red]Could not find settings.model_type in the source config.[/bold red]\n\n"
                        "The clone flow needs a regular YOLO config with a model recorded "
                        "under the settings section.",
                        border_style="red",
                        padding=(1, 2),
                    )
                )
                input("\nPress Enter to select a different file...")
                continue  # stay on step 0

            step = 1

        else:  # step == 1
            try:
                dataset_choice = list_datasets(wizard_steps=steps, wizard_current_step=1)
            except Exception as error:
                console.print(
                    Panel(
                        f"[bold red]Failed to list datasets:[/bold red] {error}",
                        border_style="red",
                        padding=(1, 2),
                    )
                )
                input("\nPress Enter to return to the main menu...")
                return False
            if dataset_choice == "Back":
                step = 0  # go back to YAML selection
                continue
            if dataset_choice is None:
                return False
            step = 2  # exit loop normally

    dataset_path = Path(dataset_choice)
    dataset_name = dataset_path.name
    generator = YOLOConfigGenerator(str(dataset_path))
    profile_context = generator.get_regular_yolo_profile_context(model_choice)
    base_config = generator.generate_config(
        model_choice,
        dict(profile_context["recommended_profiles"]),
        profile_context,
    )

    source_settings = source_config.get("settings", {})
    if isinstance(source_settings, dict):
        for key in ("model_type", "base_model_type", "finetune_from"):
            if source_settings.get(key):
                base_config.setdefault("settings", {})[key] = source_settings[key]

    source_clearml = source_config.get("clearml", {})
    if isinstance(source_clearml, dict):
        base_config.setdefault("clearml", {}).update(source_clearml)

    source_sections = collect_known_config_sections(source_config)
    copied_count = sum(len(values) for values in source_sections.values())

    clear_screen()
    print_stylized_header("Clone Config Preview")
    render_summary_panel(
        "Clone Source",
        {
            "Source Config": source_path.name,
            "Target Model": base_config.get("settings", {}).get(
                "model_type",
                model_choice,
            ),
            "Target Dataset": dataset_name,
            "Copied Tunable Values": copied_count,
            "New Dataset Path": base_config.get("model", {}).get("data_dir", "N/A"),
        },
    )

    result = run_fully_customized_config_flow(
        dataset_name,
        model_choice,
        profile_context,
        initial_sections=source_sections,
        title="Review Cloned Configuration",
        intro_text=(
            f"[bold yellow]Cloning:[/bold yellow] {source_path.name}\n\n"
            "YOLOmatic regenerated the dataset/model scaffolding for the new target, "
            "then preselected the tunable values found in the source YAML.\n\n"
            "• Deselect a value to fall back to the regenerated default.\n"
            "• Edit any selected value before saving the cloned config.\n"
            "• Dataset paths/classes are already updated from the selected dataset."
        ),
    )
    if result is None:
        return False

    final_sections = result.get("sections", {"training": result.get("params", {})})
    cloned_config = apply_config_sections(base_config, final_sections)

    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    config_file = clone_config_filename(source_path, dataset_name)
    config_path = config_dir / config_file

    output_yaml = YAML()
    output_yaml.indent(mapping=2, sequence=4, offset=2)
    with open(config_path, "w", encoding="utf-8") as file:
        output_yaml.dump(cloned_config, file)

    console.print(
        f"✅ Cloned configuration saved to: {config_file}", style="bold green"
    )
    display_configuration_summary(
        cloned_config.get("settings", {}).get("model_type", model_choice),
        dataset_name,
        config_file,
        generator.dataset_info,
        {"mode": "fully_customized"},
        profile_context,
    )
    return True


def select_finetune_candidate(
    wizard_steps: list[str] | None = None,
    wizard_current_step: int | None = None,
) -> FineTuneCandidate | None:
    root = project_root()
    with console.status("[bold cyan]Scanning for Ultralytics weights checkpoints...", spinner="dots"):
        candidates = find_finetune_candidates(root)
    if not candidates:
        console.print(
            Panel(
                "[bold yellow]No Ultralytics .pt weights were found in the "
                "project root or runs/**/weights.[/bold yellow]\n\n"
                "Train a model first, place a .pt checkpoint in the project root, "
                "or use the regular Configure Model flow to start from an "
                "official model.",
                border_style="yellow",
                padding=(1, 2),
            )
        )
        input("\nPress Enter to return to the main menu...")
        return None

    options = [candidate.display_name for candidate in candidates]
    descriptions = {
        candidate.display_name: (
            f"[bold cyan]{candidate.display_name}[/bold cyan]\n\n"
            f"Source: [yellow]{candidate.kind}[/yellow]\n"
            f"Inferred task: [yellow]{candidate.task}[/yellow]\n"
            "Fine-tuning starts a fresh run from these weights. It does not "
            "resume optimizer state."
        )
        for candidate in candidates
    }
    selected = get_user_choice(
        options,
        allow_back=True,
        title="Select Fine-Tune Starting Weights",
        text="Pick the Ultralytics checkpoint to adapt to another dataset:",
        descriptions=descriptions,
        breadcrumbs=["YOLOmatic", "Fine-Tune", "Weights"],
        status_fields={"Candidates": str(len(candidates))},
        tip=(
            "Use last.pt only when you intentionally want the latest checkpoint "
            "weights. For deployment-quality transfer, best.pt is usually the "
            "better start."
        ),
        wizard_steps=wizard_steps,
        wizard_current_step=wizard_current_step,
    )
    if selected == "Back":
        return None
    return candidates[options.index(selected)]


def select_finetune_strategy(
    candidate: FineTuneCandidate,
    wizard_steps: list[str] | None = None,
    wizard_current_step: int | None = None,
) -> str | None:
    selected = get_user_choice(
        ["Recommended", "Freeze Backbone", "Fully Customized"],
        allow_back=True,
        title="Fine-Tune Strategy",
        text=(
            f"Starting point: [cyan]{candidate.display_name}[/cyan]\n\n"
            "Choose how YOLOmatic should configure the new fine-tuning run:"
        ),
        descriptions={
            "Recommended": (
                "[bold green]Fresh fine-tune run using YOLOmatic's dataset and "
                "hardware profiles.[/bold green]\n\n"
                "• `resume` is false, so optimizer state is not reused.\n"
                "• No layers are frozen by default.\n"
                "• Best default when the target dataset is meaningfully different."
            ),
            "Freeze Backbone": (
                "[bold yellow]Freeze early layers and adapt the head.[/bold yellow]\n\n"
                "• Writes `freeze: 10` into the training config.\n"
                "• Useful for small datasets similar to the checkpoint's previous "
                "domain.\n"
                "• Can underfit if the new domain is very different."
            ),
            "Fully Customized": (
                "[bold cyan]Open the expert parameter editor.[/bold cyan]\n\n"
                "• Choose epochs, learning rate, freeze, augmentation, and other "
                "Ultralytics args.\n"
                "• Still starts from the selected checkpoint with `resume: false` "
                "unless you change it."
            ),
        },
        breadcrumbs=["YOLOmatic", "Fine-Tune", "Strategy"],
        wizard_steps=wizard_steps,
        wizard_current_step=wizard_current_step,
    )
    if selected == "Back":
        return None
    if selected == "Freeze Backbone":
        return "freeze_backbone"
    if selected == "Fully Customized":
        return "fully_customized"
    return "recommended"


def infer_finetune_profile_model(candidate: FineTuneCandidate) -> str:
    normalized = candidate.display_name.lower()
    if is_rfdetr_source(candidate.source):
        if "keypoint" in normalized or "pose" in normalized:
            return "RF-DETR-Keypoint"
        if "seg" in normalized:
            return "RF-DETR-Seg-Medium"
        return "RF-DETR-Medium"
    if is_detectron2_source(candidate.source):
        if "seg" in normalized or "mask_rcnn" in normalized:
            return "Mask R-CNN R50-FPN 3x"
        if "retinanet" in normalized:
            return "RetinaNet R50-FPN 3x"
        return "Faster R-CNN R50-FPN 3x"
    for family_rows in model_data_dict.values():
        for row in family_rows:
            model_name = str(row.get("Model", ""))
            if model_name and model_name.lower() in normalized:
                return model_name
    if candidate.task == "semantic":
        return "YOLO26n-sem"
    if candidate.task == "segmentation":
        return "YOLO11n-seg"
    return "YOLO11n"


# Ultralytics task suffixes, longest-first so "-pose" is stripped before a shorter
# accidental match. Mirrors generator.TASK_SUFFIXES.
_MODEL_TASK_SUFFIXES = ("-pose", "-seg", "-sem", "-cls", "-obb")


def _base_model_name(model_choice: str) -> str:
    """Strip any task suffix, so a variant swap cannot stack two of them.

    Without this, recommending "<model>-seg" for a "yolo26x-sem" model produces
    "yolo26x-sem-seg", which does not exist.
    """
    lowered = model_choice.lower()
    for suffix in _MODEL_TASK_SUFFIXES:
        if lowered.endswith(suffix):
            return model_choice[: -len(suffix)]
    return model_choice


# Removed print_model_info, now handled by yolomatic.utils.tui


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



# update_config() return value meaning "go back to model selection".
CHANGE_MODEL = "__change_model__"


def update_config(
    model_choice,
    dataset_choice,
    finetune_source: str | None = None,
    finetune_strategy: str | None = None,
    wizard_steps: list[str] | None = None,
    wizard_current_step: int | None = None,
):
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

    source_slug = Path(finetune_source).stem if finetune_source else model_choice
    config_file = f"{source_slug}_{safe_dataset_name}_{timestamp}.yaml"

    # Initialize appropriate generator
    if is_detectron2_model(model_choice):
        generator = Detectron2ConfigGenerator(str(dataset_path))
        generator.extract_dataset_info()
        profile_context = None
        profile_selection = None
    elif is_rfdetr_model(model_choice):
        generator = RFDETRConfigGenerator(str(dataset_path))
        generator.extract_dataset_info()
        profile_context = None
        profile_selection = None
    elif is_sam_model(model_choice):
        generator = SAMConfigGenerator(str(dataset_path))
        generator.extract_dataset_info()
        profile_context = None
        profile_selection = None
    elif "nas" in model_choice.lower():
        console.print(
            "[bold yellow]YOLO-NAS support is deprecated in this build.[/bold yellow]\n\n"
            "SuperGradients conflicts with RF-DETR's training dependency stack. "
            "Choose a YOLO, RF-DETR, or Detectron2 model instead."
        )
        return False
    else:
        generator = YOLOConfigGenerator(str(dataset_path))
        profile_context = generator.get_regular_yolo_profile_context(model_choice)
        profile_selection = None

    # Check dataset type compatibility
    dataset_type = generator.dataset_info.get("task_type", "unknown")
    model_task_source = finetune_source or model_choice
    inferred_model_task = (
        "segmentation"
        if is_rfdetr_model(model_choice) and "-seg-" in model_choice.lower()
        else "segmentation"
        if is_detectron2_model(model_choice) and get_detectron2_variant(model_choice).task == "segmentation"
        else infer_ultralytics_task_from_name(model_task_source)
    )
    is_seg_model = (
        inferred_model_task == "segmentation" or "-seg" in model_choice.lower()
    )
    # A "-sem" model consumes exactly the polygon labels a segmentation dataset
    # carries — Ultralytics rasterizes them into dense targets — so it must not be
    # flagged as needing a "-seg" variant. Note "-sem" is not a substring of "-seg".
    is_semantic_model = (
        inferred_model_task == "semantic" or "-sem" in model_choice.lower()
    )

    # Determine if there's a mismatch
    mismatch_type = None
    if is_sam_model(model_choice):
        pass  # SAM handles any segmentation dataset natively; skip mismatch checks
    elif is_semantic_model and dataset_type in ("segmentation", "semantic"):
        pass  # Polygons and dense masks are both valid inputs for a semantic model.
    elif dataset_type == "semantic" and not is_semantic_model:
        mismatch_type = "sem_model_needed"
    elif (
        dataset_type == "segmentation"
        and not is_seg_model
        and "nas" not in model_choice.lower()
        and not is_rfdetr_model(model_choice)
    ):
        mismatch_type = "seg_model_needed"
    elif (
        dataset_type == "detection"
        and is_seg_model
    ):
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
            recommended_model = f"{_base_model_name(model_choice)}-seg"
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
        elif mismatch_type == "sem_model_needed":
            title = "Dataset / Model Mismatch"
            model_kind = "semantic segmentation"
            detected_label_format = "Semantic segmentation (data.yaml declares task: semantic)"
            expected_by_model = f"Labels for a {inferred_model_task} model"
            recommended_model = f"{_base_model_name(model_choice)}-sem"
            summary = (
                f"[yellow]This dataset is built for semantic segmentation.[/yellow] "
                f"[cyan]{dataset_name}[/cyan] declares [bold]task: semantic[/bold], meaning its "
                "labels describe a dense per-pixel class map rather than individual objects. "
                f"[bold]{model_choice}[/bold] predicts discrete instances, so it would train "
                "on a target that does not represent what the dataset encodes."
            )
            continue_detail = (
                f"[bold yellow]Train {model_choice} on this dataset anyway.[/bold yellow]\n\n"
                "• Polygon-backed semantic datasets still load, and each polygon is "
                "treated as one instance.\n"
                "• [red]Mask-backed semantic datasets have no .txt labels at all[/red] and "
                "will fail at data loading.\n"
                "• Semantic datasets frequently merge touching objects of one class into "
                "a single region, so instance metrics on them are misleading."
            )
            change_model_detail = (
                "[bold green]Go back and pick the semantic variant.[/bold green]  [dim](recommended)[/dim]\n\n"
                f"• Recommended: [green]{recommended_model}[/green] — trains a dense "
                "per-pixel head that matches this dataset exactly.\n"
                "• Ultralytics rasterizes polygon labels automatically and adds a "
                "background class; no dataset changes are needed.\n"
                "• Reports mIoU and pixel accuracy instead of mAP."
            )
            fix_dataset_detail = (
                "[bold cyan]Keep this model and rebuild the dataset for instances.[/bold cyan]\n\n"
                "• Re-run Augment Dataset with output format "
                "[bold]YOLO Segmentation[/bold] (or [bold]YOLO Detection[/bold]) so "
                "data.yaml declares an instance task.\n"
                "• Only worth it if you genuinely need per-object outputs."
            )
            tip = (
                "[bold]Choose a Different Model[/bold] is almost always right here — the "
                "dataset already carries exactly what a semantic model wants."
            )
        elif mismatch_type == "det_model_needed":
            title = "Dataset / Model Mismatch"
            model_kind = "segmentation"
            detected_label_format = "Bounding boxes (5 values per line)"
            expected_by_model = "Segmentation polygons (7+ odd values per line)"
            recommended_model = _base_model_name(model_choice)
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

        # Say what the detector actually concluded and where it looked. Without
        # this the panel is unfalsifiable: a stale summary cache and a genuinely
        # mislabelled dataset produce the same screen.
        scanned_label_dirs = []
        try:
            train_path = Path(generator.dataset_info.get("train_path") or "")
            scanned_label_dirs = [str(path) for path in generator._candidate_label_dirs(train_path)]
        except Exception:
            pass

        status_fields: dict[str, str] = {
            "Selected Model": f"{model_choice} ({model_kind})",
            "Dataset": f"{dataset_name} ({num_classes} classes)",
            "Labels Found": detected_label_format,
            "Model Expects": expected_by_model,
            "Detected As": f"{dataset_type} (model task: {inferred_model_task})",
            "Labels Scanned": scanned_label_dirs[0] if scanned_label_dirs else "no label directory found",
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

        if choice == "Choose a Different Model":
            # Distinct from False ("fix the dataset") so the wizard goes back to
            # model selection instead of dataset selection.
            return CHANGE_MODEL
        if choice != "Continue Anyway":
            return False
        # Continue with "Continue Anyway"

    if is_detectron2_model(model_choice) or is_rfdetr_model(model_choice):
        # RF-DETR and Detectron2 generate smart defaults, then expose a
        # task-filtered interactive parameter page so users can tune the
        # parameters that actually apply to the selected model and task.
        if is_detectron2_model(model_choice):
            family = "detectron2"
            task = get_detectron2_variant(model_choice).task
            flow_title = "Detectron2 Configuration"
        else:
            family = "rfdetr"
            task = get_rfdetr_variant(model_choice).task
            flow_title = "RF-DETR Configuration"
        config = generator.generate_config(
            model_choice,
            finetune_source=finetune_source,
            finetune_strategy=finetune_strategy,
        )
        result = run_fully_customized_config_flow(
            dataset_name,
            model_choice,
            {},
            initial_sections={"training": dict(config.get("training", {}))},
            title=flow_title,
            parameters=parameters_for(family, task),
            wizard_steps=wizard_steps,
            wizard_current_step=wizard_current_step,
        )
        if result is None:
            return False
        custom_sections = result.get("sections", {"training": result.get("params", {})})
        for section_name, section_params in custom_sections.items():
            config.setdefault(section_name, {}).update(section_params)
    elif "nas" not in model_choice.lower():
        profile_selection = choose_regular_yolo_profiles(
            dataset_name,
            profile_context,
            model_choice,
            wizard_steps=wizard_steps,
            wizard_current_step=wizard_current_step,
            dataset_path=str(dataset_path),
        )
        if profile_selection is None:
            return False

        # Handle fully customized mode
        if profile_selection.get("mode") == "fully_customized":
            # Filter to the parameters this task actually consumes. A semantic run
            # otherwise offers box/dfl/mask/NMS knobs that its head never reads.
            result = run_fully_customized_config_flow(
                dataset_name,
                model_choice,
                profile_context,
                parameters=parameters_for(
                    "yolo", infer_ultralytics_task_from_name(model_choice)
                ),
                wizard_steps=wizard_steps,
                wizard_current_step=wizard_current_step,
            )
            if result is None:
                return False
            custom_sections = result.get(
                "sections", {"training": result.get("params", {})}
            )
            # Generate base config and apply custom params directly
            config = generator.generate_config(
                model_choice,
                dict(profile_context["recommended_profiles"]),
                profile_context,
                finetune_source=finetune_source,
                finetune_strategy=finetune_strategy,
            )
            # Override with custom parameters
            for section_name, section_params in custom_sections.items():
                config.setdefault(section_name, {}).update(section_params)
        elif profile_selection.get("mode") == "ai_recommendation":
            training_overrides = profile_selection["training_overrides"]
            suggested_name = profile_selection["name"]
            
            # Generate base config from recommended profile
            config = generator.generate_config(
                model_choice,
                dict(profile_context["recommended_profiles"]),
                profile_context,
                finetune_source=finetune_source,
                finetune_strategy=finetune_strategy,
            )
            # Apply AI recommendations
            config.setdefault("training", {}).update(training_overrides)
            config_file = f"{suggested_name}_ai_{timestamp}.yaml"
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
                finetune_source=finetune_source,
                finetune_strategy=finetune_strategy,
            )
    else:
        config = generator.generate_config(model_choice)

    # Save new config
    config_path = os.path.join(config_dir, config_file)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    console.print(f"✅ Configuration saved to: {config_file}", style="bold green")

    # Display summary
    display_configuration_summary(
        finetune_source or model_choice,
        dataset_name,
        config_file,
        generator.dataset_info,
        profile_selection,
        profile_context,
    )

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
    wizard_steps: list[str] | None = None,
    wizard_current_step: int | None = None,
    initial_selection: str | None = None,
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

    # Resolve raw key back to its display label for cursor pre-positioning
    resolved_initial: str | None = None
    if initial_selection is not None:
        for label, key in option_map.items():
            if key == initial_selection:
                resolved_initial = label
                break

    choice = get_user_choice(
        option_labels,
        allow_back=True,
        title=title,
        text=f"{prompt_text}{hint_block}",
        descriptions=descriptions,
        wizard_steps=wizard_steps,
        wizard_current_step=wizard_current_step,
        initial_selection=resolved_initial,
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
    wizard_steps: list[str] | None = None,
    wizard_current_step: int | None = None,
    dataset_path: str | None = None,
) -> dict[str, str] | None:
    summary_text = build_regular_yolo_profile_summary_text(
        dataset_name,
        profile_context,
        model_choice,
    )
    recommended_profiles = profile_context["recommended_profiles"]

    start_option_map = {
        "Recommended": "recommended",
        "AI Recommendation": "ai_recommendation",
        "Customize": "customize",
        "Fully Customized": "fully_customized",
    }
    start_descriptions = {
        "Recommended": "Fastest path - let YOLOmatic heuristics decide augmentation, compute, and worker settings for you.",
        "AI Recommendation": "AI path - let OpenAI/Gemini inspect your dataset samples/metadata and recommend custom hyperparameters.",
        "Customize": "Manual path - review and choose your own augmentation intensity, compute aggressiveness, and worker counts.",
        "Fully Customized": "Expert path - individually select and configure every training parameter with detailed explanations.",
        "Back": "Return to dataset selection.",
    }

    hint_block = build_hint_block(
        "Hints",
        [
            "Use the recommended option unless you already know you need more or less augmentation.",
            "AI Recommendation is ideal when you want multimodal models to optimize hyperparameters for your specific domain.",
            "Compute controls how hard YOLOmatic pushes memory and throughput.",
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
        wizard_steps=wizard_steps,
        wizard_current_step=wizard_current_step,
    )

    if initial_choice == "Back":
        return None
    if start_option_map[initial_choice] == "recommended":
        return dict(recommended_profiles)
    if start_option_map[initial_choice] == "fully_customized":
        return {"mode": "fully_customized"}
    if start_option_map[initial_choice] == "ai_recommendation":
        if not dataset_path:
            console.print("[bold red]Error: Dataset path is missing. Cannot run AI analysis.[/bold red]")
            input("\nPress Enter to return...")
            return None
        from yolomatic.utils.ai_client import run_ai_recommendation_flow
        res = run_ai_recommendation_flow(model_choice, dataset_path)
        if res is None:
            return None
        return {
            "mode": "ai_recommendation",
            "name": res["name"],
            "training_overrides": res["training_overrides"],
            "rationale": res["rationale"]
        }

    augmentation_options = {
        "minimum": "Essential training values only with almost no extra augmentation",
        "low": "Mild augmentation using flips, mosaic, and mixup",
        "medium": "Stronger generalization with color and geometric augmentation",
        "aerial_fused": (
            "Nadir aerial imagery with fused/false-colour channels - full rotation "
            "and both flips, no HSV jitter"
        ),
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

    # Step-machine so Back at sub-step N returns to sub-step N-1 with the
    # previous choice pre-selected, rather than exiting the whole flow.
    sub_state: dict[str, str] = {}
    sub_step = 0
    while 0 <= sub_step < 3:
        if sub_step == 0:
            result = select_profile_option(
                "Select Augmentation Profile",
                f"{summary_text}\n\nChoose the augmentation intensity for this dataset:",
                augmentation_options,
                recommended_profiles["augmentation"],
                [
                    "Minimum is the easiest to reason about and keeps the config close to core training values.",
                    "Low adds only basic robustness improvements.",
                    "Medium adds more color and geometric changes, which can improve generalization but also change training behavior more.",
                ],
                wizard_steps=wizard_steps,
                wizard_current_step=wizard_current_step,
                initial_selection=sub_state.get("augmentation"),
            )
        elif sub_step == 1:
            result = select_profile_option(
                "Select Compute Profile",
                f"{summary_text}\n\nChoose how strongly YOLOmatic should push system resources:",
                compute_options,
                recommended_profiles["compute"],
                [
                    "This profile mainly affects batch aggressiveness and cache behavior.",
                    "Conservative is better when GPU memory is tight or the model is heavy.",
                    "Aggressive is best only when your RAM, GPU memory, and dataset pressure all look healthy.",
                ],
                wizard_steps=wizard_steps,
                wizard_current_step=wizard_current_step,
                initial_selection=sub_state.get("compute"),
            )
        else:
            result = select_profile_option(
                "Select Worker Profile",
                f"{summary_text}\n\nChoose the dataloader worker profile:",
                worker_options,
                recommended_profiles["worker"],
                [
                    "Workers change throughput, not the optimization target, so higher values are not automatically better.",
                    "Too many workers can reduce training quality indirectly by causing CPU contention, RAM pressure, disk thrashing, and less stable batch preparation.",
                    "If you are unsure, keep the recommended worker profile and only raise it when the GPU is starved and the machine still has clear headroom.",
                ],
                wizard_steps=wizard_steps,
                wizard_current_step=wizard_current_step,
                initial_selection=sub_state.get("worker"),
            )

        if result is None:
            sub_step -= 1
        else:
            sub_state[["augmentation", "compute", "worker"][sub_step]] = result
            sub_step += 1

    if sub_step < 0:
        return None

    return {
        "augmentation": sub_state["augmentation"],
        "compute": sub_state["compute"],
        "worker": sub_state["worker"],
    }


def run_fully_customized_config_flow(
    dataset_name: str,
    model_choice: str,
    profile_context: dict[str, Any],
    initial_sections: dict[str, dict[str, Any]] | None = None,
    title: str = "Fully Customized Configuration",
    intro_text: str | None = None,
    wizard_steps: list[str] | None = None,
    wizard_current_step: int | None = None,
    parameters: list[ParameterDefinition] | None = None,
    default_selected: set[str] | None = None,
) -> dict[str, Any] | None:
    """
    Interactive flow for fully customized parameter selection.

    Allows users to check/uncheck individual parameters and set custom values
    with detailed explanations for each parameter. Supports bi-directional
    navigation and quick-select for boolean/enum types.

    ``parameters`` selects the catalog shown (defaults to the YOLO catalog); use
    :func:`parameters_for` to pass a task-filtered, family-specific catalog for
    RF-DETR or Detectron2.
    """
    from rich import box
    from rich.panel import Panel
    from rich.table import Table

    catalog = parameters if parameters is not None else YOLO_TRAINING_PARAMETERS
    param_lookup = {p.name: p for p in catalog}
    catalog_names = set(param_lookup)
    if default_selected is not None:
        default_selected_names = {n for n in default_selected if n in catalog_names}
    else:
        default_selected_names = {
            n
            for n in ("epochs", "patience", "batch", "imgsz", "device", "workers", "optimizer")
            if n in catalog_names
        }
    custom_values: dict[str, Any] = {}
    current_selected_names = set(default_selected_names)

    if initial_sections:
        current_selected_names = set()
        for param in catalog:
            section_values = initial_sections.get(param.config_section, {})
            if param.name in section_values:
                current_selected_names.add(param.name)
                custom_values[param.name] = section_values[param.name]

    while True:
        clear_screen()
        print_stylized_header(title)
        intro = intro_text or (
            "[bold yellow]Welcome to the Unified Configurator![/bold yellow]\n\n"
            "• [cyan]Left Pane[/cyan]: Select parameters with [bold yellow]Space[/bold yellow].\n"
            "• [cyan]Right Pane[/cyan]: Edit values with [bold yellow]Enter[/bold yellow] or [bold yellow]Right Arrow[/bold yellow].\n"
            "• [cyan]Navigation[/cyan]: Use [bold yellow]B[/bold yellow] or [bold yellow]Left Arrow[/bold yellow] to return to the list.\n"
            "• [cyan]Finish/Back[/cyan]: Press [bold yellow]F[/bold yellow] to finish or [bold yellow]Q[/bold yellow] to go back to the menu."
        )
        console.print(
            Panel(
                intro,
                border_style="cyan",
                padding=(1, 2),
            )
        )

        result = get_user_multi_select(
            parameters=catalog,
            title=title,
            instruction="[Space] Toggle  [Enter/→] Edit Value  [F] Finish",
            pre_selected=current_selected_names,
            pre_values=custom_values,
            wizard_steps=wizard_steps,
            wizard_current_step=wizard_current_step,
        )

        if result is None:
            return None

        selected_names, updated_values = result
        current_selected_names = selected_names
        custom_values = updated_values

        if not selected_names:
            console.print("[yellow]No parameters selected. Using defaults.[/yellow]")
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
        table.add_column("Saved To", style="dim cyan")

        # Only show parameters that are selected
        for name in sorted(selected_names):
            param = param_lookup.get(name)
            if param:
                val = custom_values.get(name, param.default)
                table.add_row(name, str(val), param.category, param.config_section)

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
            final_sections: dict[str, dict[str, Any]] = {}
            for name in selected_names:
                param = param_lookup[name]
                final_sections.setdefault(param.config_section, {})[name] = (
                    custom_values.get(name, param.default)
                )
            return {
                "mode": "fully_customized",
                "params": final_sections.get("training", {}),
                "sections": final_sections,
            }
        elif choice == "Back to Mode Selection":
            return None
        # Else choice == "Go Back and Modify", loop back to the unified configurator

    return {"mode": "fully_customized", "params": custom_values}


def get_model_menu():
    """Get the list of available model families grouped by category."""
    models = [
        "[Detection]",
        "detectron2",
        "rfdetr",
        "yolo26",
        "yolov12",
        "yolov11",
        "yolov10",
        "yolov9",
        "yolov8",
        "yolox",
        "[Segmentation]",
        "sam3.1",
        "detectron2-seg",
        "rfdetr-seg",
        "yolo26-sem",
        "yolo26-seg",
        "yolov12-seg",
        "yolov11-seg",
        "yolov9-seg",
        "yolov8-seg",
        "[Pose]",
        "rfdetr-pose",
        "yolo26-pose",
        "yolov11-pose",
        "yolov8-pose",
    ]
    return models


# Two-row block capitals, 3 cells wide each. Cheap to render, legible in any
# terminal that can show box-drawing characters, and it lines up without a font
# dependency.
_GOODBYE_ART = (
    "█▀▀ █▀█ █▀█ █▀▄ █▄▄ █▄█ █▀▀",
    "█▄█ █▄█ █▄█ █▄▀ █▄█ ░█░ █▄▄",
)
_GOODBYE_GRADIENT = ("nord.frost0.bold", "nord.frost2.bold")


def _render_goodbye() -> None:
    """Sign-off banner, matching the main menu's Nord gradient."""
    try:
        width = console.size.width
    except Exception:
        width = 80
    art_width = max(len(line) for line in _GOODBYE_ART)
    console.print()
    if width >= art_width + 4:
        for line, style in zip(_GOODBYE_ART, _GOODBYE_GRADIENT):
            console.print(Align.center(f"[{style}]{line}[/{style}]"))
    else:
        console.print(Align.center(f"[{_GOODBYE_GRADIENT[0]}]GOODBYE[/{_GOODBYE_GRADIENT[0]}]"))
    try:
        from yolomatic.__version__ import __version__

        tagline = f"Thanks for using YOLOmatic v{__version__}"
    except Exception:
        tagline = "Thanks for using YOLOmatic"
    console.print(Align.center(f"[dim]{tagline}[/dim]"))
    console.print()


def _exit_now(code: int = 0) -> None:
    """Terminate immediately after flushing, skipping interpreter teardown.

    A session that has touched training or augmentation leaves CUDA contexts,
    multiprocessing pools and large tensor graphs behind, and finalizing all of that
    can take seconds during which the TUI looks hung. Nothing in this project
    registers an ``atexit`` handler, so there is no cleanup left to lose.
    """
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(code)


# Subcommands of `yolomatic <command>`: module whose main() runs the tool.
SUBCOMMANDS: dict[str, tuple[str, str]] = {
    "train": ("yolomatic.trainers.yolo_trainer", "Train from a saved config (--config PATH)."),
    "predict": ("yolomatic.cli.predict", "Run inference on an image or folder."),
    "export": ("yolomatic.cli.export", "Export a trained model to deployment formats."),
    "benchmark": ("yolomatic.cli.benchmark", "Benchmark and compare models on a dataset."),
    "augment": ("yolomatic.cli.augment", "Offline Albumentations augmentation."),
    "prepare": ("yolomatic.cli.prepare_dataset", "Validate, convert and split a dataset."),
    "convert": ("yolomatic.cli.convert_ndjson", "Convert NDJSON exports to YOLO/COCO."),
    "sam": ("yolomatic.cli.sam_predict", "SAM 3.1 segmentation."),
    "upload": ("yolomatic.cli.upload", "Upload trained weights to Roboflow."),
    "upload-dataset": ("yolomatic.cli.upload_dataset", "Upload a dataset to Roboflow."),
    "tensorboard": ("yolomatic.cli.tensorboard_launcher", "Launch TensorBoard (--port, --logdir)."),
    "ultralytics": ("yolomatic.cli.ultralytics_platform", "Ultralytics Platform helpers."),
}


def build_cli_parser() -> argparse.ArgumentParser:

    from yolomatic.__version__ import __version__

    epilog = "commands:\n" + "\n".join(
        f"  {name:<15} {help_text}" for name, (_module, help_text) in SUBCOMMANDS.items()
    ) + (
        "\n\nRun `yolomatic <command> --help` for command options. Without a command the "
        "interactive TUI starts.\n\nexit codes: 0 success, 1 failure, 2 usage / no terminal, "
        "130 cancelled (Ctrl+C)."
    )
    parser = argparse.ArgumentParser(
        prog="yolomatic",
        description="YOLOmatic: configure, train, predict, benchmark and export vision models.",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"yolomatic {__version__}")
    parser.add_argument("--debug", action="store_true", help="Verbose logging and on-screen tracebacks.")
    parser.add_argument("--no-color", action="store_true", help="Disable colors (same as NO_COLOR=1).")
    parser.add_argument("--log-file", help="Write the session log to this file instead of the default.")
    parser.add_argument("command", nargs="?", choices=sorted(SUBCOMMANDS), metavar="command",
                        help="Optional command to run without the TUI (see below).")
    parser.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None):
    from yolomatic.utils.logging_setup import configure_logging

    cli_args = build_cli_parser().parse_args(sys.argv[1:] if argv is None else argv)
    if cli_args.debug:
        os.environ["YOLOMATIC_DEBUG"] = "1"
    if cli_args.no_color:
        os.environ["NO_COLOR"] = "1"
        console.no_color = True
    configure_logging(verbose=cli_args.debug, log_file=cli_args.log_file)

    if cli_args.command:
        import importlib

        from yolomatic.cli.result import run_standalone

        module_name, _help = SUBCOMMANDS[cli_args.command]
        sys.argv = [f"yolomatic {cli_args.command}", *cli_args.args]
        run_standalone(importlib.import_module(module_name).main)

    from yolomatic.utils.tui import NonInteractiveTerminalError, terminal_is_interactive

    if not terminal_is_interactive():
        console.print(
            "[bold red]YOLOmatic's interactive TUI needs a terminal.[/bold red] "
            "Run it in a terminal (`ssh -t` for remote sessions) or use a command: "
            "`yolomatic --help`."
        )
        raise SystemExit(2)

    while True:
        try:
            _main_loop_iteration()
        except NonInteractiveTerminalError as error:
            console.print(f"[bold red]{error}[/bold red]")
            raise SystemExit(2) from error
        except KeyboardInterrupt:
            # Ctrl+C at the main menu exits cleanly instead of dumping a trace.
            clear_screen()
            _render_goodbye()
            _exit_now()
        except _ExitTUI:
            clear_screen()
            _render_goodbye()
            _exit_now()
        except Exception as error:
            # Last-resort safety net — report and re-enter the menu rather than
            # crashing the whole TUI.
            from yolomatic.ui.theme import safe
            from yolomatic.utils.logging_setup import log_exception

            log_path = log_exception(error, "Unhandled error in the main menu")
            details = f"\n\n[dim]Full traceback: {safe(log_path)}[/dim]" if log_path else ""
            console.print(
                Panel(
                    f"[bold red]An unexpected error occurred:[/bold red] {safe(error)}{details}",
                    border_style="red",
                    padding=(1, 2),
                )
            )
            if os.environ.get("YOLOMATIC_DEBUG"):
                console.print(traceback.format_exc(), style="dim")
            try:
                input("\nPress Enter to return to the main menu...")
            except (EOFError, KeyboardInterrupt):
                return


class _ExitTUI(Exception):
    """Raised internally when the user chooses Exit from the main menu."""


def _settings_table(title: str, values: dict[str, Any]) -> None:
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="dim")
    table.add_column("Value")
    for key, value in values.items():
        if "api_key" in key:
            value = "configured" if value else "missing"
        table.add_row(key, str(value))
    console.print(Panel.fit(f"[bold]{title}[/bold]", style="bold blue"))
    console.print(table)


def _settings_definitions() -> list[ParameterDefinition]:
    return [
        ParameterDefinition(
            "enabled",
            "ClearML",
            True,
            "bool",
            "Enable ClearML",
            "Controls whether training initializes a ClearML task.",
            config_section="clearml",
        ),
        ParameterDefinition(
            "require_configured",
            "ClearML",
            False,
            "bool",
            "Require ClearML",
            "When true, training cancels if ClearML cannot initialize.",
            config_section="clearml",
        ),
        ParameterDefinition(
            "project_name_template",
            "ClearML",
            "{family} Training - {model}",
            "str",
            "Project template",
            "Supports {family} and {model}.",
            config_section="clearml",
        ),
        ParameterDefinition(
            "task_name_format",
            "ClearML",
            "%Y-%m-%d-%H-%M",
            "str",
            "Task timestamp format",
            "Python datetime format used in task names.",
            config_section="clearml",
        ),
        ParameterDefinition(
            "upload_final_model",
            "ClearML",
            True,
            "bool",
            "Upload final model",
            "Uploads best/last checkpoint as a ClearML artifact.",
            config_section="clearml",
        ),
        ParameterDefinition(
            "upload_artifacts",
            "ClearML",
            True,
            "bool",
            "Upload artifacts",
            "Reserved for generated artifacts beyond the final model.",
            config_section="clearml",
        ),
        ParameterDefinition(
            "log_hyperparameters",
            "ClearML",
            True,
            "bool",
            "Log hyperparameters",
            "Connects training, dataset, and export parameters to the task.",
            config_section="clearml",
        ),
        ParameterDefinition(
            "log_dataset_summary",
            "ClearML",
            True,
            "bool",
            "Log dataset summary",
            "Reserved for dataset summary logging.",
            config_section="clearml",
        ),
        ParameterDefinition(
            "upload_wizard_enabled",
            "Roboflow",
            True,
            "bool",
            "Enable manual upload wizard",
            "Controls whether Upload to Roboflow is available from the main TUI.",
            config_section="roboflow",
        ),
        ParameterDefinition(
            "auto_upload_after_training",
            "Roboflow",
            False,
            "bool",
            "Auto-upload after training",
            "New configs snapshot this as roboflow.upload.",
            config_section="roboflow",
        ),
        ParameterDefinition(
            "auto_upload_weight",
            "Roboflow",
            "best.pt",
            "str",
            "Auto-upload weight",
            "Usually best.pt or last.pt.",
            config_section="roboflow",
        ),
        ParameterDefinition(
            "default_model_name_template",
            "Roboflow",
            "{run_name}-best",
            "str",
            "Model name template",
            "Supports {run_name}.",
            config_section="roboflow",
        ),
        ParameterDefinition(
            "require_dataset_metadata",
            "Roboflow",
            True,
            "bool",
            "Require dataset metadata",
            "Skip auto upload when workspace/project metadata is unavailable.",
            config_section="roboflow",
        ),
        ParameterDefinition(
            "rfdetr_project_version",
            "Roboflow",
            1,
            "int",
            "RF-DETR version",
            "Default project version used for RF-DETR deploy.",
            min_value=1,
            config_section="roboflow",
        ),
        ParameterDefinition(
            "default_dataset_download_dir",
            "Ultralytics",
            "datasets/ultralytics/downloads",
            "str",
            "Dataset download dir",
            "Default directory for signed Platform dataset exports before preparation.",
            config_section="ultralytics",
        ),
        ParameterDefinition(
            "default_model_download_dir",
            "Ultralytics",
            "weights/ultralytics",
            "str",
            "Model download dir",
            "Default directory for Platform model weight downloads.",
            config_section="ultralytics",
        ),
        ParameterDefinition(
            "default_output_root",
            "Ultralytics",
            "datasets",
            "str",
            "Prepared output root",
            "Default root for datasets prepared from Platform exports.",
            config_section="ultralytics",
        ),
        ParameterDefinition(
            "mode",
            "Narratives",
            "guided",
            "str",
            "Narrative mode",
            "guided shows full panels, concise uses shorter messages, quiet only reports blockers and final results.",
            allowed_values=["guided", "concise", "quiet"],
            config_section="narratives",
        ),
        ParameterDefinition(
            "show_setup_guidance",
            "Narratives",
            True,
            "bool",
            "Show setup guidance",
            "Controls setup guidance text.",
            config_section="narratives",
        ),
        ParameterDefinition(
            "show_success_panels",
            "Narratives",
            True,
            "bool",
            "Show success panels",
            "Controls success panels.",
            config_section="narratives",
        ),
        ParameterDefinition(
            "show_skip_reasons",
            "Narratives",
            True,
            "bool",
            "Show skip reasons",
            "Controls expected skip messages.",
            config_section="narratives",
        ),
        ParameterDefinition(
            "provider",
            "AI",
            "Gemini",
            "str",
            "AI API Provider",
            "API Provider for AI dataset analysis and configuration recommendations.",
            allowed_values=["Gemini", "OpenAI"],
            config_section="ai",
        ),
        ParameterDefinition(
            "gemini_api_key",
            "AI",
            "",
            "str",
            "Gemini API Key",
            "API Key for Gemini recommendations (Google AI Studio).",
            config_section="ai",
        ),
        ParameterDefinition(
            "openai_api_key",
            "AI",
            "",
            "str",
            "OpenAI API Key",
            "API Key for OpenAI chat completions and vision suggestions.",
            config_section="ai",
        ),
        ParameterDefinition(
            "selected_model",
            "AI",
            "gemini-2.5-flash",
            "str",
            "AI Model Name",
            "Selected model for AI Recommendation flows.",
            config_section="ai",
        ),
    ]


def _settings_values(
    settings: dict[str, Any], definitions: list[ParameterDefinition]
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for definition in definitions:
        values[definition.name] = settings.get(definition.config_section, {}).get(
            definition.name,
            definition.default,
        )
    return values


def run_settings_customizer(
    title: str = "Global Integration Settings",
    section_filter: set[str] | None = None,
) -> bool:
    definitions = [
        definition
        for definition in _settings_definitions()
        if section_filter is None or definition.config_section in section_filter
    ]
    settings = load_settings()
    result = get_user_multi_select(
        parameters=definitions,
        title=title,
        instruction="[Enter/→] Edit  [F] Save Settings  [Q] Back",
        pre_selected={definition.name for definition in definitions},
        pre_values=_settings_values(settings, definitions),
    )
    if result is None:
        return False

    _selected, values = result
    for definition in definitions:
        settings.setdefault(definition.config_section, {})[definition.name] = (
            values.get(
                definition.name,
                definition.default,
            )
        )
    save_settings(settings)
    console.print("[bold green]Settings saved.[/bold green]")
    input("\nPress Enter to return...")
    return True


def settings_clearml_page() -> None:
    run_settings_customizer("ClearML Integration", {"clearml"})


def settings_roboflow_page() -> None:
    run_settings_customizer("Roboflow Integration", {"roboflow"})


def settings_ultralytics_page() -> None:
    run_settings_customizer("Ultralytics Platform", {"ultralytics"})


def settings_narratives_page() -> None:
    run_settings_customizer("Integration Narratives", {"narratives"})


def settings_credentials_page() -> None:
    status = roboflow_credential_status()
    ultralytics_status = ultralytics_credential_status()
    rows = {
        "ROBOFLOW_API_KEY": "configured" if status["api_key"] else "missing",
        "ROBOFLOW_WORKSPACE": "configured" if status["workspace"] else "missing",
        "ROBOFLOW_PROJECT_IDS": "configured" if status["project_ids"] else "missing",
        "ULTRALYTICS_API_KEY": "configured" if ultralytics_status["api_key"] else "missing",
    }
    _settings_table("Credential Status", rows)
    console.print(
        "[dim]API key values are never displayed or written to YAML settings.[/dim]"
    )
    input("\nPress Enter to return...")


def settings_ai_page() -> None:
    from yolomatic.ui.theme import gpu_badge_markup
    from yolomatic.utils.ai_client import (
        FALLBACK_MODELS,
        fetch_multimodal_models,
        is_free_tier_gemini_model,
    )
    from yolomatic.utils.tui import (
        TUI_CONSOLE,
        TUI_TERM,
        TUIState,
        get_gpu_status,
        is_enter_key,
        make_panel,
    )

    def _order_models(models: list[str], prov: str) -> list[str]:
        """For Gemini, list free-tier (Flash) models first so the default lands on
        a model that actually works on a free key. Stable within each group."""
        if prov == "Gemini":
            return sorted(models, key=lambda m: not is_free_tier_gemini_model(m))
        return list(models)
    import threading

    from rich.console import Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    class AISettingsRenderer:
        """Renderer for the unified dual-panel AI Recommendation Settings screen."""

        def __init__(
            self,
            provider: str,
            gemini_api_key: str,
            openai_api_key: str,
            selected_model: str,
            current_idx: int,
            focus: str,
            input_buffer: str,
            mask_keys: bool,
            test_result: tuple[bool, str] | None,
            is_testing: bool,
            model_list: list[str],
            model_scroll_index: int,
            is_fetching: bool = False,
            fetch_result: tuple[bool, str] | None = None,
            gemini_models: list[str] | None = None,
            openai_models: list[str] | None = None,
        ):
            self.provider = provider
            self.gemini_api_key = gemini_api_key
            self.openai_api_key = openai_api_key
            self.selected_model = selected_model
            self.current_idx = current_idx
            self.focus = focus
            self.input_buffer = input_buffer
            self.mask_keys = mask_keys
            self.test_result = test_result
            self.is_testing = is_testing
            self.model_list = model_list
            self.model_scroll_index = model_scroll_index
            self.is_fetching = is_fetching
            self.fetch_result = fetch_result
            self.gemini_models = list(gemini_models) if gemini_models is not None else list(FALLBACK_MODELS.get("Gemini", []))
            self.openai_models = list(openai_models) if openai_models is not None else list(FALLBACK_MODELS.get("OpenAI", []))

            self.sidebar_items = [
                ("provider", "API Provider"),
                ("gemini_key", "Gemini API Key"),
                ("openai_key", "OpenAI API Key"),
                ("model", "Selected Model"),
                ("fetch", "Fetch Latest Models"),
                ("test", "Test Connection"),
                ("save", "Save & Exit"),
            ]

        def _render_sidebar(self) -> Panel:
            items = []
            for i, (key, label) in enumerate(self.sidebar_items):
                is_active = i == self.current_idx
                
                # Show current value inside the label for context
                val_suffix = ""
                if key == "provider":
                    val_suffix = f": [cyan]{self.provider}[/cyan]"
                elif key == "gemini_key":
                    if self.gemini_api_key:
                        masked = f"***{self.gemini_api_key[-4:]}" if len(self.gemini_api_key) > 4 else "configured"
                        val_suffix = f": [dim]{masked}[/dim]"
                    else:
                        val_suffix = ": [red]not set[/red]"
                elif key == "openai_key":
                    if self.openai_api_key:
                        masked = f"***{self.openai_api_key[-4:]}" if len(self.openai_api_key) > 4 else "configured"
                        val_suffix = f": [dim]{masked}[/dim]"
                    else:
                        val_suffix = ": [red]not set[/red]"
                elif key == "model":
                    val_suffix = f": [yellow]{self.selected_model}[/yellow]"
                elif key == "fetch":
                    if self.is_fetching:
                        val_suffix = ": [yellow]fetching...[/yellow]"
                    elif self.fetch_result is not None and self.fetch_result[0]:
                        val_suffix = ": [green]loaded[/green]"
                    
                if is_active and self.focus == "sidebar":
                    text = Text.from_markup(f"[bold yellow]➤ [/bold yellow][bold white]{label}[/bold white]{val_suffix}")
                    text.stylize("on blue")
                else:
                    if is_active:
                        text = Text.from_markup(f"  [bold]{label}[/bold]{val_suffix}")
                    else:
                        text = Text.from_markup(f"  [dim white]{label}[/dim white]{val_suffix}")
                    
                items.append(text)
                items.append(Text(""))
                
            return Panel(
                Group(*items[:-1]),
                title="[bold cyan]AI Recommendation Configuration[/bold cyan]",
                border_style="blue" if self.focus == "sidebar" else "dim",
                padding=(1, 2),
                expand=True,
            )

        def _render_content(self) -> Panel:
            key, label = self.sidebar_items[self.current_idx]
            content_items = []
            
            # Build header for right card
            content_items.append(Text(f"{label} Configuration", style="bold yellow underline"))
            content_items.append(Text(""))
            
            if key == "provider":
                content_items.append(Text("YOLOMatic uses Large Language Models to analyze your dataset metadata (images, format, distribution, aspect ratios) and automatically recommend optimized training configs to eliminate trial-and-error.", style="dim"))
                content_items.append(Text(""))
                content_items.append(Text("Selecting a Provider:", style="bold cyan"))
                content_items.append(Text("  • Google Gemini: Fast, highly cost-effective, and powerful multimodal reasoning (Gemini 2.5 Flash is recommended). Setup with a key from Google AI Studio.", style="dim"))
                content_items.append(Text("  • OpenAI GPT: Industry standard model backend. Requires an OpenAI API key.", style="dim"))
                content_items.append(Text(""))
                
                # Show current provider selection indicator
                prov_table = Table.grid(expand=True)
                prov_table.add_column(ratio=1)
                prov_table.add_column(ratio=1)
                
                gemini_style = "bold white on purple" if self.provider == "Gemini" else "dim"
                openai_style = "bold white on green" if self.provider == "OpenAI" else "dim"
                
                gemini_box = Panel(Text("Google Gemini", justify="center", style=gemini_style), border_style="purple" if self.provider == "Gemini" else "dim")
                openai_box = Panel(Text("OpenAI GPT", justify="center", style=openai_style), border_style="green" if self.provider == "OpenAI" else "dim")
                
                prov_table.add_row(gemini_box, openai_box)
                content_items.append(prov_table)
                
                if self.focus == "select_provider":
                    content_items.append(Text(""))
                    content_items.append(Text("➤ Select provider using [Left/Right] arrows and press [Enter].", style="bold yellow"))
                else:
                    content_items.append(Text(""))
                    content_items.append(Text("Press [Enter/Space] or [→] to select provider.", style="dim green"))
                    
            elif key in ("gemini_key", "openai_key"):
                is_gemini = key == "gemini_key"
                prov_name = "Google Gemini" if is_gemini else "OpenAI"
                url = "https://aistudio.google.com/" if is_gemini else "https://platform.openai.com/"
                current_key_val = self.gemini_api_key if is_gemini else self.openai_api_key
                
                content_items.append(Text(f"Configure your API Key for {prov_name} access.", style="bold cyan"))
                content_items.append(Text(""))
                content_items.append(Text.from_markup(f"To generate a key, visit: [bold magenta underline]{url}[/bold magenta underline]", style="dim"))
                content_items.append(Text("This key is stored purely locally in your configuration file and never shared with external services.", style="dim italic"))
                content_items.append(Text(""))
                
                if self.focus == "input_key":
                    display_key = self.input_buffer
                    if self.mask_keys and display_key:
                        display_key = "*" * (len(display_key) - 4) + display_key[-4:] if len(display_key) > 4 else "*" * len(display_key)
                    
                    input_panel = Panel(
                        Text(f"  {display_key}█", style="bold yellow"),
                        title="[bold green]Enter API Key (Typing Mode)[/bold green]",
                        border_style="green",
                        padding=(1, 1)
                    )
                    content_items.append(input_panel)
                    content_items.append(Text(""))
                    content_items.append(Text("  [Enter] Confirm & Keep  [Esc] Cancel  [F2] Toggle Key Visibility", style="dim green"))
                else:
                    display_key = current_key_val
                    if not display_key:
                        display_key = "< Not Configured >"
                        key_style = "bold red"
                    else:
                        key_style = "bold cyan"
                        if self.mask_keys:
                            display_key = "*" * (len(display_key) - 4) + display_key[-4:] if len(display_key) > 4 else "*" * len(display_key)
                    
                    content_items.append(Panel(Text(f"  {display_key}", style=key_style), title="Configured API Key", border_style="dim"))
                    content_items.append(Text(""))
                    content_items.append(Text("Press [Enter] or [→] to edit API key.", style="dim green"))
                    if current_key_val:
                        content_items.append(Text("Press [F2] to toggle key visibility.", style="dim"))
                        
            elif key == "model":
                content_items.append(Text(f"Configure the active multimodal model for {self.provider} API recommendations.", style="bold cyan"))
                content_items.append(Text(""))
                content_items.append(Text("Choosing a smaller, faster model (like gemini-2.5-flash) is highly recommended for quick hyperparameter recommendations.", style="dim"))
                content_items.append(Text(""))
                content_items.append(Text.from_markup(f"Active Model: [bold yellow]{self.selected_model}[/bold yellow]"))
                content_items.append(Text(""))
                
                if self.focus == "select_model":
                    content_items.append(Text("Navigate & select a model:", style="bold cyan"))
                    content_items.append(Text(""))
                    
                    models = self.model_list if self.model_list else ["gemini-2.5-flash"]
                    visible_count = 5
                    start = max(0, min(self.model_scroll_index - visible_count // 2, len(models) - visible_count))
                    end = min(start + visible_count, len(models))
                    
                    model_rows = []
                    for idx in range(start, end):
                        m = models[idx]
                        is_current_active = m == self.selected_model
                        is_highlighted = idx == self.model_scroll_index
                        
                        row = Text()
                        if is_highlighted:
                            row.append("➤ ", style="bold yellow")
                            row.append(f"{m}", style="bold white on blue")
                        else:
                            row.append("  ")
                            row.append(f"{m}", style="dim" if not is_current_active else "yellow")

                        # Flag free-tier eligibility so users avoid Pro models that
                        # are rejected on a free key. Display-only — the stored
                        # model id (m) is never altered.
                        if self.provider == "Gemini":
                            if is_free_tier_gemini_model(m):
                                row.append(" (free tier)", style="green")
                            else:
                                row.append(" (no free tier)", style="red")

                        if is_current_active:
                            row.append(" [✓ active]", style="bold green")

                        model_rows.append(row)
                    
                    content_items.append(Panel(Group(*model_rows), title="Available Models", border_style="blue"))
                    content_items.append(Text(""))
                    content_items.append(Text("  [↑/↓] Navigate  [Enter] Choose  [Esc/←] Back", style="dim green"))
                else:
                    if self.is_fetching:
                        content_items.append(Panel(
                            Text("⟳ Fetching live model list from provider...", style="bold yellow"),
                            border_style="yellow",
                            title="API Request"
                        ))
                    else:
                        content_items.append(Text("Press [Enter] or [→] to select a model from the list.", style="dim green"))
                        content_items.append(Text("Press [F] to fetch the latest models from the API.", style="dim magenta"))
                    
            elif key == "test":
                content_items.append(Text("Verify your AI integration settings.", style="bold cyan"))
                content_items.append(Text(""))
                content_items.append(Text("Runs an authentication check and fetches models to make sure credentials are correct.", style="dim"))
                content_items.append(Text(""))
                
                active_key = self.gemini_api_key if self.provider == "Gemini" else self.openai_api_key
                if not active_key:
                    content_items.append(Panel(
                        Text("✗ Cannot Test: API key is not configured yet. Please configure the API Key first.", style="bold red"),
                        border_style="red",
                        title="Missing Configuration"
                    ))
                elif self.is_testing:
                    content_items.append(Panel(
                        Text("⟳ Connecting to API... Testing authentication credentials...", style="bold yellow"),
                        border_style="yellow",
                        title="Active Test"
                    ))
                elif self.test_result is not None:
                    success, msg = self.test_result
                    if success:
                        content_items.append(Panel(
                            Text(f"✓ Success! API key verified.\n\n{msg}", style="bold green"),
                            border_style="green",
                            title="Connection Success"
                        ))
                    else:
                        content_items.append(Panel(
                            Text(f"✗ Failed! API connection error.\n\n{msg}", style="bold red"),
                            border_style="red",
                            title="Connection Failed"
                        ))
                else:
                    content_items.append(Text("Press [Enter] to run the connection test now.", style="bold green"))
                    
            elif key == "fetch":
                content_items.append(Text("Fetch the latest available models from the provider's API.", style="bold cyan"))
                content_items.append(Text(""))
                content_items.append(Text("This queries the active provider to discover new or updated LLMs. The fetched models will instantly populate the 'Selected Model' list.", style="dim"))
                content_items.append(Text(""))
                
                active_key = self.gemini_api_key if self.provider == "Gemini" else self.openai_api_key
                if not active_key:
                    content_items.append(Panel(
                        Text("✗ Cannot Fetch: API key is not configured. Please configure the API Key first.", style="bold red"),
                        border_style="red",
                        title="Missing Configuration"
                    ))
                elif self.is_fetching:
                    content_items.append(Panel(
                        Text("⟳ Fetching model catalog from API... Please wait...", style="bold yellow"),
                        border_style="yellow",
                        title="Active Fetch"
                    ))
                elif self.fetch_result is not None:
                    success, msg = self.fetch_result
                    if success:
                        content_items.append(Panel(
                            Text(f"✓ Success! Model catalog updated.\n\n{msg}", style="bold green"),
                            border_style="green",
                            title="Fetch Success"
                        ))
                    else:
                        content_items.append(Panel(
                            Text(f"✗ Failed! Model catalog fetch failed.\n\n{msg}", style="bold red"),
                            border_style="red",
                            title="Fetch Failed"
                        ))
                else:
                    content_items.append(Text("Press [Enter] to query and update the model catalog.", style="bold green"))

            elif key == "save":
                content_items.append(Text("Save settings and exit.", style="bold cyan"))
                content_items.append(Text(""))
                content_items.append(Text("Apply and save all changes directly to yolomatic_settings.yaml and return to the main settings page.", style="dim"))
                content_items.append(Text(""))
                
                # Show summary of what will be saved
                summary_lines = [
                    f"[bold]Provider:[/bold] [cyan]{self.provider}[/cyan]",
                    "[bold]Gemini API Key:[/bold] " + (f"[cyan]***{self.gemini_api_key[-4:]}[/cyan]" if (self.gemini_api_key and len(self.gemini_api_key) > 4) else ("[green]configured[/green]" if self.gemini_api_key else "[red]not set[/red]")),
                    "[bold]OpenAI API Key:[/bold] " + (f"[cyan]***{self.openai_api_key[-4:]}[/cyan]" if (self.openai_api_key and len(self.openai_api_key) > 4) else ("[green]configured[/green]" if self.openai_api_key else "[red]not set[/red]")),
                    f"[bold]AI Model:[/bold] [yellow]{self.selected_model}[/yellow]",
                ]
                content_items.append(Panel(Text.from_markup("\n".join(summary_lines)), title="Pending Changes Summary", border_style="green"))
                content_items.append(Text(""))
                content_items.append(Text("Press [Enter] to save and return.", style="bold green"))
                
            return Panel(
                Group(*content_items),
                title=f"[bold cyan]{label} Information[/bold cyan]",
                border_style="dim",
                padding=(1, 2),
                expand=True,
            )

        def _render_status_bar(self) -> Panel:
            hints = []
            if self.focus == "sidebar":
                hints.append("[↑/↓] Navigate")
                hints.append("[Enter/→] Select/Edit")
                if self.sidebar_items[self.current_idx][0] == "model":
                    hints.append("[F] Fetch Models")
                elif self.sidebar_items[self.current_idx][0] in ("gemini_key", "openai_key"):
                    hints.append("[F2] Mask Key")
                hints.append("[Esc/Q] Back")
            elif self.focus == "select_provider":
                hints.append("[←/→] Select Provider")
                hints.append("[Enter] Confirm")
                hints.append("[Esc] Cancel")
            elif self.focus == "input_key":
                hints.append("[Type] Enter key characters")
                hints.append("[Backspace] Delete")
                hints.append("[F2] Toggle Masking")
                hints.append("[Enter] Confirm")
                hints.append("[Esc] Cancel")
            elif self.focus == "select_model":
                hints.append("[↑/↓] Navigate Models")
                hints.append("[Enter] Select Model")
                hints.append("[Esc/←] Back")
                
            status_table = Table.grid(expand=True)
            status_table.add_column(justify="left", ratio=1)
            status_table.add_column(justify="right")

            gpu_markup = gpu_badge_markup(get_gpu_status())

            status_table.add_row(
                Text.from_markup("  " + "  │  ".join(hints), style="bold white"),
                Text.from_markup(f"{gpu_markup}  ")
            )
            
            return Panel(status_table, border_style="dim", padding=(0, 1))

        def __rich__(self) -> Layout:
            layout = Layout()
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="body"),
                Layout(name="footer", size=3),
            )
            
            layout["header"].update(
                make_panel(
                    Text("AI Recommendation settings - Unified Panel", style="bold cyan", justify="center"),
                    state=TUIState.INFO,
                    padding=(0, 1),
                )
            )
            
            layout["body"].split_row(
                Layout(name="sidebar", ratio=2),
                Layout(name="content", ratio=3),
            )
            
            layout["body"]["sidebar"].update(self._render_sidebar())
            layout["body"]["content"].update(self._render_content())
            layout["footer"].update(self._render_status_bar())
            
            return layout

    settings = load_settings()
    ai_config = settings.setdefault("ai", {})
    provider = ai_config.get("provider", "Gemini")
    selected_model = ai_config.get("selected_model", "gemini-2.5-flash")
    gemini_key = ai_config.get("gemini_api_key", "")
    openai_key = ai_config.get("openai_api_key", "")
    gemini_models = ai_config.get("gemini_models", list(FALLBACK_MODELS.get("Gemini", [])))
    openai_models = ai_config.get("openai_models", list(FALLBACK_MODELS.get("OpenAI", [])))

    current_idx = 0
    focus = "sidebar"
    input_buffer = ""
    mask_keys = True
    test_result = None
    fetch_result = None
    is_testing = False
    is_fetching = False
    model_list = _order_models(list(gemini_models if provider == "Gemini" else openai_models), provider)
    if selected_model and selected_model not in model_list:
        model_list.insert(0, selected_model)
    model_scroll_index = 0
    if selected_model in model_list:
        model_scroll_index = model_list.index(selected_model)

    renderer = AISettingsRenderer(
        provider=provider,
        gemini_api_key=gemini_key,
        openai_api_key=openai_key,
        selected_model=selected_model,
        current_idx=current_idx,
        focus=focus,
        input_buffer=input_buffer,
        mask_keys=mask_keys,
        test_result=test_result,
        is_testing=is_testing,
        model_list=model_list,
        model_scroll_index=model_scroll_index,
        is_fetching=is_fetching,
        fetch_result=fetch_result,
        gemini_models=gemini_models,
        openai_models=openai_models,
    )

    from yolomatic.utils.tui import ensure_interactive_terminal, mouse_context

    ensure_interactive_terminal()
    with TUI_TERM.cbreak(), TUI_TERM.hidden_cursor(), mouse_context():
        last_scroll_time = 0.0
        with Live(renderer, console=TUI_CONSOLE, refresh_per_second=10, screen=True) as live:
            while True:
                try:
                    key = TUI_TERM.inkey(timeout=0.1)
                except KeyboardInterrupt:
                    # Ctrl-C leaves this page (unsaved edits are discarded), it no
                    # longer quits the whole application.
                    return
                if not key:
                    continue
                
                key_name = key.name
                key_lower = key.lower()

                is_scroll_up = key_name is not None and "SCROLL_UP" in key_name
                is_scroll_down = key_name is not None and "SCROLL_DOWN" in key_name

                if is_scroll_up or is_scroll_down:
                    now = time.time()
                    if now - last_scroll_time < 0.03:
                        continue
                    last_scroll_time = now

                # Active sidebar item key
                active_item_key, _ = renderer.sidebar_items[renderer.current_idx]

                if renderer.focus == "sidebar":
                    if key_name == "KEY_UP" or key_lower == "k" or is_scroll_up:
                        renderer.current_idx = (renderer.current_idx - 1) % len(renderer.sidebar_items)
                        renderer.test_result = None
                        renderer.fetch_result = None
                    elif key_name == "KEY_DOWN" or key_lower == "j" or is_scroll_down:
                        renderer.current_idx = (renderer.current_idx + 1) % len(renderer.sidebar_items)
                        renderer.test_result = None
                        renderer.fetch_result = None
                    elif key.name == "KEY_RIGHT" or is_enter_key(key):
                        if active_item_key == "provider":
                            renderer.focus = "select_provider"
                        elif active_item_key in ("gemini_key", "openai_key"):
                            renderer.focus = "input_key"
                            renderer.input_buffer = renderer.gemini_api_key if active_item_key == "gemini_key" else renderer.openai_api_key
                        elif active_item_key == "model":
                            renderer.focus = "select_model"
                            if not renderer.model_list:
                                renderer.model_list = list(renderer.gemini_models if renderer.provider == "Gemini" else renderer.openai_models)
                            if renderer.selected_model and renderer.selected_model not in renderer.model_list:
                                renderer.model_list.insert(0, renderer.selected_model)
                            try:
                                renderer.model_scroll_index = renderer.model_list.index(renderer.selected_model)
                            except ValueError:
                                renderer.model_scroll_index = 0
                        elif active_item_key == "fetch":
                            active_key = renderer.gemini_api_key if renderer.provider == "Gemini" else renderer.openai_api_key
                            if active_key and not renderer.is_fetching:
                                renderer.is_fetching = True
                                renderer.fetch_result = None
                                def run_fetch_sidebar():
                                    try:
                                        models = _order_models(fetch_multimodal_models(renderer.provider, active_key), renderer.provider)
                                        if models:
                                            renderer.fetch_result = (True, f"Connected to {renderer.provider} successfully!\nFetched {len(models)} models.")
                                            renderer.model_list = models
                                            if renderer.provider == "Gemini":
                                                renderer.gemini_models = models
                                            else:
                                                renderer.openai_models = models
                                        else:
                                            renderer.fetch_result = (True, f"Connected to {renderer.provider} successfully, but fetched 0 models.")
                                    except Exception as e:
                                        renderer.fetch_result = (False, str(e))
                                    renderer.is_fetching = False
                                threading.Thread(target=run_fetch_sidebar, daemon=True).start()
                        elif active_item_key == "test":
                            active_key = renderer.gemini_api_key if renderer.provider == "Gemini" else renderer.openai_api_key
                            if active_key and not renderer.is_testing:
                                renderer.is_testing = True
                                renderer.test_result = None
                                def run_test():
                                    try:
                                        models = _order_models(fetch_multimodal_models(renderer.provider, active_key), renderer.provider)
                                        if models:
                                            renderer.test_result = (True, f"Connected to {renderer.provider} successfully!\nFetched {len(models)} models.")
                                            # Update the model list too while we are at it
                                            renderer.model_list = models
                                            if renderer.provider == "Gemini":
                                                renderer.gemini_models = models
                                            else:
                                                renderer.openai_models = models
                                        else:
                                            renderer.test_result = (True, f"Connected to {renderer.provider} successfully, but fetched 0 models.")
                                    except Exception as e:
                                        renderer.test_result = (False, str(e))
                                    renderer.is_testing = False
                                threading.Thread(target=run_test, daemon=True).start()
                        elif active_item_key == "save":
                            # Save and exit
                            ai_config["provider"] = renderer.provider
                            ai_config["selected_model"] = renderer.selected_model
                            ai_config["gemini_api_key"] = renderer.gemini_api_key
                            ai_config["openai_api_key"] = renderer.openai_api_key
                            ai_config["gemini_models"] = renderer.gemini_models
                            ai_config["openai_models"] = renderer.openai_models
                            save_settings(settings)
                            break
                    elif key == " ":
                        if active_item_key == "provider":
                            # Instant toggle on Space
                            renderer.provider = "OpenAI" if renderer.provider == "Gemini" else "Gemini"
                            renderer.model_list = list(renderer.openai_models if renderer.provider == "OpenAI" else renderer.gemini_models)
                            renderer.selected_model = renderer.model_list[0] if renderer.model_list else ""
                    elif key.name == "KEY_F2" or key == "f2":
                        renderer.mask_keys = not renderer.mask_keys
                    elif key.lower() == "f" and active_item_key == "model":
                        active_key = renderer.gemini_api_key if renderer.provider == "Gemini" else renderer.openai_api_key
                        if active_key and not renderer.is_fetching:
                            renderer.is_fetching = True
                            def run_fetch():
                                try:
                                    models = fetch_multimodal_models(renderer.provider, active_key)
                                    if models:
                                        renderer.model_list = models
                                        if renderer.provider == "Gemini":
                                            renderer.gemini_models = models
                                        else:
                                            renderer.openai_models = models
                                except Exception:
                                    pass
                                renderer.is_fetching = False
                            threading.Thread(target=run_fetch, daemon=True).start()
                    elif key.lower() == "q" or key.name == "KEY_ESCAPE":
                        # Discard changes
                        break

                elif renderer.focus == "select_provider":
                    if key.name in ("KEY_LEFT", "KEY_RIGHT", "KEY_UP", "KEY_DOWN") or key.lower() in ("h", "l", "k", "j"):
                        renderer.provider = "OpenAI" if renderer.provider == "Gemini" else "Gemini"
                        renderer.model_list = list(renderer.openai_models if renderer.provider == "OpenAI" else renderer.gemini_models)
                        renderer.selected_model = renderer.model_list[0] if renderer.model_list else ""
                    elif is_enter_key(key):
                        renderer.focus = "sidebar"
                    elif key.name == "KEY_ESCAPE":
                        renderer.focus = "sidebar"

                elif renderer.focus == "input_key":
                    if key.name == "KEY_ESCAPE":
                        renderer.focus = "sidebar"
                    elif is_enter_key(key):
                        if active_item_key == "gemini_key":
                            renderer.gemini_api_key = renderer.input_buffer.strip()
                        else:
                            renderer.openai_api_key = renderer.input_buffer.strip()
                        renderer.focus = "sidebar"
                    elif key.name == "KEY_F2" or key == "f2":
                        renderer.mask_keys = not renderer.mask_keys
                    elif key.name == "KEY_BACKSPACE" or str(key) in {"\x08", "\x7f"}:
                        renderer.input_buffer = renderer.input_buffer[:-1]
                    elif key and not key.is_sequence:
                        renderer.input_buffer += key

                elif renderer.focus == "select_model":
                    if key_name == "KEY_UP" or key_lower == "k" or is_scroll_up:
                        if renderer.model_list:
                            renderer.model_scroll_index = (renderer.model_scroll_index - 1) % len(renderer.model_list)
                    elif key_name == "KEY_DOWN" or key_lower == "j" or is_scroll_down:
                        if renderer.model_list:
                            renderer.model_scroll_index = (renderer.model_scroll_index + 1) % len(renderer.model_list)
                    elif is_enter_key(key):
                        if renderer.model_list:
                            renderer.selected_model = renderer.model_list[renderer.model_scroll_index]
                        renderer.focus = "sidebar"
                    elif key.name in ("KEY_ESCAPE", "KEY_LEFT") or key.lower() == "h":
                        renderer.focus = "sidebar"

                live.update(renderer)
                if key:
                    live.refresh()


def settings_reset_page() -> None:
    choice = get_user_choice(
        ["Reset to Defaults", "Cancel"],
        title="Reset Settings",
        text="Restore configs/yolomatic_settings.yaml to built-in defaults?",
        back_option="Cancel",
    )
    if choice == "Reset to Defaults":
        reset_settings()
        console.print("[bold green]Settings reset to defaults.[/bold green]")
        input("\nPress Enter to return...")


def settings_menu() -> None:
    last_choice: str | None = None
    while True:
        choice = get_user_choice(
            [
                "Customize All Settings",
                "ClearML Integration",
                "Roboflow Integration",
                "Ultralytics Platform",
                "Integration Narratives",
                "AI Recommendations",
                "Credential Status",
                "Reset to Defaults",
                "Back",
            ],
            title="Settings",
            text="Configure global integration defaults:",
            initial_selection=last_choice,
        )
        last_choice = choice
        if choice == "Back":
            return
        if choice == "Customize All Settings":
            run_settings_customizer()
        elif choice == "ClearML Integration":
            settings_clearml_page()
        elif choice == "Roboflow Integration":
            settings_roboflow_page()
        elif choice == "Ultralytics Platform":
            settings_ultralytics_page()
        elif choice == "Integration Narratives":
            settings_narratives_page()
        elif choice == "AI Recommendations":
            settings_ai_page()
        elif choice == "Credential Status":
            settings_credentials_page()
        elif choice == "Reset to Defaults":
            settings_reset_page()


def _main_loop_iteration():
    last_choice: str | None = None
    while True:
        clear_screen()
        print_stylized_header("YOLOmatic Model Selector")

        # Full workflow surface: configure, train, predict, monitor, publish,
        # and curate datasets — all routed through the same TUI.
        main_menu_options = [
            "[Configure & Train]",
            "Configure Model",
            "Clone Config",
            "Configure Fine-Tune",
            "Train Model",
            "[Evaluate & Monitor]",
            "Run Prediction",
            "Launch TensorBoard",
            "Benchmark Models",
            "SAM Segment",
            "[Datasets & Deployment]",
            "Export / Compile Model",
            "Prepare / Split Dataset",
            "Convert Dataset Format",
            "Combine Datasets",
            "Augment Dataset",
            "Ultralytics Platform",
            "Upload to Roboflow",
            "[Maintenance]",
            "Settings",
            "Check for Updates",
            "About YOLOmatic",
            "Exit",
        ]
        if not load_settings().get("roboflow", {}).get("upload_wizard_enabled", True):
            main_menu_options.remove("Upload to Roboflow")

        main_choice = get_user_choice(
            main_menu_options,
            title="Main Menu",
            text="Pick a task to begin:",
            initial_selection=last_choice,
            descriptions={
                "Export / Compile Model": (
                    "Export and compile trained YOLO weights (.pt) to high-performance formats "
                    "like TensorRT (engine), ONNX, CoreML, OpenVINO, etc."
                ),
                "Convert Dataset Format": (
                    "Convert Labelbox or Ultralytics NDJSON exports into YOLO or COCO formats. "
                    "Supports concurrent image downloads and handles boxes, polygons, and pose keypoints."
                ),
                "Prepare / Split Dataset": (
                    "Create a versioned, training-ready YOLO or COCO dataset from a YOLO folder, "
                    "COCO folder, or Labelbox NDJSON export. Includes guided split presets, "
                    "class-balanced assignment, validation warnings, and manifest metadata."
                ),
                "Configure Model": (
                    "Walk through the YOLOmatic wizard to pick a YOLO or RF-DETR family, choose a "
                    "variant that fits your hardware, and auto-generate a training YAML "
                    "tailored to your dataset and system resources."
                ),
                "Configure Fine-Tune": (
                    "Find an existing Ultralytics .pt or RF-DETR .pth checkpoint, bind it to a dataset, "
                    "and generate a fresh fine-tuning YAML using YOLOmatic's current "
                    "hardware-aware recommendations."
                ),
                "Clone Config": (
                    "Start from a saved YAML in ./configs, automatically refresh the "
                    "dataset/model paths for a new target dataset, review the copied "
                    "training/export values, and save a new config."
                ),
                "Train Model": (
                    "Train (and validate + export) a YOLO or RF-DETR model using one of "
                    "the saved configs under ./configs. Routes each model family to its native trainer."
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
                "Benchmark Models": (
                    "Evaluate one or more trained checkpoints (.pt) on a COCO-format "
                    "validation dataset. Computes mAP@50, mAP@50:95, F1, precision, recall, "
                    "and per-image rankings split by object size. Generates an interactive "
                    "Plotly HTML report with UMAP vector analysis for exploring model "
                    "strengths and failure cases — works for both detection and segmentation."
                ),
                "SAM Segment": (
                    "Run SAM 3.1 (Meta's Segment Anything Model) inference on images.\n\n"
                    "Three modes: Auto (segment everything without prompts), "
                    "Text-prompted (open-vocabulary concept labels), "
                    "and Box-prompted (use YOLO detection .txt files as SAM prompts).\n\n"
                    "Outputs PNG overlays, COCO JSON annotations, and YOLO .txt files.\n\n"
                    "[dim]Requires a HuggingFace account and Meta's terms agreement "
                    "at huggingface.co/facebook/sam3.1[/dim]"
                ),
                "Combine Datasets": (
                    "Merge several YOLO datasets into a unified one — class names are "
                    "deduplicated, labels are remapped, and images are hard-linked where "
                    "possible for near-zero cost."
                ),
                "Augment Dataset": (
                    "Apply Albumentations transforms to a YOLO or COCO dataset. "
                    "Pools all images across splits, augments using named profiles "
                    "(create/edit/clone/delete), then redistributes to train/val/test "
                    "at user-specified ratios. Output format: YOLO Detection, YOLO Segmentation, or COCO."
                ),
                "Upload to Roboflow": (
                    "Validate and upload datasets or publish trained model checkpoints to Roboflow. "
                    "Supports pre-flight structural validation, fast batch chunking for slow connections, "
                    "and automatic credential resolution from .env."
                ),
                "Ultralytics Platform": (
                    "Use ULTRALYTICS_API_KEY from .env to list/download Platform datasets, "
                    "upload prepared datasets through signed URLs, download model weights, "
                    "and generate ul:// dataset URI training guidance."
                ),
                "Settings": (
                    "Edit global ClearML, Roboflow, Ultralytics, narrative, and credential-status settings. "
                    "Secrets remain in .env and are never displayed."
                ),
                "Check for Updates": (
                    "Run a dependency health check across every critical package — "
                    "ultralytics, torch, torchvision, rfdetr, tensorboard, "
                    "roboflow, onnx, onnxruntime. Each is classified by severity "
                    "(patch / minor / major / missing), with one-click upgrades."
                ),
                "About YOLOmatic": "Technical details, creator info, and version history.",
                "Exit": "Safely exit the application.",
            },
            breadcrumbs=["YOLOmatic"],
        )
        last_choice = main_choice
        if main_choice == "Exit":
            raise _ExitTUI()

        elif main_choice == "Check for Updates":
            check_for_updates()
            continue

        elif main_choice == "Settings":
            settings_menu()
            continue

        elif main_choice == "Train Model":
            from yolomatic.trainers.yolo_trainer import main as trainer_main

            _safe_subcommand("Training", trainer_main, prog="yolomatic-train")
            continue

        elif main_choice == "Run Prediction":
            from yolomatic.cli.predict import main as predict_main

            _safe_subcommand("Prediction", predict_main, prog="yolomatic-predict")
            continue

        elif main_choice == "Launch TensorBoard":
            from yolomatic.cli.tensorboard_launcher import main as tensorboard_main

            _safe_subcommand(
                "TensorBoard", tensorboard_main, prog="yolomatic-tensorboard"
            )
            continue

        elif main_choice == "Benchmark Models":
            from yolomatic.cli.benchmark import main as benchmark_main

            _safe_subcommand("Benchmark", benchmark_main, prog="yolomatic-benchmark")
            continue

        elif main_choice == "SAM Segment":
            from yolomatic.cli.sam_predict import main as sam_predict_main

            _safe_subcommand("SAM Segment", sam_predict_main, prog="yolomatic-sam")
            continue

        elif main_choice == "Convert Dataset Format":
            from yolomatic.cli.convert_ndjson import main as convert_main

            _safe_subcommand("Dataset Conversion", convert_main, prog="yolomatic-convert")
            continue

        elif main_choice == "Export / Compile Model":
            from yolomatic.cli.export import main as export_main

            _safe_subcommand("Export / Compile Model", export_main, prog="yolomatic-export")
            continue

        elif main_choice == "Prepare / Split Dataset":
            from yolomatic.cli.prepare_dataset import main as prepare_main

            _safe_subcommand("Dataset Preparation", prepare_main, prog="yolomatic-prepare")
            continue

        elif main_choice == "Upload to Roboflow":
            from yolomatic.cli.upload import main as upload_main

            _safe_subcommand("Roboflow Upload", upload_main, prog="yolomatic-upload")
            continue

        elif main_choice == "Ultralytics Platform":
            from yolomatic.cli.ultralytics_platform import main as ultralytics_main

            _safe_subcommand("Ultralytics Platform", ultralytics_main, prog="yolomatic-ultralytics")
            continue

        elif main_choice == "Combine Datasets":
            from yolomatic.utils.combine_datasets import main as combine_main

            _safe_subcommand("Dataset Combiner", combine_main, prog="yolomatic-combine")
            continue

        elif main_choice == "Augment Dataset":
            from yolomatic.cli.augment import main as augment_main

            _safe_subcommand("Dataset Augmentation", augment_main, prog="yolomatic-augment")
            continue

        elif main_choice == "Clone Config":
            try:
                if not clone_saved_config_flow():
                    continue
            except KeyboardInterrupt:
                console.print(
                    "\n[bold yellow]Config cloning cancelled by user.[/bold yellow]"
                )
                input("\nPress Enter to return to the main menu...")
                continue
            except Exception as error:
                console.print(
                    Panel(
                        f"[bold red]Config cloning failed:[/bold red] {error}",
                        border_style="red",
                        padding=(1, 2),
                    )
                )
                console.print(traceback.format_exc(), style="dim")
                input("\nPress Enter to return to the main menu...")
                continue

            input("\nPress Enter to continue...")
            continue

        elif main_choice == "Configure Fine-Tune":
            steps = ["Checkpoint Weights", "Strategy Selection", "Dataset Selection", "Profile Settings"]
            ft_state: dict[str, Any] = {}
            ft_step = 0
            while 0 <= ft_step < 4:
                if ft_step == 0:
                    candidate = select_finetune_candidate(wizard_steps=steps, wizard_current_step=0)
                    if candidate is None:
                        break
                    ft_state["candidate"] = candidate
                    ft_step = 1

                elif ft_step == 1:
                    strategy = select_finetune_strategy(
                        ft_state["candidate"], wizard_steps=steps, wizard_current_step=1
                    )
                    if strategy is None:
                        ft_step = 0
                    else:
                        ft_state["strategy"] = strategy
                        ft_step = 2

                elif ft_step == 2:
                    try:
                        dataset_choice = list_datasets(wizard_steps=steps, wizard_current_step=2)
                    except Exception as error:
                        console.print(
                            Panel(
                                f"[bold red]Failed to list datasets:[/bold red] {error}",
                                border_style="red",
                                padding=(1, 2),
                            )
                        )
                        input("\nPress Enter to return to the main menu...")
                        break
                    if dataset_choice == "Back":
                        ft_step = 1
                    elif dataset_choice is None:
                        break
                    else:
                        ft_state["dataset_choice"] = dataset_choice
                        ft_step = 3

                elif ft_step == 3:
                    _cand = ft_state["candidate"]
                    _dataset = ft_state["dataset_choice"]
                    _strategy = ft_state["strategy"]
                    model_choice = infer_finetune_profile_model(_cand)
                    print_summary(_cand.display_name, _dataset)
                    try:
                        outcome = update_config(
                            model_choice,
                            _dataset,
                            finetune_source=_cand.source,
                            finetune_strategy=_strategy,
                            wizard_steps=steps,
                            wizard_current_step=3,
                        )
                        if outcome == CHANGE_MODEL:
                            ft_step = 0
                            continue
                        if not outcome:
                            ft_step = 2
                            continue
                    except KeyboardInterrupt:
                        console.print(
                            "\n[bold yellow]Fine-tune configuration cancelled by user.[/bold yellow]"
                        )
                        input("\nPress Enter to return to the main menu...")
                        break
                    except Exception as error:
                        console.print(
                            Panel(
                                f"[bold red]Fine-tune configuration failed:[/bold red] {error}",
                                border_style="red",
                                padding=(1, 2),
                            )
                        )
                        console.print(traceback.format_exc(), style="dim")
                        input("\nPress Enter to return to the main menu...")
                        break
                    ft_step = 4

            if ft_step >= 4:
                input("\nPress Enter to continue...")
            continue

        elif main_choice == "About YOLOmatic":
            render_about_page()
            continue

        elif main_choice == "Configure Model":
            steps = ["Model Family", "Model Size", "Dataset Selection", "Profile Settings"]
            model_types = get_model_menu()
            cm_state: dict[str, Any] = {}
            cm_step = 0
            while 0 <= cm_step < 4:
                if cm_step == 0:
                    model_family = get_user_choice(
                        model_types,
                        title="Model Selector",
                        text="Choose a model family for your project:",
                        allow_back=True,
                        descriptions=_MODEL_DESCRIPTIONS,
                        breadcrumbs=["YOLOmatic", "Model Selection"],
                        wizard_steps=steps,
                        wizard_current_step=0,
                        initial_selection=cm_state.get("model_family"),
                    )
                    if model_family == "Back":
                        break
                    cm_state["model_family"] = model_family
                    cm_step = 1

                elif cm_step == 1:
                    _family = cm_state["model_family"]
                    variants = [model["Model"] for model in model_data_dict[_family]]
                    model_variant = get_user_choice(
                        variants,
                        allow_back=True,
                        title=f"Select {_family.upper()} Variant",
                        text="Choose the model size that fits your hardware:",
                        model_data=model_data_dict[_family],
                        breadcrumbs=["YOLOmatic", "Model Selection", _family],
                        wizard_steps=steps,
                        wizard_current_step=1,
                        initial_selection=cm_state.get("model_variant"),
                    )
                    if model_variant == "Back":
                        cm_step = 0
                    else:
                        cm_state["model_variant"] = model_variant
                        cm_step = 2

                elif cm_step == 2:
                    try:
                        dataset_choice = list_datasets(wizard_steps=steps, wizard_current_step=2)
                    except Exception as error:
                        console.print(
                            Panel(
                                f"[bold red]Failed to list datasets:[/bold red] {error}",
                                border_style="red",
                                padding=(1, 2),
                            )
                        )
                        input("\nPress Enter to return to the main menu...")
                        break
                    if dataset_choice == "Back":
                        cm_step = 1
                    elif dataset_choice is None:
                        break
                    else:
                        cm_state["dataset_choice"] = dataset_choice
                        cm_step = 3

                elif cm_step == 3:
                    _model = cm_state["model_variant"]
                    _dataset = cm_state["dataset_choice"]
                    print_summary(_model, _dataset)
                    try:
                        outcome = update_config(_model, _dataset, wizard_steps=steps, wizard_current_step=3)
                        if outcome == CHANGE_MODEL:
                            cm_step = 1
                            continue
                        if not outcome:
                            cm_step = 2
                            continue
                    except KeyboardInterrupt:
                        console.print(
                            "\n[bold yellow]Configuration cancelled by user.[/bold yellow]"
                        )
                        input("\nPress Enter to return to the main menu...")
                        break
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
                        break
                    cm_step = 4

            if cm_step >= 4:
                input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
