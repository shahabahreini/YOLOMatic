"""Dataset preparation and train/valid/test splitting wizard."""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Any

from rich.live import Live
from rich.panel import Panel

from src.datasets.prepare import (
    OUTPUT_FORMATS,
    PrepareDatasetConfig,
    PrepareDatasetStats,
    PrepareSplitConfig,
    SPLIT_STRATEGIES,
    prepare_dataset,
    resolve_versioned_output,
    slugify,
)
from src.utils.cli import (
    NAV_BACK,
    ParameterDefinition,
    clear_screen,
    console,
    format_path,
    get_parameter_value_input,
    get_user_choice,
    print_stylized_header,
    render_summary_panel,
)
from src.utils.project import (
    calculate_folder_size,
    format_size,
    list_dataset_directories,
)


WIZARD_STEPS = ["Source", "Format", "Split", "Strategy", "Output", "Confirm"]


SPLIT_PRESETS: dict[str, tuple[PrepareSplitConfig, str]] = {
    "Standard (70 / 20 / 10)": (
        PrepareSplitConfig(0.70, 0.20, 0.10),
        "Best default for most object-detection and segmentation datasets.",
    ),
    "Large Train (80 / 15 / 5)": (
        PrepareSplitConfig(0.80, 0.15, 0.05),
        "Good for larger datasets where training volume matters most.",
    ),
    "Train + Validation (90 / 10 / 0)": (
        PrepareSplitConfig(0.90, 0.10, 0.00),
        "Use when you have a separate held-out test set elsewhere.",
    ),
    "Tiny Dataset (80 / 20 / 0)": (
        PrepareSplitConfig(0.80, 0.20, 0.00),
        "Recommended when a test split would be too small to be meaningful.",
    ),
}


def _format_split_label(cfg: PrepareSplitConfig) -> str:
    return f"{cfg.train_ratio:.0%} / {cfg.val_ratio:.0%} / {cfg.test_ratio:.0%}"


def _safe_size(path: Path) -> str:
    try:
        return format_size(calculate_folder_size(path))
    except OSError:
        return "unknown"


def _discover_ndjson_files(root: Path = Path(".")) -> list[Path]:
    candidates: list[Path] = []
    search_roots = [root, root / "datasets"]
    for search_root in search_roots:
        if not search_root.exists():
            continue
        pattern = "*.ndjson" if search_root == root else "**/*.ndjson"
        for path in search_root.glob(pattern):
            if path.is_file() and not path.name.startswith("."):
                candidates.append(path)
    return sorted(set(candidates), key=lambda path: str(path.resolve()).lower())


def _ndjson_label(path: Path, root: Path = Path(".")) -> str:
    try:
        display = path.relative_to(root)
    except ValueError:
        display = path
    return format_path(display)


def _quick_source_description(path: Path) -> str:
    if path.is_file() and path.suffix.lower() == ".ndjson":
        try:
            line_count = sum(1 for line in path.read_text("utf-8").splitlines() if line.strip())
        except OSError:
            line_count = 0
        return (
            f"[bold cyan]{path.name}[/bold cyan]\n\n"
            "Format: [yellow]Labelbox NDJSON[/yellow]\n"
            f"Rows: [yellow]{line_count}[/yellow]\n\n"
            "[dim]Images will be downloaded into a temporary directory before splitting.[/dim]"
        )
    try:
        from src.datasets.core import summarize_dataset

        summary = summarize_dataset(path, sample_limit=200)
        class_preview = ", ".join(summary.classes[:8])
        if len(summary.classes) > 8:
            class_preview += "..."
        size_text = _safe_size(path)
        return (
            f"[bold cyan]{path.name}[/bold cyan]\n\n"
            f"Format: [yellow]{summary.format}[/yellow]  |  "
            f"Task: [yellow]{summary.task}[/yellow]\n"
            f"Images: [yellow]{summary.image_count}[/yellow]  |  "
            f"Annotations: [yellow]{summary.annotation_count}[/yellow]  |  "
            f"Classes: [yellow]{len(summary.classes)}[/yellow]\n"
            f"Size: [yellow]{size_text}[/yellow]\n\n"
            f"[dim]{class_preview or path}[/dim]"
        )
    except Exception as exc:
        return (
            f"[bold cyan]{path.name}[/bold cyan]\n\n"
            f"[yellow]Could not inspect dataset:[/yellow] {exc}\n\n"
            f"[dim]{path}[/dim]"
        )


def _wizard_kwargs(step_index: int) -> dict[str, Any]:
    return {"wizard_steps": WIZARD_STEPS, "wizard_current_step": step_index}


def _select_source() -> Path | None:
    clear_screen()
    print_stylized_header("Prepare / Split Dataset")
    options = ["Select Dataset Folder", "Enter NDJSON File Path", "Back"]
    choice = get_user_choice(
        options,
        title="Source Type",
        text="Choose the source you want to prepare for training:",
        descriptions={
            "Select Dataset Folder": (
                "Use an existing YOLO or COCO dataset under ./datasets.\n\n"
                "YOLOmatic detects data.yaml, dataset.yaml, or COCO annotations."
            ),
            "Enter NDJSON File Path": (
                "Use a Labelbox .ndjson export. Referenced images are downloaded "
                "before splitting and writing the final dataset."
            ),
        },
        breadcrumbs=["YOLOmatic", "Prepare Dataset", "Source"],
        tip="↑/↓ select  •  PgUp/PgDn jump  •  Enter confirm  •  Q back",
        **_wizard_kwargs(0),
    )
    if choice in (NAV_BACK, "Back"):
        return None

    if choice == "Enter NDJSON File Path":
        detected = _discover_ndjson_files()
        if detected:
            labels = [_ndjson_label(path) for path in detected]
            path_by_label = dict(zip(labels, detected, strict=True))
            selected = get_user_choice(
                [*labels, "Enter Manual Path", "Back"],
                allow_back=True,
                title=f"Select NDJSON Export ({len(detected)} found)",
                text="Choose a detected Labelbox NDJSON export or enter a path manually:",
                descriptions={label: _quick_source_description(path) for label, path in path_by_label.items()},
                breadcrumbs=["YOLOmatic", "Prepare Dataset", "NDJSON"],
                tip="Detected files include top-level .ndjson exports and any .ndjson under datasets/.",
                **_wizard_kwargs(0),
            )
            if selected in (NAV_BACK, "Back"):
                return None
            if selected != "Enter Manual Path":
                return path_by_label[selected]

        raw = get_parameter_value_input(
            ParameterDefinition(
                "ndjson_path",
                "source",
                "",
                "str",
                "Path to Labelbox .ndjson file",
                "Enter an absolute path or a path relative to the project root.",
            ),
            "",
        )
        if raw in (None, NAV_BACK):
            return None
        path = Path(str(raw)).expanduser()
        if not path.exists() or not path.is_file() or path.suffix.lower() != ".ndjson":
            console.print(Panel(
                f"[bold red]NDJSON file not found or invalid:[/bold red] {path}\n\n"
                "[dim]The path must point to a readable .ndjson file.[/dim]",
                border_style="red",
            ))
            input("\nPress Enter to continue...")
            return None
        return path

    datasets = list_dataset_directories(include_size=False)
    if not datasets:
        console.print(Panel(
            "[bold yellow]No datasets found under ./datasets/.[/bold yellow]\n\n"
            "Place a YOLO or COCO dataset (with a data.yaml) in the datasets/ folder "
            "first, or use 'Convert Dataset Format' to import from Labelbox.",
            border_style="yellow",
        ))
        input("\nPress Enter to continue...")
        return None
    path_by_name = {item["name"]: Path(item["path"]) for item in datasets}
    descriptions = {name: _quick_source_description(path) for name, path in path_by_name.items()}
    selected = get_user_choice(
        list(path_by_name),
        allow_back=True,
        title=f"Select Dataset Folder ({len(path_by_name)} available)",
        text="Choose a YOLO or COCO source dataset:",
        descriptions=descriptions,
        breadcrumbs=["YOLOmatic", "Prepare Dataset", "Source"],
        tip="PgUp/PgDn page through long lists  •  Home/End jump  •  Q back",
        **_wizard_kwargs(0),
    )
    if selected in (NAV_BACK, "Back"):
        return None
    return path_by_name[selected]


def _select_output_format(source: Path) -> str | None:
    choice = get_user_choice(
        ["YOLO Detection", "YOLO Segmentation", "COCO", "Back"],
        title="Output Format",
        text="Choose the format for the prepared training dataset:",
        descriptions={
            "YOLO Detection": (
                "Writes Ultralytics detection labels: class_id cx cy w h.\n\n"
                "Best for YOLO detection, RF-DETR, and general object detection."
            ),
            "YOLO Segmentation": (
                "Writes Ultralytics polygon labels: class_id x1 y1 ... xn yn.\n\n"
                "If the source has only boxes, YOLOmatic creates rectangle polygons."
            ),
            "COCO": (
                "Writes COCO instances JSON per split under annotations/.\n\n"
                "Best for Detectron2, benchmarking, and frameworks expecting COCO."
            ),
        },
        breadcrumbs=["YOLOmatic", "Prepare Dataset", "Format"],
        status_fields={"Source": str(source)},
        **_wizard_kwargs(1),
    )
    return None if choice in (NAV_BACK, "Back") else choice


def _select_split(source: Path, output_format: str) -> PrepareSplitConfig | None:
    options = [*SPLIT_PRESETS, "Custom", "Back"]
    descriptions = {
        name: f"{_format_split_label(cfg)}\n\n{desc}"
        for name, (cfg, desc) in SPLIT_PRESETS.items()
    }
    descriptions["Custom"] = "Enter custom train, validation, and test ratios. Values must sum to 1.0."
    choice = get_user_choice(
        options,
        title="Split Preset",
        text="Choose how to divide the source images:",
        descriptions=descriptions,
        breadcrumbs=["YOLOmatic", "Prepare Dataset", "Splits"],
        status_fields={"Source": str(source), "Format": output_format},
        **_wizard_kwargs(2),
    )
    if choice in (NAV_BACK, "Back"):
        return None
    if choice in SPLIT_PRESETS:
        return SPLIT_PRESETS[choice][0]

    values: dict[str, float] = {}
    for name, default, label in [
        ("train_ratio", 0.70, "Train ratio"),
        ("val_ratio", 0.20, "Validation ratio"),
        ("test_ratio", 0.10, "Test ratio"),
    ]:
        raw = get_parameter_value_input(
            ParameterDefinition(
                name,
                "split",
                default,
                "float",
                label,
                "Enter a decimal fraction, for example 0.70.",
                min_value=0.0,
                max_value=1.0,
            ),
            default,
        )
        if raw in (None, NAV_BACK):
            return None
        values[name] = float(raw)
    try:
        split = PrepareSplitConfig(values["train_ratio"], values["val_ratio"], values["test_ratio"])
        split.normalized()
        return split
    except ValueError as exc:
        console.print(Panel(f"[bold red]{exc}[/bold red]", border_style="red"))
        input("\nPress Enter to continue...")
        return None


def _select_split_strategy(source: Path, split_config: PrepareSplitConfig) -> str | None:
    choice = get_user_choice(
        ["Class Balanced", "Smart Balanced", "Back"],
        title="Split Strategy",
        text="Choose how YOLOmatic should assign images to train, valid, and test:",
        descriptions={
            "Class Balanced": (
                "Default. Keeps class coverage distributed across splits while preserving "
                "the requested image counts."
            ),
            "Smart Balanced": (
                "Balances class counts, small / medium / large object buckets, "
                "background-only images, and dense multi-object images. Use it when "
                "validation and test splits need to represent object scale and image difficulty."
            ),
        },
        breadcrumbs=["YOLOmatic", "Prepare Dataset", "Split Strategy"],
        status_fields={"Source": str(source), "Split": _format_split_label(split_config)},
        **_wizard_kwargs(3),
    )
    if choice in (NAV_BACK, "Back"):
        return None
    return "smart_balanced" if choice == "Smart Balanced" else "class_balanced"


def _collect_output_settings(
    source: Path,
    split_config: PrepareSplitConfig,
    strategy: str,
) -> tuple[str, int] | None:
    default_slug = slugify(source.stem if source.is_file() else source.name)
    raw_slug = get_parameter_value_input(
        ParameterDefinition(
            "output_slug",
            "output",
            default_slug,
            "str",
            "Output dataset name",
            "YOLOmatic appends _v001, _v002, etc. automatically unless you include a _vNNN suffix.",
        ),
        default_slug,
    )
    if raw_slug in (None, NAV_BACK):
        return None
    raw_seed = get_parameter_value_input(
        ParameterDefinition(
            "seed",
            "split",
            42,
            "int",
            "Split seed",
            "Use the same seed to reproduce the same train/valid/test assignment.",
            min_value=0,
            max_value=999999,
        ),
        42,
    )
    if raw_seed in (None, NAV_BACK):
        return None
    return slugify(str(raw_slug)), int(raw_seed)


def _run_with_progress(config: PrepareDatasetConfig) -> PrepareDatasetStats | None:
    progress_state: dict[str, Any] = {"done": 0, "total": 0, "message": "Initializing..."}
    stats_holder: list[PrepareDatasetStats] = []
    errors: list[Exception] = []
    done = threading.Event()

    def worker() -> None:
        try:
            def callback(current: int, total: int, message: str) -> None:
                progress_state.update({"done": current, "total": total, "message": message})

            stats_holder.append(prepare_dataset(config, progress_callback=callback))
        except Exception as exc:
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        with Live(refresh_per_second=4) as live:
            while not done.is_set():
                total = progress_state["total"]
                current = progress_state["done"]
                pct = f"{current / total * 100:.0f}%" if total else "..."
                live.update(
                    Panel(
                        f"{progress_state['message']}\n\n[cyan]{current}/{total}[/cyan]",
                        title=f"[cyan]Preparing Dataset [{pct}][/cyan]",
                        border_style="cyan",
                    )
                )
                time.sleep(0.25)
    except KeyboardInterrupt:
        console.print(Panel(
            "[bold yellow]Interrupted.[/bold yellow] Waiting for in-flight work to finish so output stays consistent...",
            border_style="yellow",
        ))
    thread.join()

    if errors:
        exc = errors[0]
        console.print(Panel(
            f"[bold red]Dataset preparation failed:[/bold red]\n\n{type(exc).__name__}: {exc}",
            border_style="red",
        ))
        return None
    stats = stats_holder[0]
    summary_fields: dict[str, Any] = {
        "Output Path": stats.output_path,
        "Output Format": stats.output_format,
        "Version": f"v{stats.version:03d}",
        "Source Format": stats.source_format,
        "Classes": f"{len(stats.classes)} ({', '.join(stats.classes[:6])}{'...' if len(stats.classes) > 6 else ''})",
        "Images": stats.total_images,
        "Annotations": stats.total_annotations,
        "Train": stats.split_counts.get("train", 0),
        "Valid": stats.split_counts.get("valid", 0),
        "Test": stats.split_counts.get("test", 0),
        "Warnings": len(stats.warnings),
        "Time": f"{stats.elapsed_seconds:.1f}s",
    }
    render_summary_panel("Prepared Dataset", summary_fields)
    if stats.warnings:
        preview = "\n".join(f"• {w}" for w in stats.warnings[:8])
        if len(stats.warnings) > 8:
            preview += f"\n[dim]... and {len(stats.warnings) - 8} more (see manifest.json)[/dim]"
        console.print(Panel(preview, title="[yellow]Warnings[/yellow]", border_style="yellow"))
    return stats


def interactive_main() -> None:
    source = _select_source()
    if source is None:
        return
    output_format = _select_output_format(source)
    if output_format is None:
        return
    split_config = _select_split(source, output_format)
    if split_config is None:
        return
    split_strategy = _select_split_strategy(source, split_config)
    if split_strategy is None:
        return

    output_settings = _collect_output_settings(source, split_config, split_strategy)
    if output_settings is None:
        return
    slug, seed = output_settings

    try:
        output_path, version = resolve_versioned_output(Path("datasets"), slug)
    except FileExistsError as exc:
        console.print(Panel(
            f"[bold red]{exc}[/bold red]\n\n"
            "[dim]Pick a different name, or remove the existing directory and try again.[/dim]",
            border_style="red",
        ))
        input("\nPress Enter to return to main menu...")
        return

    clear_screen()
    print_stylized_header("Prepare Dataset — Confirm")
    render_summary_panel(
        "Preparation Plan",
        {
            "Source": source,
            "Output Format": output_format,
            "Split": _format_split_label(split_config),
            "Split Strategy": split_strategy.replace("_", " "),
            "Seed": seed,
            "Output": format_path(str(output_path), max_chars=64),
            "Version": f"v{version:03d}",
        },
    )
    confirm = get_user_choice(
        ["Start Preparation", "Back"],
        title="Confirm",
        text="Review the plan. The source dataset is not modified.",
        descriptions={
            "Start Preparation": "Create the versioned dataset and write training-ready files.",
            "Back": "Return to the main menu without writing files.",
        },
        breadcrumbs=["YOLOmatic", "Prepare Dataset", "Confirm"],
        **_wizard_kwargs(5),
    )
    if confirm in (NAV_BACK, "Back"):
        return
    _run_with_progress(
        PrepareDatasetConfig(
            source_path=source,
            output_format=output_format,
            output_root=Path("datasets"),
            output_slug=slug,
            split_config=split_config,
            split_strategy=split_strategy,
            seed=seed,
        )
    )
    input("\nPress Enter to return to main menu...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and split YOLO, COCO, or Labelbox NDJSON datasets.")
    parser.add_argument("--source", type=Path, help="Source dataset directory or Labelbox .ndjson file.")
    parser.add_argument("--format", choices=sorted(OUTPUT_FORMATS), default="YOLO Detection", help="Output dataset format.")
    parser.add_argument("--output-root", type=Path, default=Path("datasets"), help="Directory where versioned outputs are created.")
    parser.add_argument("--slug", help="Output dataset slug. _vNNN is added automatically when omitted.")
    parser.add_argument("--train", type=float, default=0.70, help="Train split ratio.")
    parser.add_argument("--val", type=float, default=0.20, help="Validation split ratio.")
    parser.add_argument("--test", type=float, default=0.10, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument(
        "--smart-split",
        action="store_true",
        help="Balance classes, object-size buckets, background images, and object density across splits.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=sorted(SPLIT_STRATEGIES),
        help="Explicit split strategy. Overrides --smart-split when provided.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the selected versioned output if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source is None:
        interactive_main()
        return
    stats = prepare_dataset(
        PrepareDatasetConfig(
            source_path=args.source,
            output_format=args.format,
            output_root=args.output_root,
            output_slug=args.slug,
            split_config=PrepareSplitConfig(args.train, args.val, args.test),
            split_strategy=args.split_strategy or ("smart_balanced" if args.smart_split else "class_balanced"),
            seed=args.seed,
            overwrite=args.overwrite,
        )
    )
    console.print(f"[bold green]Prepared dataset:[/bold green] [cyan]{stats.output_path}[/cyan]")
    console.print(
        f"Images: {stats.total_images}  Train: {stats.split_counts.get('train', 0)}  "
        f"Valid: {stats.split_counts.get('valid', 0)}  Test: {stats.split_counts.get('test', 0)}"
    )


if __name__ == "__main__":
    main()
