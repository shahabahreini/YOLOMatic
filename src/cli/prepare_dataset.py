"""Dataset preparation and train/valid/test splitting wizard."""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

from rich.live import Live
from rich.panel import Panel

from src.datasets.prepare import (
    OUTPUT_FORMATS,
    PrepareDatasetConfig,
    PrepareDatasetStats,
    PrepareSplitConfig,
    prepare_dataset,
    resolve_versioned_output,
    slugify,
)
from src.utils.cli import (
    NAV_BACK,
    ParameterDefinition,
    clear_screen,
    console,
    get_parameter_value_input,
    get_user_choice,
    print_stylized_header,
    render_summary_panel,
)
from src.utils.project import list_dataset_directories


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


def _quick_source_description(path: Path) -> str:
    if path.suffix.lower() == ".ndjson":
        return (
            f"[bold cyan]{path.name}[/bold cyan]\n\n"
            "Format: [yellow]Labelbox NDJSON[/yellow]\n"
            "YOLOmatic will download referenced images, parse boxes/polygons, "
            "then split the prepared output."
        )
    try:
        from src.datasets.core import summarize_dataset

        summary = summarize_dataset(path, sample_limit=200)
        class_preview = ", ".join(summary.classes[:8])
        if len(summary.classes) > 8:
            class_preview += "..."
        return (
            f"[bold cyan]{path.name}[/bold cyan]\n\n"
            f"Format: [yellow]{summary.format}[/yellow]  |  "
            f"Task: [yellow]{summary.task}[/yellow]\n"
            f"Images: [yellow]{summary.image_count}[/yellow]  |  "
            f"Annotations: [yellow]{summary.annotation_count}[/yellow]  |  "
            f"Classes: [yellow]{len(summary.classes)}[/yellow]\n\n"
            f"[dim]{class_preview or path}[/dim]"
        )
    except Exception:
        return f"[bold cyan]{path.name}[/bold cyan]\n\n[dim]{path}[/dim]"


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
    )
    if choice in (NAV_BACK, "Back"):
        return None

    if choice == "Enter NDJSON File Path":
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
        if not path.exists() or path.suffix.lower() != ".ndjson":
            console.print(Panel(f"[bold red]NDJSON file not found:[/bold red] {path}", border_style="red"))
            input("\nPress Enter to continue...")
            return None
        return path

    datasets = list_dataset_directories(include_size=False)
    if not datasets:
        console.print(Panel("[bold yellow]No datasets found under ./datasets/.[/bold yellow]", border_style="yellow"))
        input("\nPress Enter to continue...")
        return None
    path_by_name = {item["name"]: Path(item["path"]) for item in datasets}
    descriptions = {name: _quick_source_description(path) for name, path in path_by_name.items()}
    selected = get_user_choice(
        list(path_by_name),
        allow_back=True,
        title="Select Dataset Folder",
        text="Choose a YOLO or COCO source dataset:",
        descriptions=descriptions,
        breadcrumbs=["YOLOmatic", "Prepare Dataset", "Source"],
    )
    if selected in (NAV_BACK, "Back"):
        return None
    return path_by_name[selected]


def _select_output_format() -> str | None:
    choice = get_user_choice(
        ["YOLO Detection", "YOLO Segmentation", "COCO"],
        allow_back=True,
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
    )
    return None if choice in (NAV_BACK, "Back") else choice


def _select_split() -> PrepareSplitConfig | None:
    options = [*SPLIT_PRESETS, "Custom", "Back"]
    descriptions = {
        name: f"{cfg.train_ratio:.0%} train / {cfg.val_ratio:.0%} valid / {cfg.test_ratio:.0%} test\n\n{desc}"
        for name, (cfg, desc) in SPLIT_PRESETS.items()
    }
    descriptions["Custom"] = "Enter custom train, validation, and test ratios. Values must sum to 1.0."
    choice = get_user_choice(
        options,
        title="Split Preset",
        text="Choose how to divide the source images:",
        descriptions=descriptions,
        breadcrumbs=["YOLOmatic", "Prepare Dataset", "Splits"],
    )
    if choice in (NAV_BACK, "Back"):
        return None
    if choice in SPLIT_PRESETS:
        return SPLIT_PRESETS[choice][0]

    values = {}
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


def _run_with_progress(config: PrepareDatasetConfig) -> None:
    progress_state = {"done": 0, "total": 0, "message": "Initializing..."}
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
    thread.join()

    if errors:
        console.print(Panel(f"[bold red]Dataset preparation failed:[/bold red] {errors[0]}", border_style="red"))
        return
    stats = stats_holder[0]
    render_summary_panel(
        "Prepared Dataset",
        {
            "Output Path": stats.output_path,
            "Output Format": stats.output_format,
            "Version": f"v{stats.version:03d}",
            "Source Format": stats.source_format,
            "Images": stats.total_images,
            "Annotations": stats.total_annotations,
            "Train": stats.split_counts.get("train", 0),
            "Valid": stats.split_counts.get("valid", 0),
            "Test": stats.split_counts.get("test", 0),
            "Warnings": len(stats.warnings),
            "Time": f"{stats.elapsed_seconds:.1f}s",
        },
    )


def interactive_main() -> None:
    source = _select_source()
    if source is None:
        return
    output_format = _select_output_format()
    if output_format is None:
        return
    split_config = _select_split()
    if split_config is None:
        return

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
        return
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
        return

    slug = slugify(str(raw_slug))
    output_path, version = resolve_versioned_output(Path("datasets"), slug)
    clear_screen()
    print_stylized_header("Prepare Dataset — Confirm")
    render_summary_panel(
        "Preparation Plan",
        {
            "Source": source,
            "Output Format": output_format,
            "Split": f"{split_config.train_ratio:.0%} / {split_config.val_ratio:.0%} / {split_config.test_ratio:.0%}",
            "Seed": raw_seed,
            "Output": output_path,
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
            seed=int(raw_seed),
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
