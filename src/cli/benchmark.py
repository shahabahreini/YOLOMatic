"""Interactive benchmark wizard for YOLOMatic."""
from __future__ import annotations

import threading
import webbrowser
from datetime import datetime
from pathlib import Path

from rich.panel import Panel

from src.utils.cli import (
    NAV_BACK,
    ParameterDefinition,
    clear_screen,
    console,
    expected_error_panel,
    get_parameter_value_input,
    get_user_choice,
    print_stylized_header,
    render_summary_panel,
)
from src.utils.project import (
    find_available_weights,
    format_size,
    format_weight_label,
    infer_ultralytics_task_from_name,
    is_rfdetr_source,
    project_root,
)

_DEFAULT_VAL_DIR = "output/nir/valid"
_DEFAULT_OUT_DIR = "output/benchmark_reports"

# Prefix characters kept plain; the shared TUI renderer applies color.
_CHECK = "✓"
_EMPTY = " "


# ---------------------------------------------------------------------------
# Weight metadata helpers
# ---------------------------------------------------------------------------

def _infer_task(path: Path) -> str:
    if is_rfdetr_source(path):
        return "detection (RF-DETR)"
    task = infer_ultralytics_task_from_name(path)
    return task


def _infer_family(path: Path) -> str:
    name = path.stem.lower()
    families = [
        ("yolo26", "YOLO26"), ("yolov12", "YOLOv12"), ("yolov11", "YOLOv11"),
        ("yolov10", "YOLOv10"), ("yolov9", "YOLOv9"), ("yolov8", "YOLOv8"),
        ("yolox", "YOLOX"), ("rfdetr", "RF-DETR"), ("rf-detr", "RF-DETR"),
    ]
    for key, label in families:
        if key in name:
            return label
    if "-seg" in name:
        return "YOLO-Seg"
    if "best" in name or "last" in name:
        return "YOLO (best/last)"
    return "Unknown"


def _find_run_context(path: Path) -> tuple[str, str]:
    """Return (run_name, config_name) from parent directory names."""
    parts = path.parts
    run_name = ""
    config_name = ""
    for i, part in enumerate(parts):
        if part == "weights" and i > 0:
            run_name = parts[i - 1]
        if part in ("runs", "segment", "detect", "classify", "pose"):
            if i + 1 < len(parts):
                config_name = parts[i + 1]
    return run_name, config_name


def _weight_description(path: Path, root: Path, selected: bool) -> str:
    """Rich markup for the right panel: full metadata for one weight file."""
    try:
        stat = path.stat()
        size_str = format_size(stat.st_size)
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        size_str = "unknown"
        mtime = "unknown"

    try:
        rel_path = str(path.relative_to(root))
    except ValueError:
        rel_path = str(path)

    task = _infer_task(path)
    family = _infer_family(path)
    run_name, config_name = _find_run_context(path)

    status_line = (
        f"[bold green]● SELECTED[/bold green]"
        if selected
        else "[dim]○ not selected[/dim]"
    )

    lines = [
        f"{status_line}\n",
        f"[bold cyan]{path.name}[/bold cyan]\n",
        f"[dim]Path:[/dim]  {rel_path}",
        f"[dim]Size:[/dim]  {size_str}   [dim]Modified:[/dim]  {mtime}",
        "",
        f"[dim]Family:[/dim]  [bold]{family}[/bold]",
        f"[dim]Task:[/dim]    [bold]{task}[/bold]",
    ]

    if run_name:
        lines.append(f"[dim]Run:[/dim]     {run_name}")
    if config_name and config_name != run_name:
        lines.append(f"[dim]Config:[/dim]  {config_name}")

    lines += [
        "",
        "[dim]Press [bold yellow]Enter[/bold yellow] to toggle selection.[/dim]",
        "[dim]Navigate with [bold yellow]↑↓[/bold yellow] arrows.[/dim]",
        "[dim]Choose [bold yellow]Confirm[/bold yellow] when ready.[/dim]",
    ]

    return "\n".join(lines)


def _confirm_description(count: int, names: list[str]) -> str:
    bullet_list = "\n".join(f"  [green]✓[/green]  {n}" for n in names)
    return (
        f"[bold green]{count} weight{'s' if count != 1 else ''} selected:[/bold green]\n\n"
        + bullet_list
        + "\n\n[bold yellow]Press Enter to proceed to the next step.[/bold yellow]"
    )


# ---------------------------------------------------------------------------
# Multi-select weight picker
# ---------------------------------------------------------------------------

def _select_weights() -> list[Path] | str:
    root = project_root()
    available = find_available_weights(root)

    if not available:
        console.print(expected_error_panel(
            "No Weights Found",
            "No .pt or .pth files found in the project root or runs/ directory.",
            next_step="Train a model first, then run the benchmark.",
        ))
        input("\nPress Enter to return...")
        return NAV_BACK

    selected_indices: set[int] = set()

    while True:
        options: list[str] = []
        descriptions: dict[str, str] = {}
        idx_by_label: dict[str, int] = {}

        for i, path in enumerate(available):
            sel = i in selected_indices
            mark = _CHECK if sel else _EMPTY
            label = f"{mark} {format_weight_label(root, path)}"
            options.append(label)
            descriptions[label] = _weight_description(path, root, sel)
            idx_by_label[label] = i

        # Separator + confirm/back
        if selected_indices:
            confirm_label = (
                f"  Confirm Selection  ({len(selected_indices)} "
                f"weight{'s' if len(selected_indices) != 1 else ''} selected)"
            )
            selected_names = [
                format_weight_label(root, available[i])
                for i in sorted(selected_indices)
            ]
            descriptions[confirm_label] = _confirm_description(
                len(selected_indices), selected_names
            )
        else:
            confirm_label = "  (Select at least one weight above, then Confirm)"
            descriptions[confirm_label] = (
                "[dim]No weights selected yet.\n\n"
                "Navigate with [bold yellow]↑↓[/bold yellow] and press "
                "[bold yellow]Enter[/bold yellow] to toggle each weight.[/dim]"
            )

        options.append("[──────────────]")
        options.append(confirm_label)

        tip = (
            "[bold yellow]↑↓[/bold yellow] navigate  "
            "[bold yellow]Enter[/bold yellow] toggle  "
            "[bold yellow]B[/bold yellow] back"
        )

        choice = get_user_choice(
            options,
            allow_back=True,
            title="Select Weights for Benchmark",
            text="Toggle each weight file to include. Confirm when ready.",
            descriptions=descriptions,
            breadcrumbs=["YOLOmatic", "Benchmark", "Weight Selection"],
            tip=tip,
        )

        if choice in (NAV_BACK, "Back"):
            return NAV_BACK

        # Confirm action
        if choice == confirm_label:
            if selected_indices:
                return [available[i] for i in sorted(selected_indices)]
            # nothing selected — stay in loop
            continue

        # Toggle weight
        if choice in idx_by_label:
            idx = idx_by_label[choice]
            if idx in selected_indices:
                selected_indices.remove(idx)
            else:
                selected_indices.add(idx)


# ---------------------------------------------------------------------------
# Validation directory selection
# ---------------------------------------------------------------------------

def _val_dir_description(path: Path, root: Path) -> str:
    from src.benchmark.engine import detect_annotation_format, _find_yolo_labels_dir
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
    img_count = sum(1 for _ in path.rglob("*") if _.suffix.lower() in img_exts)
    ann_fmt = detect_annotation_format(path)
    if ann_fmt == "coco":
        ann = list(path.glob("_annotations.coco.json")) + list(path.glob("*.json"))
        ann_note = f"[green]✓ COCO JSON:[/green]  {ann[0].name}" if ann else "[green]✓ COCO format[/green]"
    elif ann_fmt == "yolo":
        labels_dir = _find_yolo_labels_dir(path)
        label_count = sum(1 for _ in labels_dir.glob("*.txt")) if labels_dir else 0
        ann_note = f"[green]✓ YOLO labels:[/green]  {label_count} files"
    else:
        ann_note = "[yellow]⚠ No annotations detected[/yellow]"
    return (
        f"[bold cyan]{path.name}[/bold cyan]\n\n"
        f"[dim]Full path:[/dim]  {path}\n"
        f"[dim]Images:[/dim]    {img_count} found\n"
        f"{ann_note}"
    )


def _collect_val_candidates(root: Path) -> list[Path]:
    """Return candidate validation directories, including dataset roots."""
    seen: set[Path] = set()
    candidates: list[Path] = []

    def _add(p: Path) -> None:
        if p not in seen and p.exists() and p.is_dir():
            seen.add(p)
            candidates.append(p)

    # Explicit default first
    _add(root / _DEFAULT_VAL_DIR)

    # output/**/valid  and  output/**/train  and output/**/test
    for split in ("valid", "val", "train", "test"):
        for p in sorted((root / "output").glob(f"**/{split}"), key=str)[:6]:
            _add(p)

    # datasets: look for split dirs (valid/val/train) AND dataset roots
    datasets_root = root / "datasets"
    if datasets_root.exists():
        for split in ("valid", "val", "train", "test"):
            for p in sorted(datasets_root.glob(f"**/{split}"), key=str)[:10]:
                _add(p)
        # Dataset root dirs that have an images/ subdir or YOLO labels
        for ds_dir in sorted(datasets_root.iterdir()):
            if not ds_dir.is_dir():
                continue
            # Include root if it directly contains images
            img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            has_images = any(ds_dir.glob("images/*")) or any(
                f for f in ds_dir.iterdir() if f.suffix.lower() in img_exts
            )
            if has_images:
                _add(ds_dir)
            # Include each split subdir
            for split_dir in ds_dir.iterdir():
                if split_dir.is_dir() and split_dir.name.lower() in ("train", "valid", "val", "test"):
                    _add(split_dir)

    return candidates


def _select_validation_dir() -> Path | str:
    root = project_root()

    candidates = _collect_val_candidates(root)

    labels = [
        str(p.relative_to(root)) if p.is_relative_to(root) else str(p)
        for p in candidates
    ]
    labels.append("Enter custom path...")

    descriptions: dict[str, str] = {}
    for label, path in zip(labels[:-1], candidates):
        descriptions[label] = _val_dir_description(path, root)
    descriptions["Enter custom path..."] = (
        "Type a custom path to a validation directory.\n"
        "It must contain images and a COCO annotation JSON file."
    )

    choice = get_user_choice(
        labels,
        allow_back=True,
        title="Select Validation Dataset Directory",
        text="Choose the directory containing validation images and annotations:",
        descriptions=descriptions,
        breadcrumbs=["YOLOmatic", "Benchmark", "Validation Dir"],
    )
    if choice in (NAV_BACK, "Back"):
        return NAV_BACK

    if choice == "Enter custom path...":
        raw = get_parameter_value_input(
            ParameterDefinition(
                name="validation_dir",
                category="benchmark",
                default=str(root / _DEFAULT_VAL_DIR),
                value_type="str",
                description="Path to validation images + annotation JSON",
                help_text="Use an absolute path or a path relative to the project root.",
            ),
            current_value=str(root / _DEFAULT_VAL_DIR),
        )
        if raw is None or raw == NAV_BACK:
            return NAV_BACK
        path = Path(str(raw))
    else:
        idx = labels.index(choice)
        path = candidates[idx]

    if not path.exists():
        console.print(Panel(
            f"[bold red]Directory not found:[/bold red] {path}",
            border_style="red", padding=(1, 2),
        ))
        input("\nPress Enter to return...")
        return NAV_BACK

    from src.benchmark.engine import detect_annotation_format
    ann_fmt = detect_annotation_format(path)
    if ann_fmt == "coco":
        ann_candidates = list(path.glob("_annotations.coco.json")) + list(path.glob("*.json"))
        console.print(
            f"[dim]  Auto-detected COCO annotation file: {ann_candidates[0].name}[/dim]\n"
        )
    elif ann_fmt == "yolo":
        console.print("[dim]  Auto-detected YOLO label format.[/dim]\n")
    else:
        console.print(
            "[yellow]  Warning: No annotations detected in this directory. "
            "You will be asked to provide a COCO annotation path.[/yellow]\n"
        )

    return path


# ---------------------------------------------------------------------------
# Options configuration
# ---------------------------------------------------------------------------

def _configure_options(val_dir: Path) -> dict | str:
    from src.benchmark.engine import detect_annotation_format
    root = project_root()

    ann_fmt = detect_annotation_format(val_dir)
    ann_file: Path | None = None

    if ann_fmt == "coco":
        ann_candidates = list(val_dir.glob("_annotations.coco.json")) + list(val_dir.glob("*.json"))
        ann_file = ann_candidates[0] if ann_candidates else None

    if ann_fmt not in ("coco", "yolo") or (ann_fmt == "coco" and ann_file is None):
        raw = get_parameter_value_input(
            ParameterDefinition(
                name="annotations_file",
                category="benchmark",
                default="",
                value_type="str",
                description="Path to COCO annotation JSON (absolute or relative to project root)",
                help_text="Use the COCO JSON file that matches the selected validation images.",
            ),
            current_value="",
        )
        if raw is None or raw == NAV_BACK:
            return NAV_BACK
        ann_file = Path(str(raw))
        if not ann_file.exists():
            ann_file = root / str(raw)
        if not ann_file.exists():
            console.print(Panel(
                f"[bold red]Annotation file not found:[/bold red] {ann_file}",
                border_style="red", padding=(1, 2),
            ))
            input("\nPress Enter to return...")
            return NAV_BACK

    conf_raw = get_parameter_value_input(
        ParameterDefinition(
            name="conf_threshold",
            category="benchmark",
            default=0.25,
            value_type="float",
            description="Minimum confidence to count a prediction (0.01-0.99)",
            help_text="Lower values keep more detections; higher values count only more confident predictions.",
            min_value=0.01,
            max_value=0.99,
        ),
        current_value=0.25,
    )
    if conf_raw is None or conf_raw == NAV_BACK:
        return NAV_BACK
    conf = float(conf_raw)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out = root / _DEFAULT_OUT_DIR / ts
    out_raw = get_parameter_value_input(
        ParameterDefinition(
            name="output_dir",
            category="benchmark",
            default=str(default_out),
            value_type="str",
            description="Directory where the HTML report will be saved",
            help_text="A timestamped directory under output/benchmark_reports is used by default.",
        ),
        current_value=str(default_out),
    )
    if out_raw is None or out_raw == NAV_BACK:
        return NAV_BACK

    return {
        "annotations_file": ann_file,
        "conf_threshold": conf,
        "output_dir": Path(str(out_raw)),
    }


# ---------------------------------------------------------------------------
# Confirmation summary
# ---------------------------------------------------------------------------

def _confirm(weights: list[Path], val_dir: Path, options: dict) -> bool:
    root = project_root()
    ann_val = str(options["annotations_file"]) if options.get("annotations_file") else "YOLO labels (auto)"
    rows = {
        "Weights": ", ".join(format_weight_label(root, w) for w in weights),
        "Validation Dir": str(val_dir),
        "Annotations": ann_val,
        "Confidence Threshold": str(options["conf_threshold"]),
        "Output Directory": str(options["output_dir"]),
    }
    clear_screen()
    print_stylized_header("Benchmark Configuration")
    render_summary_panel("Ready to Run", rows)

    choice = get_user_choice(
        ["Start Benchmark", "Cancel"],
        title="Confirm",
        text="Review the settings above and confirm to begin:",
        descriptions={
            "Start Benchmark": (
                "[bold green]All settings look good.[/bold green]\n\n"
                "Benchmark will:\n"
                f"  • Load [bold]{len(weights)}[/bold] weight file(s)\n"
                "  • Run inference on all validation images\n"
                "  • Compute mAP@50, mAP@50:95, F1, P, R per model\n"
                "  • Generate interactive Plotly HTML report with UMAP scatter\n\n"
                "[dim]This may take several minutes depending on dataset size.[/dim]"
            ),
            "Cancel": "Return to the main menu without running the benchmark.",
        },
        breadcrumbs=["YOLOmatic", "Benchmark", "Confirm"],
    )
    return choice == "Start Benchmark"


# ---------------------------------------------------------------------------
# Running with live log
# ---------------------------------------------------------------------------

def _run_with_live_log(weights: list[Path], val_dir: Path, options: dict) -> Path | None:
    from rich.live import Live

    from src.benchmark import BenchmarkConfig, run_benchmark, write_benchmark_report

    clear_screen()
    print_stylized_header("Running Benchmark")

    log_lines: list[str] = []
    result_holder: dict = {}
    error_holder: dict = {}

    def _logger(msg: str) -> None:
        log_lines.append(msg)

    def _worker():
        try:
            config = BenchmarkConfig(
                weights=weights,
                validation_dir=val_dir,
                annotations_file=options["annotations_file"],
                output_dir=options["output_dir"],
                conf_threshold=options["conf_threshold"],
            )
            result = run_benchmark(config, logger_fn=_logger)
            report_path = write_benchmark_report(result, options["output_dir"])
            result_holder["path"] = report_path
        except Exception as exc:
            import traceback
            error_holder["error"] = str(exc)
            error_holder["tb"] = traceback.format_exc()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    with Live(console=console, refresh_per_second=4) as live:
        while thread.is_alive():
            thread.join(timeout=0.25)
            panel_content = "\n".join(log_lines[-30:]) or "[dim]Starting...[/dim]"
            live.update(Panel(
                panel_content,
                title="Benchmark Log",
                border_style="cyan",
                padding=(0, 1),
            ))
        panel_content = "\n".join(log_lines[-30:]) or "[dim]Done.[/dim]"
        live.update(Panel(panel_content, title="Benchmark Log", border_style="cyan", padding=(0, 1)))

    if "error" in error_holder:
        console.print(Panel(
            f"[bold red]Benchmark failed:[/bold red] {error_holder['error']}\n\n"
            f"[dim]{error_holder.get('tb', '')}[/dim]",
            border_style="red", padding=(1, 2),
        ))
        input("\nPress Enter to return to the main menu...")
        return None

    return result_holder.get("path")


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------

def _handle_completion(report_path: Path) -> None:
    console.print(Panel(
        f"[bold green]Report saved:[/bold green] {report_path}",
        border_style="green", padding=(1, 2),
    ))
    choice = get_user_choice(
        ["Open in Browser", "Continue"],
        title="Report Ready",
        text="Would you like to open the report in your browser?",
        descriptions={
            "Open in Browser": (
                f"[bold]Open:[/bold]  {report_path}\n\n"
                "Opens the interactive Plotly report in your default browser.\n"
                "The report includes summary cards, model comparison, object-size "
                "charts, per-image rankings, and UMAP vector scatter."
            ),
            "Continue": "Return to the main menu. You can open the report manually later.",
        },
        breadcrumbs=["YOLOmatic", "Benchmark", "Complete"],
    )
    if choice == "Open in Browser":
        try:
            webbrowser.open(report_path.as_uri())
        except Exception as exc:
            console.print(f"[yellow]Could not open browser automatically: {exc}[/yellow]")
            console.print(f"Open manually: [cyan]{report_path}[/cyan]")
            input("\nPress Enter to continue...")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def main() -> None:
    clear_screen()
    print_stylized_header("YOLOmatic — Benchmark & Evaluate")

    weights = _select_weights()
    if weights == NAV_BACK or not isinstance(weights, list):
        return

    val_dir = _select_validation_dir()
    if val_dir == NAV_BACK:
        return

    options = _configure_options(val_dir)
    if options == NAV_BACK:
        return

    if not _confirm(weights, val_dir, options):
        return

    report_path = _run_with_live_log(weights, val_dir, options)
    if report_path is None:
        return

    _handle_completion(report_path)
