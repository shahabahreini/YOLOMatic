from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from rich.panel import Panel
from rich.table import Table
from ultralytics import YOLO

try:
    from src.cli.run import console, get_user_choice, print_stylized_header
except ImportError:
    from rich.console import Console

    console = Console()

    def print_stylized_header(text: str) -> None:
        console.print(f"[bold cyan]{text}[/bold cyan]")

    def get_user_choice(
        options: list[str],
        allow_back: bool = False,
        title: str = "Select an Option",
        text: str = "Use ↑↓ keys to navigate, Enter to select:",
        model_data: object | None = None,
    ) -> str:
        del allow_back, title, text, model_data
        for index, option in enumerate(options, start=1):
            console.print(f"{index}. {option}")
        while True:
            raw_choice = input("Select an option: ").strip()
            try:
                choice_index = int(raw_choice) - 1
            except ValueError:
                continue
            if 0 <= choice_index < len(options):
                return options[choice_index]


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODE_LABELS: dict[str, str] = {
    "single": "Single Image",
    "folder": "Folder",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="yolomatic-predict")
    parser.add_argument(
        "--mode",
        choices=tuple(MODE_LABELS.keys()),
        help="Prediction mode.",
    )
    parser.add_argument(
        "--weight",
        help="Path to a .pt weight file. If omitted, an interactive selector is shown.",
    )
    parser.add_argument(
        "--source",
        help="Image file or folder path. If omitted, you will be prompted in the TUI.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions.",
    )
    return parser.parse_args()


def find_available_weights(project_root: Path) -> list[Path]:
    discovered: dict[Path, Path] = {}
    for weight_path in project_root.glob("*.pt"):
        discovered[weight_path.resolve()] = weight_path
    runs_dir = project_root / "runs"
    if runs_dir.exists():
        for weight_path in runs_dir.glob("**/weights/*.pt"):
            discovered[weight_path.resolve()] = weight_path
    return sorted(
        discovered.values(),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def format_weight_label(project_root: Path, weight_path: Path) -> str:
    try:
        return str(weight_path.relative_to(project_root))
    except ValueError:
        return str(weight_path)


def select_weight(project_root: Path, available_weights: Sequence[Path]) -> Path:
    options = [format_weight_label(project_root, path) for path in available_weights]
    selected = get_user_choice(
        options,
        title="Select Weight",
        text="Use ↑↓ keys to navigate, Enter to select:",
    )
    return available_weights[options.index(selected)]


def resolve_weight(
    project_root: Path,
    requested_weight: str | None,
    available_weights: Sequence[Path],
) -> Path:
    if requested_weight is None:
        return select_weight(project_root, available_weights)
    requested_path = Path(requested_weight).expanduser()
    if not requested_path.is_absolute():
        requested_path = project_root / requested_path
    if not requested_path.exists() or requested_path.suffix.lower() != ".pt":
        raise FileNotFoundError(f"Weight file not found: {requested_path}")
    return requested_path.resolve()


def select_mode() -> str:
    selected = get_user_choice(
        [MODE_LABELS["single"], MODE_LABELS["folder"], "Exit"],
        title="Prediction Mode",
        text="Use ↑↓ keys to navigate, Enter to select, 'q' to exit:",
    )
    if selected == "Exit":
        raise SystemExit(0)
    return next(mode for mode, label in MODE_LABELS.items() if label == selected)


def prompt_source(mode: str) -> Path:
    while True:
        prompt_text = (
            "Enter the image path: " if mode == "single" else "Enter the folder path: "
        )
        raw_value = input(prompt_text).strip()
        if not raw_value:
            console.print("[bold red]A path is required.[/bold red]")
            continue
        source_path = Path(raw_value).expanduser()
        if validate_source(source_path, mode):
            return source_path.resolve()


def validate_source(source_path: Path, mode: str) -> bool:
    if not source_path.exists():
        console.print(f"[bold red]Path not found: {source_path}[/bold red]")
        return False
    if mode == "single":
        if not source_path.is_file():
            console.print(f"[bold red]Expected an image file: {source_path}[/bold red]")
            return False
        if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
            console.print(
                f"[bold red]Unsupported image type: {source_path.suffix}[/bold red]"
            )
            return False
        return True
    if not source_path.is_dir():
        console.print(f"[bold red]Expected a folder: {source_path}[/bold red]")
        return False
    has_supported_images = any(
        child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
        for child in source_path.iterdir()
    )
    if not has_supported_images:
        console.print(
            "[bold red]No supported image files found in the selected folder.[/bold red]"
        )
        return False
    return True


def resolve_source(mode: str, requested_source: str | None) -> Path:
    if requested_source is None:
        return prompt_source(mode)
    source_path = Path(requested_source).expanduser()
    if not validate_source(source_path, mode):
        raise ValueError(f"Invalid source path: {source_path}")
    return source_path.resolve()


def render_weight_table(project_root: Path, available_weights: Sequence[Path]) -> None:
    table = Table(title="Available Weights", title_style="bold green")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Weight", style="white")
    for index, weight_path in enumerate(available_weights, start=1):
        table.add_row(str(index), format_weight_label(project_root, weight_path))
    console.print(table)


def render_prediction_summary(weight_path: Path, mode: str, source_path: Path) -> None:
    table = Table(title="Prediction Summary", title_style="bold green")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Weight", str(weight_path))
    table.add_row("Mode", MODE_LABELS[mode])
    table.add_row("Source", str(source_path))
    console.print(table)


def run_prediction(weight_path: Path, source_path: Path, conf: float) -> Path | None:
    model = YOLO(str(weight_path))
    results = model.predict(source=str(source_path), conf=conf, save=True)
    if not results:
        return None
    save_dir = getattr(results[0], "save_dir", None)
    if save_dir is None:
        return None
    return Path(save_dir)


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    print_stylized_header("YOLO Predictor")
    available_weights = find_available_weights(project_root)
    if not available_weights:
        console.print(
            Panel(
                "[bold red]No .pt weights were found in the project root or runs directory.[/bold red]",
                border_style="red",
            )
        )
        raise SystemExit(1)

    render_weight_table(project_root, available_weights)

    try:
        selected_weight = resolve_weight(project_root, args.weight, available_weights)
        mode = args.mode or select_mode()
        source_path = resolve_source(mode, args.source)
        render_prediction_summary(selected_weight, mode, source_path)
        console.print("\n[bold green]Running prediction...[/bold green]")
        output_dir = run_prediction(selected_weight, source_path, args.conf)
    except FileNotFoundError as error:
        console.print(f"[bold red]Error: {error}[/bold red]")
        raise SystemExit(1) from error
    except ValueError as error:
        console.print(f"[bold red]Error: {error}[/bold red]")
        raise SystemExit(1) from error
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Prediction cancelled.[/bold yellow]")
        raise SystemExit(0)
    except SystemExit:
        raise
    except Exception as error:
        console.print(f"[bold red]Prediction failed: {error}[/bold red]")
        raise SystemExit(1) from error

    if output_dir is None:
        console.print(
            "[bold yellow]Prediction completed, but no output directory was reported.[/bold yellow]"
        )
    else:
        console.print(
            Panel(
                f"[bold green]Prediction completed successfully.[/bold green]\nSaved results to: [bold]{output_dir}[/bold]",
                border_style="green",
            )
        )


if __name__ == "__main__":
    main()
