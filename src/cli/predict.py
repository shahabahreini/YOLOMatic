from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from rich import box
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.panel import Panel
from rich.table import Table

from src.utils.cli import (
    console,
    get_user_choice,
    print_stylized_header,
    render_summary_panel,
    render_table,
)
from src.utils.ml_dependencies import MLDependencyError, import_ultralytics_yolo
from src.utils.ml_dependencies import import_rfdetr_model_class
from src.utils.project import (
    find_available_weights,
    format_weight_label,
    project_root,
    render_weight_rows,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODE_LABELS: dict[str, str] = {
    "single": "Single Image",
    "folder": "Folder",
}
_WORKER_MODEL = None
_WORKER_CONF: float = 0.25


@dataclass(frozen=True)
class BatchPredictionResult:
    image_path: Path
    output_dir: Path | None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
        "--input-dir",
        dest="source",
        help="Folder path for batch prediction. Alias for --source in folder mode.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of worker processes for folder prediction. Default 1 keeps "
            "GPU prediction stable; set >1 to opt in to multiprocessing."
        ),
    )
    return parser.parse_args(argv)


def select_weight(project_root: Path, available_weights: Sequence[Path]) -> Path:
    options = [format_weight_label(project_root, path) for path in available_weights]
    selected = get_user_choice(
        options,
        title="Select Weight",
        text="Pick the trained model weights to use:",
        breadcrumbs=["YOLOmatic", "Predictor", "Weight Selection"],
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
    if not requested_path.exists() or requested_path.suffix.lower() not in {".pt", ".pth"}:
        raise FileNotFoundError(f"Weight file not found: {requested_path}")
    return requested_path.resolve()


def is_rfdetr_weight(weight_path: Path) -> bool:
    return weight_path.suffix.lower() == ".pth"


def infer_rfdetr_class_from_weight(weight_path: Path) -> str:
    text = str(weight_path).lower()
    is_seg = "seg" in text
    prefix = "RFDETRSeg" if is_seg else "RFDETR"
    if "2xlarge" in text or "2xl" in text:
        return f"{prefix}2XLarge"
    if "xlarge" in text or "xl" in text:
        return f"{prefix}XLarge"
    if "large" in text:
        return f"{prefix}Large"
    if "small" in text:
        return f"{prefix}Small"
    if "nano" in text:
        return f"{prefix}Nano"
    return f"{prefix}Medium"


def load_rfdetr_model(weight_path: Path) -> object:
    model_class = import_rfdetr_model_class(infer_rfdetr_class_from_weight(weight_path))
    return model_class(pretrain_weights=str(weight_path))


def select_mode() -> str:
    selected = get_user_choice(
        [MODE_LABELS["single"], MODE_LABELS["folder"], "Exit"],
        title="Prediction Mode",
        text="Choose how you want to run predictions:",
        descriptions={
            MODE_LABELS["single"]: "Run prediction on a single image file.",
            MODE_LABELS[
                "folder"
            ]: "Run prediction on all supported images within a folder.",
            "Exit": "Exit the predictor.",
        },
        breadcrumbs=["YOLOmatic", "Predictor", "Mode Selection"],
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
    has_supported_images = bool(discover_images(source_path))
    if not has_supported_images:
        console.print(
            "[bold red]No supported image files found in the selected folder.[/bold red]"
        )
        return False
    return True


def discover_images(source_dir: Path) -> list[Path]:
    return sorted(
        (
            child.resolve()
            for child in source_dir.iterdir()
            if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )


def validate_worker_count(workers: int) -> int:
    if workers < 1:
        raise ValueError("--workers must be 1 or greater.")
    return workers


def resolve_source(mode: str, requested_source: str | None) -> Path:
    if requested_source is None:
        return prompt_source(mode)
    source_path = Path(requested_source).expanduser()
    if not validate_source(source_path, mode):
        raise ValueError(f"Invalid source path: {source_path}")
    return source_path.resolve()


def render_weight_table(project_root: Path, available_weights: Sequence[Path]) -> None:
    render_table(
        "Available Weights",
        ["#", "Weight"],
        render_weight_rows(project_root, available_weights),
    )


def render_prediction_summary(weight_path: Path, mode: str, source_path: Path) -> None:
    fields = {
        "Weight": weight_path,
        "Model Family": "RF-DETR" if is_rfdetr_weight(weight_path) else "Ultralytics YOLO",
        "Mode": MODE_LABELS[mode],
        "Source": source_path,
    }
    render_summary_panel("Prediction Summary", fields)


def run_prediction(weight_path: Path, source_path: Path, conf: float) -> Path | None:
    if is_rfdetr_weight(weight_path):
        return run_rfdetr_prediction(weight_path, source_path, conf)
    yolo_class = import_ultralytics_yolo()
    model = yolo_class(str(weight_path))
    results = model.predict(source=str(source_path), conf=conf, save=True)
    if not results:
        return None
    save_dir = getattr(results[0], "save_dir", None)
    if save_dir is None:
        return None
    return Path(save_dir)


def run_rfdetr_prediction(weight_path: Path, source_path: Path, conf: float) -> Path:
    from PIL import Image
    import supervision as sv

    model = load_rfdetr_model(weight_path)
    image = Image.open(source_path).convert("RGB")
    detections = model.predict(image, threshold=conf)
    annotated = image.copy()
    if hasattr(detections, "mask") and getattr(detections, "mask", None) is not None:
        annotated = sv.MaskAnnotator().annotate(annotated, detections)
    annotated = sv.BoxAnnotator().annotate(annotated, detections)

    output_dir = Path("runs") / "predict" / "rfdetr"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / source_path.name
    annotated.save(output_path)
    return output_dir


def _predict_single_image(
    model: object, image_path: Path, conf: float
) -> BatchPredictionResult:
    try:
        if getattr(model, "_yolomatic_rfdetr", False):
            output_dir = _predict_single_rfdetr_image(model, image_path, conf)
            return BatchPredictionResult(image_path=image_path, output_dir=output_dir)
        results = model.predict(
            source=str(image_path),
            conf=conf,
            save=True,
            verbose=False,
        )
        output_dir = _output_dir_from_results(results)
        return BatchPredictionResult(image_path=image_path, output_dir=output_dir)
    except Exception as error:
        return BatchPredictionResult(
            image_path=image_path,
            output_dir=None,
            error=str(error),
        )


def _predict_single_rfdetr_image(model: object, image_path: Path, conf: float) -> Path:
    from PIL import Image
    import supervision as sv

    image = Image.open(image_path).convert("RGB")
    detections = model.predict(image, threshold=conf)
    annotated = image.copy()
    if hasattr(detections, "mask") and getattr(detections, "mask", None) is not None:
        annotated = sv.MaskAnnotator().annotate(annotated, detections)
    annotated = sv.BoxAnnotator().annotate(annotated, detections)
    output_dir = Path("runs") / "predict" / "rfdetr"
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated.save(output_dir / image_path.name)
    return output_dir


def _output_dir_from_results(results: object) -> Path | None:
    if not results:
        return None
    save_dir = getattr(results[0], "save_dir", None)
    if save_dir is None:
        return None
    return Path(save_dir)


def _initialize_prediction_worker(weight_path: str, conf: float) -> None:
    global _WORKER_MODEL, _WORKER_CONF

    if Path(weight_path).suffix.lower() == ".pth":
        _WORKER_MODEL = load_rfdetr_model(Path(weight_path))
        setattr(_WORKER_MODEL, "_yolomatic_rfdetr", True)
    else:
        yolo_class = import_ultralytics_yolo()
        _WORKER_MODEL = yolo_class(weight_path)
    _WORKER_CONF = conf


def _predict_image_in_worker(image_path: str) -> BatchPredictionResult:
    if _WORKER_MODEL is None:
        raise RuntimeError("Prediction worker was not initialized.")
    return _predict_single_image(_WORKER_MODEL, Path(image_path), _WORKER_CONF)


def build_batch_summary(
    results: Sequence[BatchPredictionResult], elapsed_seconds: float, workers: int
) -> dict[str, object]:
    succeeded = [result for result in results if result.succeeded]
    failed = [result for result in results if not result.succeeded]
    output_dirs = sorted(
        {
            str(result.output_dir)
            for result in succeeded
            if result.output_dir is not None
        }
    )
    return {
        "total": len(results),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "elapsed_seconds": elapsed_seconds,
        "workers": workers,
        "output_dirs": output_dirs,
    }


def render_batch_summary(
    results: Sequence[BatchPredictionResult], elapsed_seconds: float, workers: int
) -> None:
    summary = build_batch_summary(results, elapsed_seconds, workers)
    output_dirs = summary["output_dirs"]
    output_text = (
        "\n".join(str(path) for path in output_dirs)
        if output_dirs
        else "Not reported"
    )
    panel_style = "green" if summary["failed"] == 0 else "yellow"
    console.print(
        Panel(
            "\n".join(
                [
                    "[bold green]Folder prediction completed.[/bold green]",
                    (
                        f"Images: [bold]{summary['succeeded']}[/bold] succeeded / "
                        f"[bold]{summary['total']}[/bold] total"
                    ),
                    f"Failed: [bold]{summary['failed']}[/bold]",
                    f"Workers: [bold]{summary['workers']}[/bold]",
                    f"Elapsed: [bold]{summary['elapsed_seconds']:.1f}s[/bold]",
                    f"Saved results to:\n[bold]{output_text}[/bold]",
                ]
            ),
            title="Prediction Summary",
            border_style=panel_style,
        )
    )

    failures = [result for result in results if not result.succeeded]
    if not failures:
        return

    table = Table(title="Failed Images", box=box.SIMPLE_HEAVY)
    table.add_column("Image", overflow="fold")
    table.add_column("Error", overflow="fold")
    for result in failures:
        table.add_row(str(result.image_path), result.error or "Unknown error")
    console.print(table)


def make_prediction_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[dim]{task.fields[current]}"),
        console=console,
    )


def run_folder_prediction(
    weight_path: Path, source_dir: Path, conf: float, workers: int
) -> list[BatchPredictionResult]:
    image_paths = discover_images(source_dir)
    if not image_paths:
        raise ValueError("No supported image files found in the selected folder.")

    if workers == 1:
        if is_rfdetr_weight(weight_path):
            model = load_rfdetr_model(weight_path)
            setattr(model, "_yolomatic_rfdetr", True)
        else:
            yolo_class = import_ultralytics_yolo()
            model = yolo_class(str(weight_path))
        results: list[BatchPredictionResult] = []
        with make_prediction_progress() as progress:
            task_id = progress.add_task(
                "Predicting images",
                total=len(image_paths),
                current="Starting",
            )
            for image_path in image_paths:
                progress.update(task_id, current=image_path.name)
                results.append(_predict_single_image(model, image_path, conf))
                progress.advance(task_id)
            progress.update(task_id, current="Complete")
        return results

    results = []
    with make_prediction_progress() as progress:
        task_id = progress.add_task(
            f"Predicting images ({workers} workers)",
            total=len(image_paths),
            current="Starting",
        )
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_prediction_worker,
            initargs=(str(weight_path), conf),
        ) as executor:
            futures = {
                executor.submit(_predict_image_in_worker, str(image_path)): image_path
                for image_path in image_paths
            }
            for future in as_completed(futures):
                image_path = futures[future]
                progress.update(task_id, current=image_path.name)
                try:
                    results.append(future.result())
                except Exception as error:
                    results.append(
                        BatchPredictionResult(
                            image_path=image_path,
                            output_dir=None,
                            error=str(error),
                        )
                    )
                progress.advance(task_id)
        progress.update(task_id, current="Complete")
    return sorted(results, key=lambda result: result.image_path.name.lower())


def main() -> None:
    args = parse_args()
    root = project_root()
    print_stylized_header("YOLO Predictor")
    available_weights = find_available_weights(root)
    if not available_weights:
        console.print(
            Panel(
                "[bold red]No .pt weights were found in the project root or runs directory.[/bold red]",
                border_style="red",
            )
        )
        raise SystemExit(1)

    render_weight_table(root, available_weights)

    try:
        workers = validate_worker_count(args.workers)
        selected_weight = resolve_weight(root, args.weight, available_weights)
        mode = args.mode or select_mode()
        source_path = resolve_source(mode, args.source)
        render_prediction_summary(selected_weight, mode, source_path)
        console.print("\n[bold green]Running prediction...[/bold green]")
        if mode == "folder":
            started_at = time.perf_counter()
            results = run_folder_prediction(
                selected_weight,
                source_path,
                args.conf,
                workers,
            )
            elapsed_seconds = time.perf_counter() - started_at
            render_batch_summary(results, elapsed_seconds, workers)
            output_dir = None
        else:
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
    except MLDependencyError as error:
        console.print(f"[bold red]Prediction failed: {error}[/bold red]")
        raise SystemExit(1) from error
    except Exception as error:
        console.print(f"[bold red]Prediction failed: {error}[/bold red]")
        raise SystemExit(1) from error

    if mode == "folder":
        return

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
