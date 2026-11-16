from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
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
from rich.table import Column, Table

from yolomatic.cli.result import CommandResult
from yolomatic.utils.cli import (
    console,
    get_user_choice,
    print_stylized_header,
    render_summary_panel,
    render_table,
)
from yolomatic.utils.jobs import cancellable_executor
from yolomatic.utils.logging_setup import log_exception
from yolomatic.utils.ml_dependencies import (
    MLDependencyError,
    import_cv2,
    import_detectron2,
    import_rfdetr_model_class,
    import_torch,
    import_ultralytics_yolo,
)
from yolomatic.utils.project import (
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
    return weight_path.suffix.lower() == ".pth" and not is_detectron2_weight(weight_path)


def _detectron2_run_dir(weight_path: Path) -> Path | None:
    """The training run folder (holding YOLOmatic's Detectron2 metadata), if any."""
    from yolomatic.trainers.detectron2_trainer import RUN_METADATA_FILE

    for folder in (weight_path.parent, weight_path.parent.parent):
        if (folder / RUN_METADATA_FILE).is_file():
            return folder
    return None


def is_detectron2_weight(weight_path: Path) -> bool:
    if weight_path.suffix.lower() != ".pth":
        return False
    if _detectron2_run_dir(weight_path) is not None:
        return True
    # Fallback for runs trained before YOLOmatic wrote metadata: whole path
    # components with Detectron2 naming only, so e.g. "/data/retinanet_notes/"
    # or an RF-DETR run under such a folder is not enough.
    import re

    pattern = re.compile(r"^(detectron2|(faster|mask|cascade_mask)_rcnn(_r_\d+.*)?|retinanet(_r_\d+.*)?)$")
    return any(pattern.match(part.lower()) for part in weight_path.parts)


def infer_rfdetr_class_from_weight(weight_path: Path) -> str:
    from yolomatic.models.rfdetr import infer_rfdetr_class

    return infer_rfdetr_class(weight_path)


def load_rfdetr_model(weight_path: Path) -> object:
    model_class = import_rfdetr_model_class(infer_rfdetr_class_from_weight(weight_path))
    return model_class(pretrain_weights=str(weight_path))


def build_detectron2_predict_cfg(weight_path: Path, conf: float, get_cfg: Any, model_zoo: Any) -> tuple[Any, list[str]]:
    """Rebuild the training config so trained weights load into the right model."""
    import json

    from yolomatic.trainers.detectron2_trainer import RUN_METADATA_FILE

    cfg = get_cfg()
    class_names: list[str] = []
    run_dir = _detectron2_run_dir(weight_path)
    if run_dir is not None and (run_dir / "config.yaml").is_file():
        cfg.merge_from_file(str(run_dir / "config.yaml"))
        meta = json.loads((run_dir / RUN_METADATA_FILE).read_text(encoding="utf-8"))
        class_names = [str(name) for name in meta.get("classes", [])]
    else:
        raise FileNotFoundError(
            f"No YOLOmatic Detectron2 metadata (config.yaml, {RUN_METADATA_FILE}) next to {weight_path}. "
            "Detectron2 checkpoints need their training config to be rebuilt; retrain with this "
            "version of YOLOmatic or copy the run's config.yaml beside the checkpoint."
        )
    cfg.MODEL.WEIGHTS = str(weight_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(conf)
    if hasattr(cfg.MODEL, "RETINANET"):
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = float(conf)
    try:
        cuda = import_torch().cuda.is_available()
    except Exception:  # noqa: BLE001
        cuda = False
    cfg.MODEL.DEVICE = "cuda" if cuda else "cpu"
    return cfg, class_names


def load_detectron2_predictor(weight_path: Path, conf: float = 0.25) -> tuple[Any, list[str]]:
    import_detectron2()
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg, class_names = build_detectron2_predict_cfg(weight_path, conf, get_cfg, model_zoo)
    return DefaultPredictor(cfg), class_names


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
            "Enter the image path (:b to go back): "
            if mode == "single"
            else "Enter the folder path (:b to go back): "
        )
        try:
            raw_value = input(prompt_text).strip()
        except EOFError:
            # No interactive input (piped stdin): stop instead of looping forever.
            raise KeyboardInterrupt from None
        if raw_value.lower() in {":b", ":back", ":q"}:
            raise KeyboardInterrupt
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
        "Model Family": "Detectron2" if is_detectron2_weight(weight_path) else "RF-DETR" if is_rfdetr_weight(weight_path) else "Ultralytics YOLO",
        "Mode": MODE_LABELS[mode],
        "Source": source_path,
    }
    render_summary_panel("Prediction Summary", fields)


def run_prediction(weight_path: Path, source_path: Path, conf: float) -> Path | None:
    if is_detectron2_weight(weight_path):
        return run_detectron2_prediction(weight_path, source_path, conf)
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


def detectron2_instances_to_records(instances: Any, class_names: Sequence[str]) -> list[dict[str, Any]]:
    """JSON-friendly predictions (boxes are xyxy pixels)."""
    boxes = instances.pred_boxes.tensor.tolist() if instances.has("pred_boxes") else []
    scores = instances.scores.tolist() if instances.has("scores") else []
    classes = instances.pred_classes.tolist() if instances.has("pred_classes") else []
    records = []
    for xyxy, score, cls in zip(boxes, scores, classes, strict=False):
        cls = int(cls)
        records.append({
            "class_id": cls,
            "class_name": class_names[cls] if 0 <= cls < len(class_names) else str(cls),
            "score": round(float(score), 4),
            "box_xyxy": [round(float(v), 2) for v in xyxy],
        })
    return records


def run_detectron2_prediction(weight_path: Path, source_path: Path, conf: float) -> Path:
    """Run a trained Detectron2 model and save annotated images plus JSON.

    The old implementation discarded the predictor output and only wrote a
    "prediction completed" text file per image.
    """
    import json
    from datetime import datetime

    from detectron2.data import MetadataCatalog
    from detectron2.utils.visualizer import Visualizer

    cv2 = import_cv2()
    predictor, class_names = load_detectron2_predictor(weight_path, conf)
    metadata = MetadataCatalog.get(f"yolomatic_predict_{weight_path.stem}")
    if class_names:
        metadata.thing_classes = class_names
    output_dir = Path("runs") / "predict" / "detectron2" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = [source_path] if source_path.is_file() else discover_images(source_path)
    for image_path in sources:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        instances = predictor(image)["instances"].to("cpu")
        drawn = Visualizer(image[:, :, ::-1], metadata=metadata).draw_instance_predictions(instances)
        cv2.imwrite(str(output_dir / f"{image_path.stem}.jpg"), drawn.get_image()[:, :, ::-1])
        (output_dir / f"{image_path.stem}.json").write_text(
            json.dumps({"image": str(image_path), "predictions": detectron2_instances_to_records(instances, class_names)}, indent=2),
            encoding="utf-8",
        )
    return output_dir


def run_rfdetr_prediction(weight_path: Path, source_path: Path, conf: float) -> Path:
    import supervision as sv
    from PIL import Image

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
    import supervision as sv
    from PIL import Image

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
        _WORKER_MODEL._yolomatic_rfdetr = True
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


def make_prediction_progress(target_console: Console | None = None) -> Progress:
    c = target_console or console
    width = c.width
    columns: list = [
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=15 if width < 100 else 30),
        TaskProgressColumn(),
        MofNCompleteColumn(),
    ]
    if width >= 100:
        columns.append(TimeRemainingColumn())
    if width >= 120:
        columns.append(TimeElapsedColumn())
        columns.append(TextColumn("[dim]{task.fields[current]}", table_column=Column(overflow="ellipsis", no_wrap=True)))
    elif width >= 90:
        columns.append(TextColumn("[dim]{task.fields[current]}", table_column=Column(overflow="ellipsis", no_wrap=True)))
    return Progress(*columns, console=c)


def run_folder_prediction(
    weight_path: Path, source_dir: Path, conf: float, workers: int
) -> list[BatchPredictionResult]:
    image_paths = discover_images(source_dir)
    if not image_paths:
        raise ValueError("No supported image files found in the selected folder.")

    if is_detectron2_weight(weight_path):
        # Detectron2 runs in-process (one model, sequential images); it used to
        # be loaded as YOLO (workers=1) or as RF-DETR (workers>1).
        results: list[BatchPredictionResult] = []
        for image_path in image_paths:
            try:
                output_dir = run_detectron2_prediction(weight_path, image_path, conf)
                results.append(BatchPredictionResult(image_path=image_path, output_dir=output_dir))
            except Exception as error:  # noqa: BLE001 - per-image failure is reported
                results.append(BatchPredictionResult(image_path=image_path, output_dir=None, error=str(error)))
        return results

    if workers == 1:
        if is_rfdetr_weight(weight_path):
            model = load_rfdetr_model(weight_path)
            model._yolomatic_rfdetr = True
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
        # cancellable_executor: on Ctrl-C queued images are dropped instead of the
        # pool silently predicting the rest of the folder before returning.
        with cancellable_executor(ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_prediction_worker,
            initargs=(str(weight_path), conf),
        )) as executor:
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


def main() -> CommandResult:
    args = parse_args()
    root = project_root()
    print_stylized_header("YOLO Predictor")
    available_weights = find_available_weights(root)
    if not available_weights:
        return CommandResult.failed(
            "No .pt weights were found in the project root or runs directory.",
            hint="Train a model first or pass a valid weights path with --weight.",
        )

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
            summary = build_batch_summary(results, elapsed_seconds, workers)
            outputs = {f"Output {i + 1}": path for i, path in enumerate(summary["output_dirs"])}
            if summary["failed"] and not summary["succeeded"]:
                return CommandResult.failed(f"All {summary['failed']} predictions failed.")
            return CommandResult.ok(
                f"{summary['succeeded']} of {summary['total']} images predicted.",
                outputs=outputs,
                review=True,
            )
        output_dir = run_prediction(selected_weight, source_path, args.conf)
    except FileNotFoundError as error:
        return CommandResult.failed(str(error), hint="Choose an existing image, folder, or weights file.")
    except ValueError as error:
        return CommandResult.failed(str(error), hint="Review the selected mode, source path, and confidence value.")
    except KeyboardInterrupt:
        return CommandResult.cancelled("Prediction cancelled.")
    except SystemExit as error:
        if error.code in (None, 0):
            return CommandResult.cancelled()
        raise
    except MLDependencyError as error:
        return CommandResult.failed(str(error), hint="Install the missing ML dependency and retry prediction.")
    except Exception as error:  # noqa: BLE001
        log_exception(error, "Prediction")
        return CommandResult.failed(str(error), hint="Check the selected model and source path.")

    if output_dir is None:
        return CommandResult.ok("Prediction completed, but no output directory was reported.", review=True)
    return CommandResult.ok("Prediction completed.", outputs={"Results": output_dir})

if __name__ == "__main__":
    main()
