"""Responsive TUI Wizard for Uploading Datasets to Roboflow."""

from __future__ import annotations

import argparse
import logging
import concurrent.futures
import contextlib
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from rich import box
from rich.align import Align
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    FileSizeColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TotalFileSizeColumn,
    TransferSpeedColumn,
)
from rich.table import Table
from rich.text import Text
from roboflow import Roboflow
from roboflow.adapters import rfapi
from roboflow.adapters.rfapi import RoboflowError

from src.datasets.validate_roboflow import (
    TASK_CLASSIFY,
    TASK_INSTANCE_SEG,
    TASK_OBJECT_DETECT,
    TASK_POSE,
    TASK_SEMANTIC_SEG,
    RoboflowValidationResult,
    validate_roboflow_dataset,
)
from src.integrations.roboflow_dataset import (
    create_batch_zip_archive,
    partition_dataset_into_batches,
    poll_zip_status,
    upload_zip_stream,
)
from src.utils.cli import (
    NAV_BACK,
    ParameterDefinition,
    clear_screen,
    console,
    expected_error_panel,
    format_label,
    format_path,
    get_parameter_value_input,
    get_user_choice,
    print_stylized_header,
    warning_panel,
)
from src.utils.project import list_dataset_directories, project_root

logger = logging.getLogger(__name__)

WIZARD_STEPS = ["Dataset", "Validation", "Workspace", "Project", "Settings", "Upload"]


def _wizard_kwargs(step_index: int) -> Dict[str, Any]:
    return {"wizard_steps": WIZARD_STEPS, "wizard_current_step": step_index}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="yolomatic-upload-dataset", description="Upload dataset to Roboflow with validation and progress tracking.")
    parser.add_argument("--api-key", help="Roboflow Private API Key (if omitted, will use .env or prompt).")
    parser.add_argument("--workspace", help="Roboflow workspace slug.")
    parser.add_argument("--project", help="Roboflow project slug / ID.")
    parser.add_argument("--dataset", help="Path to dataset directory.")
    parser.add_argument(
        "--type",
        default="instance-segmentation",
        choices=[TASK_OBJECT_DETECT, TASK_INSTANCE_SEG, TASK_SEMANTIC_SEG, TASK_CLASSIFY, TASK_POSE],
        help="Roboflow project type when creating a new project.",
    )
    parser.add_argument(
        "--license",
        default="private",
        choices=["private", "MIT", "CC BY 4.0"],
        help="Project license (default: private).",
    )
    parser.add_argument("--batch-name", default="dataset-upload", help="Batch tag/name for this upload.")
    parser.add_argument("--workers", type=int, default=10, help="Parallel upload workers for per-image flow.")
    parser.add_argument("--zip", action="store_true", default=True, help="Use fast Zip upload flow (default: True).")
    parser.add_argument("--no-zip", dest="zip", action="store_false", help="Use per-image upload flow.")
    parser.add_argument(
        "--max-batch-size-mb",
        type=float,
        default=500.0,
        help="Maximum uncompressed batch size in MB per zip upload to prevent GCS timeouts (default: 500.0 MB).",
    )
    parser.add_argument(
        "--chunk-images",
        type=int,
        default=1000,
        help="Maximum number of images per zip upload batch (default: 1000).",
    )
    parser.add_argument(
        "--upload-timeout",
        type=int,
        default=1800,
        help="Socket timeout in seconds for archive PUT request (default: 1800s).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per batch upload on connection error/timeout (default: 3).",
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt.")
    # `argv=None` defers to sys.argv; callers that delegate into this CLI (e.g.
    # `yolomatic-upload --dataset`) must pass an explicit list, because their own
    # flags are not defined here and would abort argparse with SystemExit(2).
    return parser.parse_args(argv)


def load_env_credentials(root_dir: Path) -> Tuple[str, Optional[str]]:
    try:
        from dotenv import load_dotenv
        load_dotenv(root_dir / ".env")
    except ImportError:
        pass

    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    workspace = os.getenv("ROBOFLOW_WORKSPACE", "").strip() or None
    return api_key, workspace


def prompt_tui_text(
    name: str,
    description: str,
    help_text: str,
    default: Optional[str] = None,
    breadcrumbs: Optional[List[str]] = None,
) -> Optional[str]:
    param = ParameterDefinition(
        name=name,
        category="Roboflow",
        default=default or "",
        value_type="str",
        description=description,
        help_text=help_text,
    )
    result = get_parameter_value_input(param, current_value=default)
    if result in (NAV_BACK, "Back"):
        return NAV_BACK  # type: ignore
    return str(result)


def step_select_dataset(root_dir: Path, requested_dataset: Optional[str]) -> Optional[Path]:
    if requested_dataset:
        cand = Path(requested_dataset).expanduser()
        if not cand.is_absolute():
            cand = root_dir / cand
        if cand.exists():
            return cand.resolve()
        console.print(expected_error_panel(f"Specified dataset path not found: {cand}"))
        return None

    discovered = list_dataset_directories(root_dir)
    options: List[str] = []
    descriptions: Dict[str, str] = {}

    for d in discovered:
        label = format_label(d.name)
        options.append(label)
        has_yaml = (d / "data.yaml").exists() or (d / "dataset.yaml").exists()
        descriptions[label] = (
            f"[bold cyan]{d.name}[/bold cyan]\n\n"
            f"Path: {format_path(d)}\n"
            f"Config: {'[green]data.yaml found[/green]' if has_yaml else '[yellow]No data.yaml[/yellow]'}\n\n"
            "[dim]Press Enter to select and validate this dataset.[/dim]"
        )

    options.append("Enter Manual Path")
    descriptions["Enter Manual Path"] = "Enter a custom directory path outside the ./datasets directory."
    options.append("Back")
    descriptions["Back"] = "Return to previous menu."

    selected = get_user_choice(
        options,
        allow_back=True,
        title="Select Dataset to Upload",
        text="Choose a local dataset directory to validate and upload to Roboflow:",
        descriptions=descriptions,
        breadcrumbs=["YOLOmatic", "Roboflow Upload", "Dataset"],
        tip="Use up/down arrows to navigate, Enter to select, B to go back.",
        **_wizard_kwargs(0),
    )

    if selected in (NAV_BACK, "Back"):
        return None

    if selected == "Enter Manual Path":
        manual_path_str = prompt_tui_text(
            "Dataset Path",
            "Enter path to local dataset folder",
            "Path to the folder containing data.yaml or images/ and labels/.",
            default="datasets/vegetation_fused",
            breadcrumbs=["YOLOmatic", "Roboflow Upload", "Manual Path"],
        )
        if manual_path_str in (NAV_BACK, "Back") or not manual_path_str:
            return None
        p = Path(manual_path_str).expanduser()
        if not p.is_absolute():
            p = root_dir / p
        if not p.exists():
            console.print(expected_error_panel(f"Dataset path does not exist: {p}"))
            return None
        return p.resolve()

    for d in discovered:
        if format_label(d.name) == selected:
            return d.resolve()

    return None


def step_validate_dataset(
    dataset_path: Path,
    target_task: Optional[str] = None,
) -> Optional[RoboflowValidationResult]:
    clear_screen()
    print_stylized_header("Roboflow Dataset Upload")

    with console.status("[cyan]Running pre-flight structural validation...[/cyan]", spinner="dots"):
        result = validate_roboflow_dataset(dataset_path, target_task=target_task)

    # Render split breakdown table
    table = Table(title=f"Dataset Structure: {dataset_path.name}", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Split", style="bold white")
    table.add_column("Images", justify="right", style="cyan")
    table.add_column("Annotations", justify="right", style="green")
    table.add_column("Clean Negatives", justify="right", style="dim green")
    table.add_column("Missing Labels", justify="right", style="yellow")
    table.add_column("Coverage", justify="right", style="bold green")

    for split_name, stats in sorted(result.splits.items()):
        cov_str = f"{stats.coverage_pct:.1f}%"
        missing_style = f"[red]{stats.missing_labels:,}[/red]" if stats.missing_labels > 0 else "0"
        table.add_row(
            split_name,
            f"{stats.images:,}",
            f"{stats.annotated:,}",
            f"{stats.negatives:,}",
            missing_style,
            cov_str,
        )

    table.add_section()
    total_imgs = result.total_images
    total_anns = result.total_annotated
    total_negs = result.total_negatives
    overall_cov = f"{result.overall_coverage_pct:.1f}%"
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{total_imgs:,}[/bold]",
        f"[bold]{total_anns:,}[/bold]",
        f"[bold]{total_negs:,}[/bold]",
        "-",
        f"[bold]{overall_cov}[/bold]",
    )

    console.print(table)
    console.print(f"[dim]Detected Task: [bold cyan]{result.detected_task}[/bold cyan]  |  Classes ({result.num_classes}): {', '.join(result.class_names[:8])}{'...' if len(result.class_names) > 8 else ''}[/dim]\n")

    if not result.is_valid:
        error_lines = "\n".join(f"  * {err}" for err in result.errors[:10])
        console.print(expected_error_panel(f"Dataset has structural errors that must be resolved:\n\n{error_lines}"))
        options = ["Back to Dataset Selection", "Cancel"]
        get_user_choice(
            options,
            allow_back=True,
            title="Validation Failed",
            text="The dataset cannot be uploaded in its current state.",
            breadcrumbs=["YOLOmatic", "Roboflow Upload", "Validation Error"],
            **_wizard_kwargs(1),
        )
        return None

    if result.warnings:
        warn_lines = "\n".join(f"  * {w}" for w in result.warnings[:5])
        console.print(warning_panel(f"Validation warnings:\n\n{warn_lines}"))

    options = ["Continue with Upload", "Select Different Dataset", "Back"]
    choice = get_user_choice(
        options,
        allow_back=True,
        title="Validation Passed [OK]",
        text=f"Dataset verified successfully ({total_imgs:,} images, {overall_cov} coverage).",
        descriptions={
            "Continue with Upload": "Proceed to Roboflow workspace and project selection.",
            "Select Different Dataset": "Choose a different dataset folder.",
            "Back": "Return to the dataset picker.",
        },
        breadcrumbs=["YOLOmatic", "Roboflow Upload", "Validation"],
        **_wizard_kwargs(1),
    )

    if choice in (NAV_BACK, "Back", "Select Different Dataset"):
        return None

    return result


def step_resolve_auth(root_dir: Path, cli_key: Optional[str]) -> Tuple[Roboflow, str]:
    env_key, _ = load_env_credentials(root_dir)
    api_key = cli_key or env_key

    while not api_key:
        prompt_val = prompt_tui_text(
            "Roboflow API Key",
            "Enter your Roboflow Private API Key",
            "Your Private API Key from app.roboflow.com/settings/api.",
            breadcrumbs=["YOLOmatic", "Roboflow Upload", "API Key"],
        )
        if prompt_val in (NAV_BACK, "Back") or not prompt_val:
            raise RuntimeError("API key is required.")
        api_key = prompt_val.strip()

    with console.status("[cyan]Authenticating with Roboflow...[/cyan]", spinner="dots"):
        try:
            rf = Roboflow(api_key=api_key)
            return rf, api_key
        except Exception as exc:
            raise RuntimeError(f"Authentication failed: {exc}") from exc


def step_select_workspace(rf: Roboflow, root_dir: Path, cli_ws: Optional[str]) -> Tuple[Any, str]:
    _, env_ws = load_env_credentials(root_dir)
    default_ws = cli_ws or getattr(rf, "current_workspace", None) or env_ws

    if cli_ws:
        try:
            ws = rf.workspace(cli_ws)
            return ws, cli_ws
        except Exception as exc:
            console.print(expected_error_panel(f"Failed to access workspace '{cli_ws}': {exc}"))

    options: List[str] = []
    descriptions: Dict[str, str] = {}

    if default_ws:
        options.append(f"Default: {default_ws}")
        descriptions[f"Default: {default_ws}"] = f"Use workspace slug '{default_ws}' associated with your account."

    options.extend(["Enter Workspace Slug", "Back"])
    descriptions["Enter Workspace Slug"] = "Type the slug of the workspace you wish to upload into."
    descriptions["Back"] = "Return to validation step."

    choice = get_user_choice(
        options,
        allow_back=True,
        title="Select Roboflow Workspace",
        text="Choose which workspace to upload the dataset into:",
        descriptions=descriptions,
        breadcrumbs=["YOLOmatic", "Roboflow Upload", "Workspace"],
        **_wizard_kwargs(2),
    )

    if choice in (NAV_BACK, "Back"):
        return NAV_BACK, ""  # type: ignore

    if choice.startswith("Default: "):
        slug = choice.replace("Default: ", "").strip()
    else:
        manual_slug = prompt_tui_text(
            "Workspace Slug",
            "Enter Roboflow workspace slug",
            "The URL slug for your workspace (e.g. 'my-workspace').",
            default=default_ws or "",
            breadcrumbs=["YOLOmatic", "Roboflow Upload", "Workspace Entry"],
        )
        if manual_slug in (NAV_BACK, "Back") or not manual_slug:
            return NAV_BACK, ""  # type: ignore
        slug = manual_slug.strip()

    with console.status(f"[cyan]Connecting to workspace '{slug}'...[/cyan]", spinner="dots"):
        try:
            ws = rf.workspace(slug)
            return ws, slug
        except Exception as exc:
            console.print(expected_error_panel(f"Workspace '{slug}' could not be loaded: {exc}"))
            return NAV_BACK, ""  # type: ignore


def step_select_or_create_project(
    ws: Any,
    ws_slug: str,
    validation_res: RoboflowValidationResult,
    cli_project: Optional[str] = None,
    cli_type: Optional[str] = None,
    cli_license: str = "private",
) -> Tuple[str, str, str]:
    if cli_project and cli_project.strip():
        return cli_project.strip(), cli_type or validation_res.detected_task, cli_license

    options: List[str] = []
    descriptions: Dict[str, str] = {}
    project_map: Dict[str, str] = {}

    try:
        raw_projects = ws.projects() or []
        for p in raw_projects:
            pid = p.split("/")[-1] if "/" in p else p
            label = f"Existing: {pid}"
            options.append(label)
            project_map[label] = pid
            descriptions[label] = f"Upload dataset into existing project '{pid}'."
    except Exception:
        pass

    options.append("Create New Project")
    descriptions["Create New Project"] = "Initialize a new project in this workspace."
    options.append("Back")
    descriptions["Back"] = "Return to workspace selection."

    choice = get_user_choice(
        options,
        allow_back=True,
        title="Select Roboflow Project",
        text=f"Choose a project in workspace '{ws_slug}' or create a new one:",
        descriptions=descriptions,
        breadcrumbs=["YOLOmatic", "Roboflow Upload", "Project"],
        **_wizard_kwargs(3),
    )

    if choice in (NAV_BACK, "Back"):
        return NAV_BACK, "", ""  # type: ignore

    if choice in project_map:
        return project_map[choice], validation_res.detected_task, cli_license

    # Create new project
    suggested_name = validation_res.dataset_path.name.replace("_", "-").lower()
    new_name = prompt_tui_text(
        "Project Name",
        "Enter new project name/slug",
        "A URL-friendly name using letters, numbers, and dashes.",
        default=suggested_name,
        breadcrumbs=["YOLOmatic", "Roboflow Upload", "New Project Name"],
    )
    if new_name in (NAV_BACK, "Back") or not new_name:
        return NAV_BACK, "", ""  # type: ignore

    # Project type picker
    type_options = [
        "1. Instance Segmentation (Polygons / Masks)",
        "2. Object Detection (Bounding Boxes)",
        "3. Semantic Segmentation (Dense Pixels)",
        "4. Pose Estimation (Keypoints)",
        "5. Classification",
    ]
    type_descriptions = {
        type_options[0]: "Polygons and masks for distinct object instances.",
        type_options[1]: "Bounding box annotations.",
        type_options[2]: "Dense per-pixel semantic class masks.",
        type_options[3]: "Keypoint and pose estimation landmarks.",
        type_options[4]: "Image-level single or multi-class labels.",
    }

    type_choice = get_user_choice(
        type_options,
        allow_back=True,
        title="Select Project Type",
        text=f"Detected dataset task: {validation_res.detected_task}. Choose project type:",
        descriptions=type_descriptions,
        breadcrumbs=["YOLOmatic", "Roboflow Upload", "Project Type"],
        **_wizard_kwargs(3),
    )
    if type_choice in (NAV_BACK, "Back"):
        return NAV_BACK, "", ""  # type: ignore

    type_mapping = {
        type_options[0]: TASK_INSTANCE_SEG,
        type_options[1]: TASK_OBJECT_DETECT,
        type_options[2]: TASK_SEMANTIC_SEG,
        type_options[3]: TASK_POSE,
        type_options[4]: TASK_CLASSIFY,
    }
    chosen_type = type_mapping.get(type_choice, validation_res.detected_task)

    lic_choice = get_user_choice(
        ["Private (Paid / Private Quota)", "Public / MIT", "Back"],
        allow_back=True,
        title="Select Project License",
        text="Choose license visibility for the project:",
        breadcrumbs=["YOLOmatic", "Roboflow Upload", "License"],
        **_wizard_kwargs(3),
    )
    if lic_choice in (NAV_BACK, "Back"):
        return NAV_BACK, "", ""  # type: ignore

    chosen_lic = "private" if "Private" in lic_choice else "MIT"
    return new_name.strip(), chosen_type, chosen_lic


def step_configure_settings(
    args: argparse.Namespace,
    validation_res: RoboflowValidationResult,
) -> Tuple[bool, str, float, int, int, int]:
    options = [
        "Fast Zip Upload (Recommended)",
        "Concurrent Per-Image Upload",
        "Back",
    ]
    descriptions = {
        options[0]: (
            "Compresses dataset into optimized chunks and uploads directly to signed GCS URLs.\n\n"
            "Fastest method for datasets with 500+ images. Handles network drops with chunk retries."
        ),
        options[1]: (
            "Uploads images and annotations one-by-one using concurrent threads.\n\n"
            "Good for small batches (<200 images) or live monitoring."
        ),
        options[2]: "Return to project selection.",
    }

    choice = get_user_choice(
        options,
        allow_back=True,
        title="Select Upload Strategy",
        text="Choose how dataset files will be transferred to Roboflow:",
        descriptions=descriptions,
        breadcrumbs=["YOLOmatic", "Roboflow Upload", "Settings"],
        **_wizard_kwargs(4),
    )
    if choice in (NAV_BACK, "Back"):
        return False, "", 0.0, 0, 0, 0

    use_zip = "Fast Zip" in choice
    batch_name = args.batch_name or "dataset-upload"
    max_batch_mb = args.max_batch_size_mb
    chunk_images = args.chunk_images
    workers = args.workers
    max_retries = args.max_retries

    return use_zip, batch_name, max_batch_mb, chunk_images, workers, max_retries


def execute_zip_upload(
    api_key: str,
    ws: Any,
    project: Any,
    dataset_path: Path,
    parsed_images: Sequence[Dict[str, Any]],
    batch_name: str,
    max_batch_size_mb: float,
    chunk_images: int,
    timeout_secs: int,
    max_retries: int,
) -> Dict[str, Any]:
    batches = partition_dataset_into_batches(
        dataset_path=dataset_path,
        parsed_images=parsed_images,
        max_batch_size_mb=max_batch_size_mb,
        max_images_per_batch=chunk_images,
    )
    total_batches = len(batches)

    if total_batches > 1:
        console.print(
            f"[cyan]Dataset partitioned into [bold]{total_batches} batches[/bold] "
            f"(max {max_batch_size_mb:.0f} MB / {chunk_images} images each) to ensure network stability.[/cyan]\n"
        )

    uploaded_count = 0
    server_task_ids: List[str] = []
    project_slug = project.id.rsplit("/")[-1]

    for b_idx, batch_info in enumerate(batches, start=1):
        batch_label = f"[{b_idx}/{total_batches}]" if total_batches > 1 else ""
        batch_tag = f"{batch_name}-part{b_idx}" if total_batches > 1 else batch_name

        temp_zip: Optional[Path] = None
        for attempt in range(1, max_retries + 1):
            try:
                # 1. Create Zip Archive with Progress
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold cyan]{task.description}"),
                    BarColumn(bar_width=40, style="dim white", complete_style="green", finished_style="bold green"),
                    TaskProgressColumn(),
                    MofNCompleteColumn(),
                    FileSizeColumn(),
                    TextColumn("/"),
                    TotalFileSizeColumn(),
                    TimeRemainingColumn(),
                    TimeElapsedColumn(),
                    console=console,
                ) as comp_progress:
                    est_bytes = batch_info.estimated_bytes or 1024 * 1024
                    t_task = comp_progress.add_task(f"Compressing batch {batch_label}...", total=est_bytes)
                    temp_zip = create_batch_zip_archive(
                        dataset_path=dataset_path,
                        batch_info=batch_info,
                        progress_callback=lambda n: comp_progress.update(t_task, advance=n),
                    )

                # 2. Get Signed Upload URL
                status_msg = f"Requesting signed upload URL {batch_label} from Roboflow..." if batch_label else "Requesting signed upload URL from Roboflow..."
                with console.status(f"[cyan]{status_msg}[/cyan]", spinner="dots"):
                    init = rfapi.init_zip_upload(
                        api_key,
                        ws.url,
                        project_slug,
                        batch_name=batch_tag,
                    )
                    task_id = init["taskId"]
                    signed_url = init["signedUrl"]
                    server_task_ids.append(task_id)
                    # Record before the PUT: if the run dies mid-upload this is the
                    # only handle on the server-side task.
                    logger.info(
                        "Roboflow batch %d/%d: task_id=%s batch_tag=%s images=%d",
                        b_idx, total_batches, task_id, batch_tag, len(batch_info.images),
                    )

                # 3. Stream Upload with Progress
                upload_desc = f"Uploading batch {batch_label} to Roboflow..." if total_batches > 1 else "Uploading archive to Roboflow..."
                total_bytes = temp_zip.stat().st_size

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold magenta]{task.description}"),
                    BarColumn(bar_width=40, style="dim white", complete_style="bright_cyan", finished_style="bold green"),
                    TaskProgressColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeRemainingColumn(),
                    TimeElapsedColumn(),
                    console=console,
                ) as up_progress:
                    u_task = up_progress.add_task(upload_desc, total=total_bytes)
                    upload_zip_stream(
                        signed_url=signed_url,
                        zip_path=temp_zip,
                        progress_callback=lambda n: up_progress.update(u_task, advance=n),
                        timeout_secs=timeout_secs,
                    )

                # 4. Poll Server Status
                with Progress(
                    SpinnerColumn("dots12", style="bold cyan"),
                    TextColumn("[bold yellow]{task.description}"),
                    TimeElapsedColumn(),
                    console=console,
                ) as poll_progress:
                    prefix = f"{batch_label} " if batch_label else ""
                    p_task = poll_progress.add_task(f"{prefix}Roboflow server is parsing and indexing dataset...", total=None)

                    def _on_stage(stage_text: str) -> None:
                        poll_progress.update(p_task, description=f"{prefix}Roboflow processing: [bold cyan]{stage_text}[/bold cyan]")

                    poll_zip_status(
                        api_key=api_key,
                        workspace_url=ws.url,
                        task_id=task_id,
                        status_callback=_on_stage,
                    )

                uploaded_count += len(batch_info.images)
                console.print(f"[green][OK] Batch {batch_label} uploaded and processed successfully.[/green]\n")
                break

            except (requests.exceptions.RequestException, TimeoutError, RoboflowError, OSError) as exc:
                console.print(f"[bold red][!] Batch {batch_label} upload error (attempt {attempt}/{max_retries}): {exc}[/bold red]")
                if attempt < max_retries:
                    backoff = 2 ** attempt
                    console.print(f"[yellow]Retrying in {backoff}s with a fresh upload URL...[/yellow]\n")
                    time.sleep(backoff)
                else:
                    raise
            finally:
                if temp_zip and temp_zip.exists():
                    with contextlib.suppress(Exception):
                        temp_zip.unlink()

    return {
        "uploaded": uploaded_count,
        "batches": total_batches,
        "server_task_ids": server_task_ids,
    }


def execute_per_image_upload(
    ws: Any,
    project: Any,
    parsed_images: Sequence[Dict[str, Any]],
    batch_name: str,
    num_workers: int = 10,
) -> Dict[str, Any]:
    total = len(parsed_images)
    stats = {
        "uploaded": 0,
        "duplicate": 0,
        "annotations_saved": 0,
        "errors": 0,
    }
    stats_lock = threading.Lock()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, style="dim white", complete_style="bright_blue", finished_style="bold green"),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("[cyan]{task.speed:.1f} img/s[/cyan]"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Uploading images and annotations...", total=total)

        def _worker(imagedesc: Dict[str, Any]) -> None:
            image_path = imagedesc["file"]
            split = imagedesc.get("split", "train")

            try:
                # 1. Upload image
                image_res, _, _ = project.upload_image(
                    image_path=image_path,
                    split=split,
                    batch_name=batch_name,
                    sequence_number=imagedesc.get("index"),
                    sequence_size=total,
                    num_retry_uploads=3,
                )

                is_dup = image_res.get("duplicate", False) if isinstance(image_res, dict) else False
                image_id = image_res.get("id") if isinstance(image_res, dict) else None

                with stats_lock:
                    if is_dup:
                        stats["duplicate"] += 1
                    else:
                        stats["uploaded"] += 1

                # 2. Save annotation
                annotationdesc = imagedesc.get("annotationfile")
                if annotationdesc and image_id:
                    annotation_path = annotationdesc.get("file")
                    labelmap = annotationdesc.get("labelmap")

                    if annotation_path:
                        annot_res, _ = project.save_annotation(
                            annotation_path=annotation_path,
                            annotation_labelmap=labelmap,
                            image_id=image_id,
                            job_name=batch_name,
                            num_retry_uploads=3,
                        )
                        if annot_res and isinstance(annot_res, dict) and annot_res.get("success"):
                            with stats_lock:
                                stats["annotations_saved"] += 1

            except Exception as exc:
                logger.exception("Failed to upload image %s (split=%s)", image_path, split)
                with stats_lock:
                    stats["errors"] += 1
                    stats.setdefault("first_error", f"{type(exc).__name__}: {exc}")
            finally:
                progress.advance(task)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(executor.map(_worker, parsed_images))

    return stats


def display_final_report(
    ws_name: str,
    project_name: str,
    dataset_name: str,
    mode: str,
    duration: float,
    stats: Dict[str, Any],
) -> None:
    dashboard_url = f"https://app.roboflow.com/{ws_name}/{project_name}"

    # Report the outcome the run actually had. A run that uploaded nothing, or that
    # errored on some images, must never be titled a success.
    error_count = stats.get("errors", 0)
    uploaded_count = stats.get("uploaded", 0)
    if error_count and not uploaded_count:
        title, header_style = "Upload Failed", "bold red"
    elif error_count:
        title, header_style = "Upload Completed With Errors", "bold yellow"
    else:
        title, header_style = "Upload Completed Successfully", "bold green"

    table = Table(title=title, box=box.ROUNDED, header_style=header_style)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="bold white")

    table.add_row("Workspace", ws_name)
    table.add_row("Project", project_name)
    table.add_row("Dataset", dataset_name)
    table.add_row("Upload Strategy", mode)
    table.add_row("Duration", f"{duration:.2f} seconds ({duration / 60:.1f} mins)")

    if "batches" in stats and stats["batches"] > 1:
        table.add_row("Batches Uploaded", f"{stats['batches']} batches")
    if "uploaded" in stats:
        table.add_row("Images Uploaded", f"{stats['uploaded']:,}")
    if "duplicate" in stats and stats["duplicate"] > 0:
        table.add_row("Duplicate Images", f"{stats['duplicate']:,}")
    if "annotations_saved" in stats and stats["annotations_saved"] > 0:
        table.add_row("Annotations Saved", f"{stats['annotations_saved']:,}")
    if "errors" in stats and stats["errors"] > 0:
        table.add_row("Errors Encountered", f"[red]{stats['errors']:,}[/red]")
        if stats.get("first_error"):
            table.add_row("First Error", f"[red]{stats['first_error']}[/red]")
    if duration > 0 and "uploaded" in stats:
        total_p = stats.get("uploaded", 0) + stats.get("duplicate", 0)
        avg_speed = total_p / duration
        table.add_row("Average Throughput", f"{avg_speed:.1f} images/sec")

    console.print("\n")
    console.print(table)

    link_panel = Panel(
        Align.center(
            Text.assemble(
                ("Roboflow Project Dashboard: ", "bold cyan"),
                (dashboard_url, "bold underline bright_cyan"),
            )
        ),
        box=box.ROUNDED,
        border_style="bright_cyan",
    )
    console.print(link_panel)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    root = project_root()

    clear_screen()
    print_stylized_header("Roboflow Dataset Upload")

    # Step 0: Dataset Selection
    dataset_path = step_select_dataset(root, args.dataset)
    if not dataset_path:
        return

    # Step 1: Validation
    validation_res = step_validate_dataset(dataset_path, target_task=args.type)
    if not validation_res:
        return

    # Step 2: Authentication & Workspace Selection
    try:
        rf, api_key = step_resolve_auth(root, args.api_key)
    except Exception as exc:
        console.print(expected_error_panel(f"Authentication failed: {exc}"))
        return

    ws, ws_slug = step_select_workspace(rf, root, args.workspace)
    if ws == NAV_BACK or not ws_slug:
        return

    # Step 3: Project Selection / Creation
    project_name, project_type, project_license = step_select_or_create_project(
        ws=ws,
        ws_slug=ws_slug,
        validation_res=validation_res,
        cli_project=args.project,
        cli_type=args.type,
        cli_license=args.license,
    )
    if project_name == NAV_BACK or not project_name:
        return

    # Initialize project on server
    with console.status(f"[cyan]Initializing project '{project_name}' on Roboflow...[/cyan]", spinner="dots"):
        try:
            project, created = ws._get_or_create_project(
                project_id=project_name,
                license=project_license,
                type=project_type,
            )
        except Exception as exc:
            console.print(expected_error_panel(f"Failed to initialize project '{project_name}': {exc}"))
            return

    # Step 4: Configure Upload Parameters
    if not args.yes:
        use_zip, batch_name, max_batch_mb, chunk_images, workers, max_retries = step_configure_settings(args, validation_res)
        if not batch_name:
            return
    else:
        use_zip = args.zip
        batch_name = args.batch_name
        max_batch_mb = args.max_batch_size_mb
        chunk_images = args.chunk_images
        workers = args.workers
        max_retries = args.max_retries

    # Confirm Screen
    summary_table = Table(title="Upload Parameters", box=box.SIMPLE_HEAD, header_style="bold cyan")
    summary_table.add_column("Parameter", style="dim cyan")
    summary_table.add_column("Value", style="bold white")
    summary_table.add_row("Workspace", ws_slug)
    summary_table.add_row("Project", project_name)
    summary_table.add_row("Dataset Directory", format_path(dataset_path))
    summary_table.add_row("Total Images", f"{validation_res.total_images:,}")
    summary_table.add_row("Batch Name", batch_name)
    strategy_str = f"Fast Zip Flow (Max {max_batch_mb:.0f} MB / {chunk_images} imgs per batch)" if use_zip else f"Concurrent Per-Image Flow ({workers} workers)"
    summary_table.add_row("Upload Strategy", strategy_str)

    console.print("\n")
    console.print(summary_table)

    if not args.yes:
        choice = get_user_choice(
            ["Proceed with Upload", "Cancel"],
            allow_back=True,
            title="Confirm Upload",
            text="Verify the parameters above and start the upload:",
            breadcrumbs=["YOLOmatic", "Roboflow Upload", "Confirm"],
            **_wizard_kwargs(5),
        )
        if choice in (NAV_BACK, "Cancel"):
            console.print("[yellow]Upload cancelled by user.[/yellow]")
            return

    console.print("\n[bold cyan]Starting upload process...[/bold cyan]\n")
    start_time = time.monotonic()

    # Build image descriptors for upload
    image_items: List[Dict[str, Any]] = []
    classes = {i: name for i, name in enumerate(validation_res.class_names)}
    for root_dir_p, _, files in os.walk(dataset_path):
        for f in files:
            fp = Path(root_dir_p) / f
            if fp.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                split = "train"
                for p in fp.relative_to(dataset_path).parts:
                    p_lower = p.lower()
                    if p_lower in ("train", "training"):
                        split = "train"
                        break
                    if p_lower in ("val", "valid", "validation"):
                        split = "val"
                        break
                    if p_lower in ("test", "testing"):
                        split = "test"
                        break

                stem = fp.stem
                lbl_cand = dataset_path / "labels" / split / f"{stem}.txt"
                if not lbl_cand.exists() and split == "val":
                    lbl_cand = dataset_path / "labels" / "valid" / f"{stem}.txt"
                if not lbl_cand.exists():
                    lbl_cand = fp.with_suffix(".txt")

                annot_desc = None
                if lbl_cand.exists():
                    annot_desc = {"file": str(lbl_cand.resolve()), "labelmap": classes}

                image_items.append({
                    "file": str(fp.resolve()),
                    "split": "valid" if split == "val" else split,
                    "annotationfile": annot_desc,
                    "index": len(image_items),
                })

    try:
        if use_zip:
            stats = execute_zip_upload(
                api_key=api_key,
                ws=ws,
                project=project,
                dataset_path=dataset_path,
                parsed_images=image_items,
                batch_name=batch_name,
                max_batch_size_mb=max_batch_mb,
                chunk_images=chunk_images,
                timeout_secs=args.upload_timeout,
                max_retries=max_retries,
            )
        else:
            stats = execute_per_image_upload(
                ws=ws,
                project=project,
                parsed_images=image_items,
                batch_name=batch_name,
                num_workers=workers,
            )
    except KeyboardInterrupt:
        console.print("\n[yellow]Upload interrupted. Images already sent remain in the Roboflow project.[/yellow]")
        return
    except Exception as exc:
        # A failure here can leave earlier batches already committed server-side,
        # so say so plainly rather than surfacing a bare traceback.
        logger.exception("Roboflow dataset upload failed")
        console.print(
            expected_error_panel(
                f"Upload failed: {type(exc).__name__}: {exc}\n\n"
                f"Any batches that completed before this point are already in the "
                f"Roboflow project. Review the project before retrying to avoid duplicates:\n"
                f"  https://app.roboflow.com/{ws_slug}/{project_name}"
            )
        )
        return

    duration = time.monotonic() - start_time
    display_final_report(
        ws_name=ws_slug,
        project_name=project_name,
        dataset_name=dataset_path.name,
        mode="Fast Zip Flow" if use_zip else "Concurrent Per-Image Flow",
        duration=duration,
        stats=stats,
    )


if __name__ == "__main__":
    main()
