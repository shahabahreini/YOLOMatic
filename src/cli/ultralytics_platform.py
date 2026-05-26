from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from rich.panel import Panel

from src.config.settings import load_settings, ultralytics_credential_status
from src.datasets.prepare import PrepareDatasetConfig, PrepareSplitConfig, prepare_dataset, slugify
from src.integrations.ultralytics_platform import (
    DEFAULT_BASE_URL,
    UltralyticsPlatformClient,
    UltralyticsPlatformError,
    load_api_key,
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
    render_table,
)
from src.utils.project import calculate_folder_size, format_size, list_dataset_directories


WIZARD_STEPS = ["Action", "Select", "Configure", "Run"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yolomatic-ultralytics", description="Ultralytics Platform dataset and weight workflows.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Ultralytics Platform API base URL.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list-datasets", help="List Platform datasets.")

    download_dataset = subparsers.add_parser("download-dataset", help="Download a Platform dataset export and prepare it locally.")
    download_dataset.add_argument("dataset_id")
    download_dataset.add_argument("--version", type=int, help="Dataset version number. Defaults to latest.")
    download_dataset.add_argument("--output-root", type=Path, default=Path("datasets"))
    download_dataset.add_argument("--slug", help="Prepared output dataset slug.")
    download_dataset.add_argument("--format", choices=["YOLO Detection", "YOLO Segmentation", "COCO"], default="YOLO Detection")
    download_dataset.add_argument("--train", type=float, default=0.70)
    download_dataset.add_argument("--val", type=float, default=0.20)
    download_dataset.add_argument("--test", type=float, default=0.10)
    download_dataset.add_argument("--seed", type=int, default=42)
    download_dataset.add_argument("--smart-split", action="store_true")
    download_dataset.add_argument("--max-workers", type=int, default=10, help="Concurrent workers for NDJSON image downloads.")
    download_dataset.add_argument(
        "--multiprocessing",
        action="store_true",
        help="Use worker processes for Ultralytics NDJSON annotation parsing.",
    )

    upload_dataset = subparsers.add_parser("upload-dataset", help="Archive and upload a prepared dataset directory.")
    upload_dataset.add_argument("dataset_path", type=Path)
    upload_dataset.add_argument("--dataset-id", help="Existing Platform dataset ID to ingest into.")
    upload_dataset.add_argument("--name", help="Name to use when creating/selecting a dataset.")

    subparsers.add_parser("list-projects", help="List Platform projects.")
    list_models = subparsers.add_parser("list-models", help="List Platform models.")
    list_models.add_argument("--completed", action="store_true", help="Use the completed models endpoint.")

    download_model = subparsers.add_parser("download-model", help="Download signed model files.")
    download_model.add_argument("model_id")
    download_model.add_argument("--output-dir", type=Path, help="Destination directory.")
    download_model.add_argument("--project", default="platform", help="Directory segment under weights/ultralytics.")
    download_model.add_argument("--model", default=None, help="Directory segment under weights/ultralytics/<project>.")

    uri = subparsers.add_parser("uri-helper", help="Print Ultralytics Platform training URI guidance.")
    uri.add_argument("username")
    uri.add_argument("dataset_slug")
    uri.add_argument("--project", help="Ultralytics project name to sync metrics under.")
    uri.add_argument("--name", help="Ultralytics run name to sync final weights under.")
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client(args: argparse.Namespace) -> UltralyticsPlatformClient:
    return UltralyticsPlatformClient(load_api_key(), base_url=args.base_url)


def _row_name(item: dict[str, Any]) -> str:
    return str(item.get("name") or item.get("slug") or item.get("id") or item)


def _row_id(item: dict[str, Any]) -> str:
    return str(item.get("id") or item.get("dataset_id") or item.get("model_id") or "")


def _print_items(title: str, items: list[dict[str, Any]]) -> None:
    if not items:
        console.print(Panel(f"[yellow]No {title.lower()} returned by the API.[/yellow]", border_style="yellow"))
        return
    rows = [
        [
            _row_id(item),
            _row_name(item),
            str(item.get("version") or item.get("status") or item.get("created_at") or ""),
        ]
        for item in items
    ]
    render_table(title, ["ID", "Name", "Version / Status"], rows)


def _wizard_kwargs(step_index: int) -> dict[str, Any]:
    return {"wizard_steps": WIZARD_STEPS, "wizard_current_step": step_index}


def _preflight_api_key() -> bool:
    """Verify ULTRALYTICS_API_KEY is configured. Returns False if missing."""
    status = ultralytics_credential_status()
    if status.get("api_key"):
        return True
    console.print(Panel(
        "[bold red]ULTRALYTICS_API_KEY is not configured.[/bold red]\n\n"
        "Add it to your project's [cyan].env[/cyan] file or export it in your shell:\n\n"
        "  [yellow]export ULTRALYTICS_API_KEY=...[/yellow]\n\n"
        "You can generate or rotate keys at https://platform.ultralytics.com/.",
        border_style="red",
        title="[red]Missing Credential[/red]",
    ))
    input("\nPress Enter to return...")
    return False


def _safely_fetch(label: str, fn: Callable[[], list[dict[str, Any]]]) -> list[dict[str, Any]] | None:
    try:
        return fn()
    except UltralyticsPlatformError as exc:
        console.print(Panel(
            f"[bold red]Could not load {label}:[/bold red] {exc}",
            border_style="red",
        ))
        if exc.retry_after is not None:
            console.print(f"[yellow]Retry-After:[/yellow] {exc.retry_after:g}s")
        input("\nPress Enter to continue...")
        return None


def _select_remote_item(
    title: str,
    items: list[dict[str, Any]],
    step_index: int,
    *,
    describe: Callable[[dict[str, Any]], str] | None = None,
) -> dict[str, Any] | None:
    if not items:
        console.print(Panel(
            f"[yellow]No {title.lower()} are available on the Platform for this API key.[/yellow]",
            border_style="yellow",
        ))
        input("\nPress Enter to return...")
        return None
    labels: list[str] = []
    descriptions: dict[str, str] = {}
    by_label: dict[str, dict[str, Any]] = {}
    for item in items:
        identifier = _row_id(item) or "(unknown)"
        name = _row_name(item)
        label = f"{name}  [dim]({identifier})[/dim]"
        # Strip rich markup when registering as menu option (rich strips automatically)
        clean = f"{name}  ({identifier})"
        labels.append(clean)
        by_label[clean] = item
        descriptions[clean] = describe(item) if describe else (
            f"[bold cyan]{name}[/bold cyan]\n\n"
            f"ID: [yellow]{identifier}[/yellow]\n"
            f"Status: [yellow]{item.get('status', 'unknown')}[/yellow]\n"
            f"Version: [yellow]{item.get('version', '—')}[/yellow]"
        )
    selected = get_user_choice(
        labels,
        allow_back=True,
        title=f"Select {title} ({len(labels)} available)",
        text=f"Pick a {title.rstrip('s').lower()}:",
        descriptions=descriptions,
        breadcrumbs=["YOLOmatic", "Ultralytics Platform", title],
        tip="PgUp/PgDn page • Home/End jump • Q back",
        **_wizard_kwargs(step_index),
    )
    if selected in (NAV_BACK, "Back"):
        return None
    return by_label[selected]


def _select_local_prepared_dataset() -> Path | None:
    datasets = list_dataset_directories(include_size=False)
    if not datasets:
        console.print(Panel(
            "[bold yellow]No datasets found under ./datasets/.[/bold yellow]\n\n"
            "Prepare or convert a dataset first.",
            border_style="yellow",
        ))
        input("\nPress Enter to continue...")
        return None
    path_by_name: dict[str, Path] = {item["name"]: Path(item["path"]) for item in datasets}
    descriptions: dict[str, str] = {}
    for name, path in path_by_name.items():
        try:
            size = format_size(calculate_folder_size(path))
        except OSError:
            size = "unknown"
        has_yaml = (path / "data.yaml").exists() or (path / "dataset.yaml").exists()
        descriptions[name] = (
            f"[bold cyan]{name}[/bold cyan]\n\n"
            f"Size: [yellow]{size}[/yellow]  |  "
            f"data.yaml: [yellow]{'yes' if has_yaml else 'no'}[/yellow]\n\n"
            f"[dim]{path}[/dim]"
        )
    selected = get_user_choice(
        list(path_by_name),
        allow_back=True,
        title=f"Select Prepared Dataset ({len(path_by_name)} available)",
        text="Choose a local dataset to archive and upload:",
        descriptions=descriptions,
        breadcrumbs=["YOLOmatic", "Ultralytics Platform", "Upload"],
        tip="PgUp/PgDn page • Home/End jump • Q back",
        **_wizard_kwargs(1),
    )
    if selected in (NAV_BACK, "Back"):
        return None
    return path_by_name[selected]


# ---------------------------------------------------------------------------
# Action handlers (CLI + interactive)
# ---------------------------------------------------------------------------


def download_dataset(args: argparse.Namespace) -> None:
    settings = load_settings()
    download_dir = Path(settings.get("ultralytics", {}).get("default_dataset_download_dir", "datasets/ultralytics/downloads"))
    download_dir.mkdir(parents=True, exist_ok=True)
    ndjson_path = download_dir / f"{slugify(args.dataset_id)}.ndjson"
    downloaded = _client(args).download_dataset_export(args.dataset_id, ndjson_path, args.version)
    console.print(f"[green]Downloaded export:[/green] {downloaded.path} ({downloaded.bytes_written} bytes)")
    stats = prepare_dataset(
        PrepareDatasetConfig(
            source_path=downloaded.path,
            output_root=args.output_root,
            output_slug=args.slug or slugify(args.dataset_id),
            output_format=args.format,
            split_config=PrepareSplitConfig(args.train, args.val, args.test),
            split_strategy="smart_balanced" if args.smart_split else "class_balanced",
            seed=args.seed,
            max_workers=args.max_workers,
            use_multiprocessing=args.multiprocessing,
        )
    )
    console.print(f"[bold green]Prepared dataset:[/bold green] [cyan]{stats.output_path}[/cyan]")


def _make_archive(dataset_path: Path) -> Path:
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    temp_dir = Path(tempfile.mkdtemp(prefix="yolomatic-ultralytics-"))
    archive_base = temp_dir / slugify(dataset_path.name)
    archive_name = shutil.make_archive(str(archive_base), "zip", root_dir=dataset_path)
    return Path(archive_name)


def upload_dataset(args: argparse.Namespace) -> None:
    archive_path = _make_archive(args.dataset_path)
    try:
        client = _client(args)
        dataset_id = args.dataset_id
        if not dataset_id and args.name:
            created = client.create_dataset({"name": args.name})
            dataset_id = str(created.get("id") or created.get("dataset_id") or "")
        result = client.upload_archive(archive_path, dataset_id=dataset_id or None)
        console.print("[bold green]Upload complete.[/bold green]")
        console.print(json.dumps(result, indent=2))
    finally:
        shutil.rmtree(archive_path.parent, ignore_errors=True)


def download_model(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    if output_dir is None:
        model_segment = args.model or args.model_id
        output_dir = Path("weights") / "ultralytics" / slugify(args.project) / slugify(model_segment)
    downloads = _client(args).download_model_files(args.model_id, output_dir)
    if not downloads:
        console.print("[yellow]No downloadable model files were returned.[/yellow]")
        return
    for item in downloads:
        console.print(f"[green]Downloaded:[/green] {item.path} ({item.bytes_written} bytes)")


def uri_helper(args: argparse.Namespace) -> None:
    uri = f"ul://{args.username}/datasets/{args.dataset_slug}"
    rows = {
        "Dataset URI": uri,
        "YOLO data": f"data={uri}",
        "Project": args.project or "Set project=<platform project name> when training",
        "Run name": args.name or "Set name=<run name> when training",
        "Deferred": "Cloud training, deploy, export, and browser-session controls are not automated in this pass.",
    }
    render_summary_panel("Ultralytics Platform Training URI", rows)


# ---------------------------------------------------------------------------
# Interactive flows
# ---------------------------------------------------------------------------


def _flow_list_overview(args: argparse.Namespace) -> None:
    client = _client(args)
    projects = _safely_fetch("projects", client.list_projects)
    if projects is None:
        return
    models = _safely_fetch("models", client.list_models)
    if models is None:
        return
    _print_items("Projects", projects)
    _print_items("Models", models)
    input("\nPress Enter to continue...")


def _flow_download_dataset(args: argparse.Namespace) -> None:
    client = _client(args)
    datasets = _safely_fetch("datasets", client.list_datasets)
    if datasets is None:
        return
    chosen = _select_remote_item(
        "Datasets",
        datasets,
        step_index=1,
        describe=lambda item: (
            f"[bold cyan]{_row_name(item)}[/bold cyan]\n\n"
            f"ID: [yellow]{_row_id(item)}[/yellow]\n"
            f"Latest version: [yellow]{item.get('version') or item.get('latest_version') or '—'}[/yellow]\n"
            f"Format: [yellow]{item.get('format', 'NDJSON')}[/yellow]\n"
            f"Created: [yellow]{item.get('created_at', '—')}[/yellow]"
        ),
    )
    if chosen is None:
        return

    dataset_id = _row_id(chosen)
    raw_version = get_parameter_value_input(
        ParameterDefinition("version", "ultralytics", 0, "int", "Dataset version", "Use 0 for latest.", min_value=0),
        0,
    )
    if raw_version in (None, NAV_BACK):
        return

    raw_format = get_user_choice(
        ["YOLO Detection", "YOLO Segmentation", "COCO", "Back"],
        title="Output Format",
        text="Output format for the prepared local dataset:",
        breadcrumbs=["YOLOmatic", "Ultralytics Platform", "Download Dataset"],
        **_wizard_kwargs(2),
    )
    if raw_format in (NAV_BACK, "Back"):
        return

    use_smart = get_user_choice(
        ["Smart Balanced", "Class Balanced", "Back"],
        title="Split Strategy",
        text="Split strategy to apply after download:",
        descriptions={
            "Smart Balanced": "Balances class, object size, density, and background images.",
            "Class Balanced": "Default class-balanced assignment.",
        },
        breadcrumbs=["YOLOmatic", "Ultralytics Platform", "Download Dataset"],
        **_wizard_kwargs(2),
    )
    if use_smart in (NAV_BACK, "Back"):
        return

    raw_workers = get_parameter_value_input(
        ParameterDefinition(
            "max_workers",
            "ultralytics",
            10,
            "int",
            "Download workers",
            "Concurrent workers for signed image downloads.",
            min_value=1,
            max_value=64,
        ),
        10,
    )
    if raw_workers in (None, NAV_BACK):
        return

    raw_multiprocessing = get_parameter_value_input(
        ParameterDefinition(
            "multiprocessing",
            "ultralytics",
            True,
            "bool",
            "Multiprocessing parse",
            "Use worker processes for large segmentation annotation parsing.",
        ),
        True,
    )
    if raw_multiprocessing in (None, NAV_BACK):
        return

    args.dataset_id = dataset_id
    args.version = int(raw_version) or None
    args.output_root = Path(load_settings().get("ultralytics", {}).get("default_output_root", "datasets"))
    args.slug = slugify(_row_name(chosen) or dataset_id)
    args.format = raw_format
    args.train, args.val, args.test = 0.70, 0.20, 0.10
    args.seed = 42
    args.smart_split = use_smart == "Smart Balanced"
    args.max_workers = int(raw_workers)
    args.multiprocessing = bool(raw_multiprocessing)

    render_summary_panel(
        "Download Plan",
        {
            "Dataset": f"{_row_name(chosen)} ({dataset_id})",
            "Version": "latest" if args.version is None else args.version,
            "Output Format": args.format,
            "Output Root": args.output_root,
            "Strategy": "smart_balanced" if args.smart_split else "class_balanced",
            "Workers": args.max_workers,
            "Multiprocessing": "enabled" if args.multiprocessing else "disabled",
        },
    )
    confirm = get_user_choice(
        ["Start Download", "Back"],
        title="Confirm",
        text="Download and prepare locally?",
        breadcrumbs=["YOLOmatic", "Ultralytics Platform", "Download Dataset"],
        **_wizard_kwargs(3),
    )
    if confirm in (NAV_BACK, "Back"):
        return
    try:
        download_dataset(args)
    except UltralyticsPlatformError as exc:
        console.print(Panel(f"[bold red]{exc}[/bold red]", border_style="red"))
    input("\nPress Enter to continue...")


def _flow_upload_dataset(args: argparse.Namespace) -> None:
    dataset_path = _select_local_prepared_dataset()
    if dataset_path is None:
        return

    client = _client(args)
    remote_datasets = _safely_fetch("datasets", client.list_datasets)
    if remote_datasets is None:
        return

    target_options = ["Create new dataset"]
    target_descriptions: dict[str, str] = {
        "Create new dataset": f"Create a new Platform dataset named '{dataset_path.name}' and ingest into it.",
    }
    target_by_label: dict[str, dict[str, Any] | None] = {"Create new dataset": None}
    for item in remote_datasets:
        label = f"{_row_name(item)}  ({_row_id(item)})"
        target_options.append(label)
        target_by_label[label] = item
        target_descriptions[label] = (
            f"[bold cyan]{_row_name(item)}[/bold cyan]\n\n"
            f"ID: [yellow]{_row_id(item)}[/yellow]\n"
            "Upload will be ingested into this existing dataset."
        )
    target_options.append("Back")

    target_label = get_user_choice(
        target_options,
        title="Upload Destination",
        text="Choose whether to create a new Platform dataset or ingest into an existing one:",
        descriptions=target_descriptions,
        breadcrumbs=["YOLOmatic", "Ultralytics Platform", "Upload"],
        **_wizard_kwargs(2),
    )
    if target_label in (NAV_BACK, "Back"):
        return

    target = target_by_label[target_label]
    args.dataset_path = dataset_path
    args.dataset_id = _row_id(target) if target else None
    args.name = dataset_path.name if not args.dataset_id else None

    render_summary_panel(
        "Upload Plan",
        {
            "Source": format_path(str(dataset_path), max_chars=64),
            "Destination": args.dataset_id or f"new dataset '{args.name}'",
        },
    )
    confirm = get_user_choice(
        ["Start Upload", "Back"],
        title="Confirm",
        text="Archive and upload?",
        breadcrumbs=["YOLOmatic", "Ultralytics Platform", "Upload"],
        **_wizard_kwargs(3),
    )
    if confirm in (NAV_BACK, "Back"):
        return
    try:
        upload_dataset(args)
    except (UltralyticsPlatformError, FileNotFoundError) as exc:
        console.print(Panel(f"[bold red]{exc}[/bold red]", border_style="red"))
    input("\nPress Enter to continue...")


def _flow_download_model(args: argparse.Namespace) -> None:
    client = _client(args)
    completed_choice = get_user_choice(
        ["Completed Models", "All Models", "Back"],
        title="Source",
        text="Which models endpoint to browse?",
        breadcrumbs=["YOLOmatic", "Ultralytics Platform", "Download Model"],
        **_wizard_kwargs(1),
    )
    if completed_choice in (NAV_BACK, "Back"):
        return
    use_completed = completed_choice == "Completed Models"
    models = _safely_fetch("models", lambda: client.list_models(completed=use_completed))
    if models is None:
        return

    chosen = _select_remote_item(
        "Models",
        models,
        step_index=1,
        describe=lambda item: (
            f"[bold cyan]{_row_name(item)}[/bold cyan]\n\n"
            f"ID: [yellow]{_row_id(item)}[/yellow]\n"
            f"Status: [yellow]{item.get('status', 'unknown')}[/yellow]\n"
            f"Created: [yellow]{item.get('created_at', '—')}[/yellow]"
        ),
    )
    if chosen is None:
        return

    args.model_id = _row_id(chosen)
    args.output_dir = None
    args.project = "platform"
    args.model = slugify(_row_name(chosen) or args.model_id)

    render_summary_panel(
        "Download Plan",
        {
            "Model": f"{_row_name(chosen)} ({args.model_id})",
            "Destination": f"weights/ultralytics/{args.project}/{args.model}",
        },
    )
    confirm = get_user_choice(
        ["Start Download", "Back"],
        title="Confirm",
        text="Download weights?",
        breadcrumbs=["YOLOmatic", "Ultralytics Platform", "Download Model"],
        **_wizard_kwargs(3),
    )
    if confirm in (NAV_BACK, "Back"):
        return
    try:
        download_model(args)
    except UltralyticsPlatformError as exc:
        console.print(Panel(f"[bold red]{exc}[/bold red]", border_style="red"))
    input("\nPress Enter to continue...")


def _flow_uri_helper(args: argparse.Namespace) -> None:
    username = get_parameter_value_input(
        ParameterDefinition("username", "ultralytics", "", "str", "Username", "Platform username or org slug."),
        "",
    )
    if username in (None, NAV_BACK) or not str(username).strip():
        return
    dataset_slug = get_parameter_value_input(
        ParameterDefinition("dataset_slug", "ultralytics", "", "str", "Dataset slug", "Platform dataset slug."),
        "",
    )
    if dataset_slug in (None, NAV_BACK) or not str(dataset_slug).strip():
        return
    args.username = str(username).strip()
    args.dataset_slug = str(dataset_slug).strip()
    args.project = None
    args.name = None
    uri_helper(args)
    input("\nPress Enter to continue...")


def interactive_main(args: argparse.Namespace) -> None:
    last_choice: str | None = None
    while True:
        clear_screen()
        print_stylized_header("Ultralytics Platform")
        api_key_ok = ultralytics_credential_status().get("api_key", False)
        status_fields = {
            "API key": "configured" if api_key_ok else "missing (set ULTRALYTICS_API_KEY)",
            "Base URL": args.base_url,
        }
        choice = get_user_choice(
            [
                "List Projects / Models",
                "Download Dataset Version",
                "Upload Prepared Dataset",
                "Download Model Weights",
                "Platform Training URI Helper",
                "Back",
            ],
            title="Ultralytics Platform",
            text="Choose a Platform workflow:",
            descriptions={
                "List Projects / Models": "Shows projects, active models, and completed models available to the API key.",
                "Download Dataset Version": "Browse Platform datasets, download a signed NDJSON export, then prepare it locally.",
                "Upload Prepared Dataset": "Archive a local prepared dataset and ingest it into a new or existing Platform dataset.",
                "Download Model Weights": "Browse Platform models and download signed files into weights/ultralytics/.",
                "Platform Training URI Helper": "Builds ul://username/datasets/slug guidance for local Ultralytics training.",
            },
            breadcrumbs=["YOLOmatic", "Ultralytics Platform"],
            initial_selection=last_choice,
            status_fields=status_fields,
            tip="Set ULTRALYTICS_API_KEY in .env before any remote action.",
            **_wizard_kwargs(0),
        )
        last_choice = choice
        if choice in (NAV_BACK, "Back"):
            return
        if choice == "Platform Training URI Helper":
            _flow_uri_helper(args)
            continue
        if not _preflight_api_key():
            continue
        try:
            if choice == "List Projects / Models":
                _flow_list_overview(args)
            elif choice == "Download Dataset Version":
                _flow_download_dataset(args)
            elif choice == "Upload Prepared Dataset":
                _flow_upload_dataset(args)
            elif choice == "Download Model Weights":
                _flow_download_model(args)
        except UltralyticsPlatformError as exc:
            console.print(Panel(f"[bold red]{exc}[/bold red]", border_style="red"))
            if exc.retry_after is not None:
                console.print(f"[yellow]Retry-After:[/yellow] {exc.retry_after:g}s")
            input("\nPress Enter to continue...")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command is None:
            interactive_main(args)
        elif args.command == "list-datasets":
            _print_items("Ultralytics Datasets", _client(args).list_datasets())
        elif args.command == "download-dataset":
            download_dataset(args)
        elif args.command == "upload-dataset":
            upload_dataset(args)
        elif args.command == "list-projects":
            _print_items("Ultralytics Projects", _client(args).list_projects())
        elif args.command == "list-models":
            _print_items("Ultralytics Models", _client(args).list_models(completed=args.completed))
        elif args.command == "download-model":
            download_model(args)
        elif args.command == "uri-helper":
            uri_helper(args)
    except UltralyticsPlatformError as error:
        console.print(Panel(f"[bold red]{error}[/bold red]", border_style="red"))
        if error.retry_after is not None:
            console.print(f"[yellow]Retry-After:[/yellow] {error.retry_after:g}s")


if __name__ == "__main__":
    main()
