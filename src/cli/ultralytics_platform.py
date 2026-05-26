from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from rich.panel import Panel

from src.config.settings import load_settings
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
    get_parameter_value_input,
    get_user_choice,
    print_stylized_header,
    render_summary_panel,
    render_table,
)


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


def _client(args: argparse.Namespace) -> UltralyticsPlatformClient:
    return UltralyticsPlatformClient(load_api_key(), base_url=args.base_url)


def _row_name(item: dict[str, Any]) -> str:
    return str(item.get("name") or item.get("slug") or item.get("id") or item)


def _print_items(title: str, items: list[dict[str, Any]]) -> None:
    rows = []
    for item in items:
        rows.append(
            [
                str(item.get("id") or item.get("dataset_id") or item.get("model_id") or ""),
                _row_name(item),
                str(item.get("version") or item.get("status") or item.get("created_at") or ""),
            ]
        )
    render_table(title, ["ID", "Name", "Version / Status"], rows)


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


def interactive_main(args: argparse.Namespace) -> None:
    while True:
        clear_screen()
        print_stylized_header("Ultralytics Platform")
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
                "Download Dataset Version": "Downloads a signed NDJSON dataset export, then runs Prepare / Split Dataset.",
                "Upload Prepared Dataset": "Archives a local prepared dataset and uploads it through signed Platform upload URLs.",
                "Download Model Weights": "Downloads signed files from a Platform model into weights/ultralytics/.",
                "Platform Training URI Helper": "Builds ul://username/datasets/slug guidance for local Ultralytics training.",
            },
            breadcrumbs=["YOLOmatic", "Ultralytics Platform"],
        )
        if choice in (NAV_BACK, "Back"):
            return
        if choice == "List Projects / Models":
            client = _client(args)
            _print_items("Projects", client.list_projects())
            _print_items("Models", client.list_models())
            input("\nPress Enter to continue...")
        elif choice == "Download Dataset Version":
            dataset_id = get_parameter_value_input(ParameterDefinition("dataset_id", "ultralytics", "", "str", "Dataset ID", "Platform dataset ID."), "")
            if dataset_id in (None, NAV_BACK):
                continue
            version = get_parameter_value_input(ParameterDefinition("version", "ultralytics", 0, "int", "Version", "Use 0 for latest.", min_value=0), 0)
            if version in (None, NAV_BACK):
                continue
            args.dataset_id = str(dataset_id)
            args.version = int(version) or None
            args.output_root = Path("datasets")
            args.slug = slugify(str(dataset_id))
            args.format = "YOLO Detection"
            args.train, args.val, args.test = 0.70, 0.20, 0.10
            args.seed = 42
            args.smart_split = True
            download_dataset(args)
            input("\nPress Enter to continue...")
        elif choice == "Upload Prepared Dataset":
            raw = get_parameter_value_input(ParameterDefinition("dataset_path", "ultralytics", "datasets", "str", "Dataset path", "Prepared dataset directory to archive."), "datasets")
            if raw in (None, NAV_BACK):
                continue
            args.dataset_path = Path(str(raw)).expanduser()
            args.dataset_id = None
            args.name = args.dataset_path.name
            upload_dataset(args)
            input("\nPress Enter to continue...")
        elif choice == "Download Model Weights":
            model_id = get_parameter_value_input(ParameterDefinition("model_id", "ultralytics", "", "str", "Model ID", "Platform model ID."), "")
            if model_id in (None, NAV_BACK):
                continue
            args.model_id = str(model_id)
            args.output_dir = None
            args.project = "platform"
            args.model = str(model_id)
            download_model(args)
            input("\nPress Enter to continue...")
        elif choice == "Platform Training URI Helper":
            username = get_parameter_value_input(ParameterDefinition("username", "ultralytics", "", "str", "Username", "Platform username or org slug."), "")
            dataset_slug = get_parameter_value_input(ParameterDefinition("dataset_slug", "ultralytics", "", "str", "Dataset slug", "Platform dataset slug."), "")
            if username in (None, NAV_BACK) or dataset_slug in (None, NAV_BACK):
                continue
            args.username = str(username)
            args.dataset_slug = str(dataset_slug)
            args.project = None
            args.name = None
            uri_helper(args)
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

