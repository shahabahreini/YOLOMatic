from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import yaml
from rich.panel import Panel

from src.utils.cli import (
    clear_screen,
    console,
    get_user_choice,
    print_stylized_header,
    render_summary_panel,
    render_table,
)
from src.utils.ml_dependencies import MLDependencyError, import_torch
from src.utils.project import (
    find_available_weights,
    format_weight_label,
    project_root,
)


COMMON_MODEL_TYPES = [
    "yolov12",
    "yolov11",
    "yolov10",
    "yolov9",
    "yolov8",
    "yolo26n",
    "yolo26s",
    "yolo26m",
    "yolo26l",
    "yolo26x",
    "yolonas",
    "yolox",
]
MODEL_NAME_PATTERN = re.compile(r"^(?=.*[A-Za-z])[A-Za-z0-9-]+$")
EXCLUDED_WEIGHT_FILENAMES = {"state_dict.pt"}


@dataclass(frozen=True)
class UploadCandidate:
    weight_path: Path
    model_path: Path
    filename: str
    detected_model_type: str | None
    source_model_name: str | None
    task: str | None


@dataclass(frozen=True)
class EnvConfig:
    api_key: str
    workspace: str | None
    project_ids: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="yolomatic-upload")
    parser.add_argument(
        "--weight",
        help="Path to a trained weight file. If omitted, an interactive selector is shown.",
    )
    parser.add_argument(
        "--workspace",
        help="Roboflow workspace slug. If omitted, the uploader uses .env or prompts you.",
    )
    parser.add_argument(
        "--project-ids",
        help="Comma-separated Roboflow project IDs. If omitted, the uploader uses .env or prompts you.",
    )
    parser.add_argument(
        "--model-name",
        help="Versionless model name to register in Roboflow.",
    )
    parser.add_argument(
        "--model-type",
        help="Override the auto-detected Roboflow model type.",
    )
    return parser.parse_args()


def load_env_config(project_root: Path) -> EnvConfig:
    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise RuntimeError(
            "python-dotenv is not installed. Run your environment sync before using yolomatic-upload."
        ) from error

    load_dotenv(project_root / ".env")
    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is missing. Add it to your .env file before uploading."
        )

    workspace = os.getenv("ROBOFLOW_WORKSPACE", "").strip() or None
    raw_project_ids = os.getenv("ROBOFLOW_PROJECT_IDS", "")
    return EnvConfig(
        api_key=api_key,
        workspace=workspace,
        project_ids=parse_project_ids(raw_project_ids),
    )


def parse_project_ids(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def normalize_workspace_value(raw_value: str | None) -> str | None:
    if raw_value is None:
        return None
    stripped_value = raw_value.strip()
    if not stripped_value:
        return None
    if "/" not in stripped_value:
        return stripped_value

    parsed_value = urlparse(stripped_value)
    path_value = parsed_value.path if parsed_value.scheme else stripped_value
    workspace_slug = path_value.strip("/").split("/")[0]
    return workspace_slug or None


def read_args_metadata(weight_path: Path) -> dict[str, Any]:
    args_path = weight_path.parent.parent / "args.yaml"
    if not args_path.exists():
        return {}
    with open(args_path, "r") as file:
        loaded = yaml.safe_load(file) or {}
    if isinstance(loaded, dict):
        return loaded
    return {}


def infer_model_type_from_text(raw_text: str) -> str | None:
    normalized_text = raw_text.lower()

    yolo26_match = re.search(r"yolo26([nslmx])(?:-seg)?\b", normalized_text)
    if yolo26_match is not None:
        return f"yolo26{yolo26_match.group(1)}"

    if "yolo-nas" in normalized_text or "yolonas" in normalized_text:
        return "yolonas"
    if "yolo12" in normalized_text or "yolov12" in normalized_text:
        return "yolov12"
    if "yolo11" in normalized_text or "yolov11" in normalized_text:
        return "yolov11"
    if "yolo10" in normalized_text or "yolov10" in normalized_text:
        return "yolov10"
    if "yolo9" in normalized_text or "yolov9" in normalized_text:
        return "yolov9"
    if "yolo8" in normalized_text or "yolov8" in normalized_text:
        return "yolov8"
    if "yolox" in normalized_text:
        return "yolox"
    return None


def detect_model_type(weight_path: Path, metadata: dict[str, Any]) -> str | None:
    identifiers = [
        str(metadata.get("model", "")),
        str(metadata.get("task", "")),
        weight_path.name,
        weight_path.parent.name,
        weight_path.parent.parent.name,
    ]
    return infer_model_type_from_text(" ".join(identifiers))


def build_candidate(weight_path: Path) -> UploadCandidate:
    metadata = read_args_metadata(weight_path)
    return UploadCandidate(
        weight_path=weight_path,
        model_path=weight_path.parent,
        filename=weight_path.name,
        detected_model_type=detect_model_type(weight_path, metadata),
        source_model_name=(
            str(metadata["model"]).strip()
            if metadata.get("model") is not None
            else None
        ),
        task=(
            str(metadata["task"]).strip() if metadata.get("task") is not None else None
        ),
    )


def is_uploadable_weight(weight_path: Path) -> bool:
    if weight_path.name in EXCLUDED_WEIGHT_FILENAMES:
        return False
    if weight_path.name.startswith("roboflow_"):
        return False
    return True


def format_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def render_candidate_table(
    project_root: Path, candidates: Sequence[UploadCandidate]
) -> None:
    rows = [
        [
            str(i),
            format_weight_label(project_root, c.weight_path),
            c.detected_model_type or "Prompt",
            c.task or "unknown",
            format_timestamp(c.weight_path.stat().st_mtime),
        ]
        for i, c in enumerate(candidates, 1)
    ]
    render_table("Available Trained Models", ["#", "Weight", "Model Type", "Task", "Modified"], rows)


def resolve_candidate(
    project_root: Path,
    requested_weight: str | None,
    candidates: Sequence[UploadCandidate],
) -> UploadCandidate:
    if requested_weight is None:
        options = [
            format_weight_label(project_root, item.weight_path) for item in candidates
        ]
        selected = get_user_choice(
            options,
            title="Select Trained Model",
            text="Use ↑↓ keys to navigate, Enter to select:",
        )
        return candidates[options.index(selected)]

    requested_path = Path(requested_weight).expanduser()
    if not requested_path.is_absolute():
        requested_path = project_root / requested_path
    resolved_requested_path = requested_path.resolve()
    for candidate in candidates:
        if candidate.weight_path.resolve() == resolved_requested_path:
            return candidate
    raise FileNotFoundError(
        f"Weight file not found in discovered runs: {resolved_requested_path}"
    )


def prompt_text(prompt: str, default: str | None = None) -> str:
    while True:
        suffix = f" [{default}]" if default else ""
        raw_value = input(f"{prompt}{suffix}: ").strip()
        value = raw_value or (default or "")
        if value:
            return value
        console.print("[bold red]A value is required.[/bold red]")


def prompt_project_ids(defaults: Sequence[str]) -> list[str]:
    default_text = ",".join(defaults) if defaults else None
    while True:
        raw_value = prompt_text(
            "Enter Roboflow project IDs (comma-separated)", default_text
        )
        project_ids = parse_project_ids(raw_value)
        if project_ids:
            return project_ids
        console.print("[bold red]At least one project ID is required.[/bold red]")


def prompt_model_type(detected_model_type: str | None) -> str:
    if detected_model_type:
        return detected_model_type

    selection = get_user_choice(
        COMMON_MODEL_TYPES + ["Enter manually", "Cancel"],
        title="Select Roboflow Model Type",
        text="Auto-detection was inconclusive. Choose a model type or enter one manually:",
        descriptions={
            "yolov12": "Deploy a YOLOv12 model.",
            "yolov11": "Deploy a YOLOv11 model.",
            "yolov10": "Deploy a YOLOv10 model.",
            "yolov9": "Deploy a YOLOv9 model.",
            "yolov8": "Deploy a YOLOv8 model.",
            "yolonas": "Deploy a YOLO-NAS model.",
            "yolox": "Deploy a YOLOX model.",
            "Enter manually": "Type the Roboflow model type slug manually.",
            "Cancel": "Abort the upload process.",
        },
    )
    if selection == "Cancel":
        raise SystemExit(0)
    if selection == "Enter manually":
        model_type = prompt_text("Enter Roboflow model type")
    else:
        model_type = selection

    if model_type.lower() == "yolo26":
        refined_selection = get_user_choice(
            ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x", "Cancel"],
            title="Select YOLO26 Variant",
            text="Roboflow requires a YOLO26 size variant. Choose the trained model family:",
            descriptions={
                "yolo26n": "Nano variant (fastest, least accurate).",
                "yolo26s": "Small variant.",
                "yolo26m": "Medium variant.",
                "yolo26l": "Large variant.",
                "yolo26x": "Extra-large variant (slowest, most accurate).",
                "Cancel": "Abort the upload process.",
            },
        )
        if refined_selection == "Cancel":
            raise SystemExit(0)
        return refined_selection

    normalized_model_type = infer_model_type_from_text(model_type) or model_type.lower()
    if normalized_model_type == "yolo26":
        raise RuntimeError(
            "Roboflow requires a YOLO26 size variant such as yolo26n, yolo26s, yolo26m, yolo26l, or yolo26x."
        )
    return normalized_model_type


def suggest_model_name(weight_path: Path) -> str:
    preferred_parts: list[str] = []
    if weight_path.parent.name == "weights":
        preferred_parts.append(weight_path.parent.parent.name)
    else:
        preferred_parts.append(weight_path.parent.name)
    preferred_parts.append(weight_path.stem)
    raw_name = "-".join(part for part in preferred_parts if part)
    sanitized = re.sub(r"[^A-Za-z0-9-]+", "-", raw_name).strip("-")
    if not re.search(r"[A-Za-z]", sanitized):
        sanitized = f"model-{sanitized}" if sanitized else "model-upload"
    return sanitized.lower()


def prompt_model_name(default: str) -> str:
    while True:
        model_name = prompt_text("Enter Roboflow model name", default)
        if MODEL_NAME_PATTERN.fullmatch(model_name):
            return model_name
        console.print(
            "[bold red]Model name must contain at least one letter and use only letters, numbers, and dashes.[/bold red]"
        )


def render_upload_summary(
    candidate: UploadCandidate,
    model_type: str,
    workspace: str,
    project_ids: Sequence[str],
    model_name: str,
) -> None:
    fields = {
        "Weight": candidate.weight_path,
        "Model Path": candidate.model_path,
        "Filename": candidate.filename,
        "Detected Source Model": candidate.source_model_name or "unknown",
        "Task": candidate.task or "unknown",
        "Roboflow Model Type": model_type,
        "Workspace": workspace,
        "Project IDs": ", ".join(project_ids),
        "Model Name": model_name,
    }
    render_summary_panel("Roboflow Upload Summary", fields)


def stage_upload_candidate(
    candidate: UploadCandidate, model_type: str
) -> UploadCandidate:
    normalized_model_type = model_type.lower()
    if not normalized_model_type.startswith(("yolov", "yolo26", "yolonas", "yolox")):
        return candidate

    try:
        torch = import_torch()
    except MLDependencyError:
        return candidate

    temporary_root = Path(tempfile.mkdtemp(prefix="yolomatic-roboflow-"))
    staged_model_path = temporary_root / candidate.model_path.name
    staged_model_path.mkdir(parents=True, exist_ok=True)
    staged_weight_path = staged_model_path / candidate.filename

    checkpoint = torch.load(
        candidate.weight_path, map_location="cpu", weights_only=False
    )
    if isinstance(checkpoint, dict):
        normalized_checkpoint = dict(checkpoint)
        checkpoint_model = normalized_checkpoint.get("model")
        checkpoint_ema = normalized_checkpoint.get("ema")
        if checkpoint_model is not None and checkpoint_ema is None:
            normalized_checkpoint["ema"] = checkpoint_model
        elif checkpoint_model is None and checkpoint_ema is not None:
            normalized_checkpoint["model"] = checkpoint_ema
        torch.save(normalized_checkpoint, staged_weight_path)
    else:
        shutil.copy2(candidate.weight_path, staged_weight_path)

    return UploadCandidate(
        weight_path=staged_weight_path,
        model_path=staged_model_path,
        filename=candidate.filename,
        detected_model_type=candidate.detected_model_type,
        source_model_name=candidate.source_model_name,
        task=candidate.task,
    )


def upload_model(
    api_key: str,
    candidate: UploadCandidate,
    workspace_name: str,
    project_ids: Sequence[str],
    model_type: str,
    model_name: str,
) -> Any:
    rf, workspace, _resolved_workspace_name = resolve_workspace(api_key, workspace_name)
    staged_candidate = stage_upload_candidate(candidate, model_type)
    deploy_kwargs: dict[str, Any] = {
        "model_type": model_type,
        "model_path": str(staged_candidate.model_path),
        "project_ids": list(project_ids),
        "model_name": model_name,
        "filename": staged_candidate.filename,
    }
    output_buffer = io.StringIO()
    with contextlib.redirect_stdout(output_buffer):
        deployment_response = workspace.deploy_model(**deploy_kwargs)

    captured_output = output_buffer.getvalue().strip()
    if (
        "❌ ERROR" in captured_output
        or "Failed to get model deployment URL" in captured_output
    ):
        raise RuntimeError(captured_output)
    if "An error occured when uploading the model" in captured_output:
        raise RuntimeError(captured_output)

    return deployment_response or captured_output or None


def create_roboflow_client(api_key: str) -> Any:
    try:
        from roboflow import Roboflow
    except ImportError as error:
        raise RuntimeError(
            "roboflow is not installed. Run your environment sync before using yolomatic-upload."
        ) from error

    return Roboflow(api_key=api_key)


def try_get_workspace(rf: Any, workspace_name: str) -> Any | None:
    try:
        return rf.workspace(workspace_name)
    except Exception:
        return None


def resolve_workspace(
    api_key: str, requested_workspace: str | None
) -> tuple[Any, Any, str]:
    rf = create_roboflow_client(api_key)
    default_workspace_name = normalize_workspace_value(
        getattr(rf, "current_workspace", None)
    )
    normalized_requested_workspace = normalize_workspace_value(requested_workspace)

    if normalized_requested_workspace is None:
        if default_workspace_name is None:
            raise RuntimeError(
                "Unable to determine a Roboflow workspace from your API key. Set ROBOFLOW_WORKSPACE in .env."
            )
        workspace = try_get_workspace(rf, default_workspace_name)
        if workspace is None:
            raise RuntimeError(
                f"Unable to load the default Roboflow workspace '{default_workspace_name}'."
            )
        return rf, workspace, default_workspace_name

    candidate_workspace_names = [normalized_requested_workspace]
    lowered_workspace_name = normalized_requested_workspace.lower()
    if lowered_workspace_name not in candidate_workspace_names:
        candidate_workspace_names.append(lowered_workspace_name)

    for candidate_workspace_name in candidate_workspace_names:
        workspace = try_get_workspace(rf, candidate_workspace_name)
        if workspace is not None:
            return rf, workspace, candidate_workspace_name

    if (
        default_workspace_name
        and default_workspace_name != normalized_requested_workspace
    ):
        use_default_workspace = get_user_choice(
            ["Use Default Workspace", "Enter Workspace Manually", "Cancel"],
            title="Workspace Not Found",
            text=(
                f"The configured workspace '{normalized_requested_workspace}' could not be loaded. "
                f"Use the API key's default workspace '{default_workspace_name}' instead?"
            ),
        )
        if use_default_workspace == "Use Default Workspace":
            workspace = try_get_workspace(rf, default_workspace_name)
            if workspace is None:
                raise RuntimeError(
                    f"The configured workspace '{normalized_requested_workspace}' is invalid, and the default workspace '{default_workspace_name}' could not be loaded either."
                )
            return rf, workspace, default_workspace_name
        if use_default_workspace == "Enter Workspace Manually":
            manual_workspace_name = normalize_workspace_value(
                prompt_text("Enter Roboflow workspace slug", default_workspace_name)
            )
            if manual_workspace_name is None:
                raise RuntimeError("A Roboflow workspace slug is required.")
            workspace = try_get_workspace(rf, manual_workspace_name)
            if workspace is None:
                raise RuntimeError(
                    f"Workspace '{manual_workspace_name}' could not be loaded. Use your workspace slug, not the project name."
                )
            return rf, workspace, manual_workspace_name
        raise SystemExit(0)

    raise RuntimeError(
        f"Workspace '{normalized_requested_workspace}' could not be loaded. Use your workspace slug, not the project name."
    )


def main() -> None:
    args = parse_args()
    root = project_root()
    print_stylized_header("Roboflow Model Upload")

    try:
        env_config = load_env_config(root)
        available_weights = find_available_weights(root)
        if not available_weights:
            console.print(
                Panel(
                    "[bold red]No .pt weights were found in the project root or runs directory.[/bold red]",
                    border_style="red",
                )
            )
            raise SystemExit(1)

        candidates = [
            build_candidate(weight_path)
            for weight_path in available_weights
            if is_uploadable_weight(weight_path)
        ]
        if not candidates:
            console.print(
                Panel(
                    "[bold red]No uploadable model checkpoints were found. Select a full training checkpoint such as best.pt or last.pt.[/bold red]",
                    border_style="red",
                )
            )
            raise SystemExit(1)
        render_candidate_table(root, candidates)
        selected_candidate = resolve_candidate(root, args.weight, candidates)

        workspace_input = args.workspace or prompt_text(
            "Enter Roboflow workspace slug",
            normalize_workspace_value(env_config.workspace),
        )
        _rf, _workspace, resolved_workspace_name = resolve_workspace(
            env_config.api_key,
            workspace_input,
        )
        project_ids = (
            parse_project_ids(args.project_ids)
            if args.project_ids
            else prompt_project_ids(env_config.project_ids)
        )
        model_type = args.model_type or prompt_model_type(
            selected_candidate.detected_model_type
        )
        model_name = args.model_name or prompt_model_name(
            suggest_model_name(selected_candidate.weight_path)
        )

        render_upload_summary(
            selected_candidate,
            model_type,
            resolved_workspace_name,
            project_ids,
            model_name,
        )
        confirmation = get_user_choice(
            ["Upload", "Cancel"],
            title="Confirm Upload",
            text="Review the summary above. Continue with the Roboflow upload?",
            descriptions={
                "Upload": "Proceed with uploading the model to Roboflow.",
                "Cancel": "Abort the upload and return to terminal.",
            },
        )
        if confirmation == "Cancel":
            console.print("[bold yellow]Upload cancelled.[/bold yellow]")
            raise SystemExit(0)

        with console.status("[bold]Uploading model to Roboflow...", spinner="dots"):
            response = upload_model(
                env_config.api_key,
                selected_candidate,
                resolved_workspace_name,
                project_ids,
                model_type,
                model_name,
            )

        console.print(
            Panel(
                f"[bold green]Upload completed successfully.[/bold green]\nModel: [bold]{model_name}[/bold]\nWorkspace: [bold]{resolved_workspace_name}[/bold]\nProjects: [bold]{', '.join(project_ids)}[/bold]"
                + (
                    f"\nResponse: [bold]{response}[/bold]"
                    if response is not None
                    else ""
                ),
                border_style="green",
            )
        )
    except FileNotFoundError as error:
        console.print(f"[bold red]Error: {error}[/bold red]")
        raise SystemExit(1) from error
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Upload cancelled.[/bold yellow]")
        raise SystemExit(0)
    except SystemExit:
        raise
    except Exception as error:
        console.print(f"[bold red]Upload failed: {error}[/bold red]")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
