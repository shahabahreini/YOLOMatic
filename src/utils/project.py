from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml


@dataclass(frozen=True)
class FineTuneCandidate:
    source: str
    display_name: str
    kind: str
    task: str
    weight_path: Path | None = None
    modified_time: float | None = None


def project_root() -> Path:
    current_path = Path.cwd().resolve()
    for candidate_path in [current_path, *current_path.parents]:
        if (candidate_path / "pyproject.toml").is_file():
            return candidate_path
    return current_path


def format_weight_label(project_root: Path, weight_path: Path) -> str:
    try:
        return str(weight_path.relative_to(project_root))
    except ValueError:
        return str(weight_path)


def is_sam_checkpoint(path: Path) -> bool:
    """Detect a SAM fine-tuned checkpoint directory produced by HuggingFace Trainer."""
    if not path.is_dir():
        return False
    path_lower = str(path).lower()
    return (path / "config.json").exists() and (
        "sam3" in path_lower or "sam 3" in path_lower
    )


def find_available_weights(project_root: Path) -> list[Path]:
    discovered: dict[Path, Path] = {}
    for weight_path in project_root.glob("*.pt"):
        discovered[weight_path.resolve()] = weight_path
    for weight_path in project_root.glob("*.pth"):
        discovered[weight_path.resolve()] = weight_path

    runs_dir = project_root / "runs"
    if runs_dir.exists():
        for weight_path in runs_dir.glob("**/weights/*.pt"):
            discovered[weight_path.resolve()] = weight_path
        for weight_path in runs_dir.glob("**/*.pth"):
            discovered[weight_path.resolve()] = weight_path

    # SAM fine-tuned checkpoints (HuggingFace Trainer output dirs)
    sam_runs_dir = project_root / "runs" / "sam3.1"
    if sam_runs_dir.exists():
        for ckpt_dir in sam_runs_dir.rglob("checkpoint-*"):
            if ckpt_dir.is_dir() and (ckpt_dir / "config.json").exists():
                discovered[ckpt_dir.resolve()] = ckpt_dir

    return sorted(
        discovered.values(),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def infer_ultralytics_task_from_name(value: str | Path) -> str:
    name = Path(value).name.lower()
    if "-seg" in name or "/segment/" in str(value).replace("\\", "/").lower():
        return "segmentation"
    if any(tag in name for tag in ("-cls", "-pose", "-obb")):
        return "unsupported"
    return "detection"


def is_rfdetr_source(value: str | Path) -> bool:
    text = str(value).replace("\\", "/").lower()
    return Path(value).suffix.lower() == ".pth" or "rf-detr" in text or "rfdetr" in text


def is_yolo_nas_source(value: str | Path) -> bool:
    return "nas" in Path(value).name.lower()


def find_finetune_candidates(project_root: Path) -> list[FineTuneCandidate]:
    candidates: list[FineTuneCandidate] = []
    for weight_path in find_available_weights(project_root):
        if is_yolo_nas_source(weight_path):
            continue
        candidates.append(
            FineTuneCandidate(
                source=format_weight_label(project_root, weight_path),
                display_name=format_weight_label(project_root, weight_path),
                kind="local",
                task=(
                    "segmentation"
                    if is_rfdetr_source(weight_path) and "seg" in str(weight_path).lower()
                    else infer_ultralytics_task_from_name(weight_path)
                ),
                weight_path=weight_path,
                modified_time=weight_path.stat().st_mtime,
            )
        )
    return candidates


def find_run_directories(base_dir: str | Path) -> list[Path]:
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    event_dirs: list[Path] = []
    for run_file in base_path.rglob("args.yaml"):
        parent_dir = run_file.parent
        if parent_dir not in event_dirs:
            event_dirs.append(parent_dir)

    for event_file in base_path.rglob("events.out.tfevents.*"):
        parent_dir = event_file.parent
        run_dir = parent_dir.parent if parent_dir.name == "tensorboard" else parent_dir
        if run_dir not in event_dirs:
            event_dirs.append(run_dir)
    return sorted(event_dirs)


def calculate_folder_size(folder_path: str | Path) -> int:
    total_size = 0
    for path in Path(folder_path).rglob("*"):
        if path.is_file():
            total_size += path.stat().st_size
    return total_size


def format_size(size_in_bytes: int | float) -> str:
    size = float(size_in_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def list_dataset_directories(
    datasets_root: str | Path = "datasets",
    *,
    include_size: bool = True,
) -> list[dict[str, Any]]:
    root = Path(datasets_root)
    if not root.exists():
        return []

    datasets: list[dict[str, Any]] = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        dataset = {
            "name": folder.name,
            "path": folder.resolve(),
        }
        if include_size:
            dataset["size"] = format_size(calculate_folder_size(folder))
        datasets.append(dataset)
    return datasets


def load_yaml_file(file_path: str | Path) -> dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in YAML file: {file_path}")
    return loaded


def load_dataset_config(dataset_name: str, datasets_root: str | Path = "datasets") -> tuple[dict[str, Any], str, str]:
    requested_path = Path(dataset_name)
    if requested_path.exists() or requested_path.is_absolute() or "/" in str(dataset_name):
        dataset_path = requested_path.resolve()
    else:
        dataset_path = (Path(datasets_root) / dataset_name).resolve()
    data_yaml_path = dataset_path / "data.yaml"
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found in {dataset_path}")

    dataset_config = load_yaml_file(data_yaml_path)
    yaml_directory = data_yaml_path.parent
    for key in ("train", "val", "test"):
        configured_path = str(dataset_config[key])
        normalized_configured_path = configured_path.replace("\\", "/")

        if Path(configured_path).is_absolute():
            resolved_path = Path(configured_path)
        elif normalized_configured_path.startswith("../"):
            resolved_path = dataset_path / normalized_configured_path[3:]
        else:
            resolved_path = yaml_directory / configured_path

        dataset_config[key] = str(resolved_path.resolve())

    return dataset_config, str(data_yaml_path), str(dataset_path)


def verify_dataset_directories(dataset_config: dict[str, Any]) -> list[str]:
    missing_dirs: list[str] = []
    for dir_type, dir_path in [
        ("Training", dataset_config["train"]),
        ("Validation", dataset_config["val"]),
        ("Test", dataset_config["test"]),
    ]:
        if not Path(dir_path).exists():
            missing_dirs.append(f"{dir_type} directory: {dir_path}")
    return missing_dirs


def resolve_config_path(config_path: str | None, config_folder: str | Path = "configs") -> str | None:
    config_dir = Path(config_folder)
    if config_path:
        candidate_paths = [Path(config_path), config_dir / config_path]
        for candidate_path in candidate_paths:
            if candidate_path.is_file():
                return str(candidate_path.resolve())

    if not config_dir.exists():
        missing_name = config_path or "<auto-select>"
        raise FileNotFoundError(
            f"Config file '{missing_name}' not found and config folder '{config_dir}' does not exist."
        )

    yaml_files = sorted(path.name for path in config_dir.iterdir() if path.suffix == ".yaml")
    if not yaml_files:
        missing_name = config_path or "<auto-select>"
        raise FileNotFoundError(
            f"Config file '{missing_name}' not found and no YAML files exist in '{config_dir}'."
        )

    if config_path:
        available_configs = ", ".join(yaml_files)
        raise FileNotFoundError(
            f"Config file '{config_path}' not found. Available configs: {available_configs}"
        )

    if len(yaml_files) == 1:
        return str((config_dir / yaml_files[0]).resolve())

    return None


def list_config_files(config_folder: str | Path = "configs") -> list[str]:
    config_dir = Path(config_folder)
    if not config_dir.exists():
        return []
    return sorted(path.name for path in config_dir.iterdir() if path.suffix == ".yaml")


def render_weight_rows(project_root: Path, available_weights: Sequence[Path]) -> list[list[str]]:
    return [
        [str(index), format_weight_label(project_root, weight_path)]
        for index, weight_path in enumerate(available_weights, 1)
    ]
