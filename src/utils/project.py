from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import yaml


def project_root() -> Path:
    return Path.cwd()


def format_weight_label(project_root: Path, weight_path: Path) -> str:
    try:
        return str(weight_path.relative_to(project_root))
    except ValueError:
        return str(weight_path)


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


def find_run_directories(base_dir: str | Path) -> list[Path]:
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    event_dirs: list[Path] = []
    for run_file in base_path.rglob("args.yaml"):
        parent_dir = run_file.parent
        if parent_dir not in event_dirs:
            event_dirs.append(parent_dir)
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


def list_dataset_directories(datasets_root: str | Path = "datasets") -> list[dict[str, Any]]:
    root = Path(datasets_root)
    if not root.exists():
        return []

    datasets: list[dict[str, Any]] = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        datasets.append(
            {
                "name": folder.name,
                "path": folder.resolve(),
                "size": format_size(calculate_folder_size(folder)),
            }
        )
    return datasets


def load_yaml_file(file_path: str | Path) -> dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in YAML file: {file_path}")
    return loaded


def load_dataset_config(dataset_name: str, datasets_root: str | Path = "datasets") -> tuple[dict[str, Any], str, str]:
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
