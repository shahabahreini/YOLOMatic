from __future__ import annotations

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
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
    path_str = str(value).replace("\\", "/").lower()
    if "-seg" in name or "/segment/" in path_str:
        return "segmentation"
    if "-pose" in name or "/pose/" in path_str:
        return "pose"
    if "-sem" in name or "/semantic/" in path_str or "/semseg/" in path_str:
        return "semantic"
    if any(tag in name for tag in ("-cls", "-obb")):
        return "unsupported"
    return "detection"


def is_rfdetr_source(value: str | Path) -> bool:
    text = str(value).replace("\\", "/").lower()
    return Path(value).suffix.lower() == ".pth" or "rf-detr" in text or "rfdetr" in text


def is_detectron2_source(value: str | Path) -> bool:
    text = str(value).replace("\\", "/").lower()
    return "detectron2" in text or "faster_rcnn" in text or "mask_rcnn" in text or "retinanet" in text


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
                    else "segmentation"
                    if is_detectron2_source(weight_path) and ("seg" in str(weight_path).lower() or "mask_rcnn" in str(weight_path).lower())
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
    try:
        from src.datasets.core import summarize_dataset
        summary = summarize_dataset(folder_path)
        return summary.total_size_bytes
    except Exception:
        total = 0
        try:
            if hasattr(os, "fwalk"):
                for _dirpath, _dirnames, filenames, dirfd in os.fwalk(str(folder_path)):
                    for fname in filenames:
                        try:
                            total += os.stat(fname, dir_fd=dirfd).st_size
                        except OSError:
                            pass
            else:
                for _dirpath, _dirnames, filenames in os.walk(str(folder_path)):
                    for fname in filenames:
                        try:
                            total += (Path(_dirpath) / fname).stat().st_size
                        except OSError:
                            pass
        except OSError:
            pass
        return total


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

    folders = sorted(
        folder for folder in root.iterdir()
        if folder.is_dir() and folder.name != ".yolomatic_cache"
    )

    def _process(folder: Path) -> dict[str, Any]:
        entry: dict[str, Any] = {"name": folder.name, "path": folder.resolve()}
        if include_size:
            try:
                from src.datasets.core import summarize_dataset
                summary = summarize_dataset(folder)
                entry["summary"] = summary
                entry["size"] = format_size(summary.total_size_bytes)
            except Exception:
                entry["size"] = format_size(calculate_folder_size(folder))
        return entry

    with ThreadPoolExecutor() as executor:
        return list(executor.map(_process, folders))


def load_yaml_file(file_path: str | Path) -> dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in YAML file: {file_path}")
    return loaded


DATASET_SPLIT_KEYS = ("train", "val", "test")
DATASET_MASK_KEYS = ("train_masks", "val_masks", "test_masks")
RESOLVED_DATA_YAML_NAME = "data.resolved.yaml"


def resolve_dataset_root(meta: dict[str, Any], yaml_dir: str | Path) -> Path:
    """Return the directory a data.yaml's relative split values hang off.

    Ultralytics reads ``path:`` as the dataset root but resolves a *relative*
    one against the process CWD (``check_det_dataset`` only falls back to
    ``DATASETS_DIR`` when the path does not exist, and ``path: .`` always
    exists). That silently points a perfectly good dataset at whatever
    directory the run started in, so YOLOmatic anchors a relative root at the
    yaml's own directory instead -- the only interpretation that travels with
    the dataset.
    """
    yaml_directory = Path(yaml_dir).resolve()
    raw = meta.get("path")
    if raw is None or not str(raw).strip():
        return yaml_directory
    candidate = Path(str(raw).replace("\\", "/"))
    if candidate.is_absolute():
        return candidate.resolve()
    return (yaml_directory / candidate).resolve()


def _resolve_split_entry(entry: Any, root: Path, dataset_dir: Path) -> str | None:
    if not isinstance(entry, (str, Path)) or not str(entry).strip():
        return None
    normalized = str(entry).replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute():
        return str(path.resolve())

    literal = (root / path).resolve()
    if literal.exists():
        return str(literal)

    # Roboflow exports write "../train/images" but mean "./train/images".
    stripped = normalized
    while stripped.startswith("../"):
        stripped = stripped[3:]
    if stripped and stripped != normalized:
        rerooted = (dataset_dir / stripped).resolve()
        if rerooted.exists():
            return str(rerooted)

    return str(literal)


def resolve_split_paths(
    meta: dict[str, Any],
    yaml_dir: str | Path,
    dataset_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve every split (and mask) value in ``meta`` to an absolute path.

    Mirrors ``src.datasets.core._resolve_dataset_paths`` so the two resolvers
    cannot disagree. List values (multi-directory splits) are preserved as
    lists.
    """
    yaml_directory = Path(yaml_dir).resolve()
    root = resolve_dataset_root(meta, yaml_directory)
    base = Path(dataset_dir).resolve() if dataset_dir is not None else yaml_directory

    resolved: dict[str, Any] = {}
    for key in (*DATASET_SPLIT_KEYS, *DATASET_MASK_KEYS):
        raw = meta.get(key)
        if raw is None:
            continue
        if isinstance(raw, (list, tuple)):
            entries = [_resolve_split_entry(item, root, base) for item in raw]
            values = [item for item in entries if item is not None]
            if values:
                resolved[key] = values
            continue
        value = _resolve_split_entry(raw, root, base)
        if value is not None:
            resolved[key] = value
    return resolved


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

    # Normalise 'valid' → 'val' so callers always see 'val'
    if "val" not in dataset_config and "valid" in dataset_config:
        dataset_config["val"] = dataset_config.pop("valid")

    dataset_config.update(resolve_split_paths(dataset_config, yaml_directory, dataset_path))

    return dataset_config, str(data_yaml_path), str(dataset_path)


def write_resolved_data_yaml(dataset_path: str | Path, data_yaml_path: str | Path) -> str:
    """Write a CWD-independent copy of ``data_yaml_path`` and return its path.

    Ultralytics re-reads the yaml itself and applies its own root rules, so
    handing it the user's file re-introduces the ``path:``-relative bug that
    :func:`resolve_dataset_root` exists to avoid. Every path in the copy is
    absolute, which is unambiguous under any rule. The copy is regenerated per
    run and lives inside the dataset's cache directory, so the user's own
    data.yaml stays portable and untouched.

    Never raises: an unwritable dataset directory falls back to a temp file,
    and a failure there returns the original path.
    """
    dataset_dir = Path(dataset_path).resolve()
    source = Path(data_yaml_path).resolve()
    try:
        meta = load_yaml_file(source)
    except Exception:
        return str(source)

    if "val" not in meta and "valid" in meta:
        meta["val"] = meta.pop("valid")

    resolved = dict(meta)
    resolved.update(resolve_split_paths(meta, source.parent, dataset_dir))
    resolved["path"] = str(resolve_dataset_root(meta, source.parent))
    resolved.pop("yaml_file", None)

    payload = yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True)
    for destination in (
        dataset_dir / ".yolomatic_cache" / RESOLVED_DATA_YAML_NAME,
        Path(tempfile.gettempdir()) / f"yolomatic-{dataset_dir.name}-{RESOLVED_DATA_YAML_NAME}",
    ):
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(payload, encoding="utf-8")
            return str(destination)
        except OSError:
            continue
    return str(source)


def verify_dataset_directories(dataset_config: dict[str, Any]) -> list[str]:
    missing_dirs: list[str] = []
    checks = [("Training", dataset_config.get("train")), ("Validation", dataset_config.get("val")), ("Test", dataset_config.get("test"))]
    for dir_type, dir_path in checks:
        if dir_path and not Path(dir_path).exists():
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
