from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

SETTINGS_PATH = Path("configs") / "yolomatic_settings.yaml"

DEFAULT_SETTINGS: dict[str, Any] = {
    "clearml": {
        "enabled": True,
        "require_configured": False,
        "project_name_template": "{family} Training - {model}",
        "task_name_format": "%Y-%m-%d-%H-%M",
        "upload_final_model": True,
        "upload_artifacts": True,
        "log_hyperparameters": True,
        "log_dataset_summary": True,
    },
    "roboflow": {
        "upload_wizard_enabled": True,
        "auto_upload_after_training": False,
        "auto_upload_weight": "best.pt",
        "default_model_name_template": "{run_name}-best",
        "require_dataset_metadata": True,
        "rfdetr_project_version": 1,
    },
    "narratives": {
        "mode": "guided",
        "show_setup_guidance": True,
        "show_success_panels": True,
        "show_skip_reasons": True,
    },
}

_VALID_NARRATIVE_MODES = {"guided", "concise", "quiet"}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def validate_settings(settings: dict[str, Any]) -> dict[str, Any]:
    result = deep_merge(DEFAULT_SETTINGS, settings)
    mode = str(result["narratives"].get("mode", "guided")).lower()
    if mode not in _VALID_NARRATIVE_MODES:
        mode = "guided"
    result["narratives"]["mode"] = mode
    for section in ("clearml", "roboflow", "narratives"):
        if not isinstance(result.get(section), dict):
            result[section] = copy.deepcopy(DEFAULT_SETTINGS[section])
    return result


def load_settings(path: Path | str = SETTINGS_PATH) -> dict[str, Any]:
    settings_path = Path(path)
    if not settings_path.exists():
        return copy.deepcopy(DEFAULT_SETTINGS)
    with settings_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        data = {}
    return validate_settings(data)


def save_settings(settings: dict[str, Any], path: Path | str = SETTINGS_PATH) -> None:
    settings_path = Path(path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    validated = validate_settings(settings)
    with settings_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(validated, file, sort_keys=False)


def reset_settings(path: Path | str = SETTINGS_PATH) -> dict[str, Any]:
    settings = copy.deepcopy(DEFAULT_SETTINGS)
    save_settings(settings, path)
    return settings


def format_clearml_project_name(settings: dict[str, Any], family: str, model: str) -> str:
    template = settings.get("clearml", {}).get(
        "project_name_template",
        DEFAULT_SETTINGS["clearml"]["project_name_template"],
    )
    try:
        return str(template).format(family=family, model=model)
    except (KeyError, ValueError):
        return f"{family} Training - {model}"


def snapshot_clearml_settings(settings: dict[str, Any], family: str, model: str) -> dict[str, Any]:
    clearml = copy.deepcopy(settings["clearml"])
    clearml["project_name"] = format_clearml_project_name(settings, family, model)
    return clearml


def snapshot_roboflow_settings(settings: dict[str, Any]) -> dict[str, Any]:
    roboflow = settings["roboflow"]
    return {
        "upload": bool(roboflow.get("auto_upload_after_training", False)),
        "weight": roboflow.get("auto_upload_weight", "best.pt"),
        "model_name_template": roboflow.get("default_model_name_template", "{run_name}-best"),
        "require_dataset_metadata": bool(roboflow.get("require_dataset_metadata", True)),
        "rfdetr_project_version": roboflow.get("rfdetr_project_version", 1),
        "workspace": None,
        "project_id": None,
        "version": None,
    }


def roboflow_credential_status() -> dict[str, bool]:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(".env"))
    except ImportError:
        pass
    project_ids = [item.strip() for item in os.getenv("ROBOFLOW_PROJECT_IDS", "").split(",")]
    return {
        "api_key": bool(os.getenv("ROBOFLOW_API_KEY", "").strip()),
        "workspace": bool(os.getenv("ROBOFLOW_WORKSPACE", "").strip()),
        "project_ids": any(project_ids),
    }


def narrative_mode(settings: dict[str, Any]) -> str:
    return settings.get("narratives", {}).get("mode", "guided")
