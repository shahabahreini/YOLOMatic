from __future__ import annotations

import base64
import copy
import getpass
import hashlib
import os
import platform
import uuid
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
    "ultralytics": {
        "default_dataset_download_dir": "datasets/ultralytics/downloads",
        "default_model_download_dir": "weights/ultralytics",
        "default_output_root": "datasets",
    },
    "narratives": {
        "mode": "guided",
        "show_setup_guidance": True,
        "show_success_panels": True,
        "show_skip_reasons": True,
    },
    "ai": {
        "provider": "Gemini",
        "gemini_api_key": "",
        "openai_api_key": "",
        "selected_model": "gemini-2.5-flash",
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


_TRUE = {"true", "1", "yes", "y", "on"}
_FALSE = {"false", "0", "no", "n", "off", ""}


def _coerce_like(default: Any, value: Any) -> Any:
    """Coerce ``value`` to the type of ``default``; return ``default`` if impossible.

    YAML hand-edits like ``enabled: "no"`` used to be truthy strings.
    """
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        return default
    if isinstance(default, int):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    if isinstance(default, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    if isinstance(default, str):
        return "" if value is None else str(value)
    return value


def validate_settings(settings: dict[str, Any]) -> dict[str, Any]:
    result = deep_merge(DEFAULT_SETTINGS, settings if isinstance(settings, dict) else {})
    for section, defaults in DEFAULT_SETTINGS.items():
        if not isinstance(result.get(section), dict):
            result[section] = copy.deepcopy(defaults)
            continue
        for key, default in defaults.items():
            result[section][key] = _coerce_like(default, result[section].get(key, default))
    mode = str(result["narratives"].get("mode", "guided")).lower()
    if mode not in _VALID_NARRATIVE_MODES:
        mode = "guided"
    result["narratives"]["mode"] = mode
    return result


_ENC_PREFIX = "yoloenc:"


def _get_machine_secret() -> str:
    """Derive a key unique to this machine and user."""
    try:
        mac = str(uuid.getnode())
    except Exception:
        mac = "unknown-mac"
    try:
        node = platform.node()
    except Exception:
        node = "unknown-node"
    try:
        user = getpass.getuser()
    except Exception:
        user = "unknown-user"

    mix = f"yolomatic-key-salt-{mac}-{node}-{user}"
    return hashlib.sha256(mix.encode("utf-8")).hexdigest()


def _encrypt_value(plain_text: str, secret_key: str) -> str:
    """Encrypt a string using a machine-specific secret key."""
    if not plain_text:
        return ""
    if plain_text.startswith(_ENC_PREFIX):
        return plain_text  # Already encrypted
    iv = os.urandom(16)
    key_bytes = secret_key.encode("utf-8")
    data_bytes = plain_text.encode("utf-8")
    out = bytearray()
    i = 0
    while len(out) < len(data_bytes):
        keystream = hashlib.sha256(key_bytes + iv + str(i).encode("utf-8")).digest()
        chunk_len = min(len(keystream), len(data_bytes) - len(out))
        for j in range(chunk_len):
            out.append(data_bytes[len(out)] ^ keystream[j])
        i += 1
    combined = iv + out
    return _ENC_PREFIX + base64.b64encode(combined).decode("utf-8")


def _decrypt_value(cipher_text: str, secret_key: str) -> str:
    """Decrypt a string using a machine-specific secret key."""
    if not cipher_text:
        return ""
    if not cipher_text.startswith(_ENC_PREFIX):
        return cipher_text  # Legacy plain text key
    try:
        encoded = cipher_text[len(_ENC_PREFIX) :]
        combined = base64.b64decode(encoded.encode("utf-8"))
        if len(combined) < 16:
            return ""
        iv = combined[:16]
        encrypted_bytes = combined[16:]
        key_bytes = secret_key.encode("utf-8")
        out = bytearray()
        i = 0
        while len(out) < len(encrypted_bytes):
            keystream = hashlib.sha256(key_bytes + iv + str(i).encode("utf-8")).digest()
            chunk_len = min(len(keystream), len(encrypted_bytes) - len(out))
            for j in range(chunk_len):
                out.append(encrypted_bytes[len(out)] ^ keystream[j])
            i += 1
        return out.decode("utf-8")
    except Exception:  # noqa: BLE001
        # Fall back to empty string or the raw value if decryption fails
        return ""


_SECRET_FIELDS = ("gemini_api_key", "openai_api_key")

# Stored values that could not be read back (e.g. an obfuscated key after the
# machine's MAC address or hostname changed). They are kept verbatim on save
# instead of being overwritten with an empty string.
_UNREADABLE_SECRETS: dict[tuple[str, str], str] = {}


def _legacy_decrypt(stored: str) -> str | None:
    if stored.startswith(_ENC_PREFIX):
        plain = _decrypt_value(stored, _get_machine_secret())
        return plain or None
    return stored


def _read_secret(path_key: str, name: str, stored: str) -> str:
    from yolomatic.config.secrets import load_secret

    value = load_secret(stored, fallback=_legacy_decrypt)
    if value is None:
        _UNREADABLE_SECRETS[(path_key, name)] = stored
        return ""
    _UNREADABLE_SECRETS.pop((path_key, name), None)
    return value


def _write_secret(path_key: str, name: str, value: str) -> str:
    from yolomatic.config.secrets import store_secret

    if not value and (path_key, name) in _UNREADABLE_SECRETS:
        return _UNREADABLE_SECRETS[(path_key, name)]
    return store_secret(name, value, fallback=lambda plain: _encrypt_value(plain, _get_machine_secret()))


_SETTINGS_CACHE: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}


def clear_settings_cache() -> None:
    """Clear the in-memory settings cache."""
    _SETTINGS_CACHE.clear()


def load_settings(path: Path | str = SETTINGS_PATH) -> dict[str, Any]:
    settings_path = Path(path)
    if not settings_path.exists():
        return copy.deepcopy(DEFAULT_SETTINGS)

    secret_key = _get_machine_secret()
    try:
        mtime_ns = settings_path.stat().st_mtime_ns
    except OSError:
        mtime_ns = -1

    resolved_path = str(settings_path.resolve()) if mtime_ns != -1 else str(settings_path)
    cache_key = (resolved_path, secret_key)
    if mtime_ns != -1 and cache_key in _SETTINGS_CACHE:
        cached_mtime, cached_data = _SETTINGS_CACHE[cache_key]
        if cached_mtime == mtime_ns:
            return copy.deepcopy(cached_data)

    with settings_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        data = {}
    
    # Resolve stored secrets (OS keyring reference, legacy obfuscation, or plain).
    if "ai" in data and isinstance(data["ai"], dict):
        ai_cfg = data["ai"]
        for name in _SECRET_FIELDS:
            if isinstance(ai_cfg.get(name), str):
                ai_cfg[name] = _read_secret(resolved_path, name, ai_cfg[name])

    validated = validate_settings(data)
    if mtime_ns != -1:
        _SETTINGS_CACHE[cache_key] = (mtime_ns, copy.deepcopy(validated))
    return validated


def save_settings(settings: dict[str, Any], path: Path | str = SETTINGS_PATH) -> None:
    settings_path = Path(path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    validated = validate_settings(settings)
    
    # Secrets go to the OS keyring (the YAML keeps a reference) or, without a
    # keyring backend, the legacy machine-bound obfuscation.
    to_save = copy.deepcopy(validated)
    secret_key = _get_machine_secret()
    path_key = str(settings_path.resolve())
    ai_cfg = to_save.get("ai")
    if isinstance(ai_cfg, dict):
        for name in _SECRET_FIELDS:
            if isinstance(ai_cfg.get(name), str):
                ai_cfg[name] = _write_secret(path_key, name, ai_cfg[name])

    # Atomic: a crash mid-write used to leave a truncated, unreadable settings file.
    from yolomatic.utils.fs import atomic_write_text

    atomic_write_text(settings_path, yaml.safe_dump(to_save, sort_keys=False), private=True)

    try:
        mtime_ns = settings_path.stat().st_mtime_ns
        cache_key = (str(settings_path.resolve()), secret_key)
        _SETTINGS_CACHE[cache_key] = (mtime_ns, copy.deepcopy(validated))
    except OSError:
        pass


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


def ultralytics_credential_status() -> dict[str, bool]:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(".env"))
    except ImportError:
        pass
    return {
        "api_key": bool(os.getenv("ULTRALYTICS_API_KEY", "").strip()),
    }

