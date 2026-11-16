"""Storage for API keys kept in YOLOmatic settings.

Keys live in the operating system's credential store (macOS Keychain, Windows
Credential Locker, Secret Service on Linux) via ``keyring``; the settings YAML
only holds a ``keyring:<name>`` reference. When no credential store is
available (headless Linux, CI) the previous machine-bound obfuscation is used,
so existing setups keep working.

Set ``YOLOMATIC_KEYRING=0`` to never touch the OS credential store.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

SERVICE_NAME = "yolomatic"
KEYRING_PREFIX = "keyring:"


def keyring_enabled() -> bool:
    if os.environ.get("YOLOMATIC_KEYRING", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    try:
        import keyring
        from keyring.backends import fail

        return not isinstance(keyring.get_keyring(), fail.Keyring)
    except Exception:  # noqa: BLE001 - keyring missing or broken backend
        return False


def store_secret(name: str, value: str, *, fallback) -> str:
    """Persist ``value`` and return what should be written to the YAML file.

    ``fallback`` encodes the value when the credential store is unavailable.
    """
    if keyring_enabled():
        import keyring

        try:
            if value:
                keyring.set_password(SERVICE_NAME, name, value)
                return f"{KEYRING_PREFIX}{name}"
            try:
                keyring.delete_password(SERVICE_NAME, name)
            except Exception:  # noqa: BLE001 - nothing stored yet
                pass
            return ""
        except Exception as error:  # noqa: BLE001 - locked keychain, DBus error, ...
            logger.warning("Could not use the OS credential store for %s: %s", name, error)
    return fallback(value)


def load_secret(stored: str, *, fallback) -> str | None:
    """Return the plain secret for a stored YAML value, or ``None`` if unreadable."""
    if not stored:
        return ""
    if stored.startswith(KEYRING_PREFIX):
        if not keyring_enabled():
            return None
        import keyring

        try:
            return keyring.get_password(SERVICE_NAME, stored[len(KEYRING_PREFIX):])
        except Exception as error:  # noqa: BLE001
            logger.warning("Could not read %s from the OS credential store: %s", stored, error)
            return None
    return fallback(stored)
