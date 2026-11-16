"""Test-suite defaults: never touch the developer's real OS credential store."""
import os

os.environ.setdefault("YOLOMATIC_KEYRING", "0")
