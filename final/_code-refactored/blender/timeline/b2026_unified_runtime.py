from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import os
import runpy
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def get_build_mode(default: str = "timeline") -> str:
    return os.environ.get("B2026_TIMELINE_BUILD_MODE", default).strip().lower()


@contextmanager
def temporary_env(env_updates: dict[str, str] | None = None):
    previous: dict[str, str | None] = {}
    if env_updates:
        for key, value in env_updates.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = value
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def run_local_script(filename: str, *, env_updates: dict[str, str] | None = None) -> None:
    script_path = SCRIPT_DIR / filename
    if not script_path.exists():
        raise FileNotFoundError(f"Local script not found: {script_path}")
    with temporary_env(env_updates):
        runpy.run_path(str(script_path), run_name="__main__")
