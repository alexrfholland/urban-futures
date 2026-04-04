"""Sync the active Blender input bundle into the local temp root."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

try:
    from .bV2_paths import BLENDER_REPO_ROOT, TEMP_LOCAL_ROOT
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bV2_paths import BLENDER_REPO_ROOT, TEMP_LOCAL_ROOT  # type: ignore


SOURCE_INFO_FILENAME = "_bV2_source.txt"
REQUIRED_CHILDREN = ("feature-locations", "vtks", "bioenvelopes")


def write_source_info() -> Path:
    info_path = TEMP_LOCAL_ROOT / SOURCE_INFO_FILENAME
    info_path.write_text(
        "\n".join(
            (
                f"temp_local_root={TEMP_LOCAL_ROOT}",
                f"remote_repo_root={BLENDER_REPO_ROOT}",
                f"synced_at={time.strftime('%Y-%m-%d %H:%M:%S')}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return info_path


def sync_inputs_from_remote() -> Path:
    if not BLENDER_REPO_ROOT.exists():
        raise FileNotFoundError(f"Remote repo root does not exist: {BLENDER_REPO_ROOT}")

    TEMP_LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    for child_name in REQUIRED_CHILDREN:
        remote_child = BLENDER_REPO_ROOT / child_name
        if not remote_child.exists():
            raise FileNotFoundError(f"Remote bundle is missing required folder: {remote_child}")
        local_child = TEMP_LOCAL_ROOT / child_name
        if local_child.exists():
            shutil.rmtree(local_child)
        shutil.copytree(remote_child, local_child)

    write_source_info()
    return TEMP_LOCAL_ROOT


def main() -> None:
    synced_root = sync_inputs_from_remote()
    print(f"Synced Blender inputs to local temp root: {synced_root}")


if __name__ == "__main__":
    main()
