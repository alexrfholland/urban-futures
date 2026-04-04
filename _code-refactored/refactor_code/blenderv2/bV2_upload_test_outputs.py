"""Upload a local bV2 test-output folder into the configured Mediaflux blenderv2 root.

This helper uses:
- the consuming project's `.env.mediaflux`
- the shared `mediafluxsync` package for path resolution
- the official `unimelb-mf-upload` client bundled in this repo

Typical usage:

    .\.venv\Scripts\python.exe -m refactor_code.blenderv2.bV2_upload_test_outputs `
      --local "E:\2026 Arboreal Futures\blender\tests\20260404_221500_bV2_city_existing-instancers_bioenvelope_viewport_tests_1080p_flat" `
      --remote-subpath "output/tests/20260404_221500_bV2_city_existing-instancers_bioenvelope_viewport_tests_1080p_flat"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UPLOAD_CMD = (
    REPO_ROOT
    / ".tools"
    / "mediaflux-bin"
    / "unimelb-mf-clients-0.8.5"
    / "bin"
    / "windows"
    / "unimelb-mf-upload.cmd"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload local bV2 test outputs to Mediaflux blenderv2/output/tests.")
    parser.add_argument("--local", required=True, type=Path, dest="local_path", help="Local folder to upload.")
    parser.add_argument(
        "--remote-subpath",
        required=True,
        help="Remote child path beneath the configured blenderv2 root, e.g. output/tests/20260404_foo.",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=REPO_ROOT,
        help="Project directory used to resolve .env.mediaflux.",
    )
    parser.add_argument(
        "--upload-cmd",
        type=Path,
        default=DEFAULT_UPLOAD_CMD,
        help="Path to unimelb-mf-upload.cmd.",
    )
    parser.add_argument("--nb-workers", type=int, default=4, help="Concurrent Mediaflux upload workers.")
    parser.add_argument("--dry-run", action="store_true", help="Print the upload command and exit.")
    return parser.parse_args()


def ensure_project_env() -> None:
    site_packages = REPO_ROOT / ".venv" / "Lib" / "site-packages"
    if site_packages.exists():
        sys.path.insert(0, str(site_packages))


def resolve_remote_path(project_dir: Path, remote_subpath: str) -> str:
    ensure_project_env()
    from mediafluxsync.project_config import load_project_config  # type: ignore

    project = load_project_config(start_dir=project_dir)
    return project.blenderv2_collection(remote_subpath)


def default_mflux_config_path() -> Path:
    ensure_project_env()
    from mediafluxsync.client import default_mflux_config_path as _default  # type: ignore

    return _default()


def build_upload_command(
    *,
    upload_cmd: Path,
    local_path: Path,
    remote_collection: str,
    nb_workers: int,
) -> list[str]:
    source_arg = str(local_path)
    if local_path.is_dir() and not source_arg.endswith("\\"):
        source_arg = source_arg + "\\"
    return [
        str(upload_cmd),
        "--mf.config",
        str(default_mflux_config_path()),
        "--dest",
        remote_collection,
        "--create-parents",
        "--exclude-parent",
        "--csum-check",
        "--nb-workers",
        str(nb_workers),
        source_arg,
    ]


def main() -> int:
    args = parse_args()
    local_path = args.local_path.expanduser().resolve()
    if not local_path.exists():
        raise SystemExit(f"Local path does not exist: {local_path}")
    upload_cmd = args.upload_cmd.expanduser().resolve()
    if not upload_cmd.exists():
        raise SystemExit(f"Upload command not found: {upload_cmd}")

    remote_collection = resolve_remote_path(args.project_dir, args.remote_subpath)
    command = build_upload_command(
        upload_cmd=upload_cmd,
        local_path=local_path,
        remote_collection=remote_collection,
        nb_workers=int(args.nb_workers),
    )

    print("remote_collection=", remote_collection)
    print("command=", subprocess.list2cmdline(command))
    if args.dry_run:
        return 0

    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
