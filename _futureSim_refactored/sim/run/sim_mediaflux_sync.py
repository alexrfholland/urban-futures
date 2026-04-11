from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _futureSim_refactored.paths import (
    GENERATED_STATE_RUNS_ROOT,
    mediaflux_sim_root_subpath,
)


DEFAULT_CHILDREN = ("output",)
DEBUG_CHILDREN = ("temp", "comparison")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sync a simulation run root against the repo Mediaflux contract. "
            "By default this syncs only durable output/. Use --include-debug to "
            "also sync temp/ and comparison/."
        )
    )
    parser.add_argument(
        "command",
        choices=("upload", "download", "check"),
        help="Sync action to run.",
    )
    parser.add_argument(
        "sim_root",
        help="Simulation run root name, e.g. 4.9test or v4.9.",
    )
    parser.add_argument(
        "--include-debug",
        action="store_true",
        help="Include temp/ and comparison/ in addition to output/.",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("."),
        help="Project directory used to resolve repo-local Mediaflux config.",
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        default=None,
        help="Optional explicit local simulation root override.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV report path for check mode.",
    )
    parser.add_argument(
        "--checksum",
        action="store_true",
        help="Enable checksum comparison for check mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the underlying mediafluxsync commands without running them.",
    )
    return parser


def sim_local_root(sim_root: str, override: Path | None = None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    return (GENERATED_STATE_RUNS_ROOT / sim_root).resolve()


def iter_children(include_debug: bool) -> tuple[str, ...]:
    if include_debug:
        return DEFAULT_CHILDREN + DEBUG_CHILDREN
    return DEFAULT_CHILDREN


def run_mediafluxsync(command: list[str], *, dry_run: bool) -> None:
    if dry_run:
        print(subprocess.list2cmdline(command))
        return
    subprocess.run(command, check=True)


def upload_child(*, project_dir: Path, local_root: Path, remote_root: Path, child: str, dry_run: bool) -> None:
    local_child = local_root / child
    if not local_child.exists():
        print(f"[sim_mediaflux_sync] skip missing local child: {local_child}")
        return
    command = [
        sys.executable,
        "-m",
        "mediafluxsync",
        "upload-project",
        str(local_child),
        str(remote_root / child),
        "--project-dir",
        str(project_dir),
        "--create-parents",
        "--exclude-parent",
    ]
    if dry_run:
        command.append("--dry-run")
    run_mediafluxsync(command, dry_run=dry_run)


def download_child(*, project_dir: Path, local_root: Path, remote_root: Path, child: str, dry_run: bool) -> None:
    local_child = local_root / child
    command = [
        sys.executable,
        "-m",
        "mediafluxsync",
        "download-project",
        str(remote_root / child),
        "--project-dir",
        str(project_dir),
        "--out",
        str(local_child),
    ]
    if dry_run:
        command.append("--dry-run")
    run_mediafluxsync(command, dry_run=dry_run)


def check_child(
    *,
    project_dir: Path,
    local_root: Path,
    remote_root: Path,
    child: str,
    output_csv: Path,
    checksum: bool,
    dry_run: bool,
) -> None:
    local_child = local_root / child
    if not local_child.exists():
        print(f"[sim_mediaflux_sync] skip missing local child: {local_child}")
        return
    child_output = output_csv.with_name(f"{output_csv.stem}__{child.replace('/', '_')}{output_csv.suffix}")
    command = [
        sys.executable,
        "-m",
        "mediafluxsync",
        "check-project",
        str(local_child),
        str(remote_root / child),
        "--project-dir",
        str(project_dir),
        "--direction",
        "up",
        "--output",
        str(child_output),
    ]
    command.append("--checksum" if checksum else "--no-checksum")
    if dry_run:
        command.append("--dry-run")
    run_mediafluxsync(command, dry_run=dry_run)


def main() -> None:
    args = build_parser().parse_args()
    project_dir = args.project_dir.expanduser().resolve()
    local_root = sim_local_root(args.sim_root, args.local_root)
    remote_root = mediaflux_sim_root_subpath(args.sim_root)
    children = iter_children(args.include_debug)

    if args.command == "upload":
        for child in children:
            upload_child(
                project_dir=project_dir,
                local_root=local_root,
                remote_root=remote_root,
                child=child,
                dry_run=args.dry_run,
            )
        return

    if args.command == "download":
        for child in children:
            download_child(
                project_dir=project_dir,
                local_root=local_root,
                remote_root=remote_root,
                child=child,
                dry_run=args.dry_run,
            )
        return

    output_csv = args.output
    if output_csv is None:
        raise SystemExit("--output is required for check mode")
    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    for child in children:
        check_child(
            project_dir=project_dir,
            local_root=local_root,
            remote_root=remote_root,
            child=child,
            output_csv=output_csv,
            checksum=args.checksum,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
