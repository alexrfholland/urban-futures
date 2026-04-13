from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

from _futureSim_refactored.paths import (
    blenderv2_exr_family_dir,
    mediaflux_blenderv2_exr_family_subpath,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sync one bV2 EXR family against Mediaflux. "
            "By default only .exr files are transferred; use --include-metadata "
            "to also include .blend/.txt/.json sidecars."
        )
    )
    parser.add_argument("command", choices=("upload", "download", "list-remote"))
    parser.add_argument("sim_root")
    parser.add_argument("exr_family")
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--local-dir", type=Path, default=None)
    parser.add_argument("--include-metadata", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def local_family_dir(sim_root: str, exr_family: str, override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    return blenderv2_exr_family_dir(sim_root, exr_family).resolve()


def is_selected_name(name: str, *, include_metadata: bool) -> bool:
    suffix = Path(name).suffix.lower()
    if suffix == ".exr":
        return True
    return include_metadata


def run_mediafluxsync(command: list[str], *, dry_run: bool) -> None:
    if dry_run:
        print(subprocess.list2cmdline(command))
        return
    subprocess.run(command, check=True)


def local_selected_files(local_dir: Path, *, include_metadata: bool) -> list[Path]:
    if not local_dir.exists():
        raise SystemExit(f"Local EXR family directory does not exist: {local_dir}")
    return sorted(
        path
        for path in local_dir.iterdir()
        if path.is_file() and is_selected_name(path.name, include_metadata=include_metadata)
    )


def remote_selected_names(
    *,
    project_dir: Path,
    sim_root: str,
    exr_family: str,
    include_metadata: bool,
    dry_run: bool,
) -> list[str]:
    remote_subpath = mediaflux_blenderv2_exr_family_subpath(sim_root, exr_family)
    with tempfile.TemporaryDirectory(prefix="bv2_mediaflux_remote_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        output_csv = tmp_dir / "report.csv"
        command = [
            sys.executable,
            "-m",
            "mediafluxsync",
            "check-project",
            str(tmp_dir),
            str(remote_subpath),
            "--project-dir",
            str(project_dir),
            "--direction",
            "down",
            "--no-checksum",
            "--output",
            str(output_csv),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)

        names: list[str] = []
        with output_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                dst_path = row.get("DST_PATH", "")
                if not dst_path.startswith("file:"):
                    continue
                name = Path(dst_path.removeprefix("file:")).name
                if is_selected_name(name, include_metadata=include_metadata):
                    names.append(name)
        return sorted(dict.fromkeys(names))


def upload_family(
    *,
    project_dir: Path,
    sim_root: str,
    exr_family: str,
    local_dir: Path,
    include_metadata: bool,
    dry_run: bool,
) -> None:
    remote_root = mediaflux_blenderv2_exr_family_subpath(sim_root, exr_family)
    for local_file in local_selected_files(local_dir, include_metadata=include_metadata):
        command = [
            sys.executable,
            "-m",
            "mediafluxsync",
            "upload-project",
            str(local_file),
            str(remote_root / local_file.name),
            "--project-dir",
            str(project_dir),
            "--create-parents",
        ]
        if dry_run:
            command.append("--dry-run")
        run_mediafluxsync(command, dry_run=dry_run)


def download_family(
    *,
    project_dir: Path,
    sim_root: str,
    exr_family: str,
    local_dir: Path,
    include_metadata: bool,
    dry_run: bool,
) -> None:
    if not dry_run:
        local_dir.mkdir(parents=True, exist_ok=True)

    remote_root = mediaflux_blenderv2_exr_family_subpath(sim_root, exr_family)
    for name in remote_selected_names(
        project_dir=project_dir,
        sim_root=sim_root,
        exr_family=exr_family,
        include_metadata=include_metadata,
        dry_run=dry_run,
    ):
        command = [
            sys.executable,
            "-m",
            "mediafluxsync",
            "download-project",
            str(remote_root / name),
            "--project-dir",
            str(project_dir),
            "--out",
            str(local_dir),
        ]
        if dry_run:
            command.append("--dry-run")
        run_mediafluxsync(command, dry_run=dry_run)


def list_remote(
    *,
    project_dir: Path,
    sim_root: str,
    exr_family: str,
    include_metadata: bool,
    dry_run: bool,
) -> None:
    names = remote_selected_names(
        project_dir=project_dir,
        sim_root=sim_root,
        exr_family=exr_family,
        include_metadata=include_metadata,
        dry_run=dry_run,
    )
    for name in names:
        print(name)


def main() -> None:
    args = build_parser().parse_args()
    project_dir = args.project_dir.expanduser().resolve()
    local_dir = local_family_dir(args.sim_root, args.exr_family, args.local_dir)

    if args.command == "upload":
        upload_family(
            project_dir=project_dir,
            sim_root=args.sim_root,
            exr_family=args.exr_family,
            local_dir=local_dir,
            include_metadata=args.include_metadata,
            dry_run=args.dry_run,
        )
        return

    if args.command == "download":
        download_family(
            project_dir=project_dir,
            sim_root=args.sim_root,
            exr_family=args.exr_family,
            local_dir=local_dir,
            include_metadata=args.include_metadata,
            dry_run=args.dry_run,
        )
        return

    list_remote(
        project_dir=project_dir,
        sim_root=args.sim_root,
        exr_family=args.exr_family,
        include_metadata=args.include_metadata,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
