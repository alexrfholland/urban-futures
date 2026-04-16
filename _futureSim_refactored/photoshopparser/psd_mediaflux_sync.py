from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

from _futureSim_refactored.paths import (
    PSDS_LIVE_ROOT,
    PSD_BUCKETS,
    mediaflux_psd_bucket_subpath,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sync PSD/PSB files against the `pipeline/_psds/<bucket>/` remote "
            "namespace. Buckets: " + ", ".join(PSD_BUCKETS) + "."
        )
    )
    parser.add_argument("command", choices=("upload", "download", "list-remote"))
    parser.add_argument("bucket", choices=PSD_BUCKETS)
    parser.add_argument(
        "files",
        nargs="*",
        help=(
            "File names relative to the bucket (upload/download). "
            "Ignored for list-remote."
        ),
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help=(
            "Local directory to source files from (upload) or write files to "
            "(download). Defaults to _data-refactored/_psds/psd-live/."
        ),
    )
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def resolve_local_dir(override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    return PSDS_LIVE_ROOT.resolve()


def run_mediafluxsync(command: list[str], *, dry_run: bool) -> None:
    if dry_run:
        print(subprocess.list2cmdline(command))
        return
    subprocess.run(command, check=True)


def upload_files(
    *,
    project_dir: Path,
    bucket: str,
    local_dir: Path,
    names: list[str],
    dry_run: bool,
) -> None:
    if not names:
        raise SystemExit("upload requires at least one file name")
    remote_root = mediaflux_psd_bucket_subpath(bucket)
    for name in names:
        local_file = (local_dir / name).resolve()
        if not local_file.is_file():
            raise SystemExit(f"Local file not found: {local_file}")
        # mediafluxsync upload-project's `subpath` is the *parent* container;
        # the asset is created as `<subpath>/<source_basename>`. Passing the
        # full intended asset path with --create-parents caused the initial
        # template upload to nest the file under a same-named folder
        # (`_templates/base_template.psd/base_template.psd`), so we pass the
        # bucket as the parent and let mediafluxsync append the basename.
        command = [
            sys.executable,
            "-m",
            "mediafluxsync",
            "upload-project",
            str(local_file),
            str(remote_root),
            "--project-dir",
            str(project_dir),
            "--create-parents",
        ]
        if dry_run:
            command.append("--dry-run")
        run_mediafluxsync(command, dry_run=dry_run)


def remote_names(
    *,
    project_dir: Path,
    bucket: str,
) -> list[str]:
    remote_root = mediaflux_psd_bucket_subpath(bucket)
    with tempfile.TemporaryDirectory(prefix="psd_mediaflux_remote_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        output_csv = tmp_dir / "report.csv"
        command = [
            sys.executable,
            "-m",
            "mediafluxsync",
            "check-project",
            str(tmp_dir),
            str(remote_root),
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
                names.append(Path(dst_path.removeprefix("file:")).name)
        return sorted(dict.fromkeys(names))


def download_files(
    *,
    project_dir: Path,
    bucket: str,
    local_dir: Path,
    names: list[str],
    dry_run: bool,
) -> None:
    if not names:
        raise SystemExit("download requires at least one file name")
    if not dry_run:
        local_dir.mkdir(parents=True, exist_ok=True)
    remote_root = mediaflux_psd_bucket_subpath(bucket)
    for name in names:
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


def list_remote(*, project_dir: Path, bucket: str) -> None:
    for name in remote_names(project_dir=project_dir, bucket=bucket):
        print(name)


def main() -> None:
    args = build_parser().parse_args()
    project_dir = args.project_dir.expanduser().resolve()
    local_dir = resolve_local_dir(args.local_dir)

    if args.command == "upload":
        upload_files(
            project_dir=project_dir,
            bucket=args.bucket,
            local_dir=local_dir,
            names=args.files,
            dry_run=args.dry_run,
        )
        return

    if args.command == "download":
        download_files(
            project_dir=project_dir,
            bucket=args.bucket,
            local_dir=local_dir,
            names=args.files,
            dry_run=args.dry_run,
        )
        return

    list_remote(project_dir=project_dir, bucket=args.bucket)


if __name__ == "__main__":
    main()
