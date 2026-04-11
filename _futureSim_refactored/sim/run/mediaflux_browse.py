from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_MEDIAFLUX_MOUNT_ROOT = Path(
    "/Volumes/proj-7020_research_archive-1128.4.442/MF 2026 Arboreal Futures"
)
SECTIONS = ("simulation_outputs", "blender_exrs", "compositor_pngs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Browse the mounted Mediaflux pipeline tree directly. "
            "This is faster than using check-project as a remote lister."
        )
    )
    parser.add_argument(
        "sim_root",
        nargs="?",
        default="",
        help="Optional simulation root to inspect, e.g. 4.9 or 4.9test.",
    )
    parser.add_argument(
        "--section",
        choices=(*SECTIONS, "all"),
        default="all",
        help="Pipeline section to inspect.",
    )
    parser.add_argument(
        "--pattern",
        default="",
        help="Optional case-insensitive substring filter against relative paths.",
    )
    parser.add_argument(
        "--files",
        action="store_true",
        help="List files instead of directories.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum depth beneath pipeline/<sim_root> to include.",
    )
    parser.add_argument(
        "--mount-root",
        type=Path,
        default=None,
        help="Optional mounted Mediaflux root override.",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=None,
        help=(
            "When sim_root is omitted, list only the N most recently modified "
            "pipeline roots, optionally filtered by --pattern."
        ),
    )
    parser.add_argument(
        "--map",
        action="store_true",
        help="Print a compact tree map instead of a flat path list.",
    )
    return parser


def mount_root(override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    env_override = os.environ.get("MEDIAFLUX_MOUNT_ROOT", "").strip()
    if env_override:
        return Path(env_override).expanduser().resolve()
    return DEFAULT_MEDIAFLUX_MOUNT_ROOT


def iter_paths(base: Path, *, files: bool, max_depth: int) -> list[Path]:
    results: list[Path] = []
    if not base.exists():
        return results
    for path in base.rglob("*"):
        if files and not path.is_file():
            continue
        if not files and not path.is_dir():
            continue
        rel = path.relative_to(base)
        if len(rel.parts) > max_depth:
            continue
        results.append(path)
    return sorted(results)


def iter_pipeline_roots(pipeline_root: Path) -> list[Path]:
    if not pipeline_root.exists():
        return []
    roots = [path for path in pipeline_root.iterdir() if path.is_dir()]
    return sorted(roots, key=lambda path: path.stat().st_mtime, reverse=True)


def root_matches_pattern(
    root: Path,
    *,
    sections: tuple[str, ...],
    pattern: str,
    max_depth: int,
) -> bool:
    if not pattern:
        return True
    lowered = pattern.lower()
    for section in sections:
        base = root / section
        if not base.exists():
            continue
        for path in base.rglob("*"):
            rel = path.relative_to(root)
            if len(rel.parts) > max_depth + 1:
                continue
            if lowered in rel.as_posix().lower():
                return True
    return False


def print_tree(base: Path, *, files: bool, max_depth: int, mf_root: Path, pattern: str) -> bool:
    paths = iter_paths(base, files=files, max_depth=max_depth)
    if pattern:
        filtered: list[Path] = []
        for path in paths:
            rel = path.relative_to(mf_root).as_posix().lower()
            if pattern in rel:
                filtered.append(path)
        paths = filtered
    if not paths:
        return False
    print(base.relative_to(mf_root).as_posix())
    for path in paths:
        rel = path.relative_to(base)
        indent = "  " * len(rel.parts)
        print(f"{indent}{rel.name}")
    return True


def main() -> None:
    args = build_parser().parse_args()
    mf_root = mount_root(args.mount_root)
    sections = SECTIONS if args.section == "all" else (args.section,)
    pattern = args.pattern.strip().lower()
    pipeline_root = mf_root / "pipeline"

    if not pipeline_root.exists():
        raise SystemExit(f"Mounted pipeline root does not exist: {pipeline_root}")

    if not args.sim_root:
        roots = iter_pipeline_roots(pipeline_root)
        filtered_roots = [
            root
            for root in roots
            if root_matches_pattern(
                root,
                sections=sections,
                pattern=pattern,
                max_depth=args.max_depth,
            )
        ]
        if args.last is not None:
            filtered_roots = filtered_roots[: max(args.last, 0)]
        if not filtered_roots:
            raise SystemExit(
                "No matching mounted Mediaflux sim roots found. "
                "Try a different pattern, section, or remove --last."
            )
        if args.map:
            for index, root in enumerate(filtered_roots):
                if index:
                    print()
                print(root.relative_to(mf_root).as_posix())
                for section in sections:
                    base = root / section
                    if not base.exists():
                        continue
                    print_tree(
                        base,
                        files=args.files,
                        max_depth=args.max_depth,
                        mf_root=mf_root,
                        pattern=pattern,
                    )
            return
        for root in filtered_roots:
            print(root.relative_to(mf_root).as_posix())
        return

    sim_pipeline_root = pipeline_root / args.sim_root

    if not sim_pipeline_root.exists():
        raise SystemExit(f"Mounted pipeline root does not exist: {sim_pipeline_root}")

    matched_any = False
    if args.map:
        print(sim_pipeline_root.relative_to(mf_root).as_posix())
    for section in sections:
        base = sim_pipeline_root / section
        if args.map:
            printed = print_tree(
                base,
                files=args.files,
                max_depth=args.max_depth,
                mf_root=mf_root,
                pattern=pattern,
            )
            matched_any = matched_any or printed
            continue
        paths = iter_paths(base, files=args.files, max_depth=args.max_depth)
        if pattern:
            filtered: list[Path] = []
            for path in paths:
                rel = path.relative_to(mf_root).as_posix().lower()
                if pattern in rel:
                    filtered.append(path)
            paths = filtered
        if not paths:
            continue
        matched_any = True
        for path in paths:
            print(path.relative_to(mf_root).as_posix())

    if not matched_any:
        raise SystemExit(
            "No matching mounted Mediaflux paths found. "
            "Try a different sim_root, section, or pattern."
        )


if __name__ == "__main__":
    main()
