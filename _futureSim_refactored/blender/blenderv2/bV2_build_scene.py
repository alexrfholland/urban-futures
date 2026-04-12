"""End-to-end bV2 scene build orchestration."""

from __future__ import annotations

import os
import time
from pathlib import Path

import bpy

try:
    from _futureSim_refactored.paths import (
        BLENDERV2_BLENDS_ROOT,
        blenderv2_scene_blend_path,
        mediaflux_blenderv2_exr_family_subpath,
    )
    from .bV2_build_bioenvelopes import build_bioenvelopes
    from .bV2_build_instancers import build_instancers
    from .bV2_build_world_attributes import build_world_attributes
    from .bV2_setup_template import (
        BUILD_MODE_POINTS as TEMPLATE_BUILD_MODE_POINTS,
        BUILD_MODE_SPLIT as TEMPLATE_BUILD_MODE_SPLIT,
        ensure_scene_aovs as ensure_template_scene_aovs,
        reset_site as reset_template_site,
        verify_site as verify_template_site,
    )
    from .bV2_init_scene import init_scene
    from .bV2_scene_contract import TEMPLATE_BLEND_NAME
    from .bV2_setup_render_outputs import (
        DEFAULT_RENDER_ROOT,
        default_render_output_root,
        get_runtime_exr_family,
        get_runtime_case_tag,
        render_all_isolated_exrs,
        save_scene_copy,
        setup_render_outputs,
        upload_folder_to_mediaflux,
        write_manifest,
    )
    from .bV2_validate_scene import validate_scene
except ImportError:
    import sys

    _THIS_FILE = Path(__file__).resolve()
    _REPO_ROOT = next(
        parent.parent for parent in _THIS_FILE.parents if parent.name == "_futureSim_refactored"
    )
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_THIS_FILE.parent))
    from _futureSim_refactored.paths import (
        BLENDERV2_BLENDS_ROOT,
        blenderv2_scene_blend_path,
        mediaflux_blenderv2_exr_family_subpath,
    )
    from bV2_build_bioenvelopes import build_bioenvelopes  # type: ignore
    from bV2_build_instancers import build_instancers  # type: ignore
    from bV2_build_world_attributes import build_world_attributes  # type: ignore
    from bV2_setup_template import (  # type: ignore
        BUILD_MODE_POINTS as TEMPLATE_BUILD_MODE_POINTS,
        BUILD_MODE_SPLIT as TEMPLATE_BUILD_MODE_SPLIT,
        ensure_scene_aovs as ensure_template_scene_aovs,
        reset_site as reset_template_site,
        verify_site as verify_template_site,
    )
    from bV2_init_scene import init_scene  # type: ignore
    from bV2_scene_contract import TEMPLATE_BLEND_NAME  # type: ignore
    from bV2_setup_render_outputs import (  # type: ignore
        DEFAULT_RENDER_ROOT,
        default_render_output_root,
        get_runtime_exr_family,
        get_runtime_case_tag,
        render_all_isolated_exrs,
        save_scene_copy,
        setup_render_outputs,
        upload_folder_to_mediaflux,
        write_manifest,
    )
    from bV2_validate_scene import validate_scene  # type: ignore


CODE_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "_futureSim_refactored")
REPO_ROOT = CODE_ROOT.parent
TEMPLATE_BLEND = REPO_ROOT / "_data-refactored" / "blenderv2" / TEMPLATE_BLEND_NAME
DEFAULT_BLEND_ROOT = BLENDERV2_BLENDS_ROOT
TRUTHY = {"1", "true", "yes", "on"}
LOG_PATH = os.environ.get("BV2_LOG_PATH", "").strip()


def log(*args) -> None:
    message = " ".join(str(arg) for arg in args)
    timestamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(timestamped, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(timestamped + "\n")


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    return raw.strip().lower() in TRUTHY


def get_required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable {name}")
    return value


def resolve_template_blend_path() -> Path:
    override = os.environ.get("BV2_TEMPLATE_BLEND", "").strip()
    if override:
        return Path(override).resolve()
    return TEMPLATE_BLEND


def open_template_blend() -> None:
    template_blend = resolve_template_blend_path()
    if not template_blend.exists():
        raise FileNotFoundError(f"Could not find bV2 template blend at {template_blend}")
    bpy.ops.wm.open_mainfile(filepath=str(template_blend), load_ui=False)


def get_template_setup_build_mode() -> str:
    raw = os.environ.get("BV2_POINTSORCUBES", "").strip().lower()
    if raw in {"", "points", "point"}:
        return TEMPLATE_BUILD_MODE_POINTS
    if raw in {"cubes", "split"}:
        return TEMPLATE_BUILD_MODE_SPLIT
    raise RuntimeError(
        f"Unsupported BV2_POINTSORCUBES={raw!r}. Expected one of: points, point, cubes, split"
    )


def run_optional_template_setup(site: str) -> dict[str, object] | None:
    if not env_bool("BV2_RUN_TEMPLATE_SETUP", False):
        return None

    build_mode = get_template_setup_build_mode()
    ensure_template_scene_aovs()
    setup_summary = reset_template_site(site, build_mode=build_mode)
    verification_summary = verify_template_site(site, build_mode=build_mode)
    if not verification_summary.get("ok", False):
        raise RuntimeError(
            f"Template setup verification failed for {site}: {verification_summary.get('errors', [])}"
        )
    log("TEMPLATE_SETUP_DONE", {"result": setup_summary, "verification": verification_summary})
    return {"result": setup_summary, "verification": verification_summary}


def build_scene(
    *,
    site: str,
    mode: str,
    year: int | None = None,
    camera_name: str | None = None,
    save_blend: bool = True,
    blend_output_path: Path | None = None,
    render_exrs: bool = False,
    render_output_root: Path | None = None,
    render_tag: str = "8k",
    resolution_x: int = 7680,
    resolution_y: int = 4320,
    resolution_percentage: int = 100,
    samples: int = 64,
    upload_to_mediaflux: bool = False,
    remote_subpath: str | None = None,
    validate_strict: bool = True,
) -> dict[str, object]:
    is_baseline = mode == "baseline"
    log("BUILD_SCENE_START", "site=", site, "mode=", mode, "year=", year, "render_exrs=", render_exrs)
    open_template_blend()
    template_setup_summary = run_optional_template_setup(site)
    scene = init_scene(site=site, mode=mode, year=year, camera_name=camera_name)
    instancer_summary = build_instancers(scene)
    if is_baseline:
        bio_summary = {"skipped": True, "reason": "baseline mode"}
        world_summary = build_world_attributes(scene)
    else:
        bio_summary = build_bioenvelopes(scene)
        world_summary = build_world_attributes(scene)
    validation_summary = validate_scene(scene, strict=validate_strict)

    scene["bV2_build_scene_complete"] = True
    case_tag = get_runtime_case_tag(scene)

    if save_blend:
        sim_root = os.environ.get("BV2_SIM_ROOT", "").strip()
        default_blend_path = DEFAULT_BLEND_ROOT / f"{scene.name}__full_pipeline.blend"
        if sim_root:
            default_blend_path = blenderv2_scene_blend_path(sim_root, get_runtime_exr_family(scene))
        blend_output_path = Path(
            blend_output_path
            or default_blend_path
        ).resolve()
        blend_output_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_output_path), copy=True)
        log("BUILD_SCENE_BLEND_SAVED", blend_output_path)
    else:
        blend_output_path = None

    render_summary: dict[str, object] | None = None
    rendered_paths: list[Path] = []
    if render_exrs:
        timestamp = os.environ.get("BV2_OUTPUT_TIMESTAMP", "").strip() or time.strftime("%Y%m%d_%H%M%S")
        render_output_root = Path(
            render_output_root
            or default_render_output_root(scene, timestamp=timestamp, tag=render_tag)
        ).resolve()
        render_output_root.mkdir(parents=True, exist_ok=True)
        render_summary = setup_render_outputs(
            scene,
            output_root=render_output_root,
            resolution_x=resolution_x,
            resolution_y=resolution_y,
            resolution_percentage=resolution_percentage,
            samples=samples,
            tag=render_tag,
        )
        blend_copy_path = save_scene_copy(scene, render_output_root, case_tag)
        rendered_paths = render_all_isolated_exrs(
            scene,
            render_output_root,
            basename=case_tag,
            tag=render_tag,
            resolution_x=resolution_x,
            resolution_y=resolution_y,
            resolution_percentage=resolution_percentage,
            samples=samples,
        )
        write_manifest(
            scene,
            render_output_root,
            case_tag,
            rendered_paths,
            blend_path=blend_copy_path,
            remote_subpath=remote_subpath,
        )
        if upload_to_mediaflux:
            if not remote_subpath:
                sim_root = os.environ.get("BV2_SIM_ROOT", "").strip()
                if sim_root:
                    remote_subpath = str(
                        mediaflux_blenderv2_exr_family_subpath(sim_root, get_runtime_exr_family(scene))
                    )
                else:
                    remote_subpath = f"pipeline/tests/blender_exrs/{render_tag}/{render_output_root.name}"
            upload_folder_to_mediaflux(render_output_root, remote_subpath)

    summary = {
        "scene": scene.name,
        "site": site,
        "mode": mode,
        "year": year,
        "blend_output_path": str(blend_output_path) if blend_output_path else "",
        "render_output_root": str(render_output_root) if render_exrs and render_output_root else "",
        "remote_subpath": remote_subpath or "",
        "instancers": instancer_summary,
        "bioenvelopes": bio_summary,
        "world": world_summary,
        "template_setup": template_setup_summary or {},
        "validation": validation_summary,
        "render": render_summary,
        "rendered_paths": [str(path) for path in rendered_paths],
    }
    log("BUILD_SCENE_DONE", summary)
    return summary


def main() -> None:
    site = get_required_env("BV2_SITE")
    mode = get_required_env("BV2_MODE")
    year_raw = os.environ.get("BV2_YEAR", "").strip()
    year = int(year_raw) if year_raw else None
    camera_name = os.environ.get("BV2_CAMERA_NAME", "").strip() or None
    save_blend = env_bool("BV2_SAVE_BLEND", True)
    render_exrs = env_bool("BV2_RENDER_EXRS", False)
    upload_to_mediaflux = env_bool("BV2_UPLOAD_TO_MEDIAFLUX", False)
    validate_strict = env_bool("BV2_VALIDATE_STRICT", True)
    blend_output_path_raw = os.environ.get("BV2_BLEND_OUTPUT_PATH", "").strip()
    blend_output_path = Path(blend_output_path_raw) if blend_output_path_raw else None
    render_output_root_raw = os.environ.get("BV2_RENDER_OUTPUT_ROOT", "").strip()
    render_output_root = Path(render_output_root_raw) if render_output_root_raw else None
    render_tag = os.environ.get("BV2_RENDER_TAG", "8k").strip() or "8k"
    resolution_x = int(os.environ.get("BV2_RES_X", "7680"))
    resolution_y = int(os.environ.get("BV2_RES_Y", "4320"))
    resolution_percentage = int(os.environ.get("BV2_RES_PERCENT", "100"))
    samples = int(os.environ.get("BV2_SAMPLES", "64"))
    remote_subpath = None

    build_scene(
        site=site,
        mode=mode,
        year=year,
        camera_name=camera_name,
        save_blend=save_blend,
        blend_output_path=blend_output_path,
        render_exrs=render_exrs,
        render_output_root=render_output_root,
        render_tag=render_tag,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
        resolution_percentage=resolution_percentage,
        samples=samples,
        upload_to_mediaflux=upload_to_mediaflux,
        remote_subpath=remote_subpath,
        validate_strict=validate_strict,
    )


if __name__ == "__main__":
    main()
