"""Render/output setup and isolated EXR rendering for bV2 scenes."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import bpy

try:
    from _futureSim_refactored.paths import (
        BLENDERV2_OUTPUT_ROOT,
        mediaflux_blenderv2_exr_family_subpath,
    )
    from .bV2_scene_contract import (
        MATERIAL_NAMES,
        VIEW_LAYER_NAMES,
        get_aov_names,
        get_default_mist_profile,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _futureSim_refactored.paths import (
        BLENDERV2_OUTPUT_ROOT,
        mediaflux_blenderv2_exr_family_subpath,
    )
    from bV2_scene_contract import (  # type: ignore
        MATERIAL_NAMES,
        VIEW_LAYER_NAMES,
        get_aov_names,
        get_default_mist_profile,
    )


CODE_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "_futureSim_refactored")
REPO_ROOT = CODE_ROOT.parent
DEFAULT_RENDER_ROOT = BLENDERV2_OUTPUT_ROOT
MEDIAFLUX_CLIENT_BIN_DIR = (
    REPO_ROOT
    / ".tools"
    / "mediaflux-bin"
    / "unimelb-mf-clients-0.8.5"
    / "bin"
    / "windows"
)
REPO_VENV_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
LOG_PATH = os.environ.get("BV2_LOG_PATH", "").strip()
EXR_DEPTH = "16"
EXR_CODEC = "ZIP"
SITE_LABELS = {"trimmed-parade": "parade"}
TRUTHY = {"1", "true", "yes", "on"}
DEVICE_TYPE_CANDIDATES = ("OPTIX", "CUDA", "HIP", "ONEAPI")


def log(*args) -> None:
    message = " ".join(str(arg) for arg in args)
    timestamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(timestamped, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(timestamped + "\n")


def get_active_scene() -> bpy.types.Scene:
    scene = bpy.context.scene
    if scene is None:
        raise RuntimeError("No active Blender scene")
    return scene


def get_runtime_case_tag(scene: bpy.types.Scene) -> str:
    site = str(scene.get("bV2_site", "")).strip()
    mode = str(scene.get("bV2_mode", "")).strip()
    year_raw = str(scene.get("bV2_year", "")).strip()
    year = int(year_raw) if year_raw else None
    site_label = SITE_LABELS.get(site, site)
    if mode == "timeline":
        return f"{site_label}_timeline"
    if mode == "baseline":
        if year is None:
            return f"{site_label}_baseline"
        return f"{site_label}_baseline_yr{year}"
    return f"{site_label}_single-state_yr{year}"


def normalize_runtime_note(note: str) -> str:
    cleaned = "-".join(note.strip().lower().replace("_", "-").split())
    return cleaned.strip("-")


def get_runtime_exr_family(scene: bpy.types.Scene, note: str | None = None) -> str:
    case_tag = get_runtime_case_tag(scene)
    note_value = normalize_runtime_note(note or os.environ.get("BV2_EXR_FAMILY_NOTE", ""))
    if not note_value:
        return case_tag
    return f"{case_tag}__{note_value}"


def default_render_output_root(scene: bpy.types.Scene, *, timestamp: str, tag: str) -> Path:
    sim_root = os.environ.get("BV2_SIM_ROOT", "").strip()
    if sim_root:
        return (DEFAULT_RENDER_ROOT / sim_root / get_runtime_exr_family(scene)).resolve()
    return (DEFAULT_RENDER_ROOT / f"{timestamp}_{get_runtime_case_tag(scene)}_{tag}").resolve()


def ensure_aovs_and_passes(scene: bpy.types.Scene) -> None:
    required_aovs = tuple(get_aov_names())
    for view_layer in scene.view_layers:
        existing = {aov.name: aov for aov in view_layer.aovs}
        for aov_name in required_aovs:
            aov = existing.get(aov_name)
            if aov is None:
                aov = view_layer.aovs.add()
                aov.name = aov_name
            aov.type = "VALUE"
        for attr in (
            "use_pass_combined",
            "use_pass_z",
            "use_pass_mist",
            "use_pass_normal",
            "use_pass_object_index",
            "use_pass_material_index",
            "use_pass_ambient_occlusion",
        ):
            if hasattr(view_layer, attr):
                setattr(view_layer, attr, True)


def set_object_material(obj: bpy.types.Object, material_name: str) -> None:
    material = bpy.data.materials.get(material_name)
    if material is None or obj.type != "MESH" or obj.data is None:
        return
    if len(obj.data.materials) == 0:
        obj.data.materials.append(material)
    else:
        for index in range(len(obj.data.materials)):
            obj.data.materials[index] = material


def set_modifier_material(obj: bpy.types.Object, material_name: str) -> None:
    material = bpy.data.materials.get(material_name)
    if material is None:
        return
    for modifier in obj.modifiers:
        if modifier.type != "NODES":
            continue
        node_group = getattr(modifier, "node_group", None)
        if node_group is None:
            continue
        for node in node_group.nodes:
            if node.type == "SET_MATERIAL" and "Material" in node.inputs:
                node.inputs["Material"].default_value = material


def apply_live_materials_to_world(scene: bpy.types.Scene) -> None:
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        if obj.name.startswith("buildings_") or obj.name.startswith("roads_"):
            set_object_material(obj, MATERIAL_NAMES["world"])
            set_modifier_material(obj, MATERIAL_NAMES["world"])


def apply_live_materials_to_instancers(scene: bpy.types.Scene) -> None:
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        if obj.get("bV2_source_ply") or "_models_" in obj.name or obj.name.startswith("instanceID."):
            set_object_material(obj, MATERIAL_NAMES["instancers"])
        if "_positions_" in obj.name:
            set_modifier_material(obj, MATERIAL_NAMES["instancers"])


def apply_live_materials_to_bioenvelopes(scene: bpy.types.Scene) -> None:
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        if obj.name.startswith("bioenvelope_"):
            set_object_material(obj, MATERIAL_NAMES["bioenvelope"])
            set_modifier_material(obj, MATERIAL_NAMES["bioenvelope"])


def apply_live_materials(scene: bpy.types.Scene) -> None:
    apply_live_materials_to_world(scene)
    apply_live_materials_to_instancers(scene)
    apply_live_materials_to_bioenvelopes(scene)


def apply_mist_profile(scene: bpy.types.Scene) -> dict[str, object]:
    site = str(scene.get("bV2_site", "")).strip()
    mode = str(scene.get("bV2_mode", "")).strip()
    year_raw = str(scene.get("bV2_year", "")).strip()
    year = int(year_raw) if year_raw else None
    camera_name = scene.camera.name if scene.camera else ""
    if not site or not mode:
        return {}

    profile = get_default_mist_profile(site, mode, year, camera_name=camera_name)
    world = scene.world
    if world is None:
        world = bpy.data.worlds.get("debug_timeslice_world")
    if world is None:
        world = bpy.data.worlds.new("debug_timeslice_world")
    scene.world = world

    mist = world.mist_settings
    mist.use_mist = bool(profile.get("use_mist", True))
    mist.start = float(profile.get("start", 560.0))
    mist.depth = float(profile.get("depth", 320.0))
    mist.falloff = str(profile.get("falloff", "QUADRATIC"))
    for view_layer in scene.view_layers:
        if hasattr(view_layer, "use_pass_mist"):
            view_layer.use_pass_mist = True
    scene["bV2_mist_profile"] = str(profile.get("profile", ""))
    log(
        "MIST_APPLIED",
        f"profile={scene['bV2_mist_profile']}",
        f"camera={camera_name}",
        f"use_mist={mist.use_mist}",
        f"start={mist.start}",
        f"depth={mist.depth}",
        f"falloff={mist.falloff}",
    )
    return profile


def configure_cycles_device(scene: bpy.types.Scene) -> dict[str, object]:
    summary = {"device": "CPU", "backend": "", "devices": ()}
    if not hasattr(scene, "cycles"):
        return summary

    requested_backend = os.environ.get("BV2_CYCLES_DEVICE_TYPE", "").strip().upper()
    candidates = tuple(
        backend for backend in ((requested_backend,) if requested_backend else DEVICE_TYPE_CANDIDATES) if backend
    )

    preferences = bpy.context.preferences.addons.get("cycles")
    if preferences is None:
        try:
            scene.cycles.device = "CPU"
        except Exception:
            pass
        log("CYCLES_DEVICE_FALLBACK", "reason=no_cycles_preferences")
        return summary

    cprefs = preferences.preferences
    last_error = ""
    for backend in candidates:
        try:
            cprefs.compute_device_type = backend
            try:
                cprefs.get_devices()
            except Exception:
                pass
            devices = list(getattr(cprefs, "devices", ()))
            gpu_devices = [device for device in devices if getattr(device, "type", "") != "CPU"]
            if not gpu_devices:
                continue
            for device in devices:
                device.use = getattr(device, "type", "") != "CPU"
            scene.cycles.device = "GPU"
            summary = {
                "device": "GPU",
                "backend": backend,
                "devices": tuple(f"{device.name}|{device.type}" for device in gpu_devices),
            }
            log("CYCLES_GPU_ENABLED", f"backend={backend}", f"devices={summary['devices']}")
            return summary
        except Exception as exc:
            last_error = str(exc)

    try:
        scene.cycles.device = "CPU"
    except Exception:
        pass
    if last_error:
        log("CYCLES_DEVICE_FALLBACK", f"reason={last_error}")
    else:
        log("CYCLES_DEVICE_FALLBACK", "reason=no_gpu_devices")
    return summary


def configure_render_settings(
    scene: bpy.types.Scene,
    *,
    resolution_x: int,
    resolution_y: int,
    resolution_percentage: int,
    samples: int,
) -> None:
    scene.render.engine = "CYCLES"
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.render.film_transparent = True
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = resolution_percentage
    scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = EXR_DEPTH
    scene.render.image_settings.exr_codec = EXR_CODEC
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.display_settings.display_device = "sRGB"
    scene.sequencer_colorspace_settings.name = "sRGB"
    if hasattr(scene, "cycles"):
        scene.cycles.samples = samples
        scene.cycles.preview_samples = samples
        scene.cycles.use_denoising = False
        configure_cycles_device(scene)


def setup_render_outputs(
    scene: bpy.types.Scene | None = None,
    *,
    output_root: Path | None = None,
    resolution_x: int = 7680,
    resolution_y: int = 4320,
    resolution_percentage: int = 100,
    samples: int = 1,
    tag: str = "8k",
) -> dict[str, object]:
    scene = scene or get_active_scene()
    output_root = Path(output_root or DEFAULT_RENDER_ROOT).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ensure_aovs_and_passes(scene)
    apply_live_materials(scene)
    apply_mist_profile(scene)
    configure_render_settings(
        scene,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
        resolution_percentage=resolution_percentage,
        samples=samples,
    )
    scene["bV2_render_outputs_configured"] = True
    scene["bV2_render_output_root"] = str(output_root)
    scene["bV2_render_tag"] = tag
    summary = {
        "scene": scene.name,
        "output_root": str(output_root),
        "resolution": (resolution_x, resolution_y, resolution_percentage),
        "samples": samples,
        "tag": tag,
    }
    log("RENDER_SETUP_DONE", summary)
    return summary


def ensure_link(node_tree: bpy.types.NodeTree, from_socket, to_socket) -> None:
    for link in list(to_socket.links):
        if link.from_socket == from_socket:
            return
        node_tree.links.remove(link)
    node_tree.links.new(from_socket, to_socket)


def clear_file_output_slots(output_node: bpy.types.Node) -> None:
    while len(output_node.inputs):
        output_node.file_slots.remove(output_node.inputs[0])


def clone_scene_for_layer(
    scene: bpy.types.Scene,
    view_layer_name: str,
    *,
    resolution_x: int,
    resolution_y: int,
    resolution_percentage: int,
    samples: int,
) -> bpy.types.Scene:
    temp_scene = scene.copy()
    temp_scene.name = f"{scene.name}__{view_layer_name}__exr"
    temp_scene.camera = scene.camera
    configure_render_settings(
        temp_scene,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
        resolution_percentage=resolution_percentage,
        samples=samples,
    )

    target_layer = temp_scene.view_layers.get(view_layer_name)
    if target_layer is None:
        raise ValueError(f"View layer {view_layer_name!r} not found on copied scene")
    if hasattr(target_layer, "use"):
        target_layer.use = True
    if hasattr(temp_scene.view_layers, "active"):
        temp_scene.view_layers.active = target_layer

    for layer in list(temp_scene.view_layers):
        if layer != target_layer:
            temp_scene.view_layers.remove(layer)

    target_layer = temp_scene.view_layers.get(view_layer_name)
    if target_layer is None:
        raise ValueError(f"Failed to isolate view layer {view_layer_name!r}")
    ensure_aovs_and_passes(temp_scene)
    return temp_scene


def configure_temp_scene_exr_output(
    temp_scene: bpy.types.Scene,
    view_layer_name: str,
    output_path: Path,
) -> None:
    temp_scene.use_nodes = True
    node_tree = temp_scene.node_tree
    node_tree.nodes.clear()

    render_node = node_tree.nodes.new("CompositorNodeRLayers")
    render_node.name = f"Render Layers :: {view_layer_name}"
    render_node.scene = temp_scene
    render_node.layer = view_layer_name
    render_node.location = (-520.0, 0.0)

    output_node = node_tree.nodes.new("CompositorNodeOutputFile")
    output_node.name = f"EXR Output :: {view_layer_name}"
    output_node.base_path = str(output_path.with_suffix(""))
    output_node.format.file_format = "OPEN_EXR_MULTILAYER"
    output_node.format.color_mode = "RGBA"
    output_node.format.color_depth = EXR_DEPTH
    output_node.format.exr_codec = EXR_CODEC
    output_node.location = (-40.0, 0.0)

    clear_file_output_slots(output_node)
    enabled_sockets = [socket for socket in render_node.outputs if getattr(socket, "enabled", True)]
    for socket in enabled_sockets:
        output_node.file_slots.new(socket.name)
    for socket in enabled_sockets:
        target_socket = output_node.inputs.get(socket.name)
        if target_socket is not None:
            ensure_link(node_tree, socket, target_socket)


def rename_temp_exr_output(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        output_path.parent.glob(f"{output_path.stem}*.exr"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not candidates:
        raise FileNotFoundError(f"No EXR output found for {output_path.stem}")
    rendered_path = candidates[-1]
    if output_path.exists():
        output_path.unlink()
    rendered_path.replace(output_path)


def remove_blend_backup_versions(blend_path: Path) -> None:
    # Blender may leave numbered backup siblings like `foo.blend1`.
    for backup_path in blend_path.parent.glob(f"{blend_path.name}[0-9]*"):
        if backup_path.is_file():
            backup_path.unlink()


def save_mainfile(filepath: Path, *, copy: bool) -> Path:
    filepath = Path(filepath).resolve()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(filepath), copy=copy)
    remove_blend_backup_versions(filepath)
    return filepath


def render_isolated_exr(
    scene: bpy.types.Scene,
    view_layer_name: str,
    output_path: Path,
    *,
    resolution_x: int,
    resolution_y: int,
    resolution_percentage: int,
    samples: int,
) -> Path:
    temp_scene = clone_scene_for_layer(
        scene,
        view_layer_name,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
        resolution_percentage=resolution_percentage,
        samples=samples,
    )
    try:
        temp_scene.frame_set(1)
        configure_temp_scene_exr_output(temp_scene, view_layer_name, output_path)
        bpy.ops.render.render(write_still=True, scene=temp_scene.name)
        rename_temp_exr_output(output_path)
        log("RENDER_EXR_DONE", view_layer_name, output_path)
        return output_path
    finally:
        bpy.data.scenes.remove(temp_scene)


def render_all_isolated_exrs(
    scene: bpy.types.Scene,
    output_root: Path,
    *,
    basename: str | None = None,
    tag: str = "8k",
    resolution_x: int = 7680,
    resolution_y: int = 4320,
    resolution_percentage: int = 100,
    samples: int = 1,
    target_view_layers: tuple[str, ...] | None = None,
) -> list[Path]:
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    basename = basename or get_runtime_case_tag(scene)
    layers = target_view_layers or tuple(name for name in VIEW_LAYER_NAMES if scene.view_layers.get(name) is not None)

    outputs: list[Path] = []
    for view_layer_name in layers:
        output_path = output_root / f"{basename}__{view_layer_name}__{tag}.exr"
        outputs.append(
            render_isolated_exr(
                scene,
                view_layer_name,
                output_path,
                resolution_x=resolution_x,
                resolution_y=resolution_y,
                resolution_percentage=resolution_percentage,
                samples=samples,
            )
        )
    return outputs


def save_scene_copy(scene: bpy.types.Scene, output_root: Path, basename: str) -> Path:
    blend_path = output_root / f"{basename}__full_pipeline.blend"
    return save_mainfile(blend_path, copy=True)


def write_manifest(
    scene: bpy.types.Scene,
    output_root: Path,
    basename: str,
    rendered_paths: list[Path],
    *,
    blend_path: Path | None = None,
    remote_subpath: str | None = None,
) -> Path:
    manifest_path = output_root / f"{basename}__manifest.txt"
    lines = [
        f"scene={scene.name}",
        f"site={scene.get('bV2_site', '')}",
        f"mode={scene.get('bV2_mode', '')}",
        f"year={scene.get('bV2_year', '')}",
        f"camera={scene.camera.name if scene.camera else ''}",
        f"render_tag={scene.get('bV2_render_tag', '')}",
        f"output_root={output_root}",
    ]
    if remote_subpath:
        lines.append(f"remote_subpath={remote_subpath}")
    if blend_path is not None:
        lines.append(f"blend={blend_path}")
    lines.append("renders:")
    lines.extend(str(path) for path in rendered_paths)
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def upload_folder_to_mediaflux(local_path: Path, remote_subpath: str) -> None:
    if not REPO_VENV_PYTHON.exists():
        raise FileNotFoundError(f"Could not find repo Python at {REPO_VENV_PYTHON}")
    command = [
        str(REPO_VENV_PYTHON),
        "-m",
        "mediafluxsync",
        "upload-project",
        str(local_path),
        remote_subpath,
        "--project-dir",
        str(REPO_ROOT),
        "--create-parents",
        "--exclude-parent",
    ]
    env = os.environ.copy()
    env["MEDIAFLUX_CLIENT_BIN_DIR"] = str(MEDIAFLUX_CLIENT_BIN_DIR)
    log("MEDIAFLUX_UPLOAD_START", local_path, "->", remote_subpath)
    subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=True)
    log("MEDIAFLUX_UPLOAD_DONE", local_path, "->", remote_subpath)


def main() -> None:
    scene = get_active_scene()
    timestamp = os.environ.get("BV2_OUTPUT_TIMESTAMP", "").strip() or time.strftime("%Y%m%d_%H%M%S")
    tag = os.environ.get("BV2_RENDER_TAG", "8k").strip() or "8k"
    output_root = Path(
        os.environ.get("BV2_RENDER_OUTPUT_ROOT", str(default_render_output_root(scene, timestamp=timestamp, tag=tag)))
    )
    resolution_x = int(os.environ.get("BV2_RES_X", "7680"))
    resolution_y = int(os.environ.get("BV2_RES_Y", "4320"))
    resolution_percentage = int(os.environ.get("BV2_RES_PERCENT", "100"))
    samples = int(os.environ.get("BV2_SAMPLES", "1"))
    render_now = os.environ.get("BV2_RENDER_NOW", "0").strip().lower() in TRUTHY
    upload = os.environ.get("BV2_UPLOAD_TO_MEDIAFLUX", "0").strip().lower() in TRUTHY
    remote_subpath = ""

    setup_render_outputs(
        scene,
        output_root=output_root,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
        resolution_percentage=resolution_percentage,
        samples=samples,
        tag=tag,
    )
    if render_now:
        basename = get_runtime_case_tag(scene)
        blend_path = save_scene_copy(scene, output_root, basename)
        rendered_paths = render_all_isolated_exrs(
            scene,
            output_root,
            basename=basename,
            tag=tag,
            resolution_x=resolution_x,
            resolution_y=resolution_y,
            resolution_percentage=resolution_percentage,
            samples=samples,
        )
        write_manifest(scene, output_root, basename, rendered_paths, blend_path=blend_path, remote_subpath=remote_subpath or None)
    if upload:
        if not remote_subpath:
            sim_root = os.environ.get("BV2_SIM_ROOT", "").strip()
            if sim_root:
                remote_subpath = str(
                    mediaflux_blenderv2_exr_family_subpath(sim_root, get_runtime_exr_family(scene))
                )
            else:
                remote_subpath = f"pipeline/tests/blender_exrs/{timestamp}_{get_runtime_case_tag(scene)}_{tag}"
        upload_folder_to_mediaflux(output_root, remote_subpath)


if __name__ == "__main__":
    main()
