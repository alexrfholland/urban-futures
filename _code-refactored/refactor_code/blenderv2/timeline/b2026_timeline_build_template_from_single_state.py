from __future__ import annotations

from pathlib import Path
import os
import sys

import bpy


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import b2026_timeline_scene_contract as scene_contract
import b2026_timeline_scene_setup as scene_setup


DEFAULT_SINGLE_STATE_TEMPLATE_BY_SITE = {
    "city": Path(r"D:\2026 Arboreal Futures\data\2026 futures city single-state template.blend"),
}

DEFAULT_REFERENCE_BLEND_BY_SITE = {
    "city": Path(r"D:\2026 Arboreal Futures\data\2026 futures city lightweight cleaned.blend"),
    "trimmed-parade": Path(r"D:\2026 Arboreal Futures\data\2026 futures parade lightweight cleaned.blend"),
    "street": Path(r"D:\2026 Arboreal Futures\data\2026 futures street lightweight cleaned.blend"),
}
DEFAULT_TIMESLICE_CAMERA_BLEND = Path(
    r"D:\2026 Arboreal Futures\data\renders\timeslices\camera_tests\2026 futures timeslice debug camera framing v3.blend"
)
TIMESLICE_CAMERA_SPECS = {
    "city": {
        "camera_names": (
            "city - camera - time slice",
            "city - camera - time slice - zoom",
        ),
        "active_camera_name": "city - camera - time slice - zoom",
    },
    "trimmed-parade": {
        "camera_names": (
            "parade - camera - time slice",
            "parade - camera - time slice - zoom",
        ),
        "active_camera_name": "parade - camera - time slice - zoom",
    },
    "street": {
        "camera_names": (
            "street - camera - time slice",
            "street - camera - time slice - zoom",
        ),
        "active_camera_name": "street - camera - time slice - zoom",
    },
    "uni": {
        "camera_names": (
            "street - camera - time slice",
            "street - camera - time slice - zoom",
        ),
        "active_camera_name": "street - camera - time slice - zoom",
    },
}

TIMELINE_ALIAS_VIEW_LAYERS = (
    "city_priority",
    "city_bioenvelope",
)


def detect_site() -> str:
    raw = os.environ.get("B2026_SITE_KEY", "").strip()
    if raw in scene_contract.SITE_CONTRACTS:
        return raw
    scene_name = os.environ.get("B2026_SCENE_NAME", "").strip()
    inferred = scene_contract.infer_site_from_scene_name(scene_name)
    if inferred is not None:
        return inferred
    return "city"


def detect_scene_name(site: str) -> str:
    return os.environ.get("B2026_SCENE_NAME", "").strip() or scene_contract.SITE_CONTRACTS[site]["scene_name"]


def default_output_path(source_blend: Path) -> Path:
    stem = source_blend.stem.replace("single-state template", "timeline template").strip()
    if stem == source_blend.stem:
        stem = f"{source_blend.stem} timeline template"
    return source_blend.with_name(f"{stem}.blend")


def get_source_blend(site: str) -> Path:
    env_path = os.environ.get("B2026_SOURCE_BLEND", "").strip()
    if env_path:
        return Path(env_path)
    source = DEFAULT_SINGLE_STATE_TEMPLATE_BY_SITE.get(site)
    if source is None:
        raise ValueError(f"No default single-state template configured for site '{site}'")
    return source


def get_output_blend(site: str, source_blend: Path) -> Path:
    env_path = os.environ.get("B2026_OUTPUT_BLEND", "").strip()
    if env_path:
        return Path(env_path)
    return default_output_path(source_blend)


def get_reference_blend(site: str) -> Path:
    env_path = os.environ.get("B2026_REFERENCE_BLEND", "").strip()
    if env_path:
        return Path(env_path)
    reference = DEFAULT_REFERENCE_BLEND_BY_SITE.get(site)
    if reference is None:
        raise ValueError(f"No default reference blend configured for site '{site}'")
    return reference


def get_timeslice_camera_blend() -> Path:
    env_path = os.environ.get("B2026_TIMESLICE_CAMERA_BLEND", "").strip()
    if env_path:
        return Path(env_path)
    return DEFAULT_TIMESLICE_CAMERA_BLEND


def append_materials(blend_path: Path, material_names: list[str]) -> None:
    needed = [name for name in material_names if bpy.data.materials.get(name) is None]
    if not needed:
        return
    with bpy.data.libraries.load(str(blend_path), link=False) as (data_from, data_to):
        data_to.materials = [name for name in needed if name in data_from.materials]
    for material in data_to.materials:
        if material is not None:
            material.use_fake_user = True


def set_single_material(obj_name: str, material_name: str) -> None:
    obj = bpy.data.objects.get(obj_name)
    material = bpy.data.materials.get(material_name)
    if obj is None or material is None or getattr(obj, "data", None) is None:
        return
    materials = obj.data.materials
    if len(materials) == 0:
        materials.append(material)
    else:
        materials[0] = material


def remove_camera_if_present(name: str) -> None:
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != "CAMERA":
        return
    data = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if data is not None and data.users == 0:
        bpy.data.cameras.remove(data)


def resolve_camera_target_collection(scene: bpy.types.Scene, site: str) -> bpy.types.Collection:
    candidate_names = (
        scene_contract.get_collection_name(site, "cameras"),
        scene_contract.get_collection_name(site, "cameras", legacy=True),
        scene_contract.get_single_state_top_level_name(site, "cameras"),
    )
    for name in candidate_names:
        collection = bpy.data.collections.get(name)
        if collection is not None:
            return collection
    return scene_setup.ensure_root_collection(scene, candidate_names[0])


def ensure_timeslice_cameras(scene: bpy.types.Scene, site: str) -> None:
    spec = TIMESLICE_CAMERA_SPECS.get(site)
    if spec is None:
        return

    source_blend = get_timeslice_camera_blend()
    if not source_blend.exists():
        raise FileNotFoundError(f"Timeslice camera blend not found: {source_blend}")

    for camera_name in spec["camera_names"]:
        remove_camera_if_present(camera_name)

    with bpy.data.libraries.load(str(source_blend), link=False) as (data_from, data_to):
        missing = [name for name in spec["camera_names"] if name not in data_from.objects]
        if missing:
            raise ValueError(f"Missing timeslice camera(s) in {source_blend}: {missing}")
        data_to.objects = list(spec["camera_names"])

    target_collection = resolve_camera_target_collection(scene, site)
    for obj in data_to.objects:
        if obj is None:
            raise ValueError(f"Failed to append a timeslice camera from {source_blend}")
        if target_collection.objects.get(obj.name) is None:
            target_collection.objects.link(obj)
        obj.hide_render = False
        obj.hide_viewport = False

    active_camera = bpy.data.objects.get(spec["active_camera_name"])
    if active_camera is None or active_camera.type != "CAMERA":
        raise ValueError(f"Active timeslice camera '{spec['active_camera_name']}' was not imported")
    scene.camera = active_camera
    print(f"[timeline-template] active_camera={active_camera.name} source={source_blend}")


def prepare_timeline_template(scene: bpy.types.Scene, site: str, reference_blend: Path) -> None:
    scene_setup.ensure_timeline_shell(scene, site)
    scene_setup.ensure_view_layers(
        scene,
        scene_contract.STANDARD_VIEW_LAYERS,
        remove_layers=TIMELINE_ALIAS_VIEW_LAYERS,
    )

    material_names = ["WORLD_AOV", "Envelope"]
    if site == "trimmed-parade":
        material_names.append("Envelope Parade")
    append_materials(reference_blend, material_names)

    contract = scene_contract.SITE_CONTRACTS[site]
    for object_name in contract["world_objects"].values():
        set_single_material(object_name, "WORLD_AOV")

    world = bpy.data.worlds.get("debug_timeslice_world")
    if world is not None:
        scene.world = world

    ensure_timeslice_cameras(scene, site)


def main() -> None:
    site = detect_site()
    scene_name = detect_scene_name(site)
    source_blend = get_source_blend(site)
    output_blend = get_output_blend(site, source_blend)
    reference_blend = get_reference_blend(site)

    bpy.ops.wm.open_mainfile(filepath=str(source_blend))
    scene = bpy.data.scenes.get(scene_name)
    if scene is None:
        raise ValueError(f"Scene '{scene_name}' not found in {source_blend}")

    prepare_timeline_template(scene, site, reference_blend)

    output_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))
    print(f"[timeline-template] site={site} source={source_blend}")
    print(f"[timeline-template] saved={output_blend}")


if __name__ == "__main__":
    main()
