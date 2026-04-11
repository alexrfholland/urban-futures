import importlib.util
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import b2026_timeline_scene_contract as scene_contract
import b2026_timeline_runtime_flags as runtime_flags

BASE_CAMERA_NAME = "paraview_camera"
MIN_CLIP_START = 0.1
MAX_CLIP_END = 5000.0

PARAVIEW_VIEWS = {
    "city": {
        "scene_name": "city",
        "collection_name": scene_contract.get_collection_name("city", "cameras", legacy=True),
        "view_name": "City",
        "camera_name": f"{BASE_CAMERA_NAME}_city",
        "position": (827.5660735984109, 49.13287564742965, 880.7059951185498),
        "focal_point": (288.24903018994945, -22.32975691174787, 333.8400148551478),
        "view_up": (-0.7115782188004148, -0.006298457386287757, 0.702578656068758),
        "view_angle_degrees": 28.171793383633197,
    },
    "parade": {
        "scene_name": "parade",
        "collection_name": scene_contract.get_collection_name("trimmed-parade", "cameras", legacy=True),
        "view_name": "parade",
        "camera_name": f"{BASE_CAMERA_NAME}_parade",
        "position": (-710.5998866124906, 155.04839940955577, 780.0399301944501),
        "focal_point": (52.83318934160216, 109.85652084191102, 56.99999999999977),
        "view_up": (0.687286571444865, -0.010133360187298351, 0.7263156915025839),
        "view_angle_degrees": 30.0,
    },
    "uni": {
        "scene_name": "uni",
        "collection_name": scene_contract.get_collection_name("uni", "cameras", legacy=True),
        "view_name": "uni",
        "camera_name": f"{BASE_CAMERA_NAME}_uni",
        "position": (-76.76185820395062, -879.4925282617797, 863.1531793423247),
        "focal_point": (-13.385258127015193, 44.270460390127, 63.18321407013855),
        "view_up": (-0.0380557896659148, 0.6556550948655048, 0.7541008907631722),
        "view_angle_degrees": 30.0,
    },
}


def load_local_module(module_name: str, filename: str):
    file_path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def ensure_camera_clip_ranges() -> None:
    for obj in bpy.data.objects:
        if obj.type != "CAMERA":
            continue
        obj.data.clip_start = min(obj.data.clip_start, MIN_CLIP_START)
        obj.data.clip_end = max(obj.data.clip_end, MAX_CLIP_END)


def build_camera_rotation(position, focal_point, view_up) -> Matrix:
    position_vec = Vector(position)
    focal_vec = Vector(focal_point)
    up_hint = Vector(view_up).normalized()

    back_axis = (position_vec - focal_vec).normalized()
    right_axis = up_hint.cross(back_axis)
    if right_axis.length < 1.0e-8:
        raise ValueError("ParaView view_up vector is collinear with the camera direction")
    right_axis.normalize()
    up_axis = back_axis.cross(right_axis).normalized()

    return Matrix((right_axis, up_axis, back_axis)).transposed()


def ensure_camera(spec: dict) -> bpy.types.Object:
    scene = bpy.data.scenes.get(spec["scene_name"])
    if scene is None:
        raise ValueError(f"Scene '{spec['scene_name']}' was not found")

    collection = bpy.data.collections.get(spec["collection_name"])
    if collection is None:
        raise ValueError(f"Collection '{spec['collection_name']}' was not found")

    camera_obj = bpy.data.objects.get(spec["camera_name"])
    if camera_obj is None or camera_obj.type != "CAMERA":
        camera_data = bpy.data.cameras.new(spec["camera_name"])
        camera_obj = bpy.data.objects.new(spec["camera_name"], camera_data)

    if not any(existing.name == collection.name for existing in camera_obj.users_collection):
        collection.objects.link(camera_obj)

    rotation = build_camera_rotation(
        spec["position"],
        spec["focal_point"],
        spec["view_up"],
    )
    camera_obj.matrix_world = Matrix.Translation(Vector(spec["position"])) @ rotation.to_4x4()

    camera_obj.data.type = "PERSP"
    camera_obj.data.sensor_fit = "VERTICAL"
    camera_obj.data.angle_y = math.radians(spec["view_angle_degrees"])
    camera_obj.data.clip_start = min(camera_obj.data.clip_start, MIN_CLIP_START)
    camera_obj.data.clip_end = max(camera_obj.data.clip_end, MAX_CLIP_END)
    camera_obj.data.display_size = 25.0

    camera_obj["paraview_view_name"] = spec["view_name"]
    camera_obj["paraview_base_name"] = BASE_CAMERA_NAME
    camera_obj["paraview_source_doc"] = "final/blender/info/paraview-views.md"

    scene.camera = camera_obj
    return camera_obj


def main() -> None:
    print(f"Working blend: {bpy.data.filepath}")
    ensure_camera_clip_ranges()

    created = {}
    for key, spec in PARAVIEW_VIEWS.items():
        camera = ensure_camera(spec)
        created[key] = camera.name
        print(
            f"Configured {key} camera '{camera.name}' at "
            f"{tuple(round(v, 3) for v in camera.location)}"
        )

    if runtime_flags.any_clipbox_scripts_enabled():
        clipbox_setup = load_local_module(
            "b2026_clipbox_setup_paraview_runtime",
            "b2026_timeline_clipbox_setup.py",
        )
        clipbox_setup.main()
        print("Ran b2026_timeline_clipbox_setup.py")

        camera_clipboxes = load_local_module(
            "b2026_camera_clipboxes_paraview_runtime",
            "b2026_timeline_camera_clipboxes.py",
        )
        camera_clipboxes.register()
        if hasattr(camera_clipboxes, "sync_all_scene_clipboxes"):
            camera_clipboxes.sync_all_scene_clipboxes()
        print("Ran b2026_timeline_camera_clipboxes.py")
    else:
        print("Skipped clip box follow-up scripts (disabled by runtime flags)")

    for scene_name, camera_name in created.items():
        scene = bpy.data.scenes.get(scene_name)
        if scene is None:
            continue
        proxy_name = scene.camera.get("clip_proxy_object") if scene.camera else None
        print(
            f"SCENE_CAMERA scene={scene.name} camera={scene.camera.name if scene.camera else None} "
            f"proxy={proxy_name}"
        )

    bpy.ops.wm.save_mainfile()
    print("Saved blend with ParaView cameras")


if __name__ == "__main__":
    main()
