"""City-only in-session helper.

Run this inside the currently open Blender session.
It does not save the blend.
"""

import bpy
from mathutils import Vector
from pathlib import Path


CITY_POINT_OBJECT_PREFIXES = (
    "city_buildings",
    "city_highResRoad",
)
CITY_GROUND_OBJECT_PREFIXES = (
    "city_1_envelope_scenarioYR",
    "city_1_envelopes_scenarioYR",
)
CITY_TREE_OBJECT_PREFIXES = (
    "TreePositions_city_",
    "LogPositions_city_",
)
CITY_XY_PADDING = 10.0
CITY_Z_PADDING = 10.0
DEFAULT_CLIP_BOX_SCALE = (40.0, 40.0, 20.0)

TEMP_NODE_PREFIX = "CodexCityIsoVerify::"


def iter_target_objects(prefixes):
    for obj in bpy.data.objects:
        if obj.name.endswith("_cubes"):
            continue
        if any(obj.name.startswith(prefix) for prefix in prefixes):
            yield obj


def compute_bbox(prefixes):
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    found = False
    for obj in iter_target_objects(prefixes):
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            for i in range(3):
                mins[i] = min(mins[i], world[i])
                maxs[i] = max(maxs[i], world[i])
        found = True
    if not found:
        return None
    return tuple(mins), tuple(maxs)


def find_city_iso_camera():
    preferred_collection = bpy.data.collections.get("Test_iso_full")
    if preferred_collection is not None:
        for obj in preferred_collection.all_objects:
            if obj.type == "CAMERA":
                return obj

    cam = bpy.data.objects.get("WorldCam")
    if cam and cam.type == "CAMERA":
        return cam

    city_camera_collection = bpy.data.collections.get("City_Camera")
    if city_camera_collection is not None:
        for obj in city_camera_collection.all_objects:
            if obj.type == "CAMERA" and "iso" in obj.name.lower():
                return obj
        for obj in city_camera_collection.all_objects:
            if obj.type == "CAMERA":
                return obj

    raise RuntimeError("Could not find a city iso camera")


def ensure_subcollection(parent_collection, camera):
    name = f"CityCamera__{camera.name}"
    sub = bpy.data.collections.get(name)
    if sub is None:
        sub = bpy.data.collections.new(name)
    if sub not in parent_collection.children[:]:
        parent_collection.children.link(sub)
    if camera.name not in sub.objects:
        sub.objects.link(camera)
    if camera.name in parent_collection.objects:
        parent_collection.objects.unlink(camera)
    return sub


def ensure_proxy(live_clip, subcollection, camera):
    proxy_name = f"CityClipProxy__{camera.name}"
    proxy = bpy.data.objects.get(proxy_name)
    if proxy is None:
        mesh = live_clip.data.copy()
        proxy = bpy.data.objects.new(proxy_name, mesh)
        proxy.display_type = "WIRE"
        proxy.hide_render = True
    if proxy.name not in subcollection.objects:
        subcollection.objects.link(proxy)
    for coll in list(proxy.users_collection):
        if coll != subcollection and proxy.name in coll.objects:
            coll.objects.unlink(proxy)
    return proxy


def compute_city_clip_transform():
    prefixes = CITY_POINT_OBJECT_PREFIXES + CITY_GROUND_OBJECT_PREFIXES + CITY_TREE_OBJECT_PREFIXES
    bbox = compute_bbox(prefixes)
    if bbox is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), DEFAULT_CLIP_BOX_SCALE

    bbox_min, bbox_max = bbox
    location = (
        (bbox_min[0] + bbox_max[0]) / 2.0,
        (bbox_min[1] + bbox_max[1]) / 2.0,
        (bbox_min[2] + bbox_max[2]) / 2.0,
    )
    width_x = bbox_max[0] - bbox_min[0]
    width_y = bbox_max[1] - bbox_min[1]
    height = bbox_max[2] - bbox_min[2]
    scale = (
        max(DEFAULT_CLIP_BOX_SCALE[0], width_x / 2.0 + CITY_XY_PADDING),
        max(DEFAULT_CLIP_BOX_SCALE[1], width_y / 2.0 + CITY_XY_PADDING),
        max(DEFAULT_CLIP_BOX_SCALE[2], height / 2.0 + CITY_Z_PADDING),
    )
    return location, (0.0, 0.0, 0.0), scale


def sync_live_clip(proxy):
    live_clip = bpy.data.objects["City_ClipBox"]
    live_clip.location = proxy.location.copy()
    live_clip.rotation_euler = proxy.rotation_euler.copy()
    live_clip.scale = proxy.scale.copy()
    live_clip.hide_render = True
    live_clip.hide_viewport = False


def add_verify_nodes(scene):
    tree = scene.node_tree
    for node in list(tree.nodes):
        if node.name.startswith(TEMP_NODE_PREFIX):
            tree.nodes.remove(node)

    rlayers = tree.nodes.new("CompositorNodeRLayers")
    rlayers.name = f"{TEMP_NODE_PREFIX}RenderLayers"
    rlayers.scene = scene
    rlayers.layer = "pathway_state"
    rlayers.location = (20000, 0)

    ramp = tree.nodes.new("CompositorNodeValToRGB")
    ramp.name = f"{TEMP_NODE_PREFIX}Ramp"
    ramp.location = (20250, 0)
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].position = 0.5
    ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    ramp.color_ramp.elements[1].position = 0.5
    ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    file_out = tree.nodes.new("CompositorNodeOutputFile")
    file_out.name = f"{TEMP_NODE_PREFIX}FileOutput"
    file_out.location = (20500, 0)
    file_out.base_path = "/tmp/codex_city_iso_verify"
    file_out.format.file_format = "PNG"
    file_out.format.color_mode = "RGB"
    file_out.format.color_depth = "8"
    file_out.file_slots[0].path = "city_iso_tree_mask_"

    tree.links.new(rlayers.outputs["resource_tree_mask"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Image"], file_out.inputs[0])
    return file_out


def render_and_measure(scene):
    old_res_x = scene.render.resolution_x
    old_res_y = scene.render.resolution_y
    old_res_pct = scene.render.resolution_percentage
    old_samples = getattr(scene.cycles, "samples", None) if scene.render.engine == "CYCLES" else None

    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.resolution_percentage = 100
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = min(scene.cycles.samples, 32)

    verify_dir = Path("/tmp/codex_city_iso_verify")
    verify_dir.mkdir(parents=True, exist_ok=True)
    expected_path = verify_dir / "city_iso_tree_mask_0001.png"
    if expected_path.exists():
        expected_path.unlink()

    add_verify_nodes(scene)
    bpy.ops.render.render(scene=scene.name, write_still=False)

    scene.render.resolution_x = old_res_x
    scene.render.resolution_y = old_res_y
    scene.render.resolution_percentage = old_res_pct
    if old_samples is not None:
        scene.cycles.samples = old_samples

    if not expected_path.exists():
        raise RuntimeError(f"Verification render not written: {expected_path}")

    img = bpy.data.images.load(str(expected_path), check_existing=False)
    pixels = list(img.pixels)
    bpy.data.images.remove(img)
    nonzero = sum(1 for i in range(0, len(pixels), 4) if pixels[i] > 0.01 or pixels[i + 1] > 0.01 or pixels[i + 2] > 0.01)
    total = len(pixels) // 4
    return expected_path, nonzero, total


def main():
    scene = bpy.data.scenes["city"]
    live_clip = bpy.data.objects.get("City_ClipBox")
    city_camera_collection = bpy.data.collections.get("City_Camera")
    if live_clip is None or city_camera_collection is None:
        raise RuntimeError("Missing City_ClipBox or City_Camera")

    camera = find_city_iso_camera()
    sub = ensure_subcollection(city_camera_collection, camera)
    proxy = ensure_proxy(live_clip, sub, camera)

    location, rotation, scale = compute_city_clip_transform()
    proxy.location = location
    proxy.rotation_euler = rotation
    proxy.scale = scale
    proxy.hide_render = True
    proxy.display_type = "WIRE"

    camera["clip_proxy_object"] = proxy.name
    scene.camera = camera
    sync_live_clip(proxy)

    path, nonzero, total = render_and_measure(scene)

    print(f"CITY_ISO_CAMERA={camera.name}")
    print(f"CITY_ISO_PROXY={proxy.name}")
    print(f"CITY_ISO_PROXY_LOCATION={tuple(round(v, 3) for v in proxy.location)}")
    print(f"CITY_ISO_PROXY_SCALE={tuple(round(v, 3) for v in proxy.scale)}")
    print(f"CITY_TREE_MASK_PATH={path}")
    print(f"CITY_TREE_MASK_NONZERO={nonzero}")
    print(f"CITY_TREE_MASK_TOTAL={total}")
    print(f"CITY_TREE_MASK_RATIO={nonzero / total if total else 0.0:.6f}")

if __name__ == "__main__":
    main()
