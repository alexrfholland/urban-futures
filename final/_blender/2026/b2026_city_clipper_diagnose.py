"""Diagnose city tree/log instancing and optionally bypass the city tree clipper.

Run this inside the open Blender session.

What it does:
1. Counts evaluated instances for city tree/log point clouds.
2. Ensures `WorldCam` (or the iso camera) has a city clip proxy sized to city bounds.
3. Syncs `City_ClipBox` from that proxy and recounts.
4. If counts are still zero, bypasses the `Clip Box Cull` node in city tree/log GN groups and recounts.

It does not save the blend.
"""

import bpy
from mathutils import Vector


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
CITY_TREE_GROUP_PREFIXES = (
    "tree_city_",
    "log_city_",
)
CITY_XY_PADDING = 10.0
CITY_Z_PADDING = 10.0
DEFAULT_CLIP_BOX_SCALE = (40.0, 40.0, 20.0)
CLIP_NODE_NAME = "Clip Box Cull"


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


def count_city_instances():
    depsgraph = bpy.context.evaluated_depsgraph_get()
    city_targets = {
        obj.name
        for obj in bpy.data.objects
        if obj.name.startswith("TreePositions_city_") or obj.name.startswith("LogPositions_city_")
    }
    counts = {name: 0 for name in city_targets}
    for inst in depsgraph.object_instances:
        parent = getattr(inst, "parent", None)
        if parent is None:
            continue
        parent_name = getattr(getattr(parent, "original", None), "name", None) or getattr(parent, "name", None)
        if parent_name in counts:
            counts[parent_name] += 1
    return counts


def print_counts(title):
    counts = count_city_instances()
    print(title)
    total = 0
    for name in sorted(counts):
        total += counts[name]
        print(f"  {name}: {counts[name]}")
    print(f"  TOTAL: {total}")
    return counts, total


def bypass_clip_node(node_group):
    clip_node = node_group.nodes.get(CLIP_NODE_NAME)
    if clip_node is None:
        return False
    geom_input = clip_node.inputs.get("Geometry")
    geom_output = clip_node.outputs.get("Geometry")
    if geom_input is None or geom_output is None:
        return False

    incoming = [link.from_socket for link in geom_input.links]
    outgoing = [link.to_socket for link in geom_output.links]

    if not incoming:
        return False

    source = incoming[0]
    for link in list(node_group.links):
        if link.to_socket == geom_input or link.from_socket == geom_output:
            node_group.links.remove(link)
    for target in outgoing:
        node_group.links.new(source, target)

    node_group.nodes.remove(clip_node)
    return True


def bypass_city_tree_clipper():
    changed = []
    for node_group in bpy.data.node_groups:
        if any(node_group.name.startswith(prefix) for prefix in CITY_TREE_GROUP_PREFIXES):
            if bypass_clip_node(node_group):
                changed.append(node_group.name)
    return changed


def main():
    scene = bpy.data.scenes["city"]
    live_clip = bpy.data.objects.get("City_ClipBox")
    city_camera_collection = bpy.data.collections.get("City_Camera")
    if live_clip is None or city_camera_collection is None:
        raise RuntimeError("Missing City_ClipBox or City_Camera")

    print_counts("BEFORE")

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

    print(f"ISO CAMERA: {camera.name}")
    print(f"ISO PROXY: {proxy.name}")
    print(f"ISO PROXY LOCATION: {tuple(round(v, 3) for v in proxy.location)}")
    print(f"ISO PROXY SCALE: {tuple(round(v, 3) for v in proxy.scale)}")

    _, total_after_sync = print_counts("AFTER CLIP SYNC")

    if total_after_sync == 0:
        changed = bypass_city_tree_clipper()
        print(f"BYPASSED CLIP NODES: {changed}")
        print_counts("AFTER CLIP BYPASS")
    else:
        print("CITY INSTANCING IS NONZERO AFTER CLIP SYNC; no bypass applied.")


if __name__ == "__main__":
    main()
