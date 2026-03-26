from pathlib import Path
import importlib.util
import sys

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
SCRIPT_DIR = REPO_ROOT / "final/_blender/2026"
SOURCE_BLEND = REPO_ROOT / "data/blender/2026/2026 futures heroes5.blend"
TEMP_BLEND = Path("/tmp/codex_2026_futures_heroes_work.blend")


def parse_paths():
    blend_paths = [Path(arg) for arg in sys.argv if arg.endswith(".blend")]
    source = blend_paths[-1] if blend_paths else SOURCE_BLEND
    temp = TEMP_BLEND
    extra = list(sys.argv)
    if "--" in extra:
        marker = extra.index("--")
        args = extra[marker + 1:]
        if args:
            temp = Path(args[0])
    return source, temp


def load_module(module_name, script_path):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sync_text_block(name, script_path):
    text = bpy.data.texts.get(name)
    if text is None:
        text = bpy.data.texts.new(name)
    text.from_string(script_path.read_text())
    return text


def save_to_temp(temp_path, copy=False):
    bpy.ops.wm.save_as_mainfile(filepath=str(temp_path), copy=copy)
    print(f"SAVED: {temp_path}")


def count_point_vertices(prefix):
    total = 0
    per_object = {}
    for obj in bpy.data.objects:
        if obj.name.startswith(prefix) and hasattr(obj.data, "vertices"):
            count = len(obj.data.vertices)
            per_object[obj.name] = count
            total += count
    return total, per_object


def count_live_instances(scene_name):
    scene = bpy.data.scenes.get(scene_name)
    if scene is None:
        raise ValueError(f"Scene '{scene_name}' not found.")

    view_layer = scene.view_layers.get("pathway_state")
    if view_layer is None:
        raise ValueError("View layer 'pathway_state' not found.")

    tree_prefixes = ("TreePositions_city_",)
    log_prefixes = ("LogPositions_city_",)

    tree_count = 0
    log_count = 0
    tree_emitters = set()
    log_emitters = set()

    with bpy.context.temp_override(scene=scene, view_layer=view_layer):
        depsgraph = bpy.context.evaluated_depsgraph_get()
        depsgraph.update()

        for instance in depsgraph.object_instances:
            parent = getattr(instance, "parent", None)
            if parent is None:
                continue
            original_parent = getattr(parent, "original", parent)
            parent_name = original_parent.name

            if any(parent_name.startswith(prefix) for prefix in tree_prefixes):
                tree_count += 1
                tree_emitters.add(parent_name)
            elif any(parent_name.startswith(prefix) for prefix in log_prefixes):
                log_count += 1
                log_emitters.add(parent_name)

    return {
        "tree_instances": tree_count,
        "log_instances": log_count,
        "tree_emitters": sorted(tree_emitters),
        "log_emitters": sorted(log_emitters),
    }


def report_clip_groups():
    names = []
    for node_group in bpy.data.node_groups:
        if node_group.name.startswith("tree_city_") or node_group.name.startswith("log_city_"):
            for node in node_group.nodes:
                if node.type == "GROUP" and getattr(node.node_tree, "name", "") == "City ClipBox Cull Trees":
                    names.append(node_group.name)
                    break
    return sorted(names)


def main():
    source_blend, temp_blend = parse_paths()
    print(f"SOURCE_BLEND: {source_blend}")
    print(f"TEMP_BLEND: {temp_blend}")

    material_module = load_module(
        "b2026_make_material_binaries",
        SCRIPT_DIR / "b2026_make_material_binaries.py",
    )
    material_summary = material_module.main()
    save_to_temp(temp_blend, copy=False)
    print(f"STEP 1 COMPLETE: {material_summary}")

    sync_text_block("Instancer", SCRIPT_DIR / "b2026_instancer.py")
    sync_text_block("clipbox_setup", SCRIPT_DIR / "b2026_clipbox_setup.py")
    sync_text_block("camera_clipboxes", SCRIPT_DIR / "b2026_camera_clipboxes.py")
    print("Synced embedded text blocks: Instancer, clipbox_setup, camera_clipboxes")

    instancer_module = load_module(
        "b2026_instancer_temp_run",
        SCRIPT_DIR / "b2026_instancer.py",
    )
    instancer_module.TARGET_SCENE_NAME = "city"
    instancer_module.AUTO_SITE_FROM_SCENE = True
    instancer_module.SCENARIO = "trending"
    instancer_module.YEAR = 180
    instancer_module.main()

    point_tree_total, point_tree_objects = count_point_vertices("TreePositions_city_")
    point_log_total, point_log_objects = count_point_vertices("LogPositions_city_")
    live_counts = count_live_instances("city")
    clip_groups = report_clip_groups()
    print(f"CITY POINT TREE TOTAL: {point_tree_total}")
    print(f"CITY POINT LOG TOTAL: {point_log_total}")
    print(f"CITY POINT TREE OBJECTS: {point_tree_objects}")
    print(f"CITY POINT LOG OBJECTS: {point_log_objects}")
    print(f"CITY LIVE TREE INSTANCES: {live_counts['tree_instances']}")
    print(f"CITY LIVE LOG INSTANCES: {live_counts['log_instances']}")
    print(f"CITY TREE EMITTERS SEEN IN DEPSGRAPH: {live_counts['tree_emitters']}")
    print(f"CITY LOG EMITTERS SEEN IN DEPSGRAPH: {live_counts['log_emitters']}")
    print(f"CITY CLIP GROUPS PATCHED: {clip_groups}")
    save_to_temp(temp_blend, copy=False)
    print("STEP 2 COMPLETE")

    compositor_module = load_module(
        "patch_city_resource_compositor",
        SCRIPT_DIR / "patch_city_resource_compositor.py",
    )
    compositor_module.main()
    save_to_temp(temp_blend, copy=False)
    print("STEP 3 COMPLETE")


if __name__ == "__main__":
    main()
