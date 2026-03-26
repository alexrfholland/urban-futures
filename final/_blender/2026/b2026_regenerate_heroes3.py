import bpy
import importlib.util
import json
import sys
import hashlib
from pathlib import Path


SOURCE_BLEND = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/2026 futures heroes2.blend"
)
OUTPUT_BLEND = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/2026 futures heroes3.blend"
)
VERIFICATION_DIR = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/heroes3_verification"
)

INSTANCER_PATH = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/_blender/2026/b2026_instancer.py"
)
CLIPBOX_PATH = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/_blender/2026/b2026_clipbox_setup.py"
)

SCENARIO_RUNS = (
    ("city", "positive"),
    ("city", "trending"),
    ("parade", "positive"),
    ("parade", "trending"),
)


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def sync_text_block(text_name: str, source_path: Path):
    source_text = source_path.read_text(encoding="utf-8")
    text = bpy.data.texts.get(text_name)
    if text is None:
        text = bpy.data.texts.new(text_name)
    text.clear()
    text.write(source_text)
    print(f"Synced text block: {text_name}")


def configure_scene_render(scene_name: str):
    scene = bpy.data.scenes[scene_name]
    scene.use_nodes = True
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = 3840
    scene.render.resolution_y = 2160
    scene.render.resolution_percentage = 100
    scene.cycles.samples = 512
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = f"//renders/{scene_name}.png"
    print(
        f"Configured render settings for {scene_name}: "
        f"{scene.render.resolution_x}x{scene.render.resolution_y} "
        f"{scene.render.image_settings.file_format}"
    )


def ensure_camera_clip_ranges():
    for obj in bpy.data.objects:
        if obj.type != "CAMERA":
            continue
        obj.data.clip_start = min(obj.data.clip_start, 0.1)
        obj.data.clip_end = max(obj.data.clip_end, 5000.0)


def get_site_spec(clipmod, label: str):
    for spec in clipmod.SITE_SPECS:
        if spec["label"] == label:
            return spec
    raise ValueError(f"No site spec for {label}")


def refresh_clip_for_site(clipmod, scene_name: str):
    label = "city" if scene_name == "city" else "parade"
    spec = get_site_spec(clipmod, label)

    if label == "city":
        clip_box = clipmod.ensure_clip_box(spec)
    else:
        clip_box = bpy.data.objects.get(spec["clip_box_name"])
        if clip_box is None:
            clip_box = clipmod.ensure_clip_box(spec)
        else:
            clipmod.move_object_to_collection(clip_box, spec["collection_name"])

    point_group, mesh_group, tree_group = clipmod.ensure_clip_groups(spec, clip_box)
    patched_points, patched_meshes, patched_trees = clipmod.patch_scene_groups(
        spec, point_group, mesh_group, tree_group
    )
    print(
        f"Clip refresh for {scene_name}: "
        f"points={patched_points}, meshes={patched_meshes}, trees={patched_trees}"
    )


def set_active_year_collection(scene_name: str, scenario: str):
    scene = bpy.data.scenes[scene_name]
    target_name = f"Year_{'city' if scene_name == 'city' else 'trimmed-parade'}_180_{scenario}"

    for collection in scene.collection.children:
        if not collection.name.startswith("Year_"):
            continue
        is_target = collection.name == target_name
        collection.hide_viewport = not is_target
        collection.hide_render = not is_target

    print(f"Visible year collection in {scene_name}: {target_name}")


def get_layer_collection(layer_collection, collection_name: str):
    if layer_collection.collection.name == collection_name:
        return layer_collection
    for child in layer_collection.children:
        found = get_layer_collection(child, collection_name)
        if found is not None:
            return found
    return None


def set_active_year_collection_on_view_layers(scene_name: str, scenario: str):
    scene = bpy.data.scenes[scene_name]
    target_name = f"Year_{'city' if scene_name == 'city' else 'trimmed-parade'}_180_{scenario}"

    for view_layer in scene.view_layers:
        for collection in scene.collection.children:
            if not collection.name.startswith("Year_"):
                continue
            layer_collection = get_layer_collection(view_layer.layer_collection, collection.name)
            if layer_collection is not None:
                if view_layer.name == "existing_condition":
                    layer_collection.exclude = True
                else:
                    layer_collection.exclude = collection.name != target_name


def set_envelope_render_state(scene_name: str, enabled: bool):
    collection_name = "city_envelope" if scene_name == "city" else "Parade_envelope"
    collection = bpy.data.collections.get(collection_name)
    if collection is not None:
        collection.hide_render = not enabled
        collection.hide_viewport = not enabled

    scene = bpy.data.scenes[scene_name]
    for view_layer in scene.view_layers:
        layer_collection = get_layer_collection(view_layer.layer_collection, collection_name)
        if layer_collection is not None:
            layer_collection.exclude = not enabled if view_layer.name == "pathway_state" else True


def collect_run_summary(instmod, scene_name: str, scenario: str):
    site = "city" if scene_name == "city" else "trimmed-parade"
    run_id = f"{site}_180_{scenario}"
    tree_name = f"TreePositions_{run_id}"
    log_name = f"LogPositions_{run_id}"
    tree_collection_name = f"tree_{run_id}_plyModels"
    log_collection_name = f"log_{run_id}_plyModels"

    tree_obj = bpy.data.objects.get(tree_name)
    log_obj = bpy.data.objects.get(log_name)
    tree_collection = bpy.data.collections.get(tree_collection_name)
    log_collection = bpy.data.collections.get(log_collection_name)

    summary = {
        "scene": scene_name,
        "scenario": scenario,
        "tree_object": tree_name,
        "tree_points": len(tree_obj.data.vertices) if tree_obj and tree_obj.type == "MESH" else 0,
        "tree_modifier_group": (
            tree_obj.modifiers[0].node_group.name
            if tree_obj and tree_obj.modifiers and tree_obj.modifiers[0].type == "NODES" and tree_obj.modifiers[0].node_group
            else None
        ),
        "tree_models": len(tree_collection.objects) if tree_collection else 0,
        "log_object": log_name,
        "log_points": len(log_obj.data.vertices) if log_obj and log_obj.type == "MESH" else 0,
        "log_modifier_group": (
            log_obj.modifiers[0].node_group.name
            if log_obj and log_obj.modifiers and log_obj.modifiers[0].type == "NODES" and log_obj.modifiers[0].node_group
            else None
        ),
        "log_models": len(log_collection.objects) if log_collection else 0,
    }
    print("RUN_SUMMARY", json.dumps(summary, sort_keys=True))
    return summary


def render_scene(scene_name: str, scenario: str):
    scene = bpy.data.scenes[scene_name]
    output_path = VERIFICATION_DIR / f"{scene_name}_{scenario}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.use_nodes = True
    for view_layer in scene.view_layers:
        view_layer.use = True
    set_envelope_render_state(scene_name, enabled=scenario != "trending")
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True, scene=scene.name)
    if not output_path.exists():
        raise RuntimeError(f"Expected render was not written: {output_path}")
    image_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(f"Rendered {scene_name} {scenario} to {output_path}")
    return output_path, image_hash


def main():
    if Path(bpy.data.filepath) != SOURCE_BLEND:
        print(f"Working from open file: {bpy.data.filepath}")

    sync_text_block("Instancer", INSTANCER_PATH)
    sync_text_block("clipbox_setup", CLIPBOX_PATH)

    configure_scene_render("city")
    configure_scene_render("parade")
    ensure_camera_clip_ranges()

    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_BLEND))
    print(f"Saved working copy to {OUTPUT_BLEND}")

    instmod = load_module("b2026_instancer_live", INSTANCER_PATH)
    clipmod = load_module("b2026_clipbox_live", CLIPBOX_PATH)

    summaries = []
    renders = []
    render_hashes = {}

    for scene_name, scenario in SCENARIO_RUNS:
        print("=" * 80)
        print(f"REGENERATING {scene_name} {scenario}")
        instmod.TARGET_SCENE_NAME = scene_name
        instmod.AUTO_SITE_FROM_SCENE = True
        instmod.SCENARIO = scenario
        instmod.YEAR = 180
        instmod.main()

        refresh_clip_for_site(clipmod, scene_name)
        set_active_year_collection(scene_name, scenario)
        set_active_year_collection_on_view_layers(scene_name, scenario)

        summary = collect_run_summary(instmod, scene_name, scenario)
        if summary["tree_points"] <= 0:
            raise RuntimeError(f"No tree points generated for {scene_name} {scenario}")
        if summary["tree_models"] <= 0:
            raise RuntimeError(f"No tree models imported for {scene_name} {scenario}")
        summaries.append(summary)

        render_path, render_hash = render_scene(scene_name, scenario)
        renders.append(str(render_path))
        render_hashes[f"{scene_name}:{scenario}"] = render_hash
        bpy.ops.wm.save_mainfile()

    for scene_name in ("city", "parade"):
        positive_hash = render_hashes.get(f"{scene_name}:positive")
        trending_hash = render_hashes.get(f"{scene_name}:trending")
        if positive_hash and trending_hash and positive_hash == trending_hash:
            raise RuntimeError(f"{scene_name} positive and trending renders are identical")

    summary_path = VERIFICATION_DIR / "heroes3_summary.json"
    summary_path.write_text(
        json.dumps(
            {"summaries": summaries, "renders": renders, "render_hashes": render_hashes},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote summary to {summary_path}")
    bpy.ops.wm.save_mainfile()


if __name__ == "__main__":
    main()
