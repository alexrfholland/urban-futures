from __future__ import annotations

import os
from pathlib import Path

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
COMPOSITOR_ROOT = REPO_ROOT / "_futureSim_refactored" / "blender" / "compositor"
CANONICAL_ROOT = COMPOSITOR_ROOT / "canonical_templates"

SOURCE_BLEND = Path(
    os.environ.get(
        "EDGE_LAB_SOURCE_BLEND",
        CANONICAL_ROOT / "edge_lab_final_template_safe_rebuild_20260405.blend",
    )
)

WORKFLOWS = {
    "ao": {
        "target_frame": "AO::FamilyFrame",
        "output_names": {"AO::Outputs"},
        "extra_keep_names": set(),
        "output_blend": CANONICAL_ROOT / "compositor_ao.blend",
    },
    "normals": {
        "target_frame": "Normals::FamilyFrame",
        "output_names": {"Normals::Outputs"},
        "extra_keep_names": set(),
        "output_blend": CANONICAL_ROOT / "compositor_normals.blend",
    },
    "resources": {
        "target_frame": "Resources::FamilyFrame",
        "output_names": {"Resources::Outputs"},
        "extra_keep_names": set(),
        "output_blend": CANONICAL_ROOT / "compositor_resources.blend",
    },
    "bioenvelope": {
        "target_frame": "Current BioEnvelope :: Frame",
        "output_names": {"Current BioEnvelope ::Outputs"},
        "extra_keep_names": set(),
        "output_blend": CANONICAL_ROOT / "compositor_bioenvelope.blend",
    },
    "base": {
        "target_frame": "Current Base Outputs :: Frame",
        "output_names": set(),
        "extra_keep_names": set(),
        "output_blend": CANONICAL_ROOT / "compositor_base.blend",
    },
    "shading": {
        "target_frame": "Current Shading :: Frame",
        "output_names": {"Current Shading ::Outputs"},
        "extra_keep_names": {
            "AO::EXR Pathway",
            "AO::EXR Priority",
            "AO::EXR Existing",
            "Current BioEnvelope :: EXR BioEnvelope",
            "Current BioEnvelope :: EXR Trending",
        },
        "output_blend": CANONICAL_ROOT / "compositor_shading.blend",
    },
    "depth_outliner": {
        "target_frame": "DepthOutliner::FamilyFrame",
        "output_names": {"DepthOutliner::Outputs"},
        "extra_keep_names": set(),
        "output_blend": CANONICAL_ROOT / "compositor_depth_outliner.blend",
    },
    "mist": {
        "target_frame": "MistOutlines::FamilyFrame",
        "output_names": {"MistOutlines::Outputs"},
        "extra_keep_names": set(),
        "output_blend": CANONICAL_ROOT / "compositor_mist.blend",
    },
}


def keep_due_to_frame(node: bpy.types.Node, target_frame: bpy.types.Node) -> bool:
    current = node
    while getattr(current, "parent", None) is not None:
        current = current.parent
        if current == target_frame:
            return True
    return False


def strip_scene_to_workflow(
    scene: bpy.types.Scene,
    target_frame_name: str,
    output_names: set[str],
    extra_keep_names: set[str],
) -> None:
    node_tree = scene.node_tree
    target_frame = node_tree.nodes.get(target_frame_name)
    if target_frame is None or target_frame.bl_idname != "NodeFrame":
        raise ValueError(f"Missing target frame: {target_frame_name}")

    keep_names = set(output_names) | set(extra_keep_names)
    keep_types = {"CompositorNodeComposite", "CompositorNodeViewer"}

    for node in list(node_tree.nodes):
        keep = False
        if node == target_frame:
            keep = True
        elif node.name in keep_names:
            keep = True
        elif node.bl_idname in keep_types:
            keep = True
        elif keep_due_to_frame(node, target_frame):
            keep = True

        if not keep:
            node_tree.nodes.remove(node)


def prune_scenes(target_scene_name: str = "Current") -> None:
    for scene in list(bpy.data.scenes):
        if scene.name != target_scene_name:
            bpy.data.scenes.remove(scene)


def save_workflow_blend(
    source_blend: Path,
    target_frame_name: str,
    output_names: set[str],
    extra_keep_names: set[str],
    output_blend: Path,
) -> None:
    bpy.ops.wm.open_mainfile(filepath=str(source_blend))
    scene = bpy.data.scenes.get("Current")
    if scene is None or scene.node_tree is None:
        raise ValueError("Current scene not found")

    strip_scene_to_workflow(scene, target_frame_name, output_names, extra_keep_names)
    prune_scenes("Current")

    try:
        bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    except Exception:
        pass

    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))
    print(f"[split_template_native_workflows] Wrote {output_blend}")


def main() -> None:
    if not SOURCE_BLEND.exists():
        raise FileNotFoundError(SOURCE_BLEND)

    for workflow in WORKFLOWS.values():
        save_workflow_blend(
            SOURCE_BLEND,
            workflow["target_frame"],
            workflow["output_names"],
            workflow["extra_keep_names"],
            workflow["output_blend"],
        )


if __name__ == "__main__":
    main()
