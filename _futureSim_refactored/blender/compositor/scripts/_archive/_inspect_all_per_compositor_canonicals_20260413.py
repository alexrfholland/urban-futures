"""One-shot inspector: dump scene, EXR image nodes, and File Output nodes
for every per-compositor canonical blend. Used to plan the fast-runner port.

Run with:
    blender --background --factory-startup --python <this script>
"""
from __future__ import annotations

from pathlib import Path

import bpy

REPO_ROOT = Path(__file__).resolve().parents[4]
CANONICAL_ROOT = (
    REPO_ROOT / "_futureSim_refactored" / "blender" / "compositor"
    / "canonical_templates"
)

BLENDS = [
    "compositor_ao.blend",
    "compositor_base.blend",
    "compositor_bioenvelope.blend",
    "compositor_depth_outliner.blend",
    "compositor_intervention_int.blend",
    "compositor_mist.blend",
    "compositor_normals.blend",
    "compositor_proposal_masks.blend",
    "compositor_resources.blend",
    "compositor_shading.blend",
    "compositor_sizes.blend",
    "compositor_sizes_single_input.blend",
    "proposal_and_interventions.blend",
    "proposal_colored_depth_outlines.blend",
    "proposal_only_layers.blend",
    "proposal_outline_layers.blend",
    "size_outline_layers.blend",
]


def main() -> None:
    for blend_name in BLENDS:
        blend_path = CANONICAL_ROOT / blend_name
        if not blend_path.exists():
            print(f"\n=== {blend_name}: MISSING ===")
            continue
        print(f"\n=== {blend_name} ===")
        bpy.ops.wm.open_mainfile(filepath=str(blend_path))
        for scene in bpy.data.scenes:
            nt = scene.node_tree
            if nt is None:
                continue
            image_nodes = [n for n in nt.nodes if n.bl_idname == "CompositorNodeImage"]
            output_nodes = [n for n in nt.nodes if n.bl_idname == "CompositorNodeOutputFile"]
            reroute_labels = [n.label for n in nt.nodes if n.bl_idname == "NodeReroute" and n.label]
            print(f"  SCENE {scene.name!r} use_nodes={scene.use_nodes}")
            print(f"    IMAGE_NODES ({len(image_nodes)}):")
            for n in sorted(image_nodes, key=lambda x: x.name):
                fp = n.image.filepath if n.image is not None else "<no image>"
                print(f"      - {n.name!r}  image={Path(fp).name if fp else '<no image>'}")
            print(f"    FILE_OUTPUT ({len(output_nodes)}):")
            for n in sorted(output_nodes, key=lambda x: x.name):
                slot_paths = [s.path for s in n.file_slots]
                print(f"      - {n.name!r}  slots={slot_paths}")
            if reroute_labels:
                print(f"    LABELED_REROUTES ({len(reroute_labels)}): {reroute_labels}")


if __name__ == "__main__":
    main()
