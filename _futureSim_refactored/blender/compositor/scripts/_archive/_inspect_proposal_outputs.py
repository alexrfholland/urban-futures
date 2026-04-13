#!/usr/bin/env python3
"""
Inspect compositor blend files to map proposal outputs.
Examines canonical blends and checkpoint blends for output node names and EXR input nodes.
"""

import sys
from pathlib import Path

# Must be run via Blender
import bpy


def inspect_blend(blend_path: Path) -> dict:
    """Open a blend and extract output/input node metadata."""
    if not blend_path.exists():
        return {"error": f"File not found: {blend_path}"}
    
    try:
        bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    except Exception as e:
        return {"error": f"Failed to open {blend_path}: {e}"}
    
    result = {
        "path": str(blend_path),
        "scenes": {}
    }
    
    for scene in bpy.data.scenes:
        if scene.node_tree is None:
            continue
        
        node_tree = scene.node_tree
        output_nodes = []
        input_nodes = []
        
        for node in node_tree.nodes:
            # Capture all CompositorNodeOutputFile nodes (file outputs)
            if node.bl_idname == "CompositorNodeOutputFile":
                file_slots = []
                for slot in node.file_slots:
                    file_slots.append({
                        "name": slot.name,
                        "path": slot.path
                    })
                output_nodes.append({
                    "name": node.name,
                    "type": "OutputFile",
                    "file_slots": file_slots
                })
            
            # Capture Image input nodes (EXR inputs)
            elif node.bl_idname == "CompositorNodeImage":
                output_nodes.append({
                    "name": node.name,
                    "type": "ImageInput",
                    "image": node.image.name if node.image else None
                })
        
        if output_nodes or input_nodes:
            result["scenes"][scene.name] = {
                "output_file_nodes": [n for n in output_nodes if n["type"] == "OutputFile"],
                "image_input_nodes": [n for n in output_nodes if n["type"] == "ImageInput"]
            }
    
    return result


def main():
    # Paths to inspect
    canonical_root = Path("/d/2026 Arboreal Futures/urban-futures/_futureSim_refactored/blender/compositor/canonical_templates")
    checkpoint_root = Path("/d/2026 Arboreal Futures/urban-futures/_data-refactored/compositor/temp_blends/checkpoints/proposal_colored_depth_outlines_20260410")
    template_dev_root = Path("/d/2026 Arboreal Futures/urban-futures/_data-refactored/compositor/temp_blends/template_development/proposal_subcategory_versions_20260410")
    
    blends_to_inspect = [
        canonical_root / "compositor_proposal_masks.blend",
        canonical_root / "edge_lab_final_template_safe_rebuild_20260405.blend",
        canonical_root / "proposal_outline_layers.blend",
        canonical_root / "proposal_colored_depth_outlines.blend",
        canonical_root / "proposal_only_layers.blend",
    ]
    
    print("\n" + "="*80)
    print("CANONICAL BLENDS")
    print("="*80)
    for blend_path in blends_to_inspect:
        result = inspect_blend(blend_path)
        if "error" in result:
            print(f"\n{blend_path.name}: {result['error']}")
        else:
            print(f"\n{blend_path.name}:")
            for scene_name, data in result["scenes"].items():
                print(f"  Scene '{scene_name}':")
                if data["output_file_nodes"]:
                    print(f"    Output File Nodes:")
                    for node in data["output_file_nodes"]:
                        print(f"      - {node['name']}")
                        for slot in node["file_slots"]:
                            print(f"          {slot['name']}: {slot['path']}")
                if data["image_input_nodes"]:
                    print(f"    Image Input Nodes:")
                    for node in data["image_input_nodes"]:
                        print(f"      - {node['name']} (image: {node['image']})")
    
    # Also check checkpoint and template dev versions
    checkpoint_blends = list(checkpoint_root.glob("*.blend")) if checkpoint_root.exists() else []
    template_blends = list(template_dev_root.glob("*.blend")) if template_dev_root.exists() else []
    
    print("\n" + "="*80)
    print("CHECKPOINT BLENDS")
    print("="*80)
    for blend_path in checkpoint_blends[:1]:  # Just first one to save time
        result = inspect_blend(blend_path)
        if "error" not in result:
            print(f"\n{blend_path.name}:")
            for scene_name, data in result["scenes"].items():
                if data["output_file_nodes"]:
                    for node in data["output_file_nodes"]:
                        print(f"  {node['name']}")
    
    print("\n" + "="*80)
    print("TEMPLATE DEV BLENDS")
    print("="*80)
    for blend_path in sorted(template_blends)[:3]:  # Just first few
        result = inspect_blend(blend_path)
        if "error" not in result:
            print(f"\n{blend_path.name}:")
            for scene_name, data in result["scenes"].items():
                if data["output_file_nodes"]:
                    for node in data["output_file_nodes"]:
                        print(f"  {node['name']}")


if __name__ == "__main__":
    main()
