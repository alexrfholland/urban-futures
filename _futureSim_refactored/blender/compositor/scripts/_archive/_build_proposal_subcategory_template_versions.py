"""Build candidate proposal template versions with extra subcategory outputs.

Adds two new branches to each proposal compositor template:

- recruit-smalls
  - same colour as recruit
  - mask from EXR `size == 1`
- deploy-structure-fallen-logs
  - same colour as deploy-structure
  - mask from EXR `resource_fallen_log_mask`

These are written as versioned working copies, not back into the canonical
template paths.
"""

from __future__ import annotations

import os
from pathlib import Path

import bpy


REPO_ROOT = Path(__file__).resolve().parents[4]
CANONICAL_ROOT = REPO_ROOT / "_futureSim_refactored" / "blender" / "compositor" / "canonical_templates"
OUTPUT_ROOT = Path(
    os.environ.get(
        "PROPOSAL_SUBCATEGORY_OUTPUT_ROOT",
        str(
            REPO_ROOT
            / "_data-refactored"
            / "compositor"
            / "temp_blends"
            / "template_development"
            / "proposal_subcategory_versions_20260410"
        ),
    )
).expanduser()
REFERENCE_EXR = Path(
    os.environ.get(
        "PROPOSAL_SUBCATEGORY_REFERENCE_EXR",
        r"E:\2026 Arboreal Futures\blenderv2\renders\20260410_134239_city_baseline_yr-180_8k\city_baseline_yr-180__positive_state__8k.exr",
    )
).expanduser()


def log(message: str) -> None:
    print(f"[build_proposal_subcategory_template_versions] {message}")


def ensure_output_slot(file_out: bpy.types.Node, slot_path: str):
    for i, slot in enumerate(file_out.file_slots):
        if slot.path == slot_path:
            return file_out.inputs[i]
    file_out.file_slots.new(slot_path)
    return file_out.inputs[len(file_out.file_slots) - 1]


def new_math(nodes, name: str, operation: str, location: tuple[float, float], *, clamp: bool = False):
    node = nodes.new("CompositorNodeMath")
    node.name = name
    node.label = name
    node.operation = operation
    node.use_clamp = clamp
    node.location = location
    return node


def build_exact_value_mask(
    nodes,
    links,
    value_socket,
    *,
    prefix: str,
    target_value: float,
    epsilon: float,
    location: tuple[float, float],
):
    subtract = new_math(nodes, f"{prefix}::sub", "SUBTRACT", location, clamp=False)
    subtract.inputs[1].default_value = target_value
    absolute = new_math(nodes, f"{prefix}::abs", "ABSOLUTE", (location[0] + 220.0, location[1]), clamp=False)
    less_than = new_math(nodes, f"{prefix}::lt", "LESS_THAN", (location[0] + 440.0, location[1]), clamp=False)
    less_than.inputs[1].default_value = epsilon
    links.new(value_socket, subtract.inputs[0])
    links.new(subtract.outputs["Value"], absolute.inputs[0])
    links.new(absolute.outputs["Value"], less_than.inputs[0])
    return less_than.outputs["Value"]


def build_binary_threshold(
    nodes,
    links,
    value_socket,
    *,
    prefix: str,
    threshold: float,
    location: tuple[float, float],
):
    gt = new_math(nodes, f"{prefix}::gt", "GREATER_THAN", location, clamp=False)
    gt.inputs[1].default_value = threshold
    links.new(value_socket, gt.inputs[0])
    return gt.outputs["Value"]


def add_proposal_only_variants(blend_path: Path, output_path: Path) -> None:
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    scene = bpy.data.scenes["ProposalOnly"]
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links

    exr = nodes["EXR"]
    exr.image.filepath = str(REFERENCE_EXR)
    exr.image.reload()

    file_out = nodes["ProposalOnlyOutput"]

    # recruit-smalls
    release_rgb_source = nodes["proposal-only_recruit::rgb"]
    rgb = nodes.new("CompositorNodeRGB")
    rgb.name = "proposal-only_recruit-smalls::rgb"
    rgb.label = rgb.name
    rgb.outputs[0].default_value = tuple(release_rgb_source.outputs[0].default_value)
    rgb.location = (release_rgb_source.location.x + 40.0, release_rgb_source.location.y - 260.0)

    rgba = nodes.new("CompositorNodeSetAlpha")
    rgba.name = "proposal-only_recruit-smalls::rgba"
    rgba.label = rgba.name
    rgba.mode = "APPLY"
    rgba.location = (rgb.location.x + 260.0, rgb.location.y)

    size_mask = build_exact_value_mask(
        nodes,
        links,
        exr.outputs["size"],
        prefix="proposal-only_recruit-smalls::size_eq1",
        target_value=1.0,
        epsilon=0.1,
        location=(rgb.location.x + 40.0, rgb.location.y - 180.0),
    )
    links.new(rgb.outputs[0], rgba.inputs["Image"])
    links.new(size_mask, rgba.inputs["Alpha"])
    links.new(rgba.outputs["Image"], ensure_output_slot(file_out, "proposal-only_recruit-smalls_"))

    # deploy-structure-fallen-logs
    deploy_rgb_source = nodes["proposal-only_deploy-structure::rgb"]
    rgb = nodes.new("CompositorNodeRGB")
    rgb.name = "proposal-only_deploy-structure-fallen-logs::rgb"
    rgb.label = rgb.name
    rgb.outputs[0].default_value = tuple(deploy_rgb_source.outputs[0].default_value)
    rgb.location = (deploy_rgb_source.location.x + 40.0, deploy_rgb_source.location.y - 260.0)

    rgba = nodes.new("CompositorNodeSetAlpha")
    rgba.name = "proposal-only_deploy-structure-fallen-logs::rgba"
    rgba.label = rgba.name
    rgba.mode = "APPLY"
    rgba.location = (rgb.location.x + 260.0, rgb.location.y)

    fallen_mask = build_binary_threshold(
        nodes,
        links,
        exr.outputs["resource_fallen_log_mask"],
        prefix="proposal-only_deploy-structure-fallen-logs::fallen_logs",
        threshold=0.5,
        location=(rgb.location.x + 40.0, rgb.location.y - 180.0),
    )
    links.new(rgb.outputs[0], rgba.inputs["Image"])
    links.new(fallen_mask, rgba.inputs["Alpha"])
    links.new(
        rgba.outputs["Image"],
        ensure_output_slot(file_out, "proposal-only_deploy-structure-fallen-logs_"),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path), copy=False)
    log(f"saved {output_path.name}")


def add_proposal_outline_variants(blend_path: Path, output_path: Path) -> None:
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    scene = bpy.data.scenes["ProposalOutline"]
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links

    exr = nodes["EXR"]
    exr.image.filepath = str(REFERENCE_EXR)
    exr.image.reload()

    file_out = nodes["ProposalOutlineOutput"]

    def add_outline_branch(source_prefix: str, new_prefix: str, mask_socket, y_shift: float):
        rgb_source = nodes[f"{source_prefix}::rgb"]
        gt_source = nodes[f"{source_prefix}::gt1.2"]
        dilate_source = nodes[f"{source_prefix}::dilate"]
        erode_source = nodes[f"{source_prefix}::erode"]

        mask_proxy = new_math(nodes, f"{new_prefix}::mask", "ADD", (gt_source.location.x, gt_source.location.y + y_shift), clamp=False)
        mask_proxy.inputs[1].default_value = 0.0
        links.new(mask_socket, mask_proxy.inputs[0])

        dilate = nodes.new("CompositorNodeDilateErode")
        dilate.name = f"{new_prefix}::dilate"
        dilate.label = dilate.name
        dilate.mode = dilate_source.mode
        dilate.distance = dilate_source.distance
        dilate.falloff = dilate_source.falloff
        dilate.location = (dilate_source.location.x, dilate_source.location.y + y_shift)

        erode = nodes.new("CompositorNodeDilateErode")
        erode.name = f"{new_prefix}::erode"
        erode.label = erode.name
        erode.mode = erode_source.mode
        erode.distance = erode_source.distance
        erode.falloff = erode_source.falloff
        erode.location = (erode_source.location.x, erode_source.location.y + y_shift)

        edge_band = new_math(nodes, f"{new_prefix}::edge-band", "SUBTRACT", (nodes[f"{source_prefix}::edge-band"].location.x, nodes[f"{source_prefix}::edge-band"].location.y + y_shift), clamp=True)
        edge_binary = new_math(nodes, f"{new_prefix}::edge-binary", "GREATER_THAN", (nodes[f"{source_prefix}::edge-binary"].location.x, nodes[f"{source_prefix}::edge-binary"].location.y + y_shift), clamp=False)
        edge_binary.inputs[1].default_value = nodes[f"{source_prefix}::edge-binary"].inputs[1].default_value

        rgb = nodes.new("CompositorNodeRGB")
        rgb.name = f"{new_prefix}::rgb"
        rgb.label = rgb.name
        rgb.outputs[0].default_value = tuple(rgb_source.outputs[0].default_value)
        rgb.location = (rgb_source.location.x, rgb_source.location.y + y_shift)

        rgba = nodes.new("CompositorNodeSetAlpha")
        rgba.name = f"{new_prefix}::rgba"
        rgba.label = rgba.name
        rgba.mode = nodes[f"{source_prefix}::rgba"].mode
        rgba.location = (nodes[f"{source_prefix}::rgba"].location.x, nodes[f"{source_prefix}::rgba"].location.y + y_shift)

        links.new(mask_proxy.outputs["Value"], dilate.inputs["Mask"])
        links.new(mask_proxy.outputs["Value"], erode.inputs["Mask"])
        links.new(dilate.outputs["Mask"], edge_band.inputs[0])
        links.new(erode.outputs["Mask"], edge_band.inputs[1])
        links.new(edge_band.outputs["Value"], edge_binary.inputs[0])
        links.new(rgb.outputs[0], rgba.inputs["Image"])
        links.new(edge_binary.outputs["Value"], rgba.inputs["Alpha"])
        return rgba.outputs["Image"]

    release_mask = build_exact_value_mask(
        nodes,
        links,
        exr.outputs["size"],
        prefix="proposal-outline_recruit-smalls::size_eq1",
        target_value=1.0,
        epsilon=0.1,
        location=(40.0, 80.0),
    )
    release_image = add_outline_branch(
        "proposal-outline_recruit",
        "proposal-outline_recruit-smalls",
        release_mask,
        -180.0,
    )
    links.new(release_image, ensure_output_slot(file_out, "proposal-outline_recruit-smalls_"))

    fallen_mask = build_binary_threshold(
        nodes,
        links,
        exr.outputs["resource_fallen_log_mask"],
        prefix="proposal-outline_deploy-structure-fallen-logs::fallen_logs",
        threshold=0.5,
        location=(40.0, -1120.0),
    )
    deploy_image = add_outline_branch(
        "proposal-outline_deploy-structure",
        "proposal-outline_deploy-structure-fallen-logs",
        fallen_mask,
        -180.0,
    )
    links.new(
        deploy_image,
        ensure_output_slot(file_out, "proposal-outline_deploy-structure-fallen-logs_"),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path), copy=False)
    log(f"saved {output_path.name}")


def add_colored_depth_variants(blend_path: Path, output_path: Path) -> None:
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    scene = bpy.data.scenes["ProposalColoredDepthOutlines"]
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links

    exr = nodes["EXR"]
    exr.image.filepath = str(REFERENCE_EXR)
    exr.image.reload()

    file_out = nodes["ProposalColoredDepthOutput"]
    outline_alpha = nodes["DepthOutliner::viewlayer_group"].outputs["outline alpha"]

    def add_depth_branch(
        *,
        name: str,
        color_source_name: str,
        outline_rgba_source_name: str,
        masked_rgba_source_name: str,
        mask_socket,
        slot_path: str,
        y_shift: float,
    ):
        color_source = nodes[color_source_name]
        outline_source = nodes[outline_rgba_source_name]
        masked_source = nodes[masked_rgba_source_name]

        rgb = nodes.new("CompositorNodeRGB")
        rgb.name = f"{name}::rgb"
        rgb.label = rgb.name
        rgb.outputs[0].default_value = tuple(color_source.outputs[0].default_value)
        rgb.location = (color_source.location.x, color_source.location.y + y_shift)

        outline_rgba = nodes.new("CompositorNodeSetAlpha")
        outline_rgba.name = f"{name}::outline_rgba"
        outline_rgba.label = outline_rgba.name
        outline_rgba.mode = outline_source.mode
        outline_rgba.location = (outline_source.location.x, outline_source.location.y + y_shift)

        masked_rgba = nodes.new("CompositorNodeSetAlpha")
        masked_rgba.name = f"{name}::masked_rgba"
        masked_rgba.label = masked_rgba.name
        masked_rgba.mode = masked_source.mode
        masked_rgba.location = (masked_source.location.x, masked_source.location.y + y_shift)

        links.new(rgb.outputs[0], outline_rgba.inputs["Image"])
        links.new(outline_alpha, outline_rgba.inputs["Alpha"])
        links.new(outline_rgba.outputs["Image"], masked_rgba.inputs["Image"])
        links.new(mask_socket, masked_rgba.inputs["Alpha"])
        links.new(masked_rgba.outputs["Image"], ensure_output_slot(file_out, slot_path))

    release_mask = build_exact_value_mask(
        nodes,
        links,
        exr.outputs["size"],
        prefix="proposal-colored-depth_recruit-smalls::size_eq1",
        target_value=1.0,
        epsilon=0.1,
        location=(280.0, -2220.0),
    )
    add_depth_branch(
        name="proposal-colored-depth_recruit-smalls",
        color_source_name="proposal-only_release-control::rgb.004",
        outline_rgba_source_name="proposal-only_release-control::rgba.004",
        masked_rgba_source_name="Set Alpha.004",
        mask_socket=release_mask,
        slot_path="proposal-depth-colour_recruit-smalls_",
        y_shift=-320.0,
    )

    fallen_mask = build_binary_threshold(
        nodes,
        links,
        exr.outputs["resource_fallen_log_mask"],
        prefix="proposal-colored-depth_deploy-structure-fallen-logs::fallen_logs",
        threshold=0.5,
        location=(580.0, -980.0),
    )
    add_depth_branch(
        name="proposal-colored-depth_deploy-structure-fallen-logs",
        color_source_name="proposal-only_release-control::rgb.003",
        outline_rgba_source_name="proposal-only_release-control::rgba.003",
        masked_rgba_source_name="Set Alpha.003",
        mask_socket=fallen_mask,
        slot_path="proposal-depth-colour_deploy-structure-fallen-logs_",
        y_shift=-360.0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path), copy=False)
    log(f"saved {output_path.name}")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    add_proposal_only_variants(
        CANONICAL_ROOT / "proposal_only_layers.blend",
        OUTPUT_ROOT / "proposal_only_layers_v4_subcategories.blend",
    )
    add_proposal_outline_variants(
        CANONICAL_ROOT / "proposal_outline_layers.blend",
        OUTPUT_ROOT / "proposal_outline_layers_v4_subcategories.blend",
    )
    add_colored_depth_variants(
        CANONICAL_ROOT / "proposal_colored_depth_outlines.blend",
        OUTPUT_ROOT / "proposal_colored_depth_outlines_v4_subcategories.blend",
    )


if __name__ == "__main__":
    main()
