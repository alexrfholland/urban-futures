"""Build the final proposal_colored_depth_outlines.blend from the 14:39 autosave base.

Two-stage per proposal:

    Stage 1 — generate colored depth outlines:
        RGB_<name> -> outline_rgba.Image (SetAlpha, APPLY)
        DepthOutliner.outline_alpha -> outline_rgba.Alpha

    Stage 2 — mask by proposal:
        outline_rgba.Image -> masked_rgba.Image (SetAlpha, APPLY)
        EXR.proposal-<name> -> gt(>1.2) -> masked_rgba.Alpha
        masked_rgba.Image -> FileOut.<slot>

release-control additionally drives Composite and Viewer (evaluation root).
"""
from __future__ import annotations

import os
from pathlib import Path

import bpy

BASE_BLEND = Path(os.environ["BUILD_BASE_BLEND"])
EXR_PATH = Path(os.environ["BUILD_EXR_PATH"])
OUTPUT_BLEND = Path(os.environ["BUILD_OUTPUT_BLEND"])
SCENE_NAME = "ProposalColoredDepthOutlines"

PROPOSALS = [
    # (name,              exr_socket,                  rgb,                           slot_socket)
    ("release-control",   "proposal-release-control",  (0.550, 0.100, 0.850, 1.0),    "Image"),
    ("decay",             "proposal-decay",            (0.900, 0.150, 0.150, 1.0),    "proposal-only_decay"),
    ("recruit",           "proposal-recruit",          (0.150, 0.750, 0.150, 1.0),    "proposal-only_recruit"),
    ("colonise",          "proposal-colonise",         (1.000, 0.550, 0.000, 1.0),    "proposal-only_colonise"),
    ("deploy-structure",  "proposal-deploy-structure", (0.150, 0.350, 1.000, 1.0),    "proposal-only_deploy-structure"),
]

# scaffold nodes to delete from the 14:39 autosave base
SCAFFOLD_NODE_NAMES = [
    "proposal-only_release-control::binary-mask",
    "proposal-colored-depth::release-control::combine",
    "Viewer.001",
    "Math.001",
    "Viewer.002",
    "Set Alpha",
    "Viewer.003",
]


def log(msg: str) -> None:
    print(f"[build] {msg}")


def main() -> None:
    log(f"open base {BASE_BLEND}")
    bpy.ops.wm.open_mainfile(filepath=str(BASE_BLEND))

    scene = bpy.data.scenes[SCENE_NAME]
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links

    # repath EXR to latest positive-pathway EXR
    exr_node = nodes["EXR"]
    exr_node.image.filepath = str(EXR_PATH)
    exr_node.image.reload()
    log(f"repathed EXR to {EXR_PATH.name}")

    # delete scaffold nodes
    for name in SCAFFOLD_NODE_NAMES:
        n = nodes.get(name)
        if n is not None:
            nodes.remove(n)
            log(f"deleted scaffold node {name!r}")

    # reference nodes that must exist
    composite = nodes["Composite"]
    viewer = nodes["Viewer"]
    file_out = nodes["ProposalColoredDepthOutput"]
    depth_group = nodes["DepthOutliner::viewlayer_group"]
    outline_alpha_socket = depth_group.outputs["outline alpha"]

    # delete the old release-control rgb/rgba (we rebuild all 5 fresh for consistency)
    for old_name in ("proposal-only_release-control::rgb", "proposal-only_release-control::rgba"):
        n = nodes.get(old_name)
        if n is not None:
            nodes.remove(n)
            log(f"deleted legacy node {old_name!r}")

    # common exr image node ref
    def exr_socket(name: str):
        s = exr_node.outputs.get(name)
        if s is None:
            raise RuntimeError(f"EXR has no output socket {name!r}")
        return s

    # build each proposal chain with two-stage SetAlpha
    x_base = -600.0
    y_step = -300.0
    for i, (name, channel, rgb, slot_socket) in enumerate(PROPOSALS):
        y = i * y_step

        rgb_name = f"proposal-colored-depth_{name}::rgb"
        outline_rgba_name = f"proposal-colored-depth_{name}::outline_rgba"
        gt_name = f"proposal-colored-depth_{name}::gt1.2"
        masked_rgba_name = f"proposal-colored-depth_{name}::masked_rgba"

        # stage 0: RGB color
        rgb_node = nodes.new("CompositorNodeRGB")
        rgb_node.name = rgb_name
        rgb_node.label = rgb_name
        rgb_node.outputs[0].default_value = rgb

        # stage 1: RGB + outline_alpha -> colored outline
        outline_rgba = nodes.new("CompositorNodeSetAlpha")
        outline_rgba.name = outline_rgba_name
        outline_rgba.label = outline_rgba_name
        outline_rgba.mode = "APPLY"

        # stage 2: proposal mask
        gt = nodes.new("CompositorNodeMath")
        gt.name = gt_name
        gt.label = gt_name
        gt.operation = "GREATER_THAN"
        gt.inputs[1].default_value = 1.2

        masked_rgba = nodes.new("CompositorNodeSetAlpha")
        masked_rgba.name = masked_rgba_name
        masked_rgba.label = masked_rgba_name
        masked_rgba.mode = "APPLY"

        # layout
        rgb_node.location = (x_base + 0, y)
        outline_rgba.location = (x_base + 260, y)
        gt.location = (x_base + 260, y - 160)
        masked_rgba.location = (x_base + 520, y)

        # stage 1 wiring: RGB → outline_rgba.Image, outline_alpha → outline_rgba.Alpha
        links.new(rgb_node.outputs[0], outline_rgba.inputs["Image"])
        links.new(outline_alpha_socket, outline_rgba.inputs["Alpha"])

        # stage 2 wiring: outline_rgba.Image → masked_rgba.Image, gt(proposal > 1.2) → masked_rgba.Alpha
        links.new(outline_rgba.outputs["Image"], masked_rgba.inputs["Image"])
        links.new(exr_socket(channel), gt.inputs[0])
        links.new(gt.outputs[0], masked_rgba.inputs["Alpha"])

        # final output → FileOut slot
        slot_in = file_out.inputs.get(slot_socket)
        if slot_in is None:
            raise RuntimeError(f"FileOut has no input socket {slot_socket!r}")
        for link in list(slot_in.links):
            links.remove(link)
        links.new(masked_rgba.outputs["Image"], slot_in)

        log(f"chain {name}: RGB→outline_rgba(APPLY,outline_alpha)→masked_rgba(APPLY,gt>1.2) → slot {slot_socket!r}")

        # release-control also drives Composite + Viewer (evaluation root)
        if name == "release-control":
            for link in list(composite.inputs["Image"].links):
                links.remove(link)
            links.new(masked_rgba.outputs["Image"], composite.inputs["Image"])
            for link in list(viewer.inputs["Image"].links):
                links.remove(link)
            links.new(masked_rgba.outputs["Image"], viewer.inputs["Image"])
            log("wired release-control -> Composite, Viewer")

    # layout FileOut + group + composite to the right of chains
    file_out.location = (x_base + 1000, 0)
    depth_group.location = (x_base - 600, 200)
    composite.location = (x_base + 1000, -1400)
    viewer.location = (x_base + 1000, -1600)
    exr_node.location = (x_base - 1200, 0)

    log(f"final: {len(nodes)} nodes, {len(links)} links")

    OUTPUT_BLEND.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_BLEND), copy=False)
    log(f"saved to {OUTPUT_BLEND}")


if __name__ == "__main__":
    main()
