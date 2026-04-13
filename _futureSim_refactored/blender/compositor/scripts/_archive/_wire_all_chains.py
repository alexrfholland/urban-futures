"""Wire all 5 proposal chains into the single File Output node.

Actions on proposal_colored_depth_outlines_v2.blend (saved in place):

  1. Set each frame's RGB node to its saturated proposal color.
  2. Reset File Output node to 5 named slots, one per proposal, each wired
     from that frame's masked 'Set Alpha.*' node.

Slot naming:
    proposal-depth-colour_<key>_   (trailing underscore so animation render
                                    produces e.g. proposal-depth-colour_decay_0001.png)

Frame → key mapping (frame labels are verbatim from the user's template):
    'proposal- colonise'           -> colonise
    'proposal - decay'             -> decay
    'proposal - deploy structure'  -> deploy-structure
    'proposal - recruit'           -> recruit
    'proposal - release control'   -> release-control
"""
from __future__ import annotations

import os
from pathlib import Path

import bpy

BLEND = Path(os.environ["TARGET_BLEND"])
SCENE_NAME = "ProposalColoredDepthOutlines"

FRAME_LABEL_TO_KEY = {
    "proposal- colonise":          "colonise",
    "proposal - decay":            "decay",
    "proposal - deploy structure": "deploy-structure",
    "proposal - recruit":          "recruit",
    "proposal - release control":  "release-control",
}

PROPOSAL_COLORS = {
    "colonise":         (1.000, 0.550, 0.000, 1.0),
    "decay":            (0.900, 0.150, 0.150, 1.0),
    "deploy-structure": (0.150, 0.350, 1.000, 1.0),
    "recruit":          (0.150, 0.750, 0.150, 1.0),
    "release-control":  (0.550, 0.100, 0.850, 1.0),
}

# Deterministic slot order (matches reading order of proposal states)
SLOT_ORDER = ["colonise", "decay", "deploy-structure", "recruit", "release-control"]


def log(msg: str) -> None:
    print(f"[wire_all] {msg}")


def main() -> None:
    log(f"open {BLEND}")
    bpy.ops.wm.open_mainfile(filepath=str(BLEND))

    scene = bpy.data.scenes[SCENE_NAME]
    tree = scene.node_tree

    # ---- Group chain nodes by parent frame NAME ----
    frame_by_name = {
        n.name: n for n in tree.nodes if n.bl_idname == "NodeFrame"
    }
    log(f"found {len(frame_by_name)} frames: "
        f"{[f.label for f in frame_by_name.values()]}")

    # For each frame, find the masked Set Alpha (name starts 'Set Alpha')
    # and the frame's RGB node.
    frame_rgb: dict[str, bpy.types.Node] = {}
    frame_setalpha: dict[str, bpy.types.Node] = {}
    for n in tree.nodes:
        if n.parent is None or n.parent.name not in frame_by_name:
            continue
        fname = n.parent.name
        if n.bl_idname == "CompositorNodeRGB":
            frame_rgb[fname] = n
        elif n.bl_idname == "CompositorNodeSetAlpha" and n.name.startswith("Set Alpha"):
            # Prefer the masked chain output — skip the '::rgba' unmasked one
            frame_setalpha[fname] = n

    # ---- 1. Update RGB colors per frame ----
    label_to_fname = {
        frame_by_name[fname].label: fname for fname in frame_by_name
    }

    for label, key in FRAME_LABEL_TO_KEY.items():
        fname = label_to_fname.get(label)
        if fname is None:
            raise RuntimeError(f"frame with label {label!r} not found")
        rgb = frame_rgb.get(fname)
        if rgb is None:
            raise RuntimeError(f"frame {label!r} has no RGB node")
        color = PROPOSAL_COLORS[key]
        old = tuple(rgb.outputs[0].default_value)
        rgb.outputs[0].default_value = color
        log(
            f"color[{key}] {rgb.name!r}: "
            f"({old[0]:.3f},{old[1]:.3f},{old[2]:.3f}) -> "
            f"({color[0]:.3f},{color[1]:.3f},{color[2]:.3f})"
        )

    # ---- 2. Rebuild File Output slots ----
    file_out = next(
        (n for n in tree.nodes if n.bl_idname == "CompositorNodeOutputFile"),
        None,
    )
    if file_out is None:
        raise RuntimeError("no CompositorNodeOutputFile found")
    log(f"file output: {file_out.name!r} base_path={file_out.base_path!r}")

    # Resolve the Set Alpha source for each slot in order
    key_to_sock: dict[str, bpy.types.NodeSocket] = {}
    for label, key in FRAME_LABEL_TO_KEY.items():
        fname = label_to_fname[label]
        sa = frame_setalpha.get(fname)
        if sa is None:
            raise RuntimeError(f"frame {label!r} has no 'Set Alpha' chain output")
        key_to_sock[key] = sa.outputs["Image"]
        log(f"source[{key}] <- {sa.name!r}.Image")

    # Disconnect existing slot inputs
    for i in range(len(file_out.inputs)):
        for link in list(file_out.inputs[i].links):
            tree.links.remove(link)

    # Shrink to 1 slot, then expand and relabel
    while len(file_out.file_slots) > 1:
        file_out.file_slots.remove(file_out.file_slots[-1])

    first_key = SLOT_ORDER[0]
    file_out.file_slots[0].path = f"proposal-depth-colour_{first_key}_"
    for key in SLOT_ORDER[1:]:
        file_out.file_slots.new(f"proposal-depth-colour_{key}_")

    # Wire each slot from its matching source
    for i, key in enumerate(SLOT_ORDER):
        path = f"proposal-depth-colour_{key}_"
        file_out.file_slots[i].path = path
        tree.links.new(key_to_sock[key], file_out.inputs[i])
        log(f"slot[{i}] path={path!r} <- {key_to_sock[key].node.name!r}.Image")

    # ---- Re-wire Composite sink from colonise (arbitrary but deterministic) ----
    composite = next(
        (n for n in tree.nodes if n.bl_idname == "CompositorNodeComposite"),
        None,
    )
    if composite is not None:
        for link in list(composite.inputs["Image"].links):
            tree.links.remove(link)
        tree.links.new(key_to_sock["colonise"], composite.inputs["Image"])
        log("wired Composite.Image <- colonise Set Alpha")

    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND), copy=False)
    log(f"saved {BLEND.name}")


if __name__ == "__main__":
    main()
