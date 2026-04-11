"""Apply saturated proposal colors to proposal_colored_depth_outlines version blend.

Reads the user-validated CORRECT blend (via BUILD_BASE_BLEND) and writes a
finalized version (BUILD_OUTPUT_BLEND) with:

  1. RGB node colors updated per-frame to the saturated proposal set.
  2. The dangling colonise `Set Alpha` node re-wired to the Image output slot
     (the CORRECT blend currently routes unmasked rgba to that slot).
  3. EXR repathed to BUILD_EXR_PATH (latest positive-pathway EXR).

The CORRECT blend is NEVER modified — only read.
"""
from __future__ import annotations

import os
from pathlib import Path

import bpy

BASE_BLEND = Path(os.environ["BUILD_BASE_BLEND"])
OUTPUT_BLEND = Path(os.environ["BUILD_OUTPUT_BLEND"])
EXR_PATH = Path(os.environ["BUILD_EXR_PATH"])
SCENE_NAME = "ProposalColoredDepthOutlines"

# Saturated set — matches 14:39 user-confirmed release-control render.
PROPOSAL_COLORS = {
    "colonise":         (1.000, 0.550, 0.000, 1.0),
    "decay":            (0.900, 0.150, 0.150, 1.0),
    "deploy-structure": (0.150, 0.350, 1.000, 1.0),
    "recruit":          (0.150, 0.750, 0.150, 1.0),
    "release-control":  (0.550, 0.100, 0.850, 1.0),
}

# Map frame label (as found in CORRECT blend) → canonical proposal key.
FRAME_LABEL_TO_KEY = {
    "proposal- colonise":          "colonise",
    "proposal - decay":            "decay",
    "proposal - deploy structure": "deploy-structure",
    "proposal - recruit":          "recruit",
    "proposal - release control":  "release-control",
}


def log(msg: str) -> None:
    print(f"[finalize] {msg}")


def main() -> None:
    log(f"open base {BASE_BLEND.name}")
    bpy.ops.wm.open_mainfile(filepath=str(BASE_BLEND))

    scene = bpy.data.scenes[SCENE_NAME]
    tree = scene.node_tree

    # repath EXR
    exr_node = tree.nodes["EXR"]
    exr_node.image.filepath = str(EXR_PATH)
    exr_node.image.reload()
    log(f"repathed EXR -> {EXR_PATH.name}")

    # ---- 1. Update RGB colors per frame ----
    # Group nodes by parent-frame NAME (identity comparison fails across
    # separate iterations of tree.nodes after file load).
    frame_by_name = {
        n.name: n for n in tree.nodes if n.bl_idname == "NodeFrame"
    }
    rgb_by_frame: dict[str, list] = {name: [] for name in frame_by_name}
    for n in tree.nodes:
        if n.bl_idname != "CompositorNodeRGB":
            continue
        p = n.parent
        if p is None or p.name not in frame_by_name:
            log(f"WARN: RGB node {n.name!r} has no known frame parent")
            continue
        rgb_by_frame[p.name].append(n)

    for fname, frame in frame_by_name.items():
        key = FRAME_LABEL_TO_KEY.get(frame.label)
        if key is None:
            log(f"WARN: frame {frame.label!r} not in proposal map — skipped")
            continue
        color = PROPOSAL_COLORS[key]
        rgb_nodes = rgb_by_frame.get(fname, [])
        if not rgb_nodes:
            log(f"WARN: frame {frame.label!r} has no RGB node")
            continue
        for rgb in rgb_nodes:
            old = tuple(rgb.outputs[0].default_value)
            rgb.outputs[0].default_value = color
            log(
                f"color[{key}] {rgb.name!r}: "
                f"({old[0]:.3f},{old[1]:.3f},{old[2]:.3f}) -> "
                f"({color[0]:.3f},{color[1]:.3f},{color[2]:.3f})"
            )

    # ---- 2. Fix dangling colonise Set Alpha ----
    # In CORRECT blend:
    #   'proposal-only_release-control::rgba' (unmasked) -> FileOut.Image
    #   'Set Alpha' (masked with Math.001→proposal-colonise) has no output link
    # We rewire Image slot to consume 'Set Alpha'.Image instead.
    file_out = tree.nodes["ProposalColoredDepthOutput"]
    colonise_masked = tree.nodes.get("Set Alpha")
    if colonise_masked is None:
        raise RuntimeError("expected dangling 'Set Alpha' node for colonise not found")
    image_slot = file_out.inputs["Image"]
    for link in list(image_slot.links):
        log(
            f"unlink Image slot: {link.from_node.name!r}.{link.from_socket.name!r}"
        )
        tree.links.remove(link)
    tree.links.new(colonise_masked.outputs["Image"], image_slot)
    log("rewired Image slot <- 'Set Alpha'.Image (colonise masked output)")

    # ---- 2a. Normalize File Output slot paths (trailing `_`) ----
    # The CORRECT blend's slot[3] path lacks a trailing underscore, which
    # would break the runner's `*_0001.png` rename glob. Don't rebuild the
    # node here (that's the runner's job as a Blender 4.2 render workaround).
    for i, s in enumerate(file_out.file_slots):
        if not s.path.endswith("_"):
            new_path = s.path + "_"
            log(f"normalize slot[{i}] {s.path!r} -> {new_path!r}")
            s.path = new_path

    # ---- 2b. Ensure Composite + Viewer sinks exist (compositor eval root) ----
    # Blender 4.2 compositor needs a Composite node for the render pipeline to
    # drive execution. The CORRECT blend has only File Output which causes
    # render to exit at "Initializing execution" without writing anything.
    composite = next(
        (n for n in tree.nodes if n.bl_idname == "CompositorNodeComposite"),
        None,
    )
    if composite is None:
        composite = tree.nodes.new("CompositorNodeComposite")
        composite.name = "Composite"
        composite.location = (1200, -800)
        log("created missing Composite sink node")
    viewer = next(
        (n for n in tree.nodes if n.bl_idname == "CompositorNodeViewer"),
        None,
    )
    if viewer is None:
        viewer = tree.nodes.new("CompositorNodeViewer")
        viewer.name = "Viewer"
        viewer.location = (1200, -1000)
        log("created missing Viewer sink node")
    # Drive both sinks from the colonise masked output (arbitrary; same signal
    # already feeds slot[0]). Clearing any pre-existing links to be safe.
    for sink in (composite, viewer):
        for link in list(sink.inputs["Image"].links):
            tree.links.remove(link)
        tree.links.new(colonise_masked.outputs["Image"], sink.inputs["Image"])
    log("wired Composite & Viewer <- 'Set Alpha'.Image (colonise masked)")

    # ---- 3. Verify final alignment & log summary ----
    log("final slot alignment:")
    for i, s in enumerate(file_out.file_slots):
        sock = file_out.inputs[i]
        src = "<unlinked>"
        if sock.is_linked:
            l = sock.links[0]
            src = f"{l.from_node.name!r}"
        log(f"  [{i}] path={s.path!r} socket={sock.name!r} <- {src}")

    OUTPUT_BLEND.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_BLEND), copy=False)
    log(f"saved -> {OUTPUT_BLEND.name}")


if __name__ == "__main__":
    main()
