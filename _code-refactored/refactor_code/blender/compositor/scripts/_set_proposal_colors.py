"""Set the 5 RGB node colors in proposal_colored_depth_outlines.blend.

This is the ONLY modification made — no wiring, no slot paths, no sinks,
no EXR repath. Opens the blend in place and saves over it.
"""
from __future__ import annotations

import os
from pathlib import Path

import bpy

BLEND = Path(os.environ["TARGET_BLEND"])
SCENE_NAME = "ProposalColoredDepthOutlines"

# Frame label -> RGB color. Frame labels come from the CORRECT blend verbatim.
FRAME_COLORS = {
    "proposal- colonise":          (1.000, 0.550, 0.000, 1.0),
    "proposal - decay":            (0.900, 0.150, 0.150, 1.0),
    "proposal - deploy structure": (0.150, 0.350, 1.000, 1.0),
    "proposal - recruit":          (0.150, 0.750, 0.150, 1.0),
    "proposal - release control":  (0.550, 0.100, 0.850, 1.0),
}


def main() -> None:
    bpy.ops.wm.open_mainfile(filepath=str(BLEND))
    tree = bpy.data.scenes[SCENE_NAME].node_tree

    frame_by_name = {
        n.name: n for n in tree.nodes if n.bl_idname == "NodeFrame"
    }
    rgb_by_frame: dict[str, list] = {name: [] for name in frame_by_name}
    for n in tree.nodes:
        if n.bl_idname != "CompositorNodeRGB":
            continue
        if n.parent is not None and n.parent.name in rgb_by_frame:
            rgb_by_frame[n.parent.name].append(n)

    for fname, frame in frame_by_name.items():
        color = FRAME_COLORS.get(frame.label)
        if color is None:
            print(f"[colors] WARN: frame {frame.label!r} not in color map")
            continue
        for rgb in rgb_by_frame[fname]:
            old = tuple(rgb.outputs[0].default_value)
            rgb.outputs[0].default_value = color
            print(
                f"[colors] {frame.label!r} {rgb.name!r}: "
                f"({old[0]:.3f},{old[1]:.3f},{old[2]:.3f}) -> "
                f"({color[0]:.3f},{color[1]:.3f},{color[2]:.3f})"
            )

    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND), copy=False)
    print(f"[colors] saved {BLEND.name}")


if __name__ == "__main__":
    main()
