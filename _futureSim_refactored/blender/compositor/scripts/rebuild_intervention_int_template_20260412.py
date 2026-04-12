"""Intentional canonical-template rebuild: create compositor_intervention_int.blend.

Single-input compositor template that reads ``intervention_bioenvelope_ply-int``
from a bioenvelope EXR and produces:

1. Per-category separated RGBA PNGs (colour + alpha mask per int code)
2. A combined RGBA PNG (all categories alpha-overed together)

Template shape (for scene ``Current``):

    EXR Image --(intervention_bioenvelope_ply-int)--> Reroute (raw hub)
                                                          |
                                          +---------------+---------------+
                                          |               |               |
                                   Compare==1       Compare==2      ... Compare==8
                                          |               |               |
                                   CombineColor    CombineColor    ... CombineColor
                                          |               |               |
                                    SetAlpha         SetAlpha        ... SetAlpha
                                          |               |               |
                                    File Output slots (per-category)
                                          |               |               |
                                          +--AlphaOver chain--+-----------+
                                                              |
                                                     File Output slot (combined)

Integer-to-colour mapping (from constants.BIOENVELOPE_PLY_INT / BIOENVELOPE_PLY_COLORS):

    0  none                     -> transparent (alpha 0)
    1  deploy-any               -> #DCC090  sand
    2  buffer-feature           -> #F0DC90  butter
    3  larger-patches-rewild    -> #8ED8C8  mint
    4  roughen-envelope         -> #D0A040  ochre
    5  enrich-envelope          -> #B8E86C  lime
    6  rewild-smaller-patch     -> #F0DC90  butter
    7  rewild-larger-patch      -> #8ED8C8  mint
    8  buffer-feature+depaved   -> #DC78A0  pink

View transform is set to Raw so sRGB hex values pass through untouched.

Contract: this is an explicit template-edit operation. It is NOT a runtime
script. It rebuilds the ``Current`` scene's compositor tree from scratch.

Usage:

    blender --background --python rebuild_intervention_int_template_20260412.py \\
        -- --src "<source blend>" --dst "<dest blend>"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


SCENE_NAME = "Current"
EXR_NODE_NAME = "InterventionInt::EXR Input"
RAW_HUB_NODE_NAME = "InterventionInt::Raw"
FILE_OUTPUT_NAME = "InterventionInt::Outputs"

# Per-category layers: (int_code, slot_suffix, label, R, G, B)
# Values are sRGB 0-255 divided to 0-1. View transform is Raw so these
# pass through to the PNG unchanged.
CATEGORY_LAYERS: list[tuple[int, str, str, float, float, float]] = [
    (1, "deploy-any",        "deploy-any",        220 / 255, 192 / 255, 144 / 255),
    (2, "decay",             "decay",             240 / 255, 220 / 255, 144 / 255),
    (3, "colonise-ground",   "colonise ground",   142 / 255, 216 / 255, 200 / 255),
    (4, "colonise-partial",  "colonise partial",  208 / 255, 160 / 255,  64 / 255),
    (5, "colonise-full",     "colonise full",     184 / 255, 232 / 255, 108 / 255),
    (6, "recruit-partial",   "recruit partial",   240 / 255, 220 / 255, 144 / 255),
    (7, "recruit-full",      "recruit full",      142 / 255, 216 / 255, 200 / 255),
    (8, "depaved",           "depaved",           220 / 255, 120 / 255, 160 / 255),
]


def log(msg: str) -> None:
    print(f"[rebuild_intervention_int] {msg}")


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    return p.parse_args(argv)


def clear_tree(tree: bpy.types.NodeTree) -> None:
    for link in list(tree.links):
        tree.links.remove(link)
    for node in list(tree.nodes):
        tree.nodes.remove(node)


def new_node(tree, bl_idname, name, label, location, color=None, parent=None):
    node = tree.nodes.new(bl_idname)
    node.name = name
    node.label = label
    node.location = location
    if color is not None:
        node.use_custom_color = True
        node.color = color
    if parent is not None:
        node.parent = parent
    return node


def new_frame(tree, name, label, location, color):
    f = new_node(tree, "NodeFrame", name, label, location, color=color)
    f.label_size = 18
    f.shrink = False
    return f


def build_compositor(tree: bpy.types.NodeTree) -> None:
    # -- Frames ---------------------------------------------------------------
    input_frame = new_frame(
        tree, "Frame::Input", "EXR Input",
        (-1400.0, 200.0), (0.16, 0.16, 0.20),
    )
    layers_frame = new_frame(
        tree, "Frame::Layers", "Per-category layers",
        (-600.0, 0.0), (0.20, 0.16, 0.14),
    )
    combine_frame = new_frame(
        tree, "Frame::Combine", "Alpha-over combine",
        (300.0, 0.0), (0.14, 0.20, 0.16),
    )
    output_frame = new_frame(
        tree, "Frame::Output", "File Output + Sinks",
        (700.0, 0.0), (0.14, 0.16, 0.14),
    )

    # -- EXR input (no image at build time) -----------------------------------
    exr = new_node(
        tree,
        "CompositorNodeImage",
        EXR_NODE_NAME,
        EXR_NODE_NAME,
        (-1300.0, 100.0),
        color=(0.12, 0.18, 0.10),
        parent=input_frame,
    )
    exr.image = None

    # -- Raw hub reroute (runner wires EXR socket -> this) --------------------
    raw_hub = new_node(
        tree,
        "NodeReroute",
        RAW_HUB_NODE_NAME,
        "raw int value",
        (-900.0, 100.0),
        parent=input_frame,
    )

    # -- Per-category: Compare → CombineColor → SetAlpha ----------------------
    set_alpha_outputs: list[bpy.types.NodeSocket] = []
    for idx, (int_code, suffix, label, r, g, b) in enumerate(CATEGORY_LAYERS):
        y_offset = 200.0 - idx * 160.0

        # Compare: abs(raw - int_code) <= 0.5  →  0/1 mask
        cmp = new_node(
            tree,
            "CompositorNodeMath",
            f"Compare::{suffix}",
            f"== {int_code} ({label})",
            (-500.0, y_offset),
            parent=layers_frame,
        )
        cmp.operation = "COMPARE"
        cmp.inputs[1].default_value = float(int_code)
        cmp.inputs[2].default_value = 0.5
        tree.links.new(raw_hub.outputs[0], cmp.inputs[0])

        # CombineColor with the category colour
        rgb = new_node(
            tree,
            "CompositorNodeCombineColor",
            f"RGB::{suffix}",
            f"{label} colour",
            (-200.0, y_offset),
            parent=layers_frame,
        )
        rgb.mode = "RGB"
        rgb.inputs["Red"].default_value = r
        rgb.inputs["Green"].default_value = g
        rgb.inputs["Blue"].default_value = b

        # SetAlpha: colour + comparison mask
        sa = new_node(
            tree,
            "CompositorNodeSetAlpha",
            f"SetAlpha::{suffix}",
            f"{label} masked",
            (0.0, y_offset),
            parent=layers_frame,
        )
        sa.mode = "REPLACE_ALPHA"
        tree.links.new(rgb.outputs[0], sa.inputs["Image"])
        tree.links.new(cmp.outputs[0], sa.inputs["Alpha"])
        set_alpha_outputs.append(sa.outputs["Image"])

    # -- Alpha-over chain to produce the combined image -----------------------
    # Layer all 8 categories on top of each other (order doesn't matter since
    # they don't overlap — each pixel has exactly one int code).
    prev_output = set_alpha_outputs[0]
    for idx, sa_out in enumerate(set_alpha_outputs[1:], start=1):
        ao = new_node(
            tree,
            "CompositorNodeAlphaOver",
            f"AlphaOver::{idx}",
            f"combine {idx}",
            (350.0, 200.0 - idx * 80.0),
            parent=combine_frame,
        )
        tree.links.new(prev_output, ao.inputs[1])
        tree.links.new(sa_out, ao.inputs[2])
        prev_output = ao.outputs["Image"]

    combined_output = prev_output

    # -- File Output ----------------------------------------------------------
    fout = new_node(
        tree,
        "CompositorNodeOutputFile",
        FILE_OUTPUT_NAME,
        FILE_OUTPUT_NAME,
        (800.0, 0.0),
        color=(0.12, 0.20, 0.14),
        parent=output_frame,
    )
    fout.base_path = "//interventions_bioenvelope_out"
    fout.format.file_format = "PNG"
    fout.format.color_mode = "RGBA"
    fout.format.color_depth = "8"

    # Slot 0: combined
    if len(fout.file_slots) == 0:
        fout.file_slots.new("interventions_bioenvelope_")
    else:
        fout.file_slots[0].path = "interventions_bioenvelope_"
    tree.links.new(combined_output, fout.inputs[0])

    # Slots 1-8: per-category layers
    for (int_code, suffix, label, r, g, b), sa_out in zip(CATEGORY_LAYERS, set_alpha_outputs):
        slot_path = f"interventions_bioenvelope_{suffix}_"
        fout.file_slots.new(slot_path)
        tree.links.new(sa_out, fout.inputs[-1])

    # -- Composite + Viewer sinks (required by Blender) -----------------------
    comp = new_node(
        tree,
        "CompositorNodeComposite",
        "Composite",
        "Composite",
        (1100.0, 100.0),
        parent=output_frame,
    )
    tree.links.new(combined_output, comp.inputs[0])

    viewer = new_node(
        tree,
        "CompositorNodeViewer",
        "Viewer",
        "Viewer",
        (1100.0, -100.0),
        parent=output_frame,
    )
    tree.links.new(combined_output, viewer.inputs[0])


def wire_exr_to_graph_if_possible(tree: bpy.types.NodeTree) -> None:
    """Wire the EXR Image node's intervention_bioenvelope_ply-int socket to the
    raw hub reroute. Only possible if the EXR Image node already has a loaded
    image that exposes that socket. Otherwise the runner handles wiring."""
    exr = tree.nodes.get(EXR_NODE_NAME)
    raw_hub = tree.nodes.get(RAW_HUB_NODE_NAME)
    if exr is None or raw_hub is None:
        return
    int_socket = next(
        (s for s in exr.outputs if "intervention_bioenvelope_ply" in s.name),
        None,
    )
    if int_socket is None:
        log("EXR socket not yet present; runner must wire after loading EXR")
        return
    tree.links.new(int_socket, raw_hub.inputs[0])
    log("wired EXR intervention_bioenvelope_ply-int -> Raw hub")


def rebuild(src: Path, dst: Path) -> None:
    log(f"opening {src}")
    bpy.ops.wm.open_mainfile(filepath=str(src))

    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None:
        raise RuntimeError(f"scene {SCENE_NAME!r} not found")
    if scene.node_tree is None:
        scene.use_nodes = True

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    try:
        scene.display_settings.display_device = "sRGB"
    except Exception:
        pass
    try:
        scene.view_settings.view_transform = "Raw"
    except Exception:
        pass
    try:
        scene.view_settings.look = "None"
    except Exception:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    tree = scene.node_tree
    log(f"clearing {len(tree.nodes)} nodes / {len(tree.links)} links from scene {SCENE_NAME!r}")
    clear_tree(tree)

    log("building intervention_int graph (per-category + alpha-over combined)")
    build_compositor(tree)
    wire_exr_to_graph_if_possible(tree)

    fout = tree.nodes.get(FILE_OUTPUT_NAME)
    log(f"final node count: {len(tree.nodes)} links: {len(tree.links)}")
    if fout is not None:
        log(f"File Output has {len(fout.file_slots)} slots:")
        for i, s in enumerate(fout.file_slots):
            sock = fout.inputs[i]
            log(f"  slot[{i}] path={s.path!r} linked={sock.is_linked}")

    log(f"saving rebuilt blend to {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(dst))
    log("done")


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise FileNotFoundError(f"source blend not found: {src}")
    if dst.resolve() == src.resolve():
        raise RuntimeError("refusing to overwrite source in place")
    rebuild(src, dst)


if __name__ == "__main__":
    main()
