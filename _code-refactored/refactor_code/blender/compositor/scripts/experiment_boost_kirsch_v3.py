"""Experiment v3 — boost weak Kirsch response before thresholding.

Key insight from v1: raw Kirsch on mist produces visible edges everywhere
including trees, but the tree edges are WEAK relative to building edges.
The hard binary threshold then kills the weak tree edges.

This experiment boosts the Kirsch response before thresholding, and tests
continuous alpha (no binary threshold) vs soft linear ramps.

Variants for MIST:
  mist_v1_boost5x          — Kirsch → ×5 → BW → mask → colour (continuous alpha, no threshold)
  mist_v2_boost10x         — Kirsch → ×10 → BW → mask → colour
  mist_v3_boost5x_floor    — Kirsch → ×5 → linear ramp [0.01, 0.3] → BW → mask → colour
  mist_v4_boost10x_floor   — Kirsch → ×10 → linear ramp [0.01, 0.5] → BW → mask → colour
  mist_v5_norm_boost5x     — Normalize → Kirsch → ×5 → linear ramp → mask → colour
  mist_v6_boost20x         — Kirsch → ×20 → clamp → mask → colour (aggressive)

Variants for DEPTH:
  depth_v1_boost5x         — same approach for depth channel
  depth_v2_boost10x_floor  — Kirsch → ×10 → linear ramp → mask → colour

Run:
  /Applications/Blender.app/Contents/MacOS/Blender --background --python \\
    _code-refactored/refactor_code/blender/compositor/scripts/experiment_boost_kirsch_v3.py
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import bpy

REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
DEFAULT_DATASET_ROOT = (
    REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab"
    / "inputs" / "LATEST_REMOTE_EXRS"
    / "simv3-7_20260405_8k64s_simv3-7" / "city_timeline"
)
POSITIVE_EXR = DEFAULT_DATASET_ROOT / "city_timeline__positive_state__8k64s.exr"
OUTPUT_ROOT = REPO_ROOT / "_data-refactored" / "compositor" / "outputs" / "experiment_boost_kirsch_20260408"

TREE_ID = 3
EDGE_COLOR_LINEAR = (0.015996, 0.006048822, 0.043735046, 1.0)


def log(msg: str) -> None:
    print(f"[experiment_v3] {msg}")


def clear_tree(nt):
    for lk in list(nt.links):
        nt.links.remove(lk)
    for nd_ in list(nt.nodes):
        nt.nodes.remove(nd_)


def lnk(nt, fr, to):
    for lk in list(to.links):
        nt.links.remove(lk)
    nt.links.new(fr, to)


def nd(nt, typ, name, loc):
    n = nt.nodes.new(typ)
    n.name = name
    n.label = name
    n.location = loc
    return n


def detect_resolution(path: Path) -> tuple[int, int]:
    try:
        r = subprocess.run(["oiiotool", "--info", "-v", str(path)],
                           check=True, capture_output=True, text=True)
        m = re.search(r"(\d+)\s*x\s*(\d+)", r.stdout)
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    return 3840, 2160


def render_socket(scene, nt, socket, out_path):
    comp = nt.nodes["Composite"]
    view = nt.nodes["Viewer"]
    lnk(nt, socket, comp.inputs[0])
    lnk(nt, socket, view.inputs[0])
    scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True, scene=scene.name)
    log(f"  Wrote {out_path.name}")


def build_boosted_variant(nt, signal_socket, mask_socket, prefix, y,
                           boost=5.0, normalize=False, ramp_mode=None,
                           ramp_low=0.01, ramp_high=0.3):
    """
    signal → [normalize] → Kirsch → boost(×N) → [ramp] → BW → mask → colour

    ramp_mode: None = no ramp (continuous alpha), "linear", "constant"
    """
    x = -1200
    signal = signal_socket

    # Optional normalize
    if normalize:
        norm = nd(nt, "CompositorNodeNormalize", f"{prefix}_norm", (x, y))
        lnk(nt, signal, norm.inputs[0])
        signal = norm.outputs[0]
        x += 200

    # Kirsch
    kirsch = nd(nt, "CompositorNodeFilter", f"{prefix}_kirsch", (x, y))
    kirsch.filter_type = "KIRSCH"
    lnk(nt, signal, kirsch.inputs["Image"])
    x += 200

    # Boost (multiply by N)
    mult = nd(nt, "CompositorNodeMath", f"{prefix}_boost", (x, y))
    mult.operation = "MULTIPLY"
    mult.use_clamp = True
    lnk(nt, kirsch.outputs["Image"], mult.inputs[0])
    mult.inputs[1].default_value = boost
    edge_signal = mult.outputs["Value"]
    x += 200

    # Optional ramp
    if ramp_mode:
        ramp = nd(nt, "CompositorNodeValToRGB", f"{prefix}_ramp", (x, y))
        if ramp_mode == "constant":
            ramp.color_ramp.interpolation = "CONSTANT"
        else:
            ramp.color_ramp.interpolation = "LINEAR"
        ramp.color_ramp.elements[0].position = ramp_low
        ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
        ramp.color_ramp.elements[1].position = ramp_high
        ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
        lnk(nt, edge_signal, ramp.inputs["Fac"])
        edge_signal = ramp.outputs["Image"]
        x += 300

    # BW (in case signal is still RGB from the ramp)
    bw = nd(nt, "CompositorNodeRGBToBW", f"{prefix}_bw", (x, y))
    lnk(nt, edge_signal, bw.inputs["Image"])
    x += 200

    # Mask
    masked = nd(nt, "CompositorNodeMath", f"{prefix}_masked", (x, y))
    masked.operation = "MULTIPLY"
    masked.use_clamp = True
    lnk(nt, bw.outputs["Val"], masked.inputs[0])
    lnk(nt, mask_socket, masked.inputs[1])
    x += 200

    # Edge colour + alpha
    rgb = nd(nt, "CompositorNodeRGB", f"{prefix}_rgb", (x, y - 180))
    rgb.outputs[0].default_value = EDGE_COLOR_LINEAR
    sa = nd(nt, "CompositorNodeSetAlpha", f"{prefix}_final", (x + 200, y))
    sa.mode = "REPLACE_ALPHA"
    lnk(nt, rgb.outputs[0], sa.inputs["Image"])
    lnk(nt, masked.outputs["Value"], sa.inputs["Alpha"])

    return sa.outputs["Image"]


def main():
    if not POSITIVE_EXR.exists():
        raise FileNotFoundError(f"EXR not found: {POSITIVE_EXR}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.use_nodes = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1
    try:
        scene.display_settings.display_device = "sRGB"
    except Exception:
        pass
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
    except Exception:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    w, h = detect_resolution(POSITIVE_EXR)
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.resolution_percentage = 100

    nt = scene.node_tree
    clear_tree(nt)

    # ── Shared inputs ──────────────────────────────────────────────────

    exr = nd(nt, "CompositorNodeImage", "EXR", (-2000, 0))
    exr.image = bpy.data.images.load(str(POSITIVE_EXR), check_existing=True)

    id_mask = nd(nt, "CompositorNodeIDMask", "arboreal_mask", (-1600, -200))
    id_mask.index = TREE_ID
    id_mask.use_antialiasing = True
    lnk(nt, exr.outputs["IndexOB"], id_mask.inputs["ID value"])

    comp = nd(nt, "CompositorNodeComposite", "Composite", (2000, 0))
    view = nd(nt, "CompositorNodeViewer", "Viewer", (2000, -200))

    variants = []

    # ── Mist variants ──────────────────────────────────────────────────

    y = 1000
    # v1: continuous alpha, boost ×5
    s = build_boosted_variant(nt, exr.outputs["Mist"], id_mask.outputs["Alpha"],
                               "mist_v1", y, boost=5.0)
    variants.append(("mist_v1_boost5x", s))

    y -= 500
    # v2: continuous alpha, boost ×10
    s = build_boosted_variant(nt, exr.outputs["Mist"], id_mask.outputs["Alpha"],
                               "mist_v2", y, boost=10.0)
    variants.append(("mist_v2_boost10x", s))

    y -= 500
    # v3: boost ×5 + linear ramp
    s = build_boosted_variant(nt, exr.outputs["Mist"], id_mask.outputs["Alpha"],
                               "mist_v3", y, boost=5.0,
                               ramp_mode="linear", ramp_low=0.01, ramp_high=0.3)
    variants.append(("mist_v3_boost5x_floor", s))

    y -= 500
    # v4: boost ×10 + linear ramp
    s = build_boosted_variant(nt, exr.outputs["Mist"], id_mask.outputs["Alpha"],
                               "mist_v4", y, boost=10.0,
                               ramp_mode="linear", ramp_low=0.01, ramp_high=0.5)
    variants.append(("mist_v4_boost10x_floor", s))

    y -= 500
    # v5: normalize + boost ×5 + linear ramp
    s = build_boosted_variant(nt, exr.outputs["Mist"], id_mask.outputs["Alpha"],
                               "mist_v5", y, boost=5.0, normalize=True,
                               ramp_mode="linear", ramp_low=0.01, ramp_high=0.3)
    variants.append(("mist_v5_norm_boost5x", s))

    y -= 500
    # v6: aggressive boost ×20
    s = build_boosted_variant(nt, exr.outputs["Mist"], id_mask.outputs["Alpha"],
                               "mist_v6", y, boost=20.0)
    variants.append(("mist_v6_boost20x", s))

    # ── Depth variants ─────────────────────────────────────────────────

    y -= 500
    # depth boost ×5 continuous
    s = build_boosted_variant(nt, exr.outputs["Depth"], id_mask.outputs["Alpha"],
                               "depth_v1", y, boost=5.0)
    variants.append(("depth_v1_boost5x", s))

    y -= 500
    # depth boost ×10 + linear ramp
    s = build_boosted_variant(nt, exr.outputs["Depth"], id_mask.outputs["Alpha"],
                               "depth_v2", y, boost=10.0,
                               ramp_mode="linear", ramp_low=0.01, ramp_high=0.5)
    variants.append(("depth_v2_boost10x_floor", s))

    # ── Render each variant ────────────────────────────────────────────

    for name, socket in variants:
        log(f"Rendering {name}...")
        render_socket(scene, nt, socket, OUTPUT_ROOT / f"{name}.png")

    # Save blend
    blend_path = OUTPUT_ROOT / "experiment_boost_kirsch.blend"
    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        log(f"Saved {blend_path}")
    except RuntimeError as exc:
        log(f"Skipping blend save: {exc}")

    log("Done.")


if __name__ == "__main__":
    main()
