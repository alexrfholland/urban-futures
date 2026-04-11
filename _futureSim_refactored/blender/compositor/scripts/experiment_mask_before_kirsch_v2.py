"""Experiment v2 — mask-before-kirsch for both mist and depth.

The v1 experiment showed that edge-then-mask produces sparse results because
the strongest edges are at building boundaries (which get masked out). Tree
areas have smooth gradients internally, so few edges survive the arboreal mask.

This experiment tests the opposite approach: mask the signal BEFORE running
Kirsch. This creates sharp edges at tree silhouette boundaries where the signal
drops from a real value to zero.

Variants:
  mist_v1_mask_then_kirsch    — (mist × mask) → Normalize → Kirsch → threshold → colour
  mist_v2_mask_then_preblur   — (mist × mask) → Normalize → 2px blur → Kirsch → threshold → colour
  mist_v3_mask_then_soft      — same as v1 but threshold=0.04
  mist_v4_no_normalize        — (mist × mask) → Kirsch → threshold → colour (skip normalize)
  depth_v1_mask_then_kirsch   — (depth × mask) → Normalize → Kirsch → threshold → colour
  depth_v2_mask_then_preblur  — (depth × mask) → Normalize → 2px blur → Kirsch → threshold → colour
  depth_v3_mask_then_soft     — same as depth_v1 but threshold=0.04

Run:
  /Applications/Blender.app/Contents/MacOS/Blender --background --python \\
    _futureSim_refactored/blender/compositor/scripts/experiment_mask_before_kirsch_v2.py
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
OUTPUT_ROOT = REPO_ROOT / "_data-refactored" / "compositor" / "outputs" / "experiment_mask_before_kirsch_20260408"

TREE_ID = 3
EDGE_COLOR_LINEAR = (0.015996, 0.006048822, 0.043735046, 1.0)


def log(msg: str) -> None:
    print(f"[experiment_v2] {msg}")


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


def build_mask_before_kirsch(nt, signal_socket, mask_socket, prefix, y,
                              threshold=0.134, pre_blur=0, skip_normalize=False):
    """Mask signal first, then normalize, then Kirsch. Returns RGBA output socket."""
    x = -1200

    # Step 1: multiply signal by mask (creates zero outside trees)
    pre_mask = nd(nt, "CompositorNodeMath", f"{prefix}_premask", (x, y))
    pre_mask.operation = "MULTIPLY"
    pre_mask.use_clamp = False
    lnk(nt, signal_socket, pre_mask.inputs[0])
    lnk(nt, mask_socket, pre_mask.inputs[1])
    signal = pre_mask.outputs["Value"]
    x += 200

    # Step 2: normalize (optional)
    if not skip_normalize:
        norm = nd(nt, "CompositorNodeNormalize", f"{prefix}_norm", (x, y))
        lnk(nt, signal, norm.inputs[0])
        signal = norm.outputs[0]
        x += 200

    # Step 3: optional pre-blur
    if pre_blur > 0:
        blur = nd(nt, "CompositorNodeBlur", f"{prefix}_preblur", (x, y))
        blur.filter_type = "GAUSS"
        blur.use_relative = False
        blur.size_x = pre_blur
        blur.size_y = pre_blur
        lnk(nt, signal, blur.inputs[0])
        signal = blur.outputs[0]
        x += 200

    # Step 4: Kirsch
    kirsch = nd(nt, "CompositorNodeFilter", f"{prefix}_kirsch", (x, y))
    kirsch.filter_type = "KIRSCH"
    lnk(nt, signal, kirsch.inputs["Image"])
    x += 200

    # Step 5: threshold
    ramp = nd(nt, "CompositorNodeValToRGB", f"{prefix}_threshold", (x, y))
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    ramp.color_ramp.elements[1].position = threshold
    ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    lnk(nt, kirsch.outputs["Image"], ramp.inputs["Fac"])
    x += 300

    # Step 6: BW
    bw = nd(nt, "CompositorNodeRGBToBW", f"{prefix}_bw", (x, y))
    lnk(nt, ramp.outputs["Image"], bw.inputs["Image"])
    x += 200

    # Step 7: edge colour + alpha
    rgb = nd(nt, "CompositorNodeRGB", f"{prefix}_rgb", (x, y - 180))
    rgb.outputs[0].default_value = EDGE_COLOR_LINEAR
    sa = nd(nt, "CompositorNodeSetAlpha", f"{prefix}_final", (x + 200, y))
    sa.mode = "REPLACE_ALPHA"
    lnk(nt, rgb.outputs[0], sa.inputs["Image"])
    lnk(nt, bw.outputs["Val"], sa.inputs["Alpha"])

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

    # ── Mist variants ──────────────────────────────────────────────────

    variants = []

    # Mist: mask → normalize → kirsch → threshold 0.134
    y = 800
    s = build_mask_before_kirsch(nt, exr.outputs["Mist"], id_mask.outputs["Alpha"],
                                  "mist_v1", y, threshold=0.134)
    variants.append(("mist_v1_mask_then_kirsch", s))

    # Mist: mask → normalize → 2px blur → kirsch → threshold 0.134
    y = 300
    s = build_mask_before_kirsch(nt, exr.outputs["Mist"], id_mask.outputs["Alpha"],
                                  "mist_v2", y, threshold=0.134, pre_blur=2)
    variants.append(("mist_v2_mask_then_preblur", s))

    # Mist: mask → normalize → kirsch → threshold 0.04
    y = -200
    s = build_mask_before_kirsch(nt, exr.outputs["Mist"], id_mask.outputs["Alpha"],
                                  "mist_v3", y, threshold=0.04)
    variants.append(("mist_v3_mask_then_soft", s))

    # Mist: mask → kirsch → threshold 0.04 (no normalize)
    y = -700
    s = build_mask_before_kirsch(nt, exr.outputs["Mist"], id_mask.outputs["Alpha"],
                                  "mist_v4", y, threshold=0.04, skip_normalize=True)
    variants.append(("mist_v4_no_normalize", s))

    # ── Depth variants ─────────────────────────────────────────────────

    # Depth: mask → normalize → kirsch → threshold 0.134
    y = -1200
    s = build_mask_before_kirsch(nt, exr.outputs["Depth"], id_mask.outputs["Alpha"],
                                  "depth_v1", y, threshold=0.134)
    variants.append(("depth_v1_mask_then_kirsch", s))

    # Depth: mask → normalize → 2px blur → kirsch → threshold 0.134
    y = -1700
    s = build_mask_before_kirsch(nt, exr.outputs["Depth"], id_mask.outputs["Alpha"],
                                  "depth_v2", y, threshold=0.134, pre_blur=2)
    variants.append(("depth_v2_mask_then_preblur", s))

    # Depth: mask → normalize → kirsch → threshold 0.04
    y = -2200
    s = build_mask_before_kirsch(nt, exr.outputs["Depth"], id_mask.outputs["Alpha"],
                                  "depth_v3", y, threshold=0.04)
    variants.append(("depth_v3_mask_then_soft", s))

    # ── Render each variant sequentially ───────────────────────────────

    for name, socket in variants:
        log(f"Rendering {name}...")
        render_socket(scene, nt, socket,
                      OUTPUT_ROOT / f"{name}.png")

    # Save blend
    blend_path = OUTPUT_ROOT / "experiment_mask_before_kirsch.blend"
    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        log(f"Saved {blend_path}")
    except RuntimeError as exc:
        log(f"Skipping blend save: {exc}")

    log("Done. All outputs in:")
    log(f"  {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
