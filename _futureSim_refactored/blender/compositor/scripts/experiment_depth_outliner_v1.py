"""Experimental depth outliner — canopy-cleanup variants.

Renders each variant sequentially through the Composite node (File Output
nodes are unreliable in background mode).

Variants:
  v1_baseline           — standard depth pipeline (threshold=0.134)
  v2_preblur2           — 2px pre-blur before Kirsch
  v3_preblur4           — 4px pre-blur
  v4_higher_threshold   — threshold=0.20
  v5_preblur2_higher    — 2px pre-blur + threshold=0.18
  v6_dilate_erode       — morphological close after threshold

Run:
  /Applications/Blender.app/Contents/MacOS/Blender --background --python \\
    _futureSim_refactored/blender/compositor/scripts/experiment_depth_outliner_v1.py
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
OUTPUT_ROOT = REPO_ROOT / "_data-refactored" / "compositor" / "outputs" / "experiment_depth_canopy_20260408"

TREE_ID = 3
EDGE_COLOR_LINEAR = (0.015996, 0.006048822, 0.043735046, 1.0)
BASELINE_THRESHOLD = 0.13409094512462616


def log(msg: str) -> None:
    print(f"[experiment_depth] {msg}")


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
    """Wire socket to Composite+Viewer and render to out_path."""
    comp = nt.nodes["Composite"]
    view = nt.nodes["Viewer"]
    lnk(nt, socket, comp.inputs[0])
    lnk(nt, socket, view.inputs[0])
    scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True, scene=scene.name)
    log(f"  Wrote {out_path.name}")


def build_depth_variant(nt, depth_socket, mask_socket, prefix, y, threshold, pre_blur=0, post_dilate=0, post_erode=0):
    """Build one depth outliner variant. Returns the final RGBA output socket."""
    x = -1000

    # Normalize
    norm = nd(nt, "CompositorNodeNormalize", f"{prefix}_norm", (x, y))
    lnk(nt, depth_socket, norm.inputs[0])
    signal = norm.outputs[0]
    x += 200

    # Optional pre-blur
    if pre_blur > 0:
        blur = nd(nt, "CompositorNodeBlur", f"{prefix}_preblur", (x, y))
        blur.filter_type = "GAUSS"
        blur.use_relative = False
        blur.size_x = pre_blur
        blur.size_y = pre_blur
        lnk(nt, signal, blur.inputs[0])
        signal = blur.outputs[0]
        x += 200

    # Kirsch
    kirsch = nd(nt, "CompositorNodeFilter", f"{prefix}_kirsch", (x, y))
    kirsch.filter_type = "KIRSCH"
    lnk(nt, signal, kirsch.inputs["Image"])
    x += 200

    # Hard threshold
    ramp = nd(nt, "CompositorNodeValToRGB", f"{prefix}_threshold", (x, y))
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    ramp.color_ramp.elements[1].position = threshold
    ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    lnk(nt, kirsch.outputs["Image"], ramp.inputs["Fac"])
    edge_signal = ramp.outputs["Image"]
    x += 300

    # RGB to BW
    bw = nd(nt, "CompositorNodeRGBToBW", f"{prefix}_bw", (x, y))
    lnk(nt, edge_signal, bw.inputs["Image"])
    edge_val = bw.outputs["Val"]
    x += 200

    # Optional post-dilate
    if post_dilate > 0:
        dilate = nd(nt, "CompositorNodeDilateErode", f"{prefix}_dilate", (x, y))
        dilate.mode = "DISTANCE"
        dilate.distance = post_dilate
        lnk(nt, edge_val, dilate.inputs[0])
        edge_val = dilate.outputs[0]
        x += 200

    # Optional post-erode
    if post_erode > 0:
        erode = nd(nt, "CompositorNodeDilateErode", f"{prefix}_erode", (x, y))
        erode.mode = "DISTANCE"
        erode.distance = -post_erode
        lnk(nt, edge_val, erode.inputs[0])
        edge_val = erode.outputs[0]
        x += 200

    # Mask multiply
    masked = nd(nt, "CompositorNodeMath", f"{prefix}_masked", (x, y))
    masked.operation = "MULTIPLY"
    masked.use_clamp = True
    lnk(nt, edge_val, masked.inputs[0])
    lnk(nt, mask_socket, masked.inputs[1])
    x += 200

    # Edge colour + set alpha
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

    # ── Build all variants ─────────────────────────────────────────────

    variant_specs = [
        {"name": "v1_baseline",         "threshold": BASELINE_THRESHOLD, "pre_blur": 0, "post_dilate": 0, "post_erode": 0},
        {"name": "v2_preblur2",         "threshold": BASELINE_THRESHOLD, "pre_blur": 2, "post_dilate": 0, "post_erode": 0},
        {"name": "v3_preblur4",         "threshold": BASELINE_THRESHOLD, "pre_blur": 4, "post_dilate": 0, "post_erode": 0},
        {"name": "v4_higher_threshold", "threshold": 0.20,               "pre_blur": 0, "post_dilate": 0, "post_erode": 0},
        {"name": "v5_preblur2_higher",  "threshold": 0.18,               "pre_blur": 2, "post_dilate": 0, "post_erode": 0},
        {"name": "v6_dilate_erode",     "threshold": BASELINE_THRESHOLD, "pre_blur": 0, "post_dilate": 1, "post_erode": 1},
    ]

    variants = []
    for i, spec in enumerate(variant_specs):
        y = 600 - i * 500
        out_socket = build_depth_variant(
            nt, exr.outputs["Depth"], id_mask.outputs["Alpha"],
            spec["name"], y, spec["threshold"], spec["pre_blur"],
            spec["post_dilate"], spec["post_erode"]
        )
        variants.append((spec["name"], out_socket))
        log(f"Built {spec['name']}")

    # ── Render each variant sequentially ───────────────────────────────

    log("Rendering ref_depth_raw...")
    render_socket(scene, nt, exr.outputs["Depth"],
                  OUTPUT_ROOT / "ref_depth_raw.png")

    log("Rendering ref_arboreal_mask...")
    render_socket(scene, nt, id_mask.outputs["Alpha"],
                  OUTPUT_ROOT / "ref_arboreal_mask.png")

    for name, socket in variants:
        log(f"Rendering {name}...")
        render_socket(scene, nt, socket,
                      OUTPUT_ROOT / f"{name}.png")

    # Save blend
    blend_path = OUTPUT_ROOT / "experiment_depth_canopy.blend"
    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        log(f"Saved {blend_path}")
    except RuntimeError as exc:
        log(f"Skipping blend save: {exc}")

    log("Done. All outputs in:")
    log(f"  {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
