"""Experiment v5 — boost + kirschsizes width-shaping pipeline for mist.

The v3 boost experiment recovered tree shapes but they were filled blobs.
This experiment combines the boost with the kirschsizes dual-threshold +
dilate/erode width-shaping pipeline that produced the clean Mar-29 reference.

The kirschsizes pipeline creates line-like outlines by:
  1. Two LINEAR ramps (presence + strong) create variable-width response
  2. Dilate/erode controls line thickness
  3. MAXIMUM merges the two responses
  4. Blur softens the edges

Variants sweep boost factor and threshold aggression:
  v1_classic_b5        — boost ×5, original kirschsizes thresholds
  v2_classic_b10       — boost ×10, original thresholds
  v3_tight_b10         — boost ×10, tighter thresholds (less canopy mess)
  v4_tight_b15         — boost ×15, tighter thresholds
  v5_silhouette_b10    — boost ×10, aggressive thresholds (silhouettes only)
  v6_silhouette_b20    — boost ×20, aggressive thresholds

Run:
  /Applications/Blender.app/Contents/MacOS/Blender --background --python \\
    _code-refactored/refactor_code/blender/compositor/scripts/experiment_mist_kirschsizes_v5.py
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
OUTPUT_ROOT = REPO_ROOT / "_data-refactored" / "compositor" / "outputs" / "experiment_mist_kirschsizes_20260408"

TREE_ID = 3
EDGE_COLOR_LINEAR = (0.015996, 0.006048822, 0.043735046, 1.0)


def log(msg: str) -> None:
    print(f"[experiment_v5] {msg}")


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


def build_kirschsizes_variant(nt, mist_socket, mask_socket, prefix, y,
                               boost,
                               presence_low, presence_high,
                               strong_low, strong_high,
                               core_distance, wide_distance,
                               blur_pixels,
                               screen_gain_bottom=0.0, screen_gain_power=0.0):
    """
    Mist → Kirsch → boost(×N) → dual-threshold width shaping → mask → colour.

    This replicates the kirschsizes pipeline from render_exr_arboreal_mist_variants_v2
    but inserts a boost multiply after the Kirsch filter.
    """
    x = -1400

    # Kirsch on raw mist (no normalize — normalize was squashing the range)
    kirsch = nd(nt, "CompositorNodeFilter", f"{prefix}_kirsch", (x, y))
    kirsch.filter_type = "KIRSCH"
    lnk(nt, mist_socket, kirsch.inputs["Image"])
    x += 200

    # Boost — amplify weak tree edges
    bmult = nd(nt, "CompositorNodeMath", f"{prefix}_boost", (x, y))
    bmult.operation = "MULTIPLY"
    bmult.use_clamp = False  # Don't clamp — let the ramps handle range
    lnk(nt, kirsch.outputs["Image"], bmult.inputs[0])
    bmult.inputs[1].default_value = boost
    boosted = bmult.outputs["Value"]
    x += 200

    # Normalize the boosted signal to use full 0-1 range
    norm = nd(nt, "CompositorNodeNormalize", f"{prefix}_norm", (x, y))
    lnk(nt, boosted, norm.inputs[0])
    normalized = norm.outputs[0]
    x += 200

    # Presence ramp (catches weak edges, thin core)
    presence = nd(nt, "CompositorNodeValToRGB", f"{prefix}_presence", (x, y + 80))
    presence.color_ramp.interpolation = "LINEAR"
    presence.color_ramp.elements[0].position = presence_low
    presence.color_ramp.elements[0].color = (0, 0, 0, 1)
    presence.color_ramp.elements[1].position = presence_high
    presence.color_ramp.elements[1].color = (1, 1, 1, 1)
    lnk(nt, normalized, presence.inputs["Fac"])

    # Strong ramp (catches strong edges, wider)
    strong = nd(nt, "CompositorNodeValToRGB", f"{prefix}_strong", (x, y - 80))
    strong.color_ramp.interpolation = "LINEAR"
    strong.color_ramp.elements[0].position = strong_low
    strong.color_ramp.elements[0].color = (0, 0, 0, 1)
    strong.color_ramp.elements[1].position = strong_high
    strong.color_ramp.elements[1].color = (1, 1, 1, 1)
    lnk(nt, normalized, strong.inputs["Fac"])
    x += 300

    # Dilate core (thin)
    core = nd(nt, "CompositorNodeDilateErode", f"{prefix}_core", (x, y + 80))
    core.mode = "DISTANCE"
    core.distance = core_distance
    lnk(nt, presence.outputs["Image"], core.inputs[0])

    # Dilate wide (thick)
    wide = nd(nt, "CompositorNodeDilateErode", f"{prefix}_wide", (x, y - 80))
    wide.mode = "DISTANCE"
    wide.distance = wide_distance
    lnk(nt, strong.outputs["Image"], wide.inputs[0])
    x += 200

    # Combine with MAXIMUM
    combine = nd(nt, "CompositorNodeMath", f"{prefix}_combine", (x, y))
    combine.operation = "MAXIMUM"
    combine.use_clamp = True
    lnk(nt, core.outputs[0], combine.inputs[0])
    lnk(nt, wide.outputs[0], combine.inputs[1])
    x += 200

    # Soften blur
    soften = nd(nt, "CompositorNodeBlur", f"{prefix}_soften", (x, y))
    soften.filter_type = "GAUSS"
    soften.use_relative = False
    soften.size_x = blur_pixels
    soften.size_y = blur_pixels
    lnk(nt, combine.outputs[0], soften.inputs[0])
    x += 200

    # Mask
    masked = nd(nt, "CompositorNodeMath", f"{prefix}_masked", (x, y))
    masked.operation = "MULTIPLY"
    masked.use_clamp = True
    lnk(nt, soften.outputs[0], masked.inputs[0])
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

    # ── Variants ───────────────────────────────────────────────────────

    # Original kirschsizes thresholds from the working Mar-29 pipeline:
    # presence: [0.07, 0.22], strong: [0.18, 0.34], core: 1, wide: 2, blur: 1

    specs = [
        {
            "name": "v1_classic_b5",
            "boost": 5.0,
            "presence_low": 0.07, "presence_high": 0.22,
            "strong_low": 0.18, "strong_high": 0.34,
            "core_distance": 1, "wide_distance": 2, "blur_pixels": 1,
        },
        {
            "name": "v2_classic_b10",
            "boost": 10.0,
            "presence_low": 0.07, "presence_high": 0.22,
            "strong_low": 0.18, "strong_high": 0.34,
            "core_distance": 1, "wide_distance": 2, "blur_pixels": 1,
        },
        {
            "name": "v3_tight_b10",
            "boost": 10.0,
            "presence_low": 0.15, "presence_high": 0.40,
            "strong_low": 0.30, "strong_high": 0.55,
            "core_distance": 1, "wide_distance": 1, "blur_pixels": 1,
        },
        {
            "name": "v4_tight_b15",
            "boost": 15.0,
            "presence_low": 0.15, "presence_high": 0.40,
            "strong_low": 0.30, "strong_high": 0.55,
            "core_distance": 1, "wide_distance": 1, "blur_pixels": 1,
        },
        {
            "name": "v5_silhouette_b10",
            "boost": 10.0,
            "presence_low": 0.25, "presence_high": 0.50,
            "strong_low": 0.40, "strong_high": 0.65,
            "core_distance": 0, "wide_distance": 1, "blur_pixels": 1,
        },
        {
            "name": "v6_silhouette_b20",
            "boost": 20.0,
            "presence_low": 0.25, "presence_high": 0.50,
            "strong_low": 0.40, "strong_high": 0.65,
            "core_distance": 0, "wide_distance": 1, "blur_pixels": 1,
        },
    ]

    variants = []
    for i, spec in enumerate(specs):
        y = 800 - i * 500
        s = build_kirschsizes_variant(
            nt, exr.outputs["Mist"], id_mask.outputs["Alpha"],
            spec["name"], y,
            spec["boost"],
            spec["presence_low"], spec["presence_high"],
            spec["strong_low"], spec["strong_high"],
            spec["core_distance"], spec["wide_distance"],
            spec["blur_pixels"],
        )
        variants.append((spec["name"], s))
        log(f"Built {spec['name']} (boost={spec['boost']}x)")

    # ── Render ─────────────────────────────────────────────────────────

    for name, socket in variants:
        log(f"Rendering {name}...")
        render_socket(scene, nt, socket, OUTPUT_ROOT / f"{name}.png")

    # Save blend
    blend_path = OUTPUT_ROOT / "experiment_mist_kirschsizes.blend"
    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        log(f"Saved {blend_path}")
    except RuntimeError as exc:
        log(f"Skipping blend save: {exc}")

    log("Done.")


if __name__ == "__main__":
    main()
