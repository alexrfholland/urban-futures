"""Experiment v6 — push boost+threshold harder on mist.

mist_v6_b10_t40 was the best so far but still has internal canopy mess.
Push higher to clean it out while keeping outer silhouettes.

Variants:
  v1_b15_t50   — boost ×15, threshold 0.50
  v2_b15_t60   — boost ×15, threshold 0.60
  v3_b20_t50   — boost ×20, threshold 0.50
  v4_b20_t60   — boost ×20, threshold 0.60
  v5_b20_t70   — boost ×20, threshold 0.70
  v6_b30_t60   — boost ×30, threshold 0.60
  v7_b30_t70   — boost ×30, threshold 0.70
  v8_b30_t80   — boost ×30, threshold 0.80

Run:
  /Applications/Blender.app/Contents/MacOS/Blender --background --python \\
    _futureSim_refactored/blender/compositor/scripts/experiment_mist_push_v6.py
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
OUTPUT_ROOT = REPO_ROOT / "_data-refactored" / "compositor" / "outputs" / "experiment_mist_push_20260408"

TREE_ID = 3
EDGE_COLOR_LINEAR = (0.015996, 0.006048822, 0.043735046, 1.0)


def log(msg: str) -> None:
    print(f"[experiment_v6] {msg}")


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


def build_variant(nt, mist_socket, mask_socket, prefix, y, boost, threshold):
    """Mist → Kirsch → boost(×N, unclamped) → hard threshold → BW → mask → colour"""
    x = -1200

    # Kirsch on raw mist
    kirsch = nd(nt, "CompositorNodeFilter", f"{prefix}_kirsch", (x, y))
    kirsch.filter_type = "KIRSCH"
    lnk(nt, mist_socket, kirsch.inputs["Image"])
    x += 200

    # Boost — UNCLAMPED so full range reaches the threshold
    mult = nd(nt, "CompositorNodeMath", f"{prefix}_boost", (x, y))
    mult.operation = "MULTIPLY"
    mult.use_clamp = False
    lnk(nt, kirsch.outputs["Image"], mult.inputs[0])
    mult.inputs[1].default_value = boost
    x += 200

    # Hard threshold
    ramp = nd(nt, "CompositorNodeValToRGB", f"{prefix}_threshold", (x, y))
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    ramp.color_ramp.elements[1].position = threshold
    ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    lnk(nt, mult.outputs["Value"], ramp.inputs["Fac"])
    x += 300

    # BW
    bw = nd(nt, "CompositorNodeRGBToBW", f"{prefix}_bw", (x, y))
    lnk(nt, ramp.outputs["Image"], bw.inputs["Image"])
    x += 200

    # Mask
    masked = nd(nt, "CompositorNodeMath", f"{prefix}_masked", (x, y))
    masked.operation = "MULTIPLY"
    masked.use_clamp = True
    lnk(nt, bw.outputs["Val"], masked.inputs[0])
    lnk(nt, mask_socket, masked.inputs[1])
    x += 200

    # Colour + alpha
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

    specs = [
        ("v1_b15_t50", 15.0, 0.50),
        ("v2_b15_t60", 15.0, 0.60),
        ("v3_b20_t50", 20.0, 0.50),
        ("v4_b20_t60", 20.0, 0.60),
        ("v5_b20_t70", 20.0, 0.70),
        ("v6_b30_t60", 30.0, 0.60),
        ("v7_b30_t70", 30.0, 0.70),
        ("v8_b30_t80", 30.0, 0.80),
    ]

    variants = []
    for i, (name, boost, threshold) in enumerate(specs):
        y = 800 - i * 400
        s = build_variant(nt, exr.outputs["Mist"], id_mask.outputs["Alpha"],
                          name, y, boost, threshold)
        variants.append((name, s))
        log(f"Built {name} (boost=×{boost}, threshold={threshold})")

    # ── Render ─────────────────────────────────────────────────────────

    for name, socket in variants:
        log(f"Rendering {name}...")
        render_socket(scene, nt, socket, OUTPUT_ROOT / f"{name}.png")

    # Save blend
    blend_path = OUTPUT_ROOT / "experiment_mist_push.blend"
    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        log(f"Saved {blend_path}")
    except RuntimeError as exc:
        log(f"Skipping blend save: {exc}")

    log("Done.")


if __name__ == "__main__":
    main()
