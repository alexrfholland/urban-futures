"""Experiment v9 — posterize mist into discrete bands before Kirsch.

Quantizing mist into N flat bands suppresses internal canopy detail
(which lives within a single band) while preserving edges at band
boundaries (which correspond to large depth jumps like tree silhouettes).

Variants sweep band count and threshold:
  v01_bands3_t05    — 3 bands, threshold 0.05
  v02_bands5_t05    — 5 bands, threshold 0.05
  v03_bands10_t05   — 10 bands, threshold 0.05
  v04_bands20_t05   — 20 bands, threshold 0.05
  v05_bands50_t05   — 50 bands, threshold 0.05
  v06_bands5_t10    — 5 bands, threshold 0.10
  v07_bands10_t10   — 10 bands, threshold 0.10
  v08_bands5_t02    — 5 bands, threshold 0.02
  v09_bands3_t10    — 3 bands, threshold 0.10
  v10_bands10_t02   — 10 bands, threshold 0.02

Run:
  /Applications/Blender.app/Contents/MacOS/Blender --background --python \\
    _code-refactored/refactor_code/blender/compositor/scripts/experiment_mist_bands_v9.py
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
OUTPUT_ROOT = REPO_ROOT / "_data-refactored" / "compositor" / "outputs" / "experiment_mist_bands_20260408"

TREE_ID = 3
EDGE_COLOR_LINEAR = (0.015996, 0.006048822, 0.043735046, 1.0)


def log(msg: str) -> None:
    print(f"[experiment_v9] {msg}")


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


def build_variant(nt, mist_socket, mask_socket, prefix, y, num_bands, threshold):
    """Mist → posterize(N bands) → Kirsch → threshold → BW → mask → colour.

    Posterize: multiply by N, floor, divide by N.
    This quantizes continuous mist into N flat steps.
    """
    x = -1600

    # Step 1: Multiply by num_bands
    mult_up = nd(nt, "CompositorNodeMath", f"{prefix}_mult_up", (x, y))
    mult_up.operation = "MULTIPLY"
    mult_up.use_clamp = False
    lnk(nt, mist_socket, mult_up.inputs[0])
    mult_up.inputs[1].default_value = float(num_bands)
    x += 200

    # Step 2: Floor
    floor = nd(nt, "CompositorNodeMath", f"{prefix}_floor", (x, y))
    floor.operation = "FLOOR"
    floor.use_clamp = False
    lnk(nt, mult_up.outputs["Value"], floor.inputs[0])
    x += 200

    # Step 3: Divide by num_bands to get back to 0-1
    div_down = nd(nt, "CompositorNodeMath", f"{prefix}_div_down", (x, y))
    div_down.operation = "DIVIDE"
    div_down.use_clamp = False
    lnk(nt, floor.outputs["Value"], div_down.inputs[0])
    div_down.inputs[1].default_value = float(num_bands)
    x += 200

    # Kirsch edge detection on banded mist
    kirsch = nd(nt, "CompositorNodeFilter", f"{prefix}_kirsch", (x, y))
    kirsch.filter_type = "KIRSCH"
    lnk(nt, div_down.outputs["Value"], kirsch.inputs["Image"])
    x += 200

    # Hard threshold
    ramp = nd(nt, "CompositorNodeValToRGB", f"{prefix}_threshold", (x, y))
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    ramp.color_ramp.elements[1].position = threshold
    ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    lnk(nt, kirsch.outputs["Image"], ramp.inputs["Fac"])
    x += 300

    # BW
    bw = nd(nt, "CompositorNodeRGBToBW", f"{prefix}_bw", (x, y))
    lnk(nt, ramp.outputs["Image"], bw.inputs["Image"])
    x += 200

    # Mask with arboreal ID
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

    exr = nd(nt, "CompositorNodeImage", "EXR", (-2200, 0))
    exr.image = bpy.data.images.load(str(POSITIVE_EXR), check_existing=True)

    id_mask = nd(nt, "CompositorNodeIDMask", "arboreal_mask", (-1800, -200))
    id_mask.index = TREE_ID
    id_mask.use_antialiasing = True
    lnk(nt, exr.outputs["IndexOB"], id_mask.inputs["ID value"])

    comp = nd(nt, "CompositorNodeComposite", "Composite", (2000, 0))
    view = nd(nt, "CompositorNodeViewer", "Viewer", (2000, -200))

    # ── Variants ───────────────────────────────────────────────────────

    specs = [
        ("v01_bands3_t05",   3, 0.05),
        ("v02_bands5_t05",   5, 0.05),
        ("v03_bands10_t05", 10, 0.05),
        ("v04_bands20_t05", 20, 0.05),
        ("v05_bands50_t05", 50, 0.05),
        ("v06_bands5_t10",   5, 0.10),
        ("v07_bands10_t10", 10, 0.10),
        ("v08_bands5_t02",   5, 0.02),
        ("v09_bands3_t10",   3, 0.10),
        ("v10_bands10_t02", 10, 0.02),
    ]

    variants = []
    for i, (name, bands, threshold) in enumerate(specs):
        y = 1000 - i * 400
        s = build_variant(nt, exr.outputs["Mist"], id_mask.outputs["Alpha"],
                          name, y, bands, threshold)
        variants.append((name, s))
        log(f"Built {name} ({bands} bands, threshold={threshold})")

    # ── Render ─────────────────────────────────────────────────────────

    for name, socket in variants:
        log(f"Rendering {name}...")
        render_socket(scene, nt, socket, OUTPUT_ROOT / f"{name}.png")

    # Save blend
    blend_path = OUTPUT_ROOT / "experiment_mist_bands.blend"
    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        log(f"Saved {blend_path}")
    except RuntimeError as exc:
        log(f"Skipping blend save: {exc}")

    log("Done.")


if __name__ == "__main__":
    main()
