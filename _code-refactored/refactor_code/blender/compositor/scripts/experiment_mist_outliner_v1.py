"""Experimental mist outliner — back-to-basics Kirsch approach.

Renders each variant sequentially through the Composite node (File Output
nodes are unreliable in background mode).

Variants (simplest first):
  v1_raw_kirsch         — raw Mist → Kirsch (no normalize, no threshold, no mask)
  v2_normalized_kirsch  — Mist → Normalize → Kirsch (no threshold, no mask)
  v3_threshold_only     — + hard threshold 0.134 (no mask)
  v4_masked             — + arboreal mask + colour (full depth-style pipeline)
  v5_soft_threshold     — same but threshold=0.04
  v6_preblur            — + 2px pre-blur, threshold=0.06

Run:
  /Applications/Blender.app/Contents/MacOS/Blender --background --python \\
    _code-refactored/refactor_code/blender/compositor/scripts/experiment_mist_outliner_v1.py
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
OUTPUT_ROOT = REPO_ROOT / "_data-refactored" / "compositor" / "outputs" / "experiment_mist_basics_20260408"

TREE_ID = 3
EDGE_COLOR_LINEAR = (0.015996, 0.006048822, 0.043735046, 1.0)


def log(msg: str) -> None:
    print(f"[experiment_mist] {msg}")


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

    # ── Build all variant pipelines first, then render each ────────────

    # We'll collect (name, output_socket) pairs and render them one by one

    variants = []

    # -- V1: raw_kirsch --
    y = 600
    v1_kirsch = nd(nt, "CompositorNodeFilter", "v1_kirsch", (-800, y))
    v1_kirsch.filter_type = "KIRSCH"
    lnk(nt, exr.outputs["Mist"], v1_kirsch.inputs["Image"])
    variants.append(("v1_raw_kirsch", v1_kirsch.outputs["Image"]))

    # -- V2: normalized_kirsch --
    y = 100
    v2_norm = nd(nt, "CompositorNodeNormalize", "v2_norm", (-1000, y))
    lnk(nt, exr.outputs["Mist"], v2_norm.inputs[0])
    v2_kirsch = nd(nt, "CompositorNodeFilter", "v2_kirsch", (-800, y))
    v2_kirsch.filter_type = "KIRSCH"
    lnk(nt, v2_norm.outputs[0], v2_kirsch.inputs["Image"])
    variants.append(("v2_normalized_kirsch", v2_kirsch.outputs["Image"]))

    # -- V3: threshold_only (normalize → kirsch → threshold 0.134) --
    y = -400
    v3_norm = nd(nt, "CompositorNodeNormalize", "v3_norm", (-1000, y))
    lnk(nt, exr.outputs["Mist"], v3_norm.inputs[0])
    v3_kirsch = nd(nt, "CompositorNodeFilter", "v3_kirsch", (-800, y))
    v3_kirsch.filter_type = "KIRSCH"
    lnk(nt, v3_norm.outputs[0], v3_kirsch.inputs["Image"])
    v3_ramp = nd(nt, "CompositorNodeValToRGB", "v3_threshold", (-500, y))
    v3_ramp.color_ramp.interpolation = "CONSTANT"
    v3_ramp.color_ramp.elements[0].position = 0.0
    v3_ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    v3_ramp.color_ramp.elements[1].position = 0.134
    v3_ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    lnk(nt, v3_kirsch.outputs["Image"], v3_ramp.inputs["Fac"])
    variants.append(("v3_threshold_only", v3_ramp.outputs["Image"]))

    # -- V4: masked (full depth-style pipeline, threshold 0.134) --
    y = -900
    v4_norm = nd(nt, "CompositorNodeNormalize", "v4_norm", (-1000, y))
    lnk(nt, exr.outputs["Mist"], v4_norm.inputs[0])
    v4_kirsch = nd(nt, "CompositorNodeFilter", "v4_kirsch", (-800, y))
    v4_kirsch.filter_type = "KIRSCH"
    lnk(nt, v4_norm.outputs[0], v4_kirsch.inputs["Image"])
    v4_ramp = nd(nt, "CompositorNodeValToRGB", "v4_threshold", (-500, y))
    v4_ramp.color_ramp.interpolation = "CONSTANT"
    v4_ramp.color_ramp.elements[0].position = 0.0
    v4_ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    v4_ramp.color_ramp.elements[1].position = 0.134
    v4_ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    lnk(nt, v4_kirsch.outputs["Image"], v4_ramp.inputs["Fac"])
    v4_bw = nd(nt, "CompositorNodeRGBToBW", "v4_bw", (-200, y))
    lnk(nt, v4_ramp.outputs["Image"], v4_bw.inputs["Image"])
    v4_masked = nd(nt, "CompositorNodeMath", "v4_masked", (0, y))
    v4_masked.operation = "MULTIPLY"
    v4_masked.use_clamp = True
    lnk(nt, v4_bw.outputs["Val"], v4_masked.inputs[0])
    lnk(nt, id_mask.outputs["Alpha"], v4_masked.inputs[1])
    v4_rgb = nd(nt, "CompositorNodeRGB", "v4_edge_rgb", (0, y - 180))
    v4_rgb.outputs[0].default_value = EDGE_COLOR_LINEAR
    v4_final = nd(nt, "CompositorNodeSetAlpha", "v4_final", (200, y))
    v4_final.mode = "REPLACE_ALPHA"
    lnk(nt, v4_rgb.outputs[0], v4_final.inputs["Image"])
    lnk(nt, v4_masked.outputs["Value"], v4_final.inputs["Alpha"])
    variants.append(("v4_masked", v4_final.outputs["Image"]))

    # -- V5: soft_threshold (same as v4 but threshold=0.04) --
    y = -1400
    v5_norm = nd(nt, "CompositorNodeNormalize", "v5_norm", (-1000, y))
    lnk(nt, exr.outputs["Mist"], v5_norm.inputs[0])
    v5_kirsch = nd(nt, "CompositorNodeFilter", "v5_kirsch", (-800, y))
    v5_kirsch.filter_type = "KIRSCH"
    lnk(nt, v5_norm.outputs[0], v5_kirsch.inputs["Image"])
    v5_ramp = nd(nt, "CompositorNodeValToRGB", "v5_threshold", (-500, y))
    v5_ramp.color_ramp.interpolation = "CONSTANT"
    v5_ramp.color_ramp.elements[0].position = 0.0
    v5_ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    v5_ramp.color_ramp.elements[1].position = 0.04
    v5_ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    lnk(nt, v5_kirsch.outputs["Image"], v5_ramp.inputs["Fac"])
    v5_bw = nd(nt, "CompositorNodeRGBToBW", "v5_bw", (-200, y))
    lnk(nt, v5_ramp.outputs["Image"], v5_bw.inputs["Image"])
    v5_masked = nd(nt, "CompositorNodeMath", "v5_masked", (0, y))
    v5_masked.operation = "MULTIPLY"
    v5_masked.use_clamp = True
    lnk(nt, v5_bw.outputs["Val"], v5_masked.inputs[0])
    lnk(nt, id_mask.outputs["Alpha"], v5_masked.inputs[1])
    v5_rgb = nd(nt, "CompositorNodeRGB", "v5_edge_rgb", (0, y - 180))
    v5_rgb.outputs[0].default_value = EDGE_COLOR_LINEAR
    v5_final = nd(nt, "CompositorNodeSetAlpha", "v5_final", (200, y))
    v5_final.mode = "REPLACE_ALPHA"
    lnk(nt, v5_rgb.outputs[0], v5_final.inputs["Image"])
    lnk(nt, v5_masked.outputs["Value"], v5_final.inputs["Alpha"])
    variants.append(("v5_soft_threshold", v5_final.outputs["Image"]))

    # -- V6: preblur (2px blur before kirsch, threshold=0.06) --
    y = -1900
    v6_norm = nd(nt, "CompositorNodeNormalize", "v6_norm", (-1000, y))
    lnk(nt, exr.outputs["Mist"], v6_norm.inputs[0])
    v6_blur = nd(nt, "CompositorNodeBlur", "v6_preblur", (-800, y))
    v6_blur.filter_type = "GAUSS"
    v6_blur.use_relative = False
    v6_blur.size_x = 2
    v6_blur.size_y = 2
    lnk(nt, v6_norm.outputs[0], v6_blur.inputs[0])
    v6_kirsch = nd(nt, "CompositorNodeFilter", "v6_kirsch", (-600, y))
    v6_kirsch.filter_type = "KIRSCH"
    lnk(nt, v6_blur.outputs[0], v6_kirsch.inputs["Image"])
    v6_ramp = nd(nt, "CompositorNodeValToRGB", "v6_threshold", (-300, y))
    v6_ramp.color_ramp.interpolation = "CONSTANT"
    v6_ramp.color_ramp.elements[0].position = 0.0
    v6_ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    v6_ramp.color_ramp.elements[1].position = 0.06
    v6_ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    lnk(nt, v6_kirsch.outputs["Image"], v6_ramp.inputs["Fac"])
    v6_bw = nd(nt, "CompositorNodeRGBToBW", "v6_bw", (-100, y))
    lnk(nt, v6_ramp.outputs["Image"], v6_bw.inputs["Image"])
    v6_masked = nd(nt, "CompositorNodeMath", "v6_masked", (100, y))
    v6_masked.operation = "MULTIPLY"
    v6_masked.use_clamp = True
    lnk(nt, v6_bw.outputs["Val"], v6_masked.inputs[0])
    lnk(nt, id_mask.outputs["Alpha"], v6_masked.inputs[1])
    v6_rgb = nd(nt, "CompositorNodeRGB", "v6_edge_rgb", (100, y - 180))
    v6_rgb.outputs[0].default_value = EDGE_COLOR_LINEAR
    v6_final = nd(nt, "CompositorNodeSetAlpha", "v6_final", (300, y))
    v6_final.mode = "REPLACE_ALPHA"
    lnk(nt, v6_rgb.outputs[0], v6_final.inputs["Image"])
    lnk(nt, v6_masked.outputs["Value"], v6_final.inputs["Alpha"])
    variants.append(("v6_preblur", v6_final.outputs["Image"]))

    # ── Render each variant sequentially via Composite node ────────────

    # First render reference diagnostics
    log("Rendering ref_mist_raw...")
    render_socket(scene, nt, exr.outputs["Mist"],
                  OUTPUT_ROOT / "ref_mist_raw.png")

    log("Rendering ref_mist_normalized...")
    norm_ref = nd(nt, "CompositorNodeNormalize", "norm_ref", (-1400, -600))
    lnk(nt, exr.outputs["Mist"], norm_ref.inputs[0])
    render_socket(scene, nt, norm_ref.outputs[0],
                  OUTPUT_ROOT / "ref_mist_normalized.png")

    log("Rendering ref_arboreal_mask...")
    render_socket(scene, nt, id_mask.outputs["Alpha"],
                  OUTPUT_ROOT / "ref_arboreal_mask.png")

    # Now render each variant
    for name, socket in variants:
        log(f"Rendering {name}...")
        render_socket(scene, nt, socket,
                      OUTPUT_ROOT / f"{name}.png")

    # Save blend for GUI inspection
    blend_path = OUTPUT_ROOT / "experiment_mist_basics.blend"
    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        log(f"Saved {blend_path}")
    except RuntimeError as exc:
        log(f"Skipping blend save: {exc}")

    log("Done. All outputs in:")
    log(f"  {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
