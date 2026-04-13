"""Option 2 test: render compositor_intervention_int.blend with animation=True.

Single render, File Output unmuted, one-frame animation render.
No per-slot Composite loop.  No blend edits.

Expected: all 9 File Output slots land on disk with '0001' frame suffix,
which we then strip.

If slots are missing, next step is to add a permanent dummy Render Layers
node to the canonical blend and retry.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import bpy

_THIS = Path(__file__).resolve()
REPO_ROOT = next(p.parent for p in _THIS.parents if p.name == "_futureSim_refactored")
_SCRIPTS_DIR = _THIS.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from _exr_header import read_exr_dimensions  # noqa: E402

BLEND_PATH = (
    REPO_ROOT / "_futureSim_refactored" / "blender" / "compositor"
    / "canonical_templates" / "compositor_intervention_int.blend"
)
EXR_PATH = (
    REPO_ROOT / "_data-refactored" / "blenderv2" / "output"
    / "4.10" / "parade_timeline"
    / "parade_timeline__bioenvelope_positive__8k.exr"
)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = (
    REPO_ROOT / "_data-refactored" / "compositor" / "outputs"
    / "_experiments" / f"option2_animation_intervention_int__{stamp}"
)

SCENE_NAME = "Current"
EXR_NODE_NAME = "InterventionInt::EXR Input"
RAW_HUB_NODE_NAME = "InterventionInt::Raw"
FILE_OUTPUT_NAME = "InterventionInt::Outputs"
AOV_SOCKET_PREFIX = "intervention_bioenvelope_ply"


def log(msg: str) -> None:
    print(f"[option2_test] {msg}")


def require_node(tree, name):
    n = tree.nodes.get(name)
    if n is None:
        raise ValueError(f"Missing node: {name}")
    return n


def main() -> None:
    if not BLEND_PATH.exists():
        raise FileNotFoundError(BLEND_PATH)
    if not EXR_PATH.exists():
        raise FileNotFoundError(EXR_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(BLEND_PATH))
    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None or scene.node_tree is None:
        raise ValueError(f"Scene '{SCENE_NAME}' not found")

    tree = scene.node_tree

    # Repath the EXR input and rewire the AOV socket to the raw hub
    exr_node = require_node(tree, EXR_NODE_NAME)
    raw_hub = require_node(tree, RAW_HUB_NODE_NAME)
    img = bpy.data.images.load(str(EXR_PATH), check_existing=True)
    exr_node.image = img
    log(f"loaded {EXR_PATH.name}")

    aov_socket = None
    for sock in exr_node.outputs:
        if sock.name.startswith(AOV_SOCKET_PREFIX):
            aov_socket = sock
            break
    if aov_socket is None:
        available = [s.name for s in exr_node.outputs]
        raise ValueError(f"No AOV socket '{AOV_SOCKET_PREFIX}*'. Available: {available}")

    for link in list(raw_hub.inputs[0].links):
        tree.links.remove(link)
    tree.links.new(aov_socket, raw_hub.inputs[0])
    log(f"wired {aov_socket.name} -> raw hub")

    # Detect resolution
    w, h = read_exr_dimensions(str(EXR_PATH))
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.resolution_percentage = 100
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    log(f"resolution {w}x{h}")

    # Single frame animation render
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1

    # Point File Output base_path at our dir; do NOT mute
    fo = require_node(tree, FILE_OUTPUT_NAME)
    fo.base_path = str(OUTPUT_DIR)
    fo.mute = False

    # Mute any OTHER File Output nodes to keep the test clean
    for n in tree.nodes:
        if n.bl_idname == "CompositorNodeOutputFile" and n.name != FILE_OUTPUT_NAME:
            n.mute = True

    # Discard target for scene.render (still needs a path)
    scene.render.filepath = str(OUTPUT_DIR / "_discard_render.png")

    log("rendering animation (frame 1 only)...")
    bpy.ops.render.render(animation=True, scene=scene.name)
    log("render done")

    # Rename *_0001.png -> *.png
    for rendered in sorted(OUTPUT_DIR.glob("*_0001.png")):
        final = rendered.with_name(rendered.name.replace("_0001", ""))
        if final.exists():
            final.unlink()
        rendered.replace(final)
        log(f"renamed {rendered.name} -> {final.name}")

    # Clean discard
    for discard in OUTPUT_DIR.glob("_discard_render*"):
        discard.unlink()

    # Audit: every file_slot should have a PNG on disk
    expected = []
    for slot in fo.file_slots:
        stem = slot.path.rstrip("_")
        expected.append(stem)

    log("=== AUDIT ===")
    missing = []
    present = []
    for stem in expected:
        png = OUTPUT_DIR / f"{stem}.png"
        if png.exists():
            present.append(stem)
            log(f"  OK   {stem}.png ({png.stat().st_size} bytes)")
        else:
            missing.append(stem)
            log(f"  MISS {stem}.png")
    log(f"=== {len(present)}/{len(expected)} slots present ===")
    if missing:
        log(f"MISSING: {missing}")
        sys.exit(2)


if __name__ == "__main__":
    main()
