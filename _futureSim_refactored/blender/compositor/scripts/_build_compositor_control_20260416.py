"""Build canonical compositor_control.blend.

Graph (all in scene "Current", compositor nodes):

    Image node ("Current Control :: EXR", multilayer)
        --[control]--> Math Divide (/5)
                         --> ColorRamp (Constant interpolation)
                                stops: 0.0->0.0, 0.2->0.8, 0.4->0.5,
                                       0.6->0.0, 0.8->0.0
                           --> File Output ("Current Control :: Outputs")
                                  slot: "control"  (BW, 8-bit PNG)

Mapping rationale:
    control.V holds integer tree-ownership classes from bV2:
      1 = street, 2 = park, 3 = reserve, 4 = improved.
    We emit a greyscale mask that drives per-class desaturation when the
    PSB clips a Hue/Sat (-100) adjustment layer to sizes:
      street   -> 0.80 (80% desat)
      park     -> 0.50 (50% desat)
      reserve  -> 0.00 (untouched)
      improved -> 0.00 (untouched)
    Dividing by 5 normalises inputs to 0.2 / 0.4 / 0.6 / 0.8 so a ColorRamp
    with 5 stops can hit each class at its own stop.

Run headless to produce the blend:
    /Applications/Blender.app/Contents/MacOS/Blender --background \
      --factory-startup \
      --python _build_compositor_control_20260416.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import bpy  # type: ignore[import-not-found]

REPO_ROOT = Path(__file__).resolve().parents[4]
CANONICAL = REPO_ROOT / "_futureSim_refactored" / "blender" / "compositor" / "canonical_templates"
BLEND_OUT = CANONICAL / "compositor_control.blend"
SAMPLE_EXR = REPO_ROOT / "_data-refactored" / "blenderv2" / "output" / "_library" / \
    "parade_library" / "parade_library__positive_state__8k.exr"


def main() -> int:
    if not SAMPLE_EXR.exists():
        print(f"ERROR: sample EXR not found for layer enumeration: {SAMPLE_EXR}",
              file=sys.stderr)
        return 1

    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Scene: Current, 8K, sRGB/Standard.
    scene = bpy.context.scene
    scene.name = "Current"
    scene.render.resolution_x = 7680
    scene.render.resolution_y = 4320
    scene.render.resolution_percentage = 100
    try:
        scene.display_settings.display_device = "sRGB"
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
    except Exception:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1

    scene.use_nodes = True
    tree = scene.node_tree
    for n in list(tree.nodes):
        tree.nodes.remove(n)

    img = bpy.data.images.load(str(SAMPLE_EXR), check_existing=True)
    img.source = "FILE"

    img_node = tree.nodes.new("CompositorNodeImage")
    img_node.name = "Current Control :: EXR"
    img_node.label = "Current Control :: EXR"
    img_node.image = img
    img_node.location = (-800, 0)

    control_sock = img_node.outputs.get("control")
    if control_sock is None:
        print(f"ERROR: EXR lacks 'control' layer; outputs: "
              f"{[o.name for o in img_node.outputs]}", file=sys.stderr)
        return 2

    divide = tree.nodes.new("CompositorNodeMath")
    divide.operation = "DIVIDE"
    divide.name = "Current Control :: Divide5"
    divide.label = "/5"
    divide.inputs[1].default_value = 5.0
    divide.location = (-500, 0)

    ramp = tree.nodes.new("CompositorNodeValToRGB")
    ramp.name = "Current Control :: Ramp"
    ramp.label = "class->desat"
    ramp.location = (-250, 0)
    cr = ramp.color_ramp
    cr.interpolation = "CONSTANT"
    # Wipe default stops (0 and 1) and install our 5.
    while len(cr.elements) > 1:
        cr.elements.remove(cr.elements[-1])
    cr.elements[0].position = 0.0
    cr.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    for pos, v in [(0.2, 0.8), (0.4, 0.5), (0.6, 0.0), (0.8, 0.0)]:
        el = cr.elements.new(pos)
        el.color = (v, v, v, 1.0)

    fo = tree.nodes.new("CompositorNodeOutputFile")
    fo.name = "Current Control :: Outputs"
    fo.label = "Current Control :: Outputs"
    fo.location = (50, 0)
    fo.format.file_format = "PNG"
    fo.format.color_mode = "BW"
    fo.format.color_depth = "8"
    fo.format.compression = 15
    while len(fo.file_slots) > 1:
        fo.file_slots.remove(fo.file_slots[-1])
    fo.file_slots[0].path = "control_"

    tree.links.new(control_sock, divide.inputs[0])
    tree.links.new(divide.outputs[0], ramp.inputs[0])
    tree.links.new(ramp.outputs[0], fo.inputs[0])

    # Drop the image again so the blend ships without a baked-in filepath —
    # the runner will load whichever EXR it's told.
    img_node.image = None
    bpy.data.images.remove(img)

    CANONICAL.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_OUT), compress=True)
    print(f"wrote {BLEND_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
