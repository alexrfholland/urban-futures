"""Build canonical proposal_and_interventions.blend.

Single-EXR-input compositor that fans out per-intervention outputs for all
five proposal families using the integer encoding from
``proposal_framebuffers.FRAMEBUFFER_STATE_MAPPINGS``.

Run once in Blender headless to create the blend::

    blender --background --factory-startup --python \\
        _futureSim_refactored/blender/compositor/scripts/_build_proposal_and_interventions.py

Saves to ``canonical_templates/proposal_and_interventions.blend``.
"""

from __future__ import annotations

from pathlib import Path

import bpy

REPO_ROOT = next(
    p.parent for p in Path(__file__).resolve().parents
    if p.name == "_futureSim_refactored"
)
CANONICAL_ROOT = (
    REPO_ROOT / "_futureSim_refactored" / "blender" / "compositor" / "canonical_templates"
)
OUTPUT_BLEND = CANONICAL_ROOT / "proposal_and_interventions.blend"

# Reference EXR for the initial Image node — runner overrides this at render time.
REFERENCE_EXR = (
    REPO_ROOT / "_data-refactored" / "blenderv2" / "output"
    / "4.10" / "city_timeline" / "city_timeline__positive_state__8k64s.exr"
)

ARBOREAL_OB_INDEX = 3
USE_ARBOREAL_MASK = False  # Set True to multiply by IndexOB == 3

# ---------------------------------------------------------------------------
# Intervention specs: (family_aov, [(integer_value, slug, hex_color), ...])
#
# Source of truth: proposal_framebuffers.FRAMEBUFFER_STATE_MAPPINGS + constants.py
# Only intervention values (>= 2) are exported, plus rejected (1) for
# release-control only — mirroring the original PROPOSAL_BRANCH_SPECS.
# Colours are carried from the archived builder by integer value.
# ---------------------------------------------------------------------------
INTERVENTION_SPECS: list[tuple[str, list[tuple[int, str, str]]]] = [
    ("proposal-decay", [
        (2, "buffer-feature", "#B83B6B"),
        (3, "brace-feature", "#D9638C"),
    ]),
    ("proposal-release-control", [
        (1, "rejected", "#333333"),
        (2, "reduce-canopy-pruning", "#808080"),
        (3, "eliminate-canopy-pruning", "#FFFFFF"),
    ]),
    ("proposal-recruit", [
        (2, "rewild-smaller-patch", "#C5E28E"),
        (3, "rewild-larger-patch", "#5CB85C"),
    ]),
    ("proposal-colonise", [
        (2, "larger-patches-rewild", "#5CB85C"),
        (3, "enrich-envelope", "#8CCC4F"),
        (4, "roughen-envelope", "#B87A38"),
    ]),
    ("proposal-deploy-structure", [
        (2, "adapt-utility-pole", "#FF0000"),
        (3, "translocate-log", "#8F89BF"),
        (4, "upgrade-feature", "#CE6DD9"),
    ]),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def srgb_to_linear(c: int) -> float:
    n = c / 255.0
    return n / 12.92 if n <= 0.04045 else ((n + 0.055) / 1.055) ** 2.4


def hex_to_rgba(h: str) -> tuple[float, float, float, float]:
    h = h.lstrip("#")
    return (
        srgb_to_linear(int(h[0:2], 16)),
        srgb_to_linear(int(h[2:4], 16)),
        srgb_to_linear(int(h[4:6], 16)),
        1.0,
    )


def log(msg: str) -> None:
    print(f"[build_proposal_and_interventions] {msg}")


def ensure_link(tree: bpy.types.NodeTree, from_socket, to_socket) -> None:
    for link in list(to_socket.links):
        tree.links.remove(link)
    tree.links.new(from_socket, to_socket)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = "Current"
    scene.use_nodes = True
    tree = scene.node_tree

    # Clear default nodes
    for node in list(tree.nodes):
        tree.nodes.remove(node)

    # --- Render Layers (required for File Output to fire in Blender 4.2) ------
    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.name = "RenderLayers"
    rl.location = (-400, 0)

    # --- EXR input -----------------------------------------------------------
    exr = tree.nodes.new("CompositorNodeImage")
    exr.name = "EXR"
    exr.label = "EXR"
    exr.location = (0, 0)
    if REFERENCE_EXR.exists():
        img = bpy.data.images.load(str(REFERENCE_EXR), check_existing=True)
        img.source = "FILE"
        exr.image = img
        log(f"loaded reference EXR: {REFERENCE_EXR.name}")
    else:
        log(f"reference EXR not found: {REFERENCE_EXR} — placeholder node created")

    # --- Optional arboreal mask (IndexOB == 3) --------------------------------
    arboreal_mask = None
    if USE_ARBOREAL_MASK:
        id_mask = tree.nodes.new("CompositorNodeIDMask")
        id_mask.name = "ArborealMask"
        id_mask.label = "ArborealMask (IndexOB == 3)"
        id_mask.index = ARBOREAL_OB_INDEX
        id_mask.use_antialiasing = True
        id_mask.location = (300, -200)
        tree.links.new(exr.outputs["IndexOB"], id_mask.inputs["ID value"])
        arboreal_mask = id_mask.outputs["Alpha"]

    # --- Per-family, per-intervention chains ---------------------------------
    wired: list[tuple[str, bpy.types.NodeSocket]] = []
    y_cursor = 200.0

    for family_aov, interventions in INTERVENTION_SPECS:
        family_short = family_aov.replace("proposal-", "")

        # Family frame
        frame = tree.nodes.new("NodeFrame")
        frame.name = f"Frame::{family_aov}"
        frame.label = family_aov
        frame.use_custom_color = True
        frame.color = (0.12, 0.14, 0.12)

        for val, slug, hex_color in interventions:
            slot_path = f"proposal-{family_short}-{slug}_"

            # Value match: subtract → absolute → less-than 0.5
            sub = tree.nodes.new("CompositorNodeMath")
            sub.name = f"{family_aov}::{slug}::sub"
            sub.label = f"{slug} sub"
            sub.operation = "SUBTRACT"
            sub.inputs[1].default_value = float(val)
            sub.location = (600, y_cursor)
            sub.parent = frame

            abso = tree.nodes.new("CompositorNodeMath")
            abso.name = f"{family_aov}::{slug}::abs"
            abso.label = f"{slug} abs"
            abso.operation = "ABSOLUTE"
            abso.location = (800, y_cursor)
            abso.parent = frame

            lt = tree.nodes.new("CompositorNodeMath")
            lt.name = f"{family_aov}::{slug}::lt"
            lt.label = f"{slug} lt"
            lt.operation = "LESS_THAN"
            lt.inputs[1].default_value = 0.5
            lt.location = (1000, y_cursor)
            lt.parent = frame

            ensure_link(tree, exr.outputs[family_aov], sub.inputs[0])
            ensure_link(tree, sub.outputs["Value"], abso.inputs[0])
            ensure_link(tree, abso.outputs["Value"], lt.inputs[0])

            # Alpha source: value match, optionally multiplied by arboreal mask
            if arboreal_mask is not None:
                mul = tree.nodes.new("CompositorNodeMath")
                mul.name = f"{family_aov}::{slug}::masked"
                mul.label = f"{slug} masked"
                mul.operation = "MULTIPLY"
                mul.use_clamp = True
                mul.location = (1200, y_cursor)
                mul.parent = frame
                ensure_link(tree, lt.outputs["Value"], mul.inputs[0])
                ensure_link(tree, arboreal_mask, mul.inputs[1])
                alpha_socket = mul.outputs["Value"]
            else:
                alpha_socket = lt.outputs["Value"]

            # RGB colour
            rgb = tree.nodes.new("CompositorNodeRGB")
            rgb.name = f"{family_aov}::{slug}::rgb"
            rgb.label = slug
            rgb.outputs[0].default_value = hex_to_rgba(hex_color)
            rgb.location = (1400 if arboreal_mask else 1200, y_cursor + 60)
            rgb.parent = frame

            # SetAlpha
            sa = tree.nodes.new("CompositorNodeSetAlpha")
            sa.name = f"{family_aov}::{slug}::rgba"
            sa.label = slug
            sa.mode = "APPLY"
            sa.location = (1600 if arboreal_mask else 1400, y_cursor)
            sa.parent = frame

            ensure_link(tree, rgb.outputs[0], sa.inputs["Image"])
            ensure_link(tree, alpha_socket, sa.inputs["Alpha"])

            wired.append((slot_path, sa.outputs["Image"]))
            y_cursor -= 160.0

        y_cursor -= 200.0  # gap between families

    # --- File Output ---------------------------------------------------------
    file_out = tree.nodes.new("CompositorNodeOutputFile")
    file_out.name = "ProposalInterventionOutput"
    file_out.label = "ProposalInterventionOutput"
    file_out.base_path = "//outputs/"
    file_out.format.file_format = "PNG"
    file_out.format.color_mode = "RGBA"
    file_out.format.color_depth = "8"
    file_out.location = (2000, 200)

    # First slot already exists — rename it; add the rest
    file_out.file_slots[0].path = wired[0][0]
    for slot_path, _ in wired[1:]:
        file_out.file_slots.new(slot_path)

    # Wire all slots
    for index, (_, socket) in enumerate(wired):
        ensure_link(tree, socket, file_out.inputs[index])

    # --- Composite + Viewer sinks (required for Blender 4.x) -----------------
    composite = tree.nodes.new("CompositorNodeComposite")
    composite.name = "Composite"
    composite.location = (2000, 400)
    ensure_link(tree, wired[0][1], composite.inputs[0])

    viewer = tree.nodes.new("CompositorNodeViewer")
    viewer.name = "Viewer"
    viewer.location = (2000, 540)
    ensure_link(tree, wired[0][1], viewer.inputs[0])

    # --- Save ----------------------------------------------------------------
    OUTPUT_BLEND.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_BLEND))
    log(f"saved: {OUTPUT_BLEND}")
    log(f"slots: {[s.path.rstrip('_') for s in file_out.file_slots]}")


if __name__ == "__main__":
    build()
