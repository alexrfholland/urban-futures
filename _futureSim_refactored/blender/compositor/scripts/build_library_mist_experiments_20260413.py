"""Build three experimental working-copy blends derived from
`compositor_mist.blend`, each tuning the kirsch/threshold/dilate chain
to produce more contiguous edge lines on isolated library-tree EXRs.

Contract-compliant: writes derivative *working copies* (not the canonical)
under `_data-refactored/compositor/working_copies/library_mist_20260413/`.
The runner addresses them via COMPOSITOR_BLEND_PATH.

Variants:
    v1_dilate      — raise DilateErode distances (1->3, 2->4) and Blur
                     size (1->2) in all three alpha subgroups. Closes
                     1-2px gaps structurally without changing contrast.
    v2_gain        — lift the outer thin/fine/extra_thin gain values
                     (1.42/1.38/1.34 -> 2.2/2.0/1.8) so more weak
                     kirsch pixels clear the ColorRamp thresholds.
    v3_threshold   — drop the ColorRamp stops (thin_alpha example:
                     0.07..0.22 and 0.18..0.34 -> 0.03..0.12 and
                     0.10..0.22). Widens the set of kirsch pixels
                     that survive the two thresholds.

Run in Blender headless. No args.
"""
from __future__ import annotations

from pathlib import Path

import bpy

REPO_ROOT = Path(r"D:/2026 Arboreal Futures/urban-futures")
CANONICAL = REPO_ROOT / "_futureSim_refactored/blender/compositor/canonical_templates/compositor_mist.blend"
OUT_ROOT = REPO_ROOT / "_data-refactored/compositor/working_copies/library_mist_20260413"

BRANCH_SUBGROUPS = ("thin_alpha", "fine_alpha", "extra_thin_alpha")


def _viewlayer_tree() -> bpy.types.NodeTree:
    tree = bpy.data.scenes["Current"].node_tree
    return tree.nodes["MistOutlines::viewlayer_group"].node_tree


def _apply_v1_dilate() -> None:
    vl = _viewlayer_tree()
    for sub_name in BRANCH_SUBGROUPS:
        sub = vl.nodes[sub_name].node_tree
        for node_name, new_dist in (("Dilate/Erode", 3), ("Dilate/Erode.001", 4)):
            n = sub.nodes[node_name]
            print(f"  {sub_name}::{node_name} distance {n.distance} -> {new_dist}")
            n.distance = new_dist
        blur = sub.nodes["Blur"]
        blur.size_x = 2
        blur.size_y = 2
        print(f"  {sub_name}::Blur size -> {blur.size_x}x{blur.size_y}")


def _apply_v2_gain() -> None:
    vl = _viewlayer_tree()
    new = {"thin_gain": 2.2, "fine_gain": 2.0, "extra_thin_gain": 1.8}
    for name, value in new.items():
        n = vl.nodes[name]
        old = n.outputs[0].default_value
        n.outputs[0].default_value = value
        print(f"  {name} {old:.3f} -> {value:.3f}")


def _apply_v3_threshold() -> None:
    vl = _viewlayer_tree()
    stops = {
        "Color Ramp":     (0.03, 0.12),
        "Color Ramp.001": (0.10, 0.22),
    }
    for sub_name in BRANCH_SUBGROUPS:
        sub = vl.nodes[sub_name].node_tree
        for cr_name, (p0, p1) in stops.items():
            cr = sub.nodes[cr_name]
            old = (cr.color_ramp.elements[0].position,
                   cr.color_ramp.elements[1].position)
            cr.color_ramp.elements[0].position = p0
            cr.color_ramp.elements[1].position = p1
            print(f"  {sub_name}::{cr_name} {old} -> {(p0, p1)}")


VARIANTS = {
    "compositor_mist_library_v1_dilate.blend":    _apply_v1_dilate,
    "compositor_mist_library_v2_gain.blend":      _apply_v2_gain,
    "compositor_mist_library_v3_threshold.blend": _apply_v3_threshold,
}


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for fname, mutator in VARIANTS.items():
        print(f"\n=== building {fname} ===")
        bpy.ops.wm.open_mainfile(filepath=str(CANONICAL))
        mutator()
        out = OUT_ROOT / fname
        bpy.ops.wm.save_as_mainfile(filepath=str(out), copy=True)
        print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
