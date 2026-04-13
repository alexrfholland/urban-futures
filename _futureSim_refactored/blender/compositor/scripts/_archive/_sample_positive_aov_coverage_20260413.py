"""Sample per-AOV coverage from two parade_timeline EXRs at low res via PNG
round-trip. Reports, per layer: mean, min, max, and count of nonzero pixels.

Goal: answer "what is up between positive_state and existing_condition_positive?"
— show which AOVs are populated where, so we can see the structural difference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import bpy


AOVS_TO_SAMPLE = [
    "Alpha",
    "IndexOB",
    "instanceID",
    "size",
    "precolonial",
    "control",
    "improvement",
    "canopy_resistance",
    "node_id",
    "sim_Turns",
    "source-year",
    "bioEnvelopeType",
    "world_design_bioenvelope",
    "world_design_bioenvelope_simp",
    "world_sim_matched",
    "world_sim_nodes",
    "world_sim_turns",
    "intervention_bioenvelope_ply-",
    "proposal-colonise",
    "proposal-decay",
    "proposal-deploy-structure",
    "proposal-recruit",
    "proposal-release-control",
    "resource_dead_branch_mask",
    "resource_epiphyte_mask",
    "resource_fallen_log_mask",
    "resource_hollow_mask",
    "resource_none_mask",
    "resource_peeling_bark_mask",
    "resource_perch_branch_mask",
]

SIZE = 512
OUT_ROOT = Path("E:/compare_positive_20260413")


def log(msg: str) -> None:
    print(f"[sample_positive] {msg}", flush=True)


def build_and_render(exr: Path, tag: str) -> Path:
    scene = bpy.data.scenes.new(f"probe_{tag}")
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    img = bpy.data.images.load(str(exr), check_existing=False)
    img_node = tree.nodes.new("CompositorNodeImage")
    img_node.image = img

    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    fout = tree.nodes.new("CompositorNodeOutputFile")
    fout.base_path = str(out_dir)
    fout.format.file_format = "OPEN_EXR"
    fout.format.color_mode = "RGBA"
    fout.format.color_depth = "32"
    while len(fout.file_slots) > 0:
        fout.file_slots.remove(fout.inputs[0])

    for aov in AOVS_TO_SAMPLE:
        sock = next((s for s in img_node.outputs if s.name == aov), None)
        if sock is None:
            log(f"  MISSING socket {aov!r}")
            continue
        fout.file_slots.new(f"{aov}_")
        tree.links.new(sock, fout.inputs[-1])

    scene.render.resolution_x = SIZE
    scene.render.resolution_y = SIZE
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.filepath = str(out_dir / "_discard.exr")
    bpy.context.window.scene = scene
    bpy.ops.render.render(write_still=True)
    return out_dir


def stats(values):
    if not values:
        return None
    mn = min(values); mx = max(values); mean = sum(values) / len(values)
    return mn, mx, mean


def report(out_dir: Path) -> dict[str, tuple[float, float, float, int, int]]:
    results: dict = {}
    for aov in AOVS_TO_SAMPLE:
        matches = sorted(out_dir.glob(f"{aov}_*.exr"))
        if not matches:
            continue
        img = bpy.data.images.load(str(matches[-1]), check_existing=False)
        w, h = img.size[0], img.size[1]
        if w == 0 or h == 0:
            img.reload()
            w, h = img.size[0], img.size[1]
        px = list(img.pixels)
        # EXR is RGBA; AOV value is R. Alpha in [3].
        vals = [px[i] for i in range(0, len(px), 4)]
        alphas = [px[i + 3] for i in range(0, len(px), 4)]
        opaque = sum(1 for a in alphas if a > 0.5)
        nonzero = sum(1 for v in vals if abs(v) > 1e-6)
        s = stats(vals)
        if s is None:
            continue
        mn, mx, mean = s
        results[aov] = (mn, mx, mean, nonzero, opaque)
    return results


def main() -> None:
    argv = sys.argv
    if "--" not in argv:
        print("usage: ... -- <positive_state.exr> <existing_positive.exr>")
        return
    args = argv[argv.index("--") + 1 :]
    a_path = Path(args[0])
    b_path = Path(args[1])

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"rendering A: {a_path.name}")
    a_dir = build_and_render(a_path, "A_positive_state")
    log(f"rendering B: {b_path.name}")
    b_dir = build_and_render(b_path, "B_existing_condition_positive")

    log("sampling A outputs")
    a_res = report(a_dir)
    log("sampling B outputs")
    b_res = report(b_dir)

    log("=" * 90)
    header = f"{'AOV':38s} | {'A.min':>8s} {'A.max':>8s} {'A.mean':>8s} {'A.nz':>7s} | {'B.min':>8s} {'B.max':>8s} {'B.mean':>8s} {'B.nz':>7s}"
    log(header)
    log("-" * len(header))
    for aov in AOVS_TO_SAMPLE:
        a = a_res.get(aov)
        b = b_res.get(aov)
        if a is None and b is None:
            continue
        a_s = f"{a[0]:8.3f} {a[1]:8.3f} {a[2]:8.3f} {a[3]:7d}" if a else f"{'-':>8s} {'-':>8s} {'-':>8s} {'-':>7s}"
        b_s = f"{b[0]:8.3f} {b[1]:8.3f} {b[2]:8.3f} {b[3]:7d}" if b else f"{'-':>8s} {'-':>8s} {'-':>8s} {'-':>7s}"
        log(f"{aov:38s} | {a_s} | {b_s}")


if __name__ == "__main__":
    main()
