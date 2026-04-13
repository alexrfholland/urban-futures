"""Inspect the raw Normal pass of a bV2 multilayer EXR.

Loads the EXR, wires its `Normal` output into a Viewer node, renders at a
downsampled resolution, reads the Viewer pixels via foreach_get, and prints
per-channel statistics plus a column sweep across screen X to diagnose any
position-dependent drift.

The goal is to answer: does the stock Cycles Normal pass on this specific
render have uniform `N_x`, `N_y`, `N_z` on flat regions, or does it drift
linearly across screen X (which would confirm something in the shader graph
or camera setup is baking a view-dependent term into world-space normals)?

Usage:

    "C:/Program Files/Blender Foundation/Blender 4.2/blender.exe" --background \\
        --python _inspect_normal_pass_20260411.py -- \\
        --exr "<path to multilayer EXR>" \\
        [--downsample 8]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import bpy


def log(msg: str) -> None:
    print(f"[inspect_normal_pass] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--exr", required=True)
    p.add_argument("--downsample", type=int, default=8, help="integer downsample factor (1 = full res)")
    return p.parse_args(argv)


def find_normal_socket(node: bpy.types.Node) -> bpy.types.NodeSocket:
    candidates = [s for s in node.outputs if s.name.lower() == "normal"]
    if candidates:
        return candidates[0]
    # Fallback: any socket containing "ormal"
    for s in node.outputs:
        if "ormal" in s.name:
            return s
    raise RuntimeError(
        "no Normal output on EXR Image node; available sockets: "
        + ", ".join(f"{s.name!r}" for s in node.outputs)
    )


def main() -> None:
    args = parse_args()
    exr_path = Path(args.exr)
    if not exr_path.exists():
        raise FileNotFoundError(f"EXR not found: {exr_path}")

    # Fresh scene; throw away anything the default startup left around.
    for scn in list(bpy.data.scenes):
        if scn.name != bpy.context.scene.name:
            bpy.data.scenes.remove(scn)
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    # Load the EXR.
    log(f"loading {exr_path.name}")
    img = bpy.data.images.load(str(exr_path), check_existing=False)
    img.source = "FILE"

    img_node = tree.nodes.new("CompositorNodeImage")
    img_node.name = "EXR"
    img_node.image = img

    # Force Blender to populate layer/pass sockets.
    bpy.context.view_layer.update()

    log(f"image.size reported as {tuple(img.size)}  (may be 0 until decoded)")
    log(f"image node exposes {len(img_node.outputs)} output sockets:")
    for s in img_node.outputs:
        log(f"  {s.name!r} type={s.type} enabled={getattr(s, 'enabled', True)}")

    normal_socket = find_normal_socket(img_node)
    log(f"using normal socket: {normal_socket.name!r} (type={normal_socket.type})")

    # Viewer node — Blender auto-converts VECTOR to COLOR in the link.
    viewer = tree.nodes.new("CompositorNodeViewer")
    viewer.name = "Viewer"
    viewer.use_alpha = False
    tree.links.new(normal_socket, viewer.inputs[0])

    # Determine render res from the loaded image. image.size can be (0,0) at
    # this point, so fall back to reading EXR dims via the sibling helper.
    w, h = img.size[0], img.size[1]
    if w == 0 or h == 0:
        # Use the shared header helper — same one the runner uses.
        import importlib.util

        helper_path = Path(__file__).resolve().parent / "_exr_header.py"
        spec = importlib.util.spec_from_file_location("_exr_header", helper_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        w, h = mod.read_exr_dimensions(exr_path)
        log(f"image.size was lazy; read EXR header directly: {w}x{h}")

    ds = max(1, int(args.downsample))
    render_w = max(16, w // ds)
    render_h = max(16, h // ds)
    scene.render.resolution_x = render_w
    scene.render.resolution_y = render_h
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
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

    log(f"rendering at {render_w}x{render_h} (downsample {ds}x from {w}x{h})")
    bpy.ops.render.render(write_still=False)

    viewer_img = bpy.data.images.get("Viewer Node")
    if viewer_img is None:
        raise RuntimeError("Viewer Node image not populated after render")

    pixels = list(viewer_img.pixels)
    n_pixels = len(pixels) // 4
    if n_pixels != render_w * render_h:
        log(f"WARN: pixel buffer has {n_pixels} pixels but expected {render_w * render_h}")

    # Rearrange into per-channel arrays without numpy (to keep zero deps).
    nx = pixels[0::4]
    ny = pixels[1::4]
    nz = pixels[2::4]

    def stats(label: str, values: list[float]) -> None:
        if not values:
            log(f"{label}: empty")
            return
        vmin = min(values)
        vmax = max(values)
        mean = sum(values) / len(values)
        # Use a cheap variance.
        m = mean
        var = sum((v - m) * (v - m) for v in values) / len(values)
        stdev = var ** 0.5
        log(
            f"{label}: min={vmin:+.4f} max={vmax:+.4f} mean={mean:+.4f} "
            f"stdev={stdev:.4f} n={len(values)}"
        )

    log("=== whole-image stats (raw, NOT remapped to [0,1]) ===")
    stats("N.x", nx)
    stats("N.y", ny)
    stats("N.z", nz)

    # Column sweep: for each of 9 evenly spaced x columns, compute the mean
    # of each channel over the full column height.
    col_xs = [int(round(i * (render_w - 1) / 8)) for i in range(9)]
    log("=== column-mean sweep across screen X ===")
    log(f"sampling columns at x = {col_xs}")
    header = f"  {'x':>6}  {'N.x':>9}  {'N.y':>9}  {'N.z':>9}  {'|N|':>9}"
    log(header)
    for cx in col_xs:
        col_nx = [nx[y * render_w + cx] for y in range(render_h)]
        col_ny = [ny[y * render_w + cx] for y in range(render_h)]
        col_nz = [nz[y * render_w + cx] for y in range(render_h)]
        mx = sum(col_nx) / len(col_nx)
        my = sum(col_ny) / len(col_ny)
        mz = sum(col_nz) / len(col_nz)
        mag = (mx * mx + my * my + mz * mz) ** 0.5
        log(f"  {cx:>6d}  {mx:>+9.4f}  {my:>+9.4f}  {mz:>+9.4f}  {mag:>9.4f}")

    # Row sweep too — just to rule out a Y-axis bias.
    row_ys = [int(round(i * (render_h - 1) / 8)) for i in range(9)]
    log("=== row-mean sweep across screen Y ===")
    log(f"sampling rows at y = {row_ys}")
    log(f"  {'y':>6}  {'N.x':>9}  {'N.y':>9}  {'N.z':>9}  {'|N|':>9}")
    for cy in row_ys:
        row_nx = [nx[cy * render_w + x] for x in range(render_w)]
        row_ny = [ny[cy * render_w + x] for x in range(render_w)]
        row_nz = [nz[cy * render_w + x] for x in range(render_w)]
        mx = sum(row_nx) / len(row_nx)
        my = sum(row_ny) / len(row_ny)
        mz = sum(row_nz) / len(row_nz)
        mag = (mx * mx + my * my + mz * mz) ** 0.5
        log(f"  {cy:>6d}  {mx:>+9.4f}  {my:>+9.4f}  {mz:>+9.4f}  {mag:>9.4f}")

    # Histogram of |N| so we can see how often it's ~1.0 vs 0.0 (background).
    mags = [
        (nx[i] * nx[i] + ny[i] * ny[i] + nz[i] * nz[i]) ** 0.5
        for i in range(0, n_pixels, max(1, n_pixels // 500_000))
    ]
    log("=== |N| histogram (sampled) ===")
    buckets = [0] * 11
    for m in mags:
        idx = min(10, max(0, int(m * 10)))
        buckets[idx] += 1
    for i, count in enumerate(buckets):
        lo = i / 10.0
        hi = (i + 1) / 10.0
        log(f"  [{lo:.1f}..{hi:.1f}): {count}")

    log("done")


if __name__ == "__main__":
    main()
