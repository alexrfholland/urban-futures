"""Inspect Normal + Alpha of a bV2 library EXR by round-tripping through PNG.

The Viewer-node pixel readout is unreliable in background mode (caps at 256
regardless of scene render resolution). Instead this script:

  1. Loads the library EXR
  2. Builds a compositor graph that remaps Normal from [-1, 1] to [0, 1] and
     writes it as an RGBA PNG (R = N.x, G = N.y, B = N.z, A = EXR Alpha)
  3. Writes the PNG at a specified size
  4. Reloads the PNG via bpy.data.images and samples pixels directly from
     image.pixels (reliable)
  5. For every opaque pixel, decodes the Normal and reports stats + samples

Usage
-----
    "C:/Program Files/Blender Foundation/Blender 4.2/blender.exe" --background \\
        --python _inspect_library_normal_20260413.py -- \\
        --exr "<absolute path>" [--size 512] [--out <dir>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


def log(msg: str) -> None:
    print(f"[inspect_library_normal] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--exr", required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def build_graph(scene: bpy.types.Scene, exr_path: Path, out_dir: Path, size: int) -> Path:
    """Write `normal_rgba.png` with N remapped to [0,1] and EXR alpha preserved."""
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    img = bpy.data.images.load(str(exr_path), check_existing=False)
    img.source = "FILE"

    img_node = tree.nodes.new("CompositorNodeImage")
    img_node.image = img
    bpy.context.view_layer.update()

    def socket(name: str):
        s = next((s for s in img_node.outputs if s.name == name), None)
        if s is None:
            raise RuntimeError(f"no socket {name!r}; have "
                               + ", ".join(repr(s.name) for s in img_node.outputs))
        return s

    normal_sock = socket("Normal")
    alpha_sock = socket("Alpha")

    # (N + 1) * 0.5 — per-channel remap.
    sep = tree.nodes.new("CompositorNodeSepRGBA")
    tree.links.new(normal_sock, sep.inputs[0])

    def remap(chan_output):
        add = tree.nodes.new("CompositorNodeMath")
        add.operation = "ADD"
        add.inputs[1].default_value = 1.0
        tree.links.new(chan_output, add.inputs[0])
        mul = tree.nodes.new("CompositorNodeMath")
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = 0.5
        tree.links.new(add.outputs[0], mul.inputs[0])
        return mul.outputs[0]

    r_out = remap(sep.outputs["R"])
    g_out = remap(sep.outputs["G"])
    b_out = remap(sep.outputs["B"])

    combine = tree.nodes.new("CompositorNodeCombRGBA")
    tree.links.new(r_out, combine.inputs["R"])
    tree.links.new(g_out, combine.inputs["G"])
    tree.links.new(b_out, combine.inputs["B"])
    tree.links.new(alpha_sock, combine.inputs["A"])

    out_dir.mkdir(parents=True, exist_ok=True)
    file_out = tree.nodes.new("CompositorNodeOutputFile")
    file_out.base_path = str(out_dir)
    file_out.format.file_format = "PNG"
    file_out.format.color_mode = "RGBA"
    file_out.format.color_depth = "16"
    while len(file_out.file_slots) > 0:
        file_out.file_slots.remove(file_out.inputs[0])
    file_out.file_slots.new("normal_rgba_")
    tree.links.new(combine.outputs["Image"], file_out.inputs[0])

    scene.render.resolution_x = size
    scene.render.resolution_y = size
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
    scene.render.filepath = str(out_dir / "_discard.png")
    bpy.ops.render.render(write_still=True)

    written = sorted(out_dir.glob("normal_rgba_*.png"))
    if not written:
        raise RuntimeError("file output produced nothing")
    return written[-1]


def main() -> None:
    args = parse_args()
    exr_path = Path(args.exr)
    if not exr_path.exists():
        raise FileNotFoundError(exr_path)
    out_dir = Path(args.out) if args.out else Path("E:/probe_normal_space_20260413")
    log(f"exr: {exr_path.name}")
    log(f"out: {out_dir}")

    scene = bpy.context.scene
    png_path = build_graph(scene, exr_path, out_dir, args.size)
    log(f"wrote: {png_path}")

    img = bpy.data.images.load(str(png_path), check_existing=False)
    w, h = img.size[0], img.size[1]
    log(f"png size: {w}x{h}")
    px = list(img.pixels)

    tree_pixels: list[tuple[int, int, float, float, float, float]] = []
    for y in range(h):
        for x in range(w):
            idx = (y * w + x) * 4
            a = px[idx + 3]
            if a > 0.5:
                # Decode N from [0,1] back to [-1,1].
                nx = px[idx + 0] * 2.0 - 1.0
                ny = px[idx + 1] * 2.0 - 1.0
                nz = px[idx + 2] * 2.0 - 1.0
                tree_pixels.append((x, y, a, nx, ny, nz))

    log(f"opaque pixels: {len(tree_pixels)} / {w*h}")
    if not tree_pixels:
        log("=> EXR Alpha appears zero everywhere after compositing. Investigate.")
        return

    nxs = [p[3] for p in tree_pixels]
    nys = [p[4] for p in tree_pixels]
    nzs = [p[5] for p in tree_pixels]

    def stats(label: str, values: list[float]) -> None:
        vmin = min(values); vmax = max(values); mean = sum(values)/len(values)
        var = sum((v-mean)**2 for v in values)/len(values)
        log(f"  {label}: min={vmin:+.4f} max={vmax:+.4f} mean={mean:+.4f} stdev={var**0.5:.4f}")

    log("=== Normal stats over tree pixels (decoded to [-1, 1]) ===")
    stats("N.x", nxs); stats("N.y", nys); stats("N.z", nzs)
    mags = [(nxs[i]**2 + nys[i]**2 + nzs[i]**2)**0.5 for i in range(len(nxs))]
    stats("|N|", mags)

    log("=== 20 evenly spaced samples (x, y, a, N.x, N.y, N.z) ===")
    step = max(1, len(tree_pixels) // 20)
    for s in tree_pixels[::step][:20]:
        log(f"  x={s[0]:4d} y={s[1]:4d} a={s[2]:+.3f}  N=({s[3]:+.4f}, {s[4]:+.4f}, {s[5]:+.4f})")

    mean_mag = sum(mags) / len(mags)
    if mean_mag < 0.05:
        log("=> Normal pass is effectively zero on tree pixels. Pass is EMPTY.")
    elif 0.9 < mean_mag < 1.1:
        log("=> Normal pass carries unit-length vectors. Pass is POPULATED.")
    else:
        log(f"=> Normal pass mean |N|={mean_mag:.4f} (suspicious magnitude).")


if __name__ == "__main__":
    main()
