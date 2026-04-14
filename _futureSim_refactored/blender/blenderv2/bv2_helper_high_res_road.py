"""Interpolate low-res road PLY voxels into high-res point clouds.

Reads PLY files from  _data-refactored/model-inputs/world/originals/
and writes interpolated results to  _data-refactored/model-inputs/world/highres/

Same logic as the legacy extract_scene.interpolate_road_voxels but operates
directly on PLY files and preserves ALL vertex properties.

Usage (from repo root):
    .tools/uv/uv.exe run python _futureSim_refactored/blender/blenderv2/bv2_helper_high_res_road.py
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement

from _futureSim_refactored.paths import WORLD_ORIGINALS_ROOT, WORLD_HIGHRES_ROOT

ORIGINALS_DIR = WORLD_ORIGINALS_ROOT
HIGHRES_DIR = WORLD_HIGHRES_ROOT

SITES = ["city", "trimmed-parade", "uni"]

# 4x4 grid offsets at 0.25m spacing (identical to legacy extract_scene)
_xy = np.array([-0.375, -0.125, 0.125, 0.375])
_xx, _yy = np.meshgrid(_xy, _xy)
OFFSETS_XY = np.column_stack((_xx.ravel(), _yy.ravel()))  # (16, 2)


def interpolate_road_ply(src: Path, dst: Path) -> None:
    ply = PlyData.read(str(src))
    verts = ply["vertex"]
    props = [p.name for p in verts.properties]
    n_total = len(verts.data)

    print(f"\n--- {src.name} ---")
    print(f"  vertices : {n_total}")
    print(f"  properties: {props}")

    scale = np.array(verts["scale"])

    hi_mask = scale <= 0.5
    lo_mask = scale > 0.5
    n_hi = int(hi_mask.sum())
    n_lo = int(lo_mask.sum())
    print(f"  high-res (scale <= 0.5): {n_hi}")
    print(f"  low-res  (scale >  0.5): {n_lo}")

    if n_lo == 0:
        print("  nothing to interpolate — copying as-is")
        ply.write(str(dst))
        return

    n_grid = len(OFFSETS_XY)  # 16
    n_interp = n_lo * n_grid
    n_out = n_hi + n_interp

    # Build a structured array with the same dtype as the source
    dtype = verts.data.dtype
    out = np.empty(n_out, dtype=dtype)

    # --- high-res pass-through ---
    if n_hi:
        out[:n_hi] = verts.data[hi_mask]

    # --- interpolated low-res ---
    lo_data = verts.data[lo_mask]

    # Tile every property: each low-res row repeated n_grid times
    interp = np.repeat(lo_data, n_grid)

    # Offset x and y
    x = interp["x"].copy()
    y = interp["y"].copy()
    offsets_tiled = np.tile(OFFSETS_XY, (n_lo, 1))  # (n_interp, 2)
    x += offsets_tiled[:, 0]
    y += offsets_tiled[:, 1]
    interp["x"] = x
    interp["y"] = y

    # Update scale to 0.25 for all interpolated points
    interp["scale"] = 0.25

    out[n_hi:] = interp

    el = PlyElement.describe(out, "vertex")
    PlyData([el], text=False).write(str(dst))
    print(f"  output   : {n_out} vertices -> {dst}")


def main() -> None:
    HIGHRES_DIR.mkdir(parents=True, exist_ok=True)

    for site in SITES:
        src = ORIGINALS_DIR / f"{site}-roadVoxels.ply"
        if not src.exists():
            print(f"SKIP {src} — not found")
            continue
        dst = HIGHRES_DIR / f"{site}-roadVoxels.ply"
        interpolate_road_ply(src, dst)

    print("\nDone.")


if __name__ == "__main__":
    main()
