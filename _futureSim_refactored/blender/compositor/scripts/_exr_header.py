"""Minimal OpenEXR header parser for reading the true pixel dimensions.

Purpose: Blender's `bpy.data.images.load()` is lazy — `img.size` stays (0, 0)
until something actually forces the file to be decoded. Relying on that for
setting `scene.render.resolution_x/y` is fragile, and a silent fallback
(e.g. defaulting to 4K) will quietly mis-render an 8K EXR down to 4K without
complaint. See COMPOSITOR_TEMPLATE_CONTRACT.md / the input-resolution rule.

This helper parses the EXR header directly in pure Python and returns the
displayWindow dimensions. It does NOT attempt to decode pixels. It raises on
any parse failure — callers should NOT silently fall back.
"""

from __future__ import annotations

import struct
from pathlib import Path


EXR_MAGIC = b"\x76\x2f\x31\x01"


def read_exr_dimensions(path: str | Path) -> tuple[int, int]:
    """Return (width, height) from an EXR file's displayWindow attribute.

    Raises ``RuntimeError`` if the file is not an EXR, is corrupted, or does
    not contain a ``displayWindow`` attribute. Never returns a fallback.
    """
    p = Path(path)
    with open(p, "rb") as f:
        magic = f.read(4)
        if magic != EXR_MAGIC:
            raise RuntimeError(f"{p.name}: not an OpenEXR file (magic={magic!r})")
        f.read(4)  # version / flags

        while True:
            name = _read_c_string(f)
            if not name:
                break  # null byte terminates attribute list
            atype = _read_c_string(f)
            size_bytes = f.read(4)
            if len(size_bytes) != 4:
                raise RuntimeError(f"{p.name}: truncated header")
            size = struct.unpack("<i", size_bytes)[0]
            data = f.read(size)
            if len(data) != size:
                raise RuntimeError(f"{p.name}: truncated attribute {name!r}")
            if name == b"displayWindow" and atype == b"box2i":
                if size != 16:
                    raise RuntimeError(f"{p.name}: box2i displayWindow wrong size {size}")
                xmin, ymin, xmax, ymax = struct.unpack("<iiii", data)
                return (xmax - xmin + 1, ymax - ymin + 1)

    raise RuntimeError(f"{p.name}: no displayWindow attribute in EXR header")


def _read_c_string(f) -> bytes:
    out = b""
    while True:
        c = f.read(1)
        if c == b"" or c == b"\x00":
            return out
        out += c
