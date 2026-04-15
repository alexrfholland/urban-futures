"""Walk a PSD/PSB and dump its layer tree (panel order) to JSON.

Usage:
    uv run python -m _futureSim_refactored.photoshopparser.read_psb [PSB_PATH] [OUT_JSON]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from psd_tools import PSDImage

DEFAULT_PSB = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia"
    "/_data-refactored/_psds/assembled/parade_timeline test.psb"
)

ADJUSTMENT_KINDS = {
    "brightnesscontrast", "curves", "exposure", "hue-saturation", "huesaturation",
    "colorbalance", "blackandwhite", "photofilter", "channelmixer", "colorlookup",
    "invert", "posterize", "threshold", "gradientmap", "selectivecolor", "levels",
    "vibrance", "solidcolorfill", "gradientfill", "patternfill",
}


def classify(layer) -> str:
    if layer.is_group():
        return "group"
    kind = getattr(layer, "kind", "") or ""
    if kind == "smartobject":
        return "smart_object"
    if kind in ADJUSTMENT_KINDS:
        return "adjustment"
    if kind in {"pixel", "type", "shape"}:
        return kind
    return kind or "other"


def describe(layer) -> dict:
    try:
        bbox = tuple(layer.bbox) if layer.bbox else None
    except Exception:
        bbox = None
    node = {
        "name": layer.name,
        "type": classify(layer),
        "kind": getattr(layer, "kind", None),
        "visible": layer.visible,
        "opacity": layer.opacity,
        "blend_mode": layer.blend_mode.name if hasattr(layer.blend_mode, "name") else str(layer.blend_mode),
        "bbox": bbox,
    }
    if layer.is_group():
        # psd-tools yields children bottom-up; flip to panel order (top-down).
        node["layers"] = [describe(c) for c in reversed(list(layer))]
    return node


def tally(nodes, counts: Counter) -> None:
    for n in nodes:
        counts[n["type"]] += 1
        if "layers" in n:
            tally(n["layers"], counts)


def main(psb: Path, out_json: Path) -> None:
    print(f"Opening {psb} ({psb.stat().st_size / 1e9:.2f} GB) ...", flush=True)
    psd = PSDImage.open(psb)

    tree = {
        "source": str(psb),
        "version": psd.version,
        "width": psd.width,
        "height": psd.height,
        "depth": psd.depth,
        "channels": psd.channels,
        "color_mode": str(psd.color_mode),
        "layers": [describe(c) for c in reversed(list(psd))],
    }

    out_json.write_text(json.dumps(tree, indent=2))

    counts: Counter = Counter()
    tally(tree["layers"], counts)
    print(f"\nWrote {out_json}")
    print(f"Totals: {dict(counts)}")


if __name__ == "__main__":
    psb = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PSB
    default_out = psb.with_name(psb.stem.replace(" ", "_") + "_layers.json")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else default_out
    main(psb, out)
