"""Compare parade_timeline positive_state vs existing_condition_positive.

Goal: user asked "look at the positive state and the base positive state. what
is up?" — enumerate layers/sockets per EXR and report alpha-coverage per layer
so we can see what's actually populated in each file and where they differ.
"""
from __future__ import annotations

import sys
from pathlib import Path

import bpy


EXRS = [
    ("positive_state",             Path(sys.argv[sys.argv.index("--") + 1])),
    ("existing_condition_positive", Path(sys.argv[sys.argv.index("--") + 2])),
]


def log(msg: str) -> None:
    print(f"[compare_positive] {msg}", flush=True)


def enumerate_sockets(exr: Path) -> list[str]:
    scene = bpy.data.scenes.new(f"probe_{exr.stem}")
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    img = bpy.data.images.load(str(exr), check_existing=False)
    node = tree.nodes.new("CompositorNodeImage")
    node.image = img
    names = [s.name for s in node.outputs]
    return names


def main() -> None:
    results: dict[str, list[str]] = {}
    for label, path in EXRS:
        log(f"=== {label} :: {path.name} ===")
        log(f"  size_bytes={path.stat().st_size}")
        socks = enumerate_sockets(path)
        results[label] = socks
        log(f"  sockets ({len(socks)}):")
        for n in socks:
            log(f"    {n!r}")
        log("")

    a_set = set(results["positive_state"])
    b_set = set(results["existing_condition_positive"])
    only_a = sorted(a_set - b_set)
    only_b = sorted(b_set - a_set)
    common = sorted(a_set & b_set)
    log("=" * 60)
    log(f"ONLY in positive_state           ({len(only_a)}): {only_a}")
    log(f"ONLY in existing_condition_positive ({len(only_b)}): {only_b}")
    log(f"common                           ({len(common)})")


if __name__ == "__main__":
    main()
