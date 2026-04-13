"""Thin runner: render mist or depth_outliner PNGs from a single per-tree
library EXR using the canonical compositor blends.

Per COMPOSITOR_TEMPLATE_CONTRACT.md Render Execution Rule — delegates the
actual render to `_fast_runner_core.run_fast_render`, which handles:
  - single-frame `animation=True` render
  - the scene.camera attach that animation=True requires
  - EXR header resolution detection
  - discard-render path + frame-suffix cleanup
  - per-slot presence audit

This runner's own responsibility is limited to:
  - picking the family-specific canonical blend and node names
  - prepending the EXR stem to each File Output slot.path so the produced
    filenames match the library batch convention
    (`<exr_stem>__<slot_stem>.png`)

One invocation = one EXR + one family. Orchestration is external.

Env vars:
    COMPOSITOR_LIBRARY_EXR  — absolute path to one library EXR (required)
    COMPOSITOR_OUTPUT_DIR   — absolute output dir for this run (required;
                              prefer a short path; Blender has MAX_PATH
                              issues on Win)
    COMPOSITOR_FAMILY       — 'mist' | 'depth_outliner' (required)
    COMPOSITOR_BLEND_PATH   — optional absolute path to an experimental
                              working-copy blend to use instead of the
                              canonical for this family. Must expose the
                              same node names.

Invocation:
    blender -b --factory-startup -P render_library_mist_or_depth.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable

import bpy

_THIS = Path(__file__).resolve()
_SCRIPTS_DIR = _THIS.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from _fast_runner_core import (  # noqa: E402
    CANONICAL_ROOT,
    FastRunnerConfig,
    run_fast_render,
)

NAME = "render_library_mist_or_depth"
SCENE_NAME = "Current"

# Library EXRs expose the tree silhouette as the `resource_tree_mask` pass
# (from the `asset.resource_tree_mask.X` channel), not as `IndexOB`. The
# canonical graph feeds its `*::viewlayer_group.Arboreal Mask` socket via
# an IDMask chain hanging off `EXR Input.IndexOB`. For library inputs we
# bypass that chain at runtime by linking `EXR Input.resource_tree_mask`
# directly into `viewlayer_group.Arboreal Mask`.
LIBRARY_TREE_MASK_SOCKET = "resource_tree_mask"

FAMILY_SPEC: dict[str, tuple[str, str, str, str]] = {
    # family -> (blend filename, exr input node, viewlayer group node, file output node)
    "mist": (
        "compositor_mist.blend",
        "MistOutlines::EXR Input",
        "MistOutlines::viewlayer_group",
        "MistOutlines::Outputs",
    ),
    "depth_outliner": (
        "compositor_depth_outliner.blend",
        "DepthOutliner::EXR Input",
        "DepthOutliner::viewlayer_group",
        "DepthOutliner::Outputs",
    ),
}


def env_required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing env: {name}")
    return value


def _find_output_socket(node: bpy.types.Node, name: str):
    for sock in node.outputs:
        if sock.name == name:
            return sock
    return None


def _find_input_socket(node: bpy.types.Node, name: str):
    for sock in node.inputs:
        if sock.name == name:
            return sock
    return None


def _relink_library_tree_mask(
    tree: bpy.types.NodeTree, exr_node_name: str, group_node_name: str
) -> None:
    exr_node = tree.nodes.get(exr_node_name)
    group_node = tree.nodes.get(group_node_name)
    if exr_node is None or group_node is None:
        raise ValueError(
            f"relink: missing nodes — exr={exr_node is not None} "
            f"group={group_node is not None}"
        )
    mask_out = _find_output_socket(exr_node, LIBRARY_TREE_MASK_SOCKET)
    if mask_out is None:
        available = sorted(s.name for s in exr_node.outputs)
        raise RuntimeError(
            f"relink: {exr_node_name!r} has no output {LIBRARY_TREE_MASK_SOCKET!r}; "
            f"available: {available}"
        )
    mask_in = _find_input_socket(group_node, "Arboreal Mask")
    if mask_in is None:
        available = sorted(s.name for s in group_node.inputs)
        raise RuntimeError(
            f"relink: {group_node_name!r} has no input 'Arboreal Mask'; "
            f"available: {available}"
        )
    for link in list(mask_in.links):
        tree.links.remove(link)
    tree.links.new(mask_out, mask_in)
    print(
        f"[{NAME}] relinked {exr_node_name!r}.{LIBRARY_TREE_MASK_SOCKET!r} "
        f"-> {group_node_name!r}.'Arboreal Mask'"
    )


def make_extra_setup(
    fo_node_name: str,
    exr_node_name: str,
    group_node_name: str,
    exr_stem: str,
) -> Callable[[bpy.types.NodeTree], None]:
    def _setup(tree: bpy.types.NodeTree) -> None:
        _relink_library_tree_mask(tree, exr_node_name, group_node_name)
        fo = tree.nodes.get(fo_node_name)
        if fo is None:
            raise ValueError(f"missing File Output node {fo_node_name!r}")
        for slot in fo.file_slots:
            original = slot.path
            slot.path = f"{exr_stem}__{original}"
            print(f"[{NAME}] slot rewrite: {original!r} -> {slot.path!r}")
    return _setup


def main() -> None:
    exr = Path(env_required("COMPOSITOR_LIBRARY_EXR"))
    out = Path(env_required("COMPOSITOR_OUTPUT_DIR"))
    family = env_required("COMPOSITOR_FAMILY").strip().lower()

    if family not in FAMILY_SPEC:
        raise ValueError(
            f"unknown COMPOSITOR_FAMILY {family!r}; allowed: {sorted(FAMILY_SPEC)}"
        )
    blend_name, exr_node, group_node, fo_node = FAMILY_SPEC[family]
    override = os.environ.get("COMPOSITOR_BLEND_PATH", "").strip()
    if override:
        blend_path = Path(override)
        if not blend_path.is_file():
            raise FileNotFoundError(f"COMPOSITOR_BLEND_PATH not found: {blend_path}")
        print(f"[{NAME}] using override blend: {blend_path}")
    else:
        blend_path = CANONICAL_ROOT / blend_name

    config = FastRunnerConfig(
        name=NAME,
        blend_path=blend_path,
        scene_name=SCENE_NAME,
        exr_inputs={exr_node: exr},
        file_output_node=fo_node,
        output_dir=out,
        extra_setup=make_extra_setup(fo_node, exr_node, group_node, exr.stem),
        # compositor_mist / compositor_depth_outliner feed their File Output
        # directly from a CompositorNodeGroup. Blender 4.2 skips those saved
        # FO nodes at render time; rebuild an equivalent FO in-memory (not
        # saved back) per the runtime-compatibility allowance in the contract.
        rebuild_file_output=True,
    )
    run_fast_render(config)


if __name__ == "__main__":
    main()
