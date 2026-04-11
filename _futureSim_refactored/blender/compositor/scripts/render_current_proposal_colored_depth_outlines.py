"""Canonical runner for proposal-colored depth outlines.

This script treats `proposal_colored_depth_outlines.blend` as the owner of the
approved proposal-depth compositor graph and colors. At runtime it only:

- opens the canonical template
- repaths the EXR input
- sets the output directory
- applies Blender 4.2 render-compatibility workarounds
- renders and normalizes `_0001` filenames

It does NOT save graph edits back into the canonical template.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import bpy

REPO_ROOT = Path(__file__).resolve().parents[5]
COMPOSITOR_ROOT = REPO_ROOT / "_code-refactored" / "refactor_code" / "blender" / "compositor"
CANONICAL_ROOT = COMPOSITOR_ROOT / "canonical_templates"


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


BLEND_PATH = env_path(
    "EDGE_LAB_BLEND_PATH",
    str(CANONICAL_ROOT / "proposal_colored_depth_outlines.blend"),
)
EXR_PATH = env_path(
    "EDGE_LAB_PROPOSAL_COLORED_DEPTH_EXR",
    "",
)
OUTPUT_DIR = env_path(
    "EDGE_LAB_PROPOSAL_COLORED_DEPTH_OUTPUT_DIR",
    "",
)
SCENE_NAME = os.environ.get("EDGE_LAB_SCENE_NAME", "ProposalColoredDepthOutlines")


def log(message: str) -> None:
    print(f"[render_current_proposal_colored_depth_outlines] {message}")


def set_standard_view(scene: bpy.types.Scene) -> None:
    try:
        scene.display_settings.display_device = "sRGB"
    except Exception:
        pass
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
    except Exception:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def detect_resolution(exr_path: Path, image: bpy.types.Image) -> tuple[int, int]:
    try:
        result = subprocess.run(
            ["oiiotool", "--info", "-v", str(exr_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        match = re.search(r"(\d+)\s*x\s*(\d+)", result.stdout)
        if match:
            return int(match.group(1)), int(match.group(2))
    except Exception:
        pass
    width, height = image.size[:]
    if width > 0 and height > 0:
        return width, height
    return 7680, 4320


def rename_outputs(output_dir: Path) -> None:
    for rendered in sorted(output_dir.glob("*_0001.png")):
        final = rendered.with_name(rendered.name.replace("_0001", ""))
        if final.exists():
            final.unlink()
        rendered.replace(final)
        log(f"Renamed {rendered.name} -> {final.name}")


def rebuild_file_output_in_memory(
    node_tree: bpy.types.NodeTree,
    node_name: str,
    base_path: str,
) -> bpy.types.Node:
    """Rebuild the saved File Output node as a transient runtime node.

    Blender 4.2 can fail to write from a File Output node that was persisted in
    the saved template. We keep the saved node as disconnected/muted graph
    state, then create a fresh in-memory replacement with the same slot wiring
    just for this render.
    """
    old = node_tree.nodes[node_name]
    source_slots: list[tuple[str, bpy.types.NodeSocket]] = []
    for i, s in enumerate(old.file_slots):
        sock = old.inputs[i]
        if not sock.is_linked:
            raise RuntimeError(f"{node_name} slot[{i}] {s.path!r} is unlinked")
        source_slots.append((s.path, sock.links[0].from_socket))
    old_location = tuple(old.location)
    # Disconnect (not remove). Removing the saved node seems to break the
    # Blender 4.2 compositor render even with a fresh replacement. Leaving
    # the old node in the tree, disconnected, lets the fresh node's slots
    # be recognized as active outputs.
    for i in range(len(old.inputs)):
        for link in list(old.inputs[i].links):
            node_tree.links.remove(link)
    old.name = f"_orphan_{node_name}"
    old.label = "_orphan"
    old.mute = True

    fresh = node_tree.nodes.new("CompositorNodeOutputFile")
    fresh.name = node_name
    fresh.label = node_name
    fresh.base_path = base_path
    fresh.format.file_format = "PNG"
    fresh.format.color_mode = "RGBA"
    fresh.format.color_depth = "8"
    fresh.location = old_location

    while len(fresh.file_slots) > 1:
        fresh.file_slots.remove(fresh.file_slots[-1])
    fresh.file_slots[0].path = source_slots[0][0]
    for path, _src in source_slots[1:]:
        fresh.file_slots.new(path)
    for i, (path, src) in enumerate(source_slots):
        fresh.file_slots[i].path = path
        node_tree.links.new(src, fresh.inputs[i])
    log(f"rebuilt {node_name} in-memory with {len(source_slots)} slots")
    return fresh


def main() -> None:
    if not BLEND_PATH.exists():
        raise FileNotFoundError(f"Canonical blend not found: {BLEND_PATH}")
    if not EXR_PATH or not Path(EXR_PATH).exists():
        raise FileNotFoundError(f"EXR not found: {EXR_PATH}")
    if not OUTPUT_DIR:
        raise ValueError("EDGE_LAB_PROPOSAL_COLORED_DEPTH_OUTPUT_DIR is required")

    bpy.ops.wm.open_mainfile(filepath=str(BLEND_PATH))
    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None or scene.node_tree is None:
        raise ValueError(f"Scene '{SCENE_NAME}' not found in {BLEND_PATH}")
    # Ensure the target scene is the active context scene so render operates
    # on it regardless of the blend's saved active scene.
    if bpy.context.window is not None:
        bpy.context.window.scene = scene

    node_tree = scene.node_tree

    exr_node = node_tree.nodes.get("EXR")
    if exr_node is None or exr_node.image is None:
        raise ValueError("Missing 'EXR' image node in canonical blend")
    # Fresh load instead of reload(): in background Blender, reload() leaves
    # image.size at (0,0), which breaks resolution detection. A fresh load
    # via bpy.data.images.load() forces the file to be opened.
    exr_node.image = bpy.data.images.load(str(EXR_PATH), check_existing=False)

    width, height = detect_resolution(Path(EXR_PATH), exr_node.image)
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1
    set_standard_view(scene)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if node_tree.nodes.get("ProposalColoredDepthOutput") is None:
        raise ValueError("Missing 'ProposalColoredDepthOutput' file output node")

    # Add transient Composite sink BEFORE rebuilding File Output. Wire from
    # the first saved File Output input source (any source would do).
    if not any(n.bl_idname == "CompositorNodeComposite" for n in node_tree.nodes):
        saved_fout = node_tree.nodes["ProposalColoredDepthOutput"]
        first_driver = None
        for sock in saved_fout.inputs:
            if sock.is_linked:
                first_driver = sock.links[0].from_socket
                break
        if first_driver is None:
            raise RuntimeError("saved File Output has no linked inputs")
        composite = node_tree.nodes.new("CompositorNodeComposite")
        composite.name = "RuntimeComposite"
        node_tree.links.new(first_driver, composite.inputs["Image"])
        log("added runtime Composite sink (workaround for render exec)")

    output_node = rebuild_file_output_in_memory(
        node_tree, "ProposalColoredDepthOutput", str(output_dir)
    )

    # Prefix (no .png suffix) — with animation=True Blender appends frame+ext.
    scene.render.filepath = str(output_dir / "_discard_render_")
    # animation=True is required for CompositorNodeOutputFile slots to write.
    # write_still=True only fires the Composite sink — File Output nodes
    # need the animation render path. No `scene=` param: we've already set
    # bpy.context.window.scene to target the correct scene.
    bpy.ops.render.render(animation=True)

    rename_outputs(output_dir)

    for discard in output_dir.glob("_discard_render*"):
        discard.unlink()

    for slot in output_node.file_slots:
        stem = slot.path.rstrip("_")
        out_path = output_dir / f"{stem}.png"
        if out_path.exists():
            log(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
