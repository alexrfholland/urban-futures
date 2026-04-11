"""Full render diagnostic on a blend file."""
import os
from pathlib import Path
import bpy

BLEND = Path(os.environ["DIAG_BLEND"])
bpy.ops.wm.open_mainfile(filepath=str(BLEND))

print(f"\n=== DIAGNOSTIC: {BLEND.name} ===\n")

for scene in bpy.data.scenes:
    print(f"[scene] {scene.name!r}")
    print(f"  camera: {scene.camera.name if scene.camera else '<NONE>'}")
    print(f"  use_nodes: {scene.use_nodes}")
    print(f"  render.use_compositing: {scene.render.use_compositing}")
    print(f"  render.use_sequencer: {scene.render.use_sequencer}")
    print(f"  render.resolution: {scene.render.resolution_x}x{scene.render.resolution_y} @ {scene.render.resolution_percentage}%")
    print(f"  render.filepath: {scene.render.filepath!r}")
    print(f"  render.image_settings.file_format: {scene.render.image_settings.file_format}")
    print(f"  frame_start/end/current: {scene.frame_start}/{scene.frame_end}/{scene.frame_current}")
    print(f"  view_layers: {[vl.name for vl in scene.view_layers]}")
    for vl in scene.view_layers:
        print(f"    vl {vl.name!r}: use={vl.use}")

    if not scene.use_nodes or scene.node_tree is None:
        continue

    tree = scene.node_tree
    print(f"  compositor: {len(tree.nodes)} nodes, {len(tree.links)} links")
    sinks_by_type = {}
    for n in tree.nodes:
        if n.bl_idname in (
            "CompositorNodeComposite",
            "CompositorNodeViewer",
            "CompositorNodeOutputFile",
        ):
            sinks_by_type.setdefault(n.bl_idname, []).append(n)

    print(f"  sinks:")
    for t in ("CompositorNodeComposite", "CompositorNodeViewer", "CompositorNodeOutputFile"):
        lst = sinks_by_type.get(t, [])
        print(f"    {t}: {len(lst)}")
        for n in lst:
            muted = " MUTED" if n.mute else ""
            print(f"      {n.name!r}{muted}")
            if t == "CompositorNodeOutputFile":
                print(f"        base_path={n.base_path!r}")
                print(f"        format: {n.format.file_format} {n.format.color_mode} {n.format.color_depth}")
                for i, s in enumerate(n.file_slots):
                    sock = n.inputs[i]
                    src = "<unlinked>"
                    if sock.is_linked:
                        l = sock.links[0]
                        src = f"{l.from_node.name!r}.{l.from_socket.name!r}"
                    print(f"        slot[{i}] path={s.path!r} <- {src}")
                # Check base path parent exists
                bp = Path(n.base_path)
                print(f"        base_path.parent.exists(): {bp.parent.exists() if bp.parent else 'N/A'}")
                print(f"        base_path.exists(): {bp.exists()}")
            elif sock_list := n.inputs:
                for i, sock in enumerate(sock_list):
                    if sock.is_linked:
                        l = sock.links[0]
                        print(f"        input[{i}] {sock.name!r} <- {l.from_node.name!r}.{l.from_socket.name!r}")
                    else:
                        print(f"        input[{i}] {sock.name!r} <unlinked>")
print()
