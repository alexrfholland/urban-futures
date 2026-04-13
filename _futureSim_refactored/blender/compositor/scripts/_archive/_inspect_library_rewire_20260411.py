"""Deep inspector for canonical compositor blends — plan single-input rewires.

Usage:
    blender -b <blend> --factory-startup -P _inspect_library_rewire_20260411.py

For the blend already opened, inspect scene 'Current' node tree and dump:
  - Every CompositorNodeImage input node: name, label, image filepath, and
    for each OUTPUT socket, the list of downstream (node, socket) targets.
  - Every CompositorNodeOutputFile: name, base_path, file_slots (index, path)
    and for each input link the from_node/from_socket.

Output is printed. Intended to be read manually to plan the rewire.
"""
from __future__ import annotations

import bpy


def main() -> None:
    scene = bpy.data.scenes.get("Current")
    if scene is None:
        if len(bpy.data.scenes) == 1:
            scene = bpy.data.scenes[0]
        else:
            print(f"NO 'Current' SCENE; scenes={[s.name for s in bpy.data.scenes]}")
            return
    if scene.node_tree is None:
        print("no compositor node tree")
        return

    tree = scene.node_tree
    image_nodes = [n for n in tree.nodes if n.bl_idname == "CompositorNodeImage"]
    out_nodes = [n for n in tree.nodes if n.bl_idname == "CompositorNodeOutputFile"]

    print("")
    print("=" * 70)
    print(f"SCENE={scene.name!r}")
    print(f"image_input_nodes={len(image_nodes)}  output_file_nodes={len(out_nodes)}")
    print("=" * 70)

    print("\n-- IMAGE INPUT NODES --")
    for n in image_nodes:
        img = n.image.filepath if n.image else "<no image>"
        print(f"\n  name={n.name!r}  label={n.label!r}")
        print(f"  image={img!r}")
        for sock in n.outputs:
            if sock.links:
                tgts = [
                    f"{lk.to_node.name!r}:{lk.to_socket.name!r}"
                    for lk in sock.links
                ]
                print(f"    output[{sock.name!r}] -> {len(sock.links)} link(s)")
                for t in tgts:
                    print(f"       -> {t}")

    print("\n-- OUTPUT FILE NODES --")
    for n in out_nodes:
        print(f"\n  name={n.name!r}  base_path={n.base_path!r}  muted={n.mute}")
        for i, slot in enumerate(n.file_slots):
            stem = slot.path
            inp = n.inputs[i] if i < len(n.inputs) else None
            src = "<unlinked>"
            if inp is not None and inp.links:
                lk = inp.links[0]
                src = f"{lk.from_node.name!r}:{lk.from_socket.name!r}"
            print(f"    slot[{i}] path={stem!r}  <- {src}")


if __name__ == "__main__":
    main()
