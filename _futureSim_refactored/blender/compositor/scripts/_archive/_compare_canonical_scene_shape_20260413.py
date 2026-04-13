"""One-off inspection: which canonical compositor blends have a Render Layers
node in their compositor scene? Compared to `compositor_intervention_int.blend`
which the contract cites as working under `animation=True`.
"""
import bpy
from pathlib import Path

ROOT = Path(r"D:/2026 Arboreal Futures/urban-futures/_futureSim_refactored/blender/compositor/canonical_templates")
NAMES = (
    "compositor_intervention_int.blend",
    "compositor_mist.blend",
    "compositor_depth_outliner.blend",
    "compositor_ao.blend",
    "compositor_sizes.blend",
    "compositor_sizes_single_input.blend",
    "compositor_normals.blend",
)

for name in NAMES:
    p = ROOT / name
    if not p.is_file():
        print(f"=== {name} === MISSING")
        continue
    bpy.ops.wm.open_mainfile(filepath=str(p))
    print(f"=== {name} ===")
    for scene in bpy.data.scenes:
        tree = scene.node_tree
        if tree is None:
            print(f"  scene={scene.name!r} no node_tree")
            continue
        rlayers = [(n.name, n.scene.name if n.scene else None, getattr(n, "layer", None))
                   for n in tree.nodes if n.bl_idname == "CompositorNodeRLayers"]
        composites = [n.name for n in tree.nodes if n.bl_idname == "CompositorNodeComposite"]
        images = [n.name for n in tree.nodes if n.bl_idname == "CompositorNodeImage"]
        fos = [n.name for n in tree.nodes if n.bl_idname == "CompositorNodeOutputFile"]
        print(f"  scene={scene.name!r} cam={scene.camera}")
        print(f"    view_layers={[vl.name for vl in scene.view_layers]}")
        print(f"    RLayers nodes: {rlayers}")
        print(f"    Composite nodes: {composites}")
        print(f"    Image nodes: {images}")
        print(f"    FileOutput nodes: {fos}")
