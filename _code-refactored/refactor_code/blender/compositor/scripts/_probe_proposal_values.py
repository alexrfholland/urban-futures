"""Sample the EXR's proposal-release-control channel to see value distribution."""
import os
from pathlib import Path
import bpy

EXR = Path(os.environ["PROBE_EXR"])

# Load in the python API directly via bpy.data.images
img = bpy.data.images.load(str(EXR))
img.colorspace_settings.name = "Non-Color"

# Force pixel read by accessing .pixels
w, h = img.size[:]
print(f"[probe] image size: {w}x{h}")
print(f"[probe] channels: {img.channels}")

# If multilayer, list render_slots or use a tiny python pixel sample
# Easier: use a scene with an image node hooked to a viewer, force evaluate
scene = bpy.data.scenes.new("probe")
scene.use_nodes = True
tree = scene.node_tree
for n in list(tree.nodes):
    tree.nodes.remove(n)
img_node = tree.nodes.new("CompositorNodeImage")
img_node.image = img

viewer = tree.nodes.new("CompositorNodeViewer")
# find the proposal-release-control output
target_socket = None
for o in img_node.outputs:
    if o.name == "proposal-release-control":
        target_socket = o
        break
if target_socket is None:
    print("[probe] ERROR: no proposal-release-control socket")
    raise SystemExit(1)

# SetAlpha white RGB with mask, feed to viewer/composite via rgb-to-bw
rgb = tree.nodes.new("CompositorNodeRGB")
rgb.outputs[0].default_value = (1, 1, 1, 1)
setalpha = tree.nodes.new("CompositorNodeSetAlpha")
setalpha.mode = "APPLY"
tree.links.new(rgb.outputs[0], setalpha.inputs["Image"])
tree.links.new(target_socket, setalpha.inputs["Alpha"])

composite = tree.nodes.new("CompositorNodeComposite")
tree.links.new(setalpha.outputs["Image"], composite.inputs["Image"])
tree.links.new(setalpha.outputs["Image"], viewer.inputs["Image"])

# Render tiny resolution for a quick histogram
scene.render.resolution_x = 512
scene.render.resolution_y = 288
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = True
scene.frame_start = 1
scene.frame_end = 1

out = Path(os.environ.get("PROBE_OUT", "/tmp/probe_raw.png"))
scene.render.filepath = str(out)
bpy.ops.render.render(write_still=True, scene=scene.name)
print(f"[probe] wrote raw-value png to {out}")

# Sample values directly from the viewer image
vi = bpy.data.images.get("Viewer Node")
if vi is not None and vi.size[0] > 0:
    pixels = list(vi.pixels)
    # Take the alpha channel (index 3 per pixel)
    alphas = pixels[3::4]
    nz = [a for a in alphas if a > 0.001]
    if nz:
        mn = min(nz)
        mx = max(nz)
        avg = sum(nz) / len(nz)
        total = len(alphas)
        frac_nonzero = len(nz) / total
        frac_gt12 = sum(1 for a in alphas if a > 1.2) / total
        frac_gt05 = sum(1 for a in alphas if a > 0.5) / total
        print(f"[probe] alpha nonzero: {len(nz)}/{total} ({frac_nonzero*100:.1f}%)")
        print(f"[probe] alpha range (nonzero): min={mn:.3f} max={mx:.3f} mean={avg:.3f}")
        print(f"[probe] fraction > 0.5: {frac_gt05*100:.1f}%")
        print(f"[probe] fraction > 1.2: {frac_gt12*100:.1f}%")
        # Histogram in buckets
        buckets = [0]*12
        for a in alphas:
            if a <= 0: continue
            b = min(int(a), 11)
            buckets[b] += 1
        print("[probe] histogram (nonzero):")
        for i, c in enumerate(buckets):
            if c > 0:
                print(f"  {i}..{i+1}: {c}")
