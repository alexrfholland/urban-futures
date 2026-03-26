from pathlib import Path

import bpy


OUTPUT_IMAGE = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/edge_detection_lab/worldcubes_hoddle_preview_pathway_ao_zoom.png"
)
BORDER = {
    "min_x": 0.15,
    "max_x": 0.65,
    "min_y": 0.18,
    "max_y": 0.68,
}


def build_ao_compositor(scene: bpy.types.Scene) -> None:
    scene.use_nodes = True
    node_tree = scene.node_tree
    node_tree.nodes.clear()

    render_layers = node_tree.nodes.new("CompositorNodeRLayers")
    render_layers.location = (-300.0, 0.0)
    render_layers.scene = scene
    render_layers.layer = "pathway_state"

    composite = node_tree.nodes.new("CompositorNodeComposite")
    composite.location = (0.0, 0.0)

    node_tree.links.new(render_layers.outputs["AO"], composite.inputs["Image"])


def main():
    scene = bpy.data.scenes.get("city")
    if scene is None:
        raise ValueError("Scene 'city' not found.")

    view_layer = scene.view_layers.get("pathway_state")
    if view_layer is None:
        raise ValueError("View layer 'pathway_state' not found.")

    view_layer.use_pass_ambient_occlusion = True

    original_layer_use = {layer.name: layer.use for layer in scene.view_layers}
    for layer in scene.view_layers:
        layer.use = layer.name == "pathway_state"

    scene.render.engine = "CYCLES"
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "BW"
    scene.render.filepath = str(OUTPUT_IMAGE)
    scene.render.resolution_percentage = 50
    scene.render.use_border = True
    scene.render.use_crop_to_border = True
    scene.render.border_min_x = BORDER["min_x"]
    scene.render.border_max_x = BORDER["max_x"]
    scene.render.border_min_y = BORDER["min_y"]
    scene.render.border_max_y = BORDER["max_y"]

    cycles = scene.cycles
    cycles.samples = 16
    cycles.preview_samples = 16
    cycles.use_denoising = False
    cycles.use_adaptive_sampling = False
    cycles.max_bounces = 1
    cycles.diffuse_bounces = 1
    cycles.glossy_bounces = 0
    cycles.transmission_bounces = 0
    cycles.transparent_max_bounces = 1
    cycles.volume_bounces = 0

    build_ao_compositor(scene)
    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)

    try:
        bpy.ops.render.render(scene=scene.name, layer="pathway_state", write_still=True, use_viewport=False)
        print(f"Wrote AO zoom preview: {OUTPUT_IMAGE}")
    finally:
        for layer in scene.view_layers:
            layer.use = original_layer_use.get(layer.name, True)


if __name__ == "__main__":
    main()
