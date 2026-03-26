from pathlib import Path

import bpy


OUTPUT_IMAGE = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/edge_detection_lab/worldcubes_hoddle_preview_pathway.png"
)


def main():
    scene = bpy.data.scenes.get("city")
    if scene is None:
        raise ValueError("Scene 'city' not found.")

    scene.render.engine = "CYCLES"
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = str(OUTPUT_IMAGE)
    scene.render.resolution_percentage = 25

    cycles = scene.cycles
    cycles.samples = 8
    cycles.use_denoising = False
    cycles.preview_samples = 8
    cycles.max_bounces = 2
    cycles.diffuse_bounces = 1
    cycles.glossy_bounces = 1
    cycles.transmission_bounces = 0
    cycles.transparent_max_bounces = 2
    cycles.volume_bounces = 0
    cycles.use_adaptive_sampling = False

    original_layer_use = {view_layer.name: view_layer.use for view_layer in scene.view_layers}
    for view_layer in scene.view_layers:
        view_layer.use = view_layer.name == "pathway_state"

    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    try:
        bpy.ops.render.render(scene=scene.name, layer="pathway_state", write_still=True, use_viewport=False)
        print(f"Wrote preview render: {OUTPUT_IMAGE}")
    finally:
        for view_layer in scene.view_layers:
            view_layer.use = original_layer_use.get(view_layer.name, True)


if __name__ == "__main__":
    main()
