from pathlib import Path

import bpy
from PIL import Image


BLEND_PATH = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/2026 hero tests 2.blend"
)
OUTPUT_PATH = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/renders/2026_hero_tests_2_ground_bw_only.png"
)
CROPPED_OUTPUT_PATH = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/renders/2026_hero_tests_2_ground_bw_only_cropped.png"
)
TEXTURE_PATH = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/texture/grass/repeat/grass_repeat_bw.png"
)
SHADOW_TEXTURE_PATH = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/texture/grass/repeat/grass_repeat_shadow.png"
)
MID_TEXTURE_PATH = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/texture/grass/repeat/grass_repeat_mid.png"
)
LIGHT_TEXTURE_PATH = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/texture/grass/repeat/grass_repeat_light.png"
)
DETAIL_TEXTURE_PATH = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/texture/grass/repeat/grass_repeat_detail.png"
)

TARGET_OBJECT_NAME = "trimmed-parade_180_rewilded"
GROUND_MATERIAL_NAME = "Ground"


def ensure_image(path: Path) -> bpy.types.Image:
    image = bpy.data.images.load(str(path), check_existing=True)
    image.colorspace_settings.name = "Non-Color"
    image.alpha_mode = "STRAIGHT"
    return image


def build_preview_material(
    material: bpy.types.Material,
    bw_image: bpy.types.Image,
    shadow_image: bpy.types.Image,
    mid_image: bpy.types.Image,
    light_image: bpy.types.Image,
    detail_image: bpy.types.Image,
) -> None:
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping_a = nodes.new("ShaderNodeMapping")
    mapping_b = nodes.new("ShaderNodeMapping")
    shadow_mapping_a = nodes.new("ShaderNodeMapping")
    shadow_mapping_b = nodes.new("ShaderNodeMapping")
    light_mapping_a = nodes.new("ShaderNodeMapping")
    light_mapping_b = nodes.new("ShaderNodeMapping")
    detail_mapping = nodes.new("ShaderNodeMapping")
    bw_tex_a = nodes.new("ShaderNodeTexImage")
    bw_tex_b = nodes.new("ShaderNodeTexImage")
    mid_tex_a = nodes.new("ShaderNodeTexImage")
    mid_tex_b = nodes.new("ShaderNodeTexImage")
    shadow_tex_a = nodes.new("ShaderNodeTexImage")
    shadow_tex_b = nodes.new("ShaderNodeTexImage")
    light_tex_a = nodes.new("ShaderNodeTexImage")
    light_tex_b = nodes.new("ShaderNodeTexImage")
    detail_tex = nodes.new("ShaderNodeTexImage")
    bw_to_val_a = nodes.new("ShaderNodeRGBToBW")
    bw_to_val_b = nodes.new("ShaderNodeRGBToBW")
    mid_to_val_a = nodes.new("ShaderNodeRGBToBW")
    mid_to_val_b = nodes.new("ShaderNodeRGBToBW")
    shadow_to_val_a = nodes.new("ShaderNodeRGBToBW")
    shadow_to_val_b = nodes.new("ShaderNodeRGBToBW")
    light_to_val_a = nodes.new("ShaderNodeRGBToBW")
    light_to_val_b = nodes.new("ShaderNodeRGBToBW")
    detail_to_val = nodes.new("ShaderNodeRGBToBW")
    bw_mix = nodes.new("ShaderNodeMath")
    mid_mix = nodes.new("ShaderNodeMath")
    shadow_mix = nodes.new("ShaderNodeMath")
    light_mix = nodes.new("ShaderNodeMath")
    base_mul = nodes.new("ShaderNodeMath")
    shadow_invert = nodes.new("ShaderNodeMath")
    shadow_mul = nodes.new("ShaderNodeMath")
    light_add = nodes.new("ShaderNodeMath")
    detail_add = nodes.new("ShaderNodeMath")
    clamp = nodes.new("ShaderNodeClamp")
    contrast = nodes.new("ShaderNodeBrightContrast")

    output.location = (680, 0)
    emission.location = (460, 0)
    contrast.location = (220, 0)
    clamp.location = (0, 0)
    detail_add.location = (-220, 0)
    light_add.location = (-440, 0)
    shadow_mul.location = (-660, 0)
    shadow_invert.location = (-880, 0)
    base_mul.location = (-1100, 0)
    bw_mix.location = (-1320, -120)
    mid_mix.location = (-1320, 120)
    shadow_mix.location = (-1320, 340)
    light_mix.location = (-1320, 560)
    contrast.location = (-20, 0)
    detail_to_val.location = (-1540, 780)
    light_to_val_b.location = (-1540, 560)
    light_to_val_a.location = (-1540, 460)
    shadow_to_val_b.location = (-1540, 340)
    shadow_to_val_a.location = (-1540, 240)
    mid_to_val_b.location = (-1540, 120)
    mid_to_val_a.location = (-1540, 20)
    bw_to_val_b.location = (-1540, -120)
    bw_to_val_a.location = (-1540, -220)
    detail_tex.location = (-1760, 780)
    light_tex_b.location = (-1760, 560)
    light_tex_a.location = (-1760, 460)
    shadow_tex_b.location = (-1760, 340)
    shadow_tex_a.location = (-1760, 240)
    mid_tex_b.location = (-1760, 120)
    mid_tex_a.location = (-1760, 20)
    bw_tex_b.location = (-1760, -120)
    bw_tex_a.location = (-1760, -220)
    detail_mapping.location = (-1980, 780)
    light_mapping_b.location = (-1980, 560)
    light_mapping_a.location = (-2200, 560)
    shadow_mapping_b.location = (-1980, 340)
    shadow_mapping_a.location = (-2200, 340)
    mapping_b.location = (-1980, 120)
    mapping_a.location = (-2200, 120)
    texcoord.location = (-2440, 120)

    for node, location_xy, scale_xy, rotation_z in (
        (mapping_a, (0.00, 0.02), (3.20, 1.02), 0.01),
        (mapping_b, (0.63, 0.18), (6.60, 1.92), -0.04),
        (shadow_mapping_a, (0.17, 0.00), (3.00, 1.04), 0.02),
        (shadow_mapping_b, (0.54, 0.11), (5.90, 1.50), -0.03),
        (light_mapping_a, (0.31, -0.05), (3.60, 0.96), 0.04),
        (light_mapping_b, (0.73, 0.15), (7.40, 2.04), -0.05),
        (detail_mapping, (0.44, 0.19), (11.20, 2.90), 0.06),
    ):
        node.inputs["Location"].default_value[0] = location_xy[0]
        node.inputs["Location"].default_value[1] = location_xy[1]
        node.inputs["Scale"].default_value = (scale_xy[0], scale_xy[1], 1.0)
        node.inputs["Rotation"].default_value[2] = rotation_z

    for node, image in (
        (bw_tex_a, bw_image),
        (bw_tex_b, bw_image),
        (mid_tex_a, mid_image),
        (mid_tex_b, mid_image),
        (shadow_tex_a, shadow_image),
        (shadow_tex_b, shadow_image),
        (light_tex_a, light_image),
        (light_tex_b, light_image),
        (detail_tex, detail_image),
    ):
        node.image = image
        node.interpolation = "Cubic"
        node.extension = "REPEAT"
        node.projection = "FLAT"

    links.new(texcoord.outputs["Generated"], mapping_a.inputs["Vector"])
    links.new(texcoord.outputs["Generated"], mapping_b.inputs["Vector"])
    links.new(texcoord.outputs["Generated"], shadow_mapping_a.inputs["Vector"])
    links.new(texcoord.outputs["Generated"], shadow_mapping_b.inputs["Vector"])
    links.new(texcoord.outputs["Generated"], light_mapping_a.inputs["Vector"])
    links.new(texcoord.outputs["Generated"], light_mapping_b.inputs["Vector"])
    links.new(texcoord.outputs["Generated"], detail_mapping.inputs["Vector"])
    links.new(mapping_a.outputs["Vector"], bw_tex_a.inputs["Vector"])
    links.new(mapping_b.outputs["Vector"], bw_tex_b.inputs["Vector"])
    links.new(mapping_a.outputs["Vector"], mid_tex_a.inputs["Vector"])
    links.new(mapping_b.outputs["Vector"], mid_tex_b.inputs["Vector"])
    links.new(shadow_mapping_a.outputs["Vector"], shadow_tex_a.inputs["Vector"])
    links.new(shadow_mapping_b.outputs["Vector"], shadow_tex_b.inputs["Vector"])
    links.new(light_mapping_a.outputs["Vector"], light_tex_a.inputs["Vector"])
    links.new(light_mapping_b.outputs["Vector"], light_tex_b.inputs["Vector"])
    links.new(detail_mapping.outputs["Vector"], detail_tex.inputs["Vector"])
    links.new(bw_tex_a.outputs["Color"], bw_to_val_a.inputs["Color"])
    links.new(bw_tex_b.outputs["Color"], bw_to_val_b.inputs["Color"])
    links.new(mid_tex_a.outputs["Color"], mid_to_val_a.inputs["Color"])
    links.new(mid_tex_b.outputs["Color"], mid_to_val_b.inputs["Color"])
    links.new(shadow_tex_a.outputs["Color"], shadow_to_val_a.inputs["Color"])
    links.new(shadow_tex_b.outputs["Color"], shadow_to_val_b.inputs["Color"])
    links.new(light_tex_a.outputs["Color"], light_to_val_a.inputs["Color"])
    links.new(light_tex_b.outputs["Color"], light_to_val_b.inputs["Color"])
    links.new(detail_tex.outputs["Color"], detail_to_val.inputs["Color"])

    for node in (bw_mix, mid_mix, shadow_mix, light_mix, base_mul, shadow_mul, light_add, detail_add):
        node.operation = "ADD"
    bw_mix.operation = "MAXIMUM"
    mid_mix.operation = "MAXIMUM"
    shadow_mix.operation = "MAXIMUM"
    light_mix.operation = "MAXIMUM"
    base_mul.operation = "MULTIPLY"
    shadow_invert.operation = "SUBTRACT"
    shadow_mul.operation = "MULTIPLY"
    light_add.operation = "ADD"
    detail_add.operation = "ADD"

    base_mul.inputs[1].default_value = 0.74
    mid_mix.inputs[1].default_value = 0.0
    shadow_invert.inputs[0].default_value = 1.0
    light_add.inputs[1].default_value = 0.0
    detail_add.inputs[1].default_value = 0.0

    contrast.inputs["Bright"].default_value = 0.02
    contrast.inputs["Contrast"].default_value = 0.20
    clamp.inputs["Min"].default_value = 0.0
    clamp.inputs["Max"].default_value = 1.0

    scale_base = nodes.new("ShaderNodeMath")
    scale_mid = nodes.new("ShaderNodeMath")
    scale_shadow = nodes.new("ShaderNodeMath")
    scale_light = nodes.new("ShaderNodeMath")
    scale_detail = nodes.new("ShaderNodeMath")
    scale_base.location = (-1100, -180)
    scale_mid.location = (-1100, 140)
    scale_shadow.location = (-1100, 360)
    scale_light.location = (-1100, 580)
    scale_detail.location = (-660, 780)
    for node, factor in (
        (scale_base, 0.74),
        (scale_mid, 0.18),
        (scale_shadow, 0.68),
        (scale_light, 0.20),
        (scale_detail, 0.10),
    ):
        node.operation = "MULTIPLY"
        node.inputs[1].default_value = factor

    links.new(bw_to_val_a.outputs["Val"], bw_mix.inputs[0])
    links.new(bw_to_val_b.outputs["Val"], bw_mix.inputs[1])
    links.new(mid_to_val_a.outputs["Val"], mid_mix.inputs[0])
    links.new(mid_to_val_b.outputs["Val"], mid_mix.inputs[1])
    links.new(shadow_to_val_a.outputs["Val"], shadow_mix.inputs[0])
    links.new(shadow_to_val_b.outputs["Val"], shadow_mix.inputs[1])
    links.new(light_to_val_a.outputs["Val"], light_mix.inputs[0])
    links.new(light_to_val_b.outputs["Val"], light_mix.inputs[1])
    links.new(bw_mix.outputs[0], scale_base.inputs[0])
    links.new(mid_mix.outputs[0], scale_mid.inputs[0])
    links.new(shadow_mix.outputs[0], scale_shadow.inputs[0])
    links.new(light_mix.outputs[0], scale_light.inputs[0])
    links.new(detail_to_val.outputs["Val"], scale_detail.inputs[0])
    links.new(scale_base.outputs[0], base_mul.inputs[0])
    links.new(scale_mid.outputs[0], base_mul.inputs[1])
    links.new(scale_shadow.outputs[0], shadow_invert.inputs[1])
    links.new(base_mul.outputs[0], shadow_mul.inputs[0])
    links.new(shadow_invert.outputs[0], shadow_mul.inputs[1])
    links.new(shadow_mul.outputs[0], light_add.inputs[0])
    links.new(scale_light.outputs[0], light_add.inputs[1])
    links.new(light_add.outputs[0], detail_add.inputs[0])
    links.new(scale_detail.outputs[0], detail_add.inputs[1])
    links.new(detail_add.outputs[0], clamp.inputs["Value"])
    links.new(clamp.outputs["Result"], contrast.inputs["Color"])
    links.new(contrast.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])


def hide_non_ground(scene: bpy.types.Scene) -> None:
    for obj in scene.objects:
        if obj.type == "CAMERA":
            obj.hide_render = False
            continue
        obj.hide_render = obj.name != TARGET_OBJECT_NAME


def disable_target_modifiers(obj: bpy.types.Object) -> None:
    for modifier in obj.modifiers:
        modifier.show_render = False
        modifier.show_viewport = False


def crop_to_alpha(path: Path, output_path: Path) -> None:
    image = Image.open(path).convert("RGBA")
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        image.save(output_path)
        return

    cropped = image.crop(bbox)
    cropped.save(output_path)


def main() -> None:
    if Path(bpy.data.filepath).resolve() != BLEND_PATH.resolve():
        raise RuntimeError(f"Open the expected blend before running this script: {BLEND_PATH}")

    scene = bpy.context.scene
    scene.use_nodes = False
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 900
    scene.render.filepath = str(OUTPUT_PATH)

    material = bpy.data.materials.get(GROUND_MATERIAL_NAME)
    if material is None:
        raise RuntimeError(f"Material not found: {GROUND_MATERIAL_NAME}")

    obj = bpy.data.objects.get(TARGET_OBJECT_NAME)
    if obj is None:
        raise RuntimeError(f"Object not found: {TARGET_OBJECT_NAME}")

    disable_target_modifiers(obj)

    bw_image = ensure_image(TEXTURE_PATH)
    shadow_image = ensure_image(SHADOW_TEXTURE_PATH)
    mid_image = ensure_image(MID_TEXTURE_PATH)
    light_image = ensure_image(LIGHT_TEXTURE_PATH)
    detail_image = ensure_image(DETAIL_TEXTURE_PATH)
    build_preview_material(
        material,
        bw_image,
        shadow_image,
        mid_image,
        light_image,
        detail_image,
    )
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)
    hide_non_ground(scene)

    bpy.ops.render.render(write_still=True)
    crop_to_alpha(OUTPUT_PATH, CROPPED_OUTPUT_PATH)
    print(f"Saved BW-only ground render to {OUTPUT_PATH}")
    print(f"Saved cropped BW-only ground render to {CROPPED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
