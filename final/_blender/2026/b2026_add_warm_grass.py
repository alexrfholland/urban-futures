import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


DEFAULT_BLEND = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/2026 hero tests 2.blend"
)

GROUND_GROUP_NAME = "Ground"
GROUND_MATERIAL_NAME = "Ground"
CLIP_GROUP_NAME = "ClipBox Cull Mesh"
ASSET_COLLECTION_NAME = "WarmGrassAssets"
COLOR_TEXTURE_NAME = "GrassRepeatBase"
BW_TEXTURE_NAME = "GrassRepeatBW"
SHADOW_TEXTURE_NAME = "GrassRepeatShadow"
MID_TEXTURE_NAME = "GrassRepeatMid"
LIGHT_TEXTURE_NAME = "GrassRepeatLight"
DETAIL_TEXTURE_NAME = "GrassRepeatDetail"
COLOR_TEXTURE_PATH = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/texture/grass/repeat/grass_repeat_bw.png"
)
BW_TEXTURE_PATH = Path(
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

PATCH_LARGE_OBJECT_NAME = "WarmGrassPatchLarge"
PATCH_MID_OBJECT_NAME = "WarmGrassPatchMid"
PATCH_DETAIL_OBJECT_NAME = "WarmGrassTuft"

PATCH_LARGE_MATERIAL_NAME = "WarmGrassShadow"
PATCH_MID_MATERIAL_NAME = "WarmGrassPaint"
PATCH_DETAIL_MATERIAL_NAME = "WarmGrassHighlight"

FIELD_MASK_AOV_NAME = "ground_field_mask"
STROKE_MASK_AOV_NAME = "ground_stroke_mask"
ACCENT_MASK_AOV_NAME = "ground_accent_mask"


def has_name_prefix(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(f"{prefix}.")


def get_target_path() -> Path:
    if "--" in sys.argv:
        extra = sys.argv[sys.argv.index("--") + 1 :]
        if extra:
            return Path(extra[0]).expanduser().resolve()
    if bpy.data.filepath:
        return Path(bpy.data.filepath).resolve()
    return DEFAULT_BLEND


def remove_object(obj: bpy.types.Object) -> None:
    data = getattr(obj, "data", None)
    bpy.data.objects.remove(obj, do_unlink=True)
    if isinstance(data, bpy.types.Mesh) and data.users == 0:
        bpy.data.meshes.remove(data)


def ensure_asset_collection(name: str) -> bpy.types.Collection:
    for obj in list(bpy.data.objects):
        if any(
            has_name_prefix(obj.name, prefix)
            for prefix in (
                PATCH_LARGE_OBJECT_NAME,
                PATCH_MID_OBJECT_NAME,
                PATCH_DETAIL_OBJECT_NAME,
            )
        ):
            remove_object(obj)

    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)

    for obj in list(collection.objects):
        remove_object(obj)

    scene_collection = bpy.context.scene.collection
    if scene_collection.children.get(collection.name) is None:
        scene_collection.children.link(collection)

    return collection


def ensure_mask_aovs() -> None:
    for scene in bpy.data.scenes:
        for view_layer in scene.view_layers:
            existing = {aov.name: aov for aov in view_layer.aovs}
            for name in (
                FIELD_MASK_AOV_NAME,
                STROKE_MASK_AOV_NAME,
                ACCENT_MASK_AOV_NAME,
            ):
                aov = existing.get(name)
                if aov is None:
                    aov = view_layer.aovs.add()
                    aov.name = name
                if hasattr(aov, "type"):
                    aov.type = "VALUE"


def ensure_texture_image(
    name: str,
    path: Path,
    *,
    non_color: bool = True,
) -> bpy.types.Image:
    image = bpy.data.images.get(name)
    if image is None:
        image = bpy.data.images.load(str(path), check_existing=True)
        image.name = name
    elif Path(bpy.path.abspath(image.filepath)).resolve() != path.resolve():
        bpy.data.images.remove(image)
        image = bpy.data.images.load(str(path), check_existing=True)
        image.name = name

    image.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    image.alpha_mode = "STRAIGHT"
    return image


def build_ribbon(angle: float, width_scale: float, bend_scale: float):
    levels = [0.00, 0.18, 0.40, 0.68, 0.95]
    widths = [0.18, 0.16, 0.12, 0.08, 0.02]
    bends = [0.00, 0.02, 0.08, 0.16, 0.24]

    rotation = Matrix.Rotation(angle, 4, "Z")
    vertices = []
    faces = []

    for index, (z, width, bend) in enumerate(zip(levels, widths, bends)):
        width *= width_scale
        bend *= bend_scale
        left = rotation @ Vector((bend, -width * 0.5, z))
        right = rotation @ Vector((bend, width * 0.5, z))
        vertices.extend((left, right))
        if index == 0:
            continue
        previous = (index - 1) * 2
        current = index * 2
        faces.append((previous, previous + 1, current + 1, current))

    return vertices, faces


def create_painterly_material(
    name: str,
    root_color,
    mid_color,
    tip_color,
    *,
    ao_distance: float,
    toon_size: float,
    emission_strength: float,
) -> bpy.types.Material:
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)

    material.use_nodes = True
    material.use_backface_culling = False
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    geometry = nodes.new("ShaderNodeNewGeometry")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    map_range = nodes.new("ShaderNodeMapRange")
    color_ramp = nodes.new("ShaderNodeValToRGB")
    ao = nodes.new("ShaderNodeAmbientOcclusion")
    toon = nodes.new("ShaderNodeBsdfToon")
    emission = nodes.new("ShaderNodeEmission")
    add_shader = nodes.new("ShaderNodeAddShader")

    output.location = (760, 0)
    add_shader.location = (540, 0)
    toon.location = (320, 100)
    emission.location = (320, -120)
    ao.location = (120, 100)
    color_ramp.location = (-120, 100)
    map_range.location = (-340, 100)
    separate.location = (-560, 100)
    geometry.location = (-760, 100)

    map_range.clamp = True
    map_range.inputs["From Min"].default_value = 0.0
    map_range.inputs["From Max"].default_value = 1.0
    map_range.inputs["To Min"].default_value = 0.0
    map_range.inputs["To Max"].default_value = 1.0

    ramp = color_ramp.color_ramp
    while len(ramp.elements) > 2:
        ramp.elements.remove(ramp.elements[-1])
    ramp.elements[0].position = 0.0
    ramp.elements[0].color = root_color
    ramp.elements[1].position = 1.0
    ramp.elements[1].color = tip_color
    middle = ramp.elements.new(0.55)
    middle.color = mid_color

    ao.inputs["Distance"].default_value = ao_distance
    toon.inputs["Size"].default_value = toon_size
    toon.inputs["Smooth"].default_value = 0.08
    emission.inputs["Strength"].default_value = emission_strength

    links.new(geometry.outputs["Position"], separate.inputs["Vector"])
    links.new(separate.outputs["Z"], map_range.inputs["Value"])
    links.new(map_range.outputs["Result"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], ao.inputs["Color"])
    links.new(ao.outputs["Color"], toon.inputs["Color"])
    links.new(color_ramp.outputs["Color"], emission.inputs["Color"])
    links.new(toon.outputs["BSDF"], add_shader.inputs[0])
    links.new(emission.outputs["Emission"], add_shader.inputs[1])
    links.new(add_shader.outputs["Shader"], output.inputs["Surface"])

    return material


def create_cluster_asset(
    name: str,
    collection: bpy.types.Collection,
    material: bpy.types.Material,
    *,
    seed: int,
    ribbon_count: int,
    footprint_x: float,
    footprint_y: float,
    height_min: float,
    height_max: float,
    width_min: float,
    width_max: float,
    bend_min: float,
    bend_max: float,
    lean_max: float,
) -> bpy.types.Object:
    rng = random.Random(seed)
    vertices = []
    faces = []

    for _ in range(ribbon_count):
        angle = rng.uniform(-math.pi, math.pi)
        width_scale = rng.uniform(width_min, width_max)
        bend_scale = rng.uniform(bend_min, bend_max)
        height_scale = rng.uniform(height_min, height_max)
        lean = rng.uniform(-lean_max, lean_max)
        offset_x = rng.uniform(-footprint_x * 0.5, footprint_x * 0.5)
        offset_y = rng.uniform(-footprint_y * 0.5, footprint_y * 0.5)

        ribbon_vertices, ribbon_faces = build_ribbon(angle, width_scale, bend_scale)
        direction = Vector((math.cos(angle), math.sin(angle), 0.0))
        offset = len(vertices)

        for vertex in ribbon_vertices:
            displaced = Vector((vertex.x, vertex.y, vertex.z * height_scale))
            displaced += direction * (lean * displaced.z)
            displaced.x += offset_x
            displaced.y += offset_y
            vertices.append(displaced)

        faces.extend(tuple(index + offset for index in face) for face in ribbon_faces)

    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata([tuple(vertex) for vertex in vertices], [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.hide_render = True
    obj.hide_viewport = True
    obj.hide_select = True

    if obj.data.materials:
        obj.data.materials.clear()
    obj.data.materials.append(material)

    for polygon in obj.data.polygons:
        polygon.use_smooth = True

    return obj


def rebuild_ground_material(
    color_image: bpy.types.Image,
    bw_image: bpy.types.Image,
    shadow_image: bpy.types.Image,
    mid_image: bpy.types.Image,
    light_image: bpy.types.Image,
    detail_image: bpy.types.Image,
) -> bpy.types.Material:
    material = bpy.data.materials.get(GROUND_MATERIAL_NAME)
    if material is None:
        material = bpy.data.materials.new(GROUND_MATERIAL_NAME)

    material.use_nodes = True
    material.use_backface_culling = False
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    texcoord = nodes.new("ShaderNodeTexCoord")
    scenario = nodes.new("ShaderNodeAttribute")
    rewild_map = nodes.new("ShaderNodeMapRange")
    macro_mapping = nodes.new("ShaderNodeMapping")
    field_mapping_a = nodes.new("ShaderNodeMapping")
    field_mapping_b = nodes.new("ShaderNodeMapping")
    mid_mapping_a = nodes.new("ShaderNodeMapping")
    mid_mapping_b = nodes.new("ShaderNodeMapping")
    shadow_mapping_a = nodes.new("ShaderNodeMapping")
    shadow_mapping_b = nodes.new("ShaderNodeMapping")
    light_mapping_a = nodes.new("ShaderNodeMapping")
    light_mapping_b = nodes.new("ShaderNodeMapping")
    detail_mapping = nodes.new("ShaderNodeMapping")
    noise_macro = nodes.new("ShaderNodeTexNoise")
    noise_breakup = nodes.new("ShaderNodeTexNoise")
    macro_density_map = nodes.new("ShaderNodeMapRange")
    breakup_map = nodes.new("ShaderNodeMapRange")
    macro_ramp = nodes.new("ShaderNodeValToRGB")
    color_tex_a = nodes.new("ShaderNodeTexImage")
    color_tex_b = nodes.new("ShaderNodeTexImage")
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
    bw_map_a = nodes.new("ShaderNodeMapRange")
    bw_map_b = nodes.new("ShaderNodeMapRange")
    mid_map_a = nodes.new("ShaderNodeMapRange")
    mid_map_b = nodes.new("ShaderNodeMapRange")
    shadow_map_a = nodes.new("ShaderNodeMapRange")
    shadow_map_b = nodes.new("ShaderNodeMapRange")
    light_map_a = nodes.new("ShaderNodeMapRange")
    light_map_b = nodes.new("ShaderNodeMapRange")
    detail_map = nodes.new("ShaderNodeMapRange")
    field_max_a = nodes.new("ShaderNodeMath")
    field_max_b = nodes.new("ShaderNodeMath")
    field_max_c = nodes.new("ShaderNodeMath")
    field_seed = nodes.new("ShaderNodeMath")
    field_factor = nodes.new("ShaderNodeMath")
    field_strength = nodes.new("ShaderNodeMath")
    shadow_max = nodes.new("ShaderNodeMath")
    stroke_seed = nodes.new("ShaderNodeMath")
    stroke_factor = nodes.new("ShaderNodeMath")
    stroke_strength = nodes.new("ShaderNodeMath")
    light_max = nodes.new("ShaderNodeMath")
    detail_gain = nodes.new("ShaderNodeMath")
    accent_base = nodes.new("ShaderNodeMath")
    accent_seed = nodes.new("ShaderNodeMath")
    accent_factor = nodes.new("ShaderNodeMath")
    accent_strength = nodes.new("ShaderNodeMath")
    dry_color = nodes.new("ShaderNodeRGB")
    field_color = nodes.new("ShaderNodeRGB")
    dark_clump_color = nodes.new("ShaderNodeRGB")
    light_clump_color = nodes.new("ShaderNodeRGB")
    color_mix = nodes.new("ShaderNodeMix")
    color_grade = nodes.new("ShaderNodeHueSaturation")
    color_contrast = nodes.new("ShaderNodeBrightContrast")
    color_tint_mix = nodes.new("ShaderNodeMix")
    macro_color_mix = nodes.new("ShaderNodeMix")
    dark_tint_mix = nodes.new("ShaderNodeMix")
    light_tint_mix = nodes.new("ShaderNodeMix")
    field_mix = nodes.new("ShaderNodeMix")
    stroke_mix = nodes.new("ShaderNodeMix")
    accent_mix = nodes.new("ShaderNodeMix")
    rewild_mix = nodes.new("ShaderNodeMix")
    field_aov = nodes.new("ShaderNodeOutputAOV")
    stroke_aov = nodes.new("ShaderNodeOutputAOV")
    accent_aov = nodes.new("ShaderNodeOutputAOV")
    ao = nodes.new("ShaderNodeAmbientOcclusion")
    toon = nodes.new("ShaderNodeBsdfToon")
    emission = nodes.new("ShaderNodeEmission")
    add_shader = nodes.new("ShaderNodeAddShader")

    output.location = (1240, 40)
    add_shader.location = (1020, 40)
    toon.location = (800, 120)
    emission.location = (800, -120)
    ao.location = (600, 120)
    rewild_mix.location = (360, 140)
    accent_mix.location = (120, 160)
    stroke_mix.location = (-120, 160)
    field_mix.location = (-360, 160)
    light_tint_mix.location = (120, 340)
    dark_tint_mix.location = (-120, 340)
    macro_color_mix.location = (-360, 340)
    color_tint_mix.location = (-600, 160)
    color_contrast.location = (-840, 160)
    color_grade.location = (-1080, 160)
    color_mix.location = (-1320, 160)
    light_clump_color.location = (120, -120)
    dark_clump_color.location = (-120, -120)
    field_color.location = (-360, -120)
    dry_color.location = (360, -120)
    field_aov.location = (-380, 420)
    stroke_aov.location = (-140, 420)
    accent_aov.location = (100, 420)
    field_strength.location = (-380, 620)
    stroke_strength.location = (-140, 620)
    accent_strength.location = (100, 620)
    field_factor.location = (-620, 620)
    stroke_factor.location = (-380, 820)
    accent_factor.location = (-140, 820)
    field_seed.location = (-860, 620)
    stroke_seed.location = (-620, 820)
    accent_seed.location = (-380, 1020)
    field_max_c.location = (-1100, 620)
    shadow_max.location = (-860, 820)
    accent_base.location = (-620, 1020)
    field_max_b.location = (-1340, 620)
    light_max.location = (-860, 1020)
    field_max_a.location = (-1580, 620)
    detail_gain.location = (-1100, 1020)
    bw_map_b.location = (-1820, 620)
    mid_map_b.location = (-1820, 820)
    shadow_map_b.location = (-1820, 1020)
    light_map_b.location = (-1820, 1220)
    detail_map.location = (-1820, 1420)
    bw_map_a.location = (-2060, 620)
    mid_map_a.location = (-2060, 820)
    shadow_map_a.location = (-2060, 1020)
    light_map_a.location = (-2060, 1220)
    breakup_map.location = (-1100, 1420)
    macro_density_map.location = (-1340, 1420)
    bw_to_val_b.location = (-2300, 620)
    mid_to_val_b.location = (-2300, 820)
    shadow_to_val_b.location = (-2300, 1020)
    light_to_val_b.location = (-2300, 1220)
    detail_to_val.location = (-2300, 1420)
    color_tex_b.location = (-2300, 420)
    bw_to_val_a.location = (-2540, 620)
    mid_to_val_a.location = (-2540, 820)
    shadow_to_val_a.location = (-2540, 1020)
    light_to_val_a.location = (-2540, 1220)
    color_tex_a.location = (-2540, 420)
    detail_tex.location = (-2780, 1420)
    bw_tex_b.location = (-2780, 620)
    mid_tex_b.location = (-2780, 820)
    shadow_tex_b.location = (-2780, 1020)
    light_tex_b.location = (-2780, 1220)
    bw_tex_a.location = (-3020, 620)
    mid_tex_a.location = (-3020, 820)
    shadow_tex_a.location = (-3020, 1020)
    light_tex_a.location = (-3020, 1220)
    macro_ramp.location = (-1580, 1420)
    noise_breakup.location = (-1580, 1620)
    noise_macro.location = (-1820, 1620)
    detail_mapping.location = (-2780, 1620)
    light_mapping_b.location = (-3020, 1620)
    light_mapping_a.location = (-3260, 1620)
    shadow_mapping_b.location = (-3020, 1820)
    shadow_mapping_a.location = (-3260, 1820)
    mid_mapping_b.location = (-3020, 2020)
    mid_mapping_a.location = (-3260, 2020)
    field_mapping_b.location = (-3020, 2220)
    field_mapping_a.location = (-3260, 2220)
    macro_mapping.location = (-1820, 1820)
    rewild_map.location = (-620, 420)
    scenario.location = (-860, 420)
    texcoord.location = (-3500, 2020)

    scenario.attribute_name = "scenario_rewilded"
    field_aov.aov_name = FIELD_MASK_AOV_NAME
    stroke_aov.aov_name = STROKE_MASK_AOV_NAME
    accent_aov.aov_name = ACCENT_MASK_AOV_NAME

    rewild_map.clamp = True
    rewild_map.inputs["From Min"].default_value = 0.0
    rewild_map.inputs["From Max"].default_value = 3.0
    rewild_map.inputs["To Min"].default_value = 0.0
    rewild_map.inputs["To Max"].default_value = 1.0

    for node, location_xy, scale_xy in (
        (macro_mapping, (0.0, 0.0), (0.72, 0.72)),
        (field_mapping_a, (0.00, 0.02), (3.20, 1.02)),
        (field_mapping_b, (0.37, 0.12), (5.10, 1.34)),
        (mid_mapping_a, (0.21, -0.04), (2.30, 0.92)),
        (mid_mapping_b, (0.63, 0.18), (6.60, 1.92)),
        (shadow_mapping_a, (0.17, 0.00), (3.00, 1.04)),
        (shadow_mapping_b, (0.54, 0.11), (5.90, 1.50)),
        (light_mapping_a, (0.31, -0.05), (3.60, 0.96)),
        (light_mapping_b, (0.73, 0.15), (7.40, 2.04)),
        (detail_mapping, (0.44, 0.19), (11.20, 2.90)),
    ):
        node.inputs["Location"].default_value[0] = location_xy[0]
        node.inputs["Location"].default_value[1] = location_xy[1]
        node.inputs["Scale"].default_value = (scale_xy[0], scale_xy[1], 1.0)

    for node, rotation_z in (
        (field_mapping_a, 0.01),
        (field_mapping_b, -0.02),
        (mid_mapping_a, 0.03),
        (mid_mapping_b, -0.04),
        (shadow_mapping_a, 0.02),
        (shadow_mapping_b, -0.03),
        (light_mapping_a, 0.04),
        (light_mapping_b, -0.05),
        (detail_mapping, 0.06),
    ):
        node.inputs["Rotation"].default_value[2] = rotation_z

    for node, scale, detail, roughness, distortion in (
        (noise_macro, 1.8, 3.0, 0.56, 0.08),
        (noise_breakup, 4.6, 2.2, 0.54, 0.14),
    ):
        node.inputs[2].default_value = scale
        node.inputs[3].default_value = detail
        node.inputs[4].default_value = roughness
        node.inputs[8].default_value = distortion

    for node, image in (
        (color_tex_a, color_image),
        (color_tex_b, color_image),
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

    macro_colors = macro_ramp.color_ramp
    while len(macro_colors.elements) > 2:
        macro_colors.elements.remove(macro_colors.elements[-1])
    macro_colors.elements[0].position = 0.0
    macro_colors.elements[0].color = (0.23, 0.13, 0.08, 1.0)
    macro_colors.elements[1].position = 1.0
    macro_colors.elements[1].color = (0.73, 0.51, 0.23, 1.0)
    macro_mid_a = macro_colors.elements.new(0.36)
    macro_mid_a.color = (0.41, 0.24, 0.12, 1.0)
    macro_mid_b = macro_colors.elements.new(0.72)
    macro_mid_b.color = (0.58, 0.38, 0.18, 1.0)

    for node, from_min, from_max, to_min, to_max in (
        (macro_density_map, 0.10, 0.88, 0.74, 1.0),
        (breakup_map, 0.16, 0.80, 0.56, 1.0),
        (bw_map_a, 0.28, 0.82, 0.0, 1.0),
        (bw_map_b, 0.34, 0.90, 0.0, 1.0),
        (mid_map_a, 0.26, 0.84, 0.0, 1.0),
        (mid_map_b, 0.34, 0.92, 0.0, 1.0),
        (shadow_map_a, 0.52, 0.96, 0.0, 1.0),
        (shadow_map_b, 0.48, 0.94, 0.0, 1.0),
        (light_map_a, 0.42, 0.90, 0.0, 1.0),
        (light_map_b, 0.46, 0.94, 0.0, 1.0),
        (detail_map, 0.56, 1.00, 0.0, 1.0),
    ):
        node.clamp = True
        node.inputs["From Min"].default_value = from_min
        node.inputs["From Max"].default_value = from_max
        node.inputs["To Min"].default_value = to_min
        node.inputs["To Max"].default_value = to_max

    for node, default_value in (
        (field_strength, 0.96),
        (stroke_strength, 0.98),
        (detail_gain, 0.78),
        (accent_strength, 0.72),
    ):
        node.operation = "MULTIPLY"
        node.inputs[1].default_value = default_value

    for node in (
        field_max_a,
        field_max_b,
        field_max_c,
        shadow_max,
        light_max,
        accent_base,
    ):
        node.operation = "MAXIMUM"

    for node in (
        field_seed,
        field_factor,
        stroke_seed,
        stroke_factor,
        accent_seed,
        accent_factor,
    ):
        node.operation = "MULTIPLY"

    dry_color.outputs[0].default_value = (0.38, 0.27, 0.18, 1.0)
    field_color.outputs[0].default_value = (0.84, 0.71, 0.40, 1.0)
    dark_clump_color.outputs[0].default_value = (0.20, 0.15, 0.11, 1.0)
    light_clump_color.outputs[0].default_value = (0.96, 0.88, 0.63, 1.0)

    color_mix.data_type = "RGBA"
    color_tint_mix.data_type = "RGBA"
    macro_color_mix.data_type = "RGBA"
    dark_tint_mix.data_type = "RGBA"
    light_tint_mix.data_type = "RGBA"
    field_mix.data_type = "RGBA"
    stroke_mix.data_type = "RGBA"
    accent_mix.data_type = "RGBA"
    rewild_mix.data_type = "RGBA"

    color_mix.inputs[0].default_value = 0.56
    color_tint_mix.inputs[0].default_value = 0.92
    macro_color_mix.inputs[0].default_value = 0.08
    dark_tint_mix.inputs[0].default_value = 0.30
    light_tint_mix.inputs[0].default_value = 0.20
    color_grade.inputs["Hue"].default_value = 0.50
    color_grade.inputs["Saturation"].default_value = 0.90
    color_grade.inputs["Value"].default_value = 1.00
    color_contrast.inputs["Bright"].default_value = 0.08
    color_contrast.inputs["Contrast"].default_value = 0.74

    ao.inputs["Distance"].default_value = 0.9
    toon.inputs["Size"].default_value = 0.82
    toon.inputs["Smooth"].default_value = 0.08
    emission.inputs["Strength"].default_value = 0.12

    links.new(texcoord.outputs["Generated"], macro_mapping.inputs["Vector"])
    links.new(texcoord.outputs["Generated"], field_mapping_a.inputs["Vector"])
    links.new(texcoord.outputs["Generated"], field_mapping_b.inputs["Vector"])
    links.new(texcoord.outputs["Generated"], mid_mapping_a.inputs["Vector"])
    links.new(texcoord.outputs["Generated"], mid_mapping_b.inputs["Vector"])
    links.new(texcoord.outputs["Generated"], shadow_mapping_a.inputs["Vector"])
    links.new(texcoord.outputs["Generated"], shadow_mapping_b.inputs["Vector"])
    links.new(texcoord.outputs["Generated"], light_mapping_a.inputs["Vector"])
    links.new(texcoord.outputs["Generated"], light_mapping_b.inputs["Vector"])
    links.new(texcoord.outputs["Generated"], detail_mapping.inputs["Vector"])
    links.new(macro_mapping.outputs["Vector"], noise_macro.inputs[0])
    links.new(macro_mapping.outputs["Vector"], noise_breakup.inputs[0])
    links.new(field_mapping_a.outputs["Vector"], color_tex_a.inputs["Vector"])
    links.new(mid_mapping_b.outputs["Vector"], color_tex_b.inputs["Vector"])
    links.new(field_mapping_a.outputs["Vector"], bw_tex_a.inputs["Vector"])
    links.new(field_mapping_b.outputs["Vector"], bw_tex_b.inputs["Vector"])
    links.new(mid_mapping_a.outputs["Vector"], mid_tex_a.inputs["Vector"])
    links.new(mid_mapping_b.outputs["Vector"], mid_tex_b.inputs["Vector"])
    links.new(shadow_mapping_a.outputs["Vector"], shadow_tex_a.inputs["Vector"])
    links.new(shadow_mapping_b.outputs["Vector"], shadow_tex_b.inputs["Vector"])
    links.new(light_mapping_a.outputs["Vector"], light_tex_a.inputs["Vector"])
    links.new(light_mapping_b.outputs["Vector"], light_tex_b.inputs["Vector"])
    links.new(detail_mapping.outputs["Vector"], detail_tex.inputs["Vector"])
    links.new(noise_macro.outputs["Fac"], macro_ramp.inputs["Fac"])
    links.new(noise_macro.outputs["Fac"], macro_density_map.inputs["Value"])
    links.new(noise_breakup.outputs["Fac"], breakup_map.inputs["Value"])
    links.new(color_tex_a.outputs["Color"], color_mix.inputs[6])
    links.new(color_tex_b.outputs["Color"], color_mix.inputs[7])
    links.new(color_mix.outputs[2], color_grade.inputs["Color"])
    links.new(color_grade.outputs["Color"], color_contrast.inputs["Color"])
    links.new(field_color.outputs["Color"], color_tint_mix.inputs[6])
    links.new(color_contrast.outputs["Color"], color_tint_mix.inputs[7])
    links.new(color_tint_mix.outputs[2], macro_color_mix.inputs[6])
    links.new(macro_ramp.outputs["Color"], macro_color_mix.inputs[7])
    links.new(bw_tex_a.outputs["Color"], bw_to_val_a.inputs["Color"])
    links.new(bw_tex_b.outputs["Color"], bw_to_val_b.inputs["Color"])
    links.new(mid_tex_a.outputs["Color"], mid_to_val_a.inputs["Color"])
    links.new(mid_tex_b.outputs["Color"], mid_to_val_b.inputs["Color"])
    links.new(shadow_tex_a.outputs["Color"], shadow_to_val_a.inputs["Color"])
    links.new(shadow_tex_b.outputs["Color"], shadow_to_val_b.inputs["Color"])
    links.new(light_tex_a.outputs["Color"], light_to_val_a.inputs["Color"])
    links.new(light_tex_b.outputs["Color"], light_to_val_b.inputs["Color"])
    links.new(detail_tex.outputs["Color"], detail_to_val.inputs["Color"])
    links.new(bw_to_val_a.outputs["Val"], bw_map_a.inputs["Value"])
    links.new(bw_to_val_b.outputs["Val"], bw_map_b.inputs["Value"])
    links.new(mid_to_val_a.outputs["Val"], mid_map_a.inputs["Value"])
    links.new(mid_to_val_b.outputs["Val"], mid_map_b.inputs["Value"])
    links.new(shadow_to_val_a.outputs["Val"], shadow_map_a.inputs["Value"])
    links.new(shadow_to_val_b.outputs["Val"], shadow_map_b.inputs["Value"])
    links.new(light_to_val_a.outputs["Val"], light_map_a.inputs["Value"])
    links.new(light_to_val_b.outputs["Val"], light_map_b.inputs["Value"])
    links.new(detail_to_val.outputs["Val"], detail_map.inputs["Value"])
    links.new(bw_map_a.outputs["Result"], field_max_a.inputs[0])
    links.new(bw_map_b.outputs["Result"], field_max_a.inputs[1])
    links.new(mid_map_a.outputs["Result"], field_max_b.inputs[0])
    links.new(mid_map_b.outputs["Result"], field_max_b.inputs[1])
    links.new(field_max_a.outputs[0], field_max_c.inputs[0])
    links.new(field_max_b.outputs[0], field_max_c.inputs[1])
    links.new(field_max_c.outputs[0], field_seed.inputs[0])
    links.new(macro_density_map.outputs["Result"], field_seed.inputs[1])
    links.new(field_seed.outputs[0], field_factor.inputs[0])
    links.new(rewild_map.outputs["Result"], field_factor.inputs[1])
    links.new(field_factor.outputs[0], field_strength.inputs[0])
    links.new(shadow_map_a.outputs["Result"], shadow_max.inputs[0])
    links.new(shadow_map_b.outputs["Result"], shadow_max.inputs[1])
    links.new(shadow_max.outputs[0], stroke_seed.inputs[0])
    links.new(breakup_map.outputs["Result"], stroke_seed.inputs[1])
    links.new(stroke_seed.outputs[0], stroke_factor.inputs[0])
    links.new(rewild_map.outputs["Result"], stroke_factor.inputs[1])
    links.new(stroke_factor.outputs[0], stroke_strength.inputs[0])
    links.new(light_map_a.outputs["Result"], light_max.inputs[0])
    links.new(light_map_b.outputs["Result"], light_max.inputs[1])
    links.new(detail_map.outputs["Result"], detail_gain.inputs[0])
    links.new(light_max.outputs[0], accent_base.inputs[0])
    links.new(detail_gain.outputs[0], accent_base.inputs[1])
    links.new(accent_base.outputs[0], accent_seed.inputs[0])
    links.new(breakup_map.outputs["Result"], accent_seed.inputs[1])
    links.new(accent_seed.outputs[0], accent_factor.inputs[0])
    links.new(rewild_map.outputs["Result"], accent_factor.inputs[1])
    links.new(accent_factor.outputs[0], accent_strength.inputs[0])
    links.new(field_strength.outputs[0], field_aov.inputs["Value"])
    links.new(stroke_strength.outputs[0], stroke_aov.inputs["Value"])
    links.new(accent_strength.outputs[0], accent_aov.inputs["Value"])
    links.new(dry_color.outputs["Color"], field_mix.inputs[6])
    links.new(macro_color_mix.outputs[2], field_mix.inputs[7])
    links.new(rewild_map.outputs["Result"], field_mix.inputs[0])
    links.new(field_mix.outputs[2], dark_tint_mix.inputs[6])
    links.new(dark_clump_color.outputs["Color"], dark_tint_mix.inputs[7])
    links.new(field_mix.outputs[2], stroke_mix.inputs[6])
    links.new(dark_tint_mix.outputs[2], stroke_mix.inputs[7])
    links.new(stroke_strength.outputs[0], stroke_mix.inputs[0])
    links.new(stroke_mix.outputs[2], light_tint_mix.inputs[6])
    links.new(light_clump_color.outputs["Color"], light_tint_mix.inputs[7])
    links.new(stroke_mix.outputs[2], accent_mix.inputs[6])
    links.new(light_tint_mix.outputs[2], accent_mix.inputs[7])
    links.new(accent_strength.outputs[0], accent_mix.inputs[0])
    links.new(scenario.outputs["Fac"], rewild_map.inputs["Value"])
    links.new(dry_color.outputs["Color"], rewild_mix.inputs[6])
    links.new(accent_mix.outputs[2], rewild_mix.inputs[7])
    links.new(rewild_map.outputs["Result"], rewild_mix.inputs[0])
    links.new(rewild_mix.outputs[2], ao.inputs["Color"])
    links.new(ao.outputs["Color"], toon.inputs["Color"])
    links.new(rewild_mix.outputs[2], emission.inputs["Color"])
    links.new(toon.outputs["BSDF"], add_shader.inputs[0])
    links.new(emission.outputs["Emission"], add_shader.inputs[1])
    links.new(add_shader.outputs["Shader"], output.inputs["Surface"])

    return material


def add_layer(
    nodes,
    links,
    *,
    prefix: str,
    base_socket,
    selection_socket,
    density_socket,
    join_node,
    asset_object: bpy.types.Object,
    distance_min: float,
    density_max: float,
    density_multiplier: float,
    scale_min: float,
    scale_max: float,
    seed: int,
    location_x: float,
    location_y: float,
):
    density_mul = nodes.new("ShaderNodeMath")
    density_mul.name = f"{prefix} Density"
    density_mul.operation = "MULTIPLY"
    density_mul.inputs[1].default_value = density_multiplier

    distribute = nodes.new("GeometryNodeDistributePointsOnFaces")
    distribute.name = f"{prefix} Scatter"
    distribute.distribute_method = "POISSON"
    distribute.inputs[2].default_value = distance_min
    distribute.inputs[3].default_value = density_max
    distribute.inputs[4].default_value = 1.0
    distribute.inputs[6].default_value = seed

    object_info = nodes.new("GeometryNodeObjectInfo")
    object_info.name = f"{prefix} Asset"
    object_info.inputs[0].default_value = asset_object
    object_info.inputs[1].default_value = True

    random_scale = nodes.new("FunctionNodeRandomValue")
    random_scale.name = f"{prefix} Scale"
    random_scale.data_type = "FLOAT"
    random_scale.inputs[2].default_value = scale_min
    random_scale.inputs[3].default_value = scale_max
    random_scale.inputs[8].default_value = seed + 1000

    scale_vector = nodes.new("ShaderNodeCombineXYZ")
    scale_vector.name = f"{prefix} Scale Vector"

    instance = nodes.new("GeometryNodeInstanceOnPoints")
    instance.name = f"{prefix} Instance"

    density_mul.location = (location_x - 220, location_y - 220)
    distribute.location = (location_x, location_y)
    object_info.location = (location_x, location_y - 440)
    random_scale.location = (location_x + 220, location_y - 440)
    scale_vector.location = (location_x + 440, location_y - 440)
    instance.location = (location_x + 660, location_y - 120)

    links.new(density_socket, density_mul.inputs[0])
    links.new(base_socket, distribute.inputs[0])
    links.new(selection_socket, distribute.inputs[1])
    links.new(density_mul.outputs[0], distribute.inputs[5])
    links.new(distribute.outputs[0], instance.inputs[0])
    links.new(distribute.outputs[2], instance.inputs[5])
    links.new(object_info.outputs[4], instance.inputs[2])
    links.new(random_scale.outputs[1], scale_vector.inputs["X"])
    links.new(random_scale.outputs[1], scale_vector.inputs["Y"])
    links.new(random_scale.outputs[1], scale_vector.inputs["Z"])
    links.new(scale_vector.outputs["Vector"], instance.inputs[6])
    links.new(instance.outputs[0], join_node.inputs[0])


def rebuild_ground_group(
    large_patch: bpy.types.Object,
    mid_patch: bpy.types.Object,
    detail_patch: bpy.types.Object,
    ground_material: bpy.types.Material,
) -> bpy.types.NodeTree:
    group = bpy.data.node_groups.get(GROUND_GROUP_NAME)
    clip_group = bpy.data.node_groups.get(CLIP_GROUP_NAME)
    if group is None:
        raise RuntimeError(f"Geometry node group '{GROUND_GROUP_NAME}' not found")
    if clip_group is None:
        raise RuntimeError(f"Geometry node group '{CLIP_GROUP_NAME}' not found")

    nodes = group.nodes
    links = group.links
    nodes.clear()

    group_input = nodes.new("NodeGroupInput")
    clip_node = nodes.new("GeometryNodeGroup")
    base_material = nodes.new("GeometryNodeSetMaterial")
    named_attr = nodes.new("GeometryNodeInputNamedAttribute")
    selection_gate = nodes.new("ShaderNodeMath")
    density_map = nodes.new("ShaderNodeMapRange")
    join = nodes.new("GeometryNodeJoinGeometry")
    group_output = nodes.new("NodeGroupOutput")

    clip_node.name = "Clip Box Cull"
    clip_node.node_tree = clip_group
    base_material.name = "Base Material"
    base_material.inputs["Material"].default_value = ground_material
    named_attr.name = "Rewilding Attribute"
    named_attr.data_type = "FLOAT"
    named_attr.inputs["Name"].default_value = "scenario_rewilded"
    selection_gate.name = "Rewilding Selection"
    selection_gate.operation = "GREATER_THAN"
    selection_gate.inputs[1].default_value = 0.05
    density_map.name = "Density Map"
    density_map.clamp = True
    density_map.inputs["From Min"].default_value = 0.0
    density_map.inputs["From Max"].default_value = 3.0
    density_map.inputs["To Min"].default_value = 0.0
    density_map.inputs["To Max"].default_value = 1.0

    group_input.location = (-1280, 0)
    clip_node.location = (-1040, 0)
    base_material.location = (-820, 240)
    named_attr.location = (-1040, -320)
    selection_gate.location = (-820, -280)
    density_map.location = (-820, -480)
    join.location = (1060, 0)
    group_output.location = (1300, 0)

    links.new(group_input.outputs["Geometry"], clip_node.inputs["Geometry"])
    links.new(clip_node.outputs["Geometry"], base_material.inputs["Geometry"])
    links.new(base_material.outputs["Geometry"], join.inputs[0])
    links.new(named_attr.outputs["Attribute"], selection_gate.inputs[0])
    links.new(named_attr.outputs["Attribute"], density_map.inputs["Value"])

    add_layer(
        nodes,
        links,
        prefix="Accent Patch",
        base_socket=clip_node.outputs["Geometry"],
        selection_socket=selection_gate.outputs[0],
        density_socket=density_map.outputs["Result"],
        join_node=join,
        asset_object=mid_patch,
        distance_min=1.35,
        density_max=0.050,
        density_multiplier=0.40,
        scale_min=0.85,
        scale_max=1.30,
        seed=41,
        location_x=-540,
        location_y=-120,
    )

    links.new(join.outputs[0], group_output.inputs["Geometry"])
    return group


def save_current_blend(path: Path) -> None:
    bpy.ops.wm.save_as_mainfile(filepath=str(path))


def main() -> None:
    target_path = get_target_path()
    print(f"Painterly warm grass patch target: {target_path}")

    asset_collection = ensure_asset_collection(ASSET_COLLECTION_NAME)
    ensure_mask_aovs()
    color_image = ensure_texture_image(COLOR_TEXTURE_NAME, COLOR_TEXTURE_PATH)
    bw_image = ensure_texture_image(BW_TEXTURE_NAME, BW_TEXTURE_PATH)
    shadow_image = ensure_texture_image(SHADOW_TEXTURE_NAME, SHADOW_TEXTURE_PATH)
    mid_image = ensure_texture_image(MID_TEXTURE_NAME, MID_TEXTURE_PATH)
    light_image = ensure_texture_image(LIGHT_TEXTURE_NAME, LIGHT_TEXTURE_PATH)
    detail_image = ensure_texture_image(DETAIL_TEXTURE_NAME, DETAIL_TEXTURE_PATH)

    shadow_material = create_painterly_material(
        PATCH_LARGE_MATERIAL_NAME,
        (0.18, 0.07, 0.09, 1.0),
        (0.42, 0.13, 0.12, 1.0),
        (0.72, 0.27, 0.16, 1.0),
        ao_distance=0.55,
        toon_size=0.88,
        emission_strength=0.18,
    )
    mid_material = create_painterly_material(
        PATCH_MID_MATERIAL_NAME,
        (0.32, 0.10, 0.08, 1.0),
        (0.86, 0.33, 0.17, 1.0),
        (1.00, 0.76, 0.38, 1.0),
        ao_distance=0.42,
        toon_size=0.78,
        emission_strength=0.32,
    )
    highlight_material = create_painterly_material(
        PATCH_DETAIL_MATERIAL_NAME,
        (0.46, 0.18, 0.10, 1.0),
        (0.98, 0.49, 0.22, 1.0),
        (1.00, 0.90, 0.60, 1.0),
        ao_distance=0.28,
        toon_size=0.68,
        emission_strength=0.40,
    )

    large_patch = create_cluster_asset(
        PATCH_LARGE_OBJECT_NAME,
        asset_collection,
        shadow_material,
        seed=101,
        ribbon_count=40,
        footprint_x=2.40,
        footprint_y=0.85,
        height_min=0.22,
        height_max=0.40,
        width_min=0.08,
        width_max=0.16,
        bend_min=0.22,
        bend_max=0.52,
        lean_max=0.06,
    )
    mid_patch = create_cluster_asset(
        PATCH_MID_OBJECT_NAME,
        asset_collection,
        mid_material,
        seed=202,
        ribbon_count=24,
        footprint_x=1.24,
        footprint_y=0.42,
        height_min=0.16,
        height_max=0.28,
        width_min=0.05,
        width_max=0.10,
        bend_min=0.20,
        bend_max=0.46,
        lean_max=0.05,
    )
    detail_patch = create_cluster_asset(
        PATCH_DETAIL_OBJECT_NAME,
        asset_collection,
        highlight_material,
        seed=303,
        ribbon_count=14,
        footprint_x=0.60,
        footprint_y=0.20,
        height_min=0.14,
        height_max=0.24,
        width_min=0.035,
        width_max=0.07,
        bend_min=0.16,
        bend_max=0.36,
        lean_max=0.04,
    )

    ground_material = rebuild_ground_material(
        color_image,
        bw_image,
        shadow_image,
        mid_image,
        light_image,
        detail_image,
    )
    rebuild_ground_group(large_patch, mid_patch, detail_patch, ground_material)

    print(
        "Created painterly assets:",
        large_patch.name,
        mid_patch.name,
        detail_patch.name,
    )
    print("Updated Ground material and rebuilt layered painterly instancing")

    save_current_blend(target_path)
    print(f"Saved patched blend to {target_path}")


if __name__ == "__main__":
    main()
