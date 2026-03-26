import json
from pathlib import Path

import bpy


TEXTURE_ROOTS = (
    Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final/trimmed-parade/textures/rewilded_sim_turns"),
    Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final/city/textures/rewilded_sim_turns"),
)
RAW_TEXTURE_KEY = "raw_texture"
MASK_TEXTURE_KEY = "mask_texture"
PREVIEW_TEXTURE_KEY = "preview_texture"
AUTO_ASSIGN_BY_PREFIX = True


def load_metadata_files():
    files = []
    for root in TEXTURE_ROOTS:
        if root.exists():
            for metadata_path in sorted(root.glob("*_meta.json")):
                try:
                    metadata = json.loads(metadata_path.read_text())
                except json.JSONDecodeError:
                    print(f"Skipping unreadable metadata: {metadata_path}")
                    continue

                if metadata_has_existing_ply(metadata_path, metadata):
                    files.append(metadata_path)
                else:
                    print(f"Skipping metadata without matching PLY on disk: {metadata_path}")
    return files


def metadata_has_existing_ply(metadata_path: Path, metadata: dict) -> bool:
    site_root = metadata_path.parents[2]
    ply_dir = site_root / "ply"
    return any((ply_dir / name).exists() for name in metadata.get("ply_candidates", []))


def clear_node_tree(node_tree):
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)


def load_image(image_path: Path, non_color: bool):
    image = bpy.data.images.load(str(image_path), check_existing=True)
    image.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    return image


def new_math(nodes, operation, location, value=None, use_clamp=False):
    node = nodes.new("ShaderNodeMath")
    node.operation = operation
    node.location = location
    node.use_clamp = use_clamp
    if value is not None:
        node.inputs[1].default_value = value
    return node


def build_normal_masks(nodes, links, separate_normal):
    abs_x = new_math(nodes, "ABSOLUTE", (-1200, 350))
    abs_y = new_math(nodes, "ABSOLUTE", (-1200, 170))
    abs_z = new_math(nodes, "ABSOLUTE", (-1200, -10))
    links.new(separate_normal.outputs["X"], abs_x.inputs[0])
    links.new(separate_normal.outputs["Y"], abs_y.inputs[0])
    links.new(separate_normal.outputs["Z"], abs_z.inputs[0])

    def mask_for(axis_name, positive, location_y):
        axis_socket = separate_normal.outputs[axis_name.upper()]
        abs_axis = {"x": abs_x, "y": abs_y, "z": abs_z}[axis_name].outputs[0]
        other_axes = [axis for axis in "xyz" if axis != axis_name]
        abs_other_a = {"x": abs_x, "y": abs_y, "z": abs_z}[other_axes[0]].outputs[0]
        abs_other_b = {"x": abs_x, "y": abs_y, "z": abs_z}[other_axes[1]].outputs[0]

        sign = new_math(nodes, "GREATER_THAN" if positive else "LESS_THAN", (-950, location_y), 0.0)
        links.new(axis_socket, sign.inputs[0])

        compare_a_sub = new_math(nodes, "SUBTRACT", (-950, location_y - 120))
        links.new(abs_axis, compare_a_sub.inputs[0])
        links.new(abs_other_a, compare_a_sub.inputs[1])
        compare_a = new_math(nodes, "GREATER_THAN", (-760, location_y - 120), -1e-5)
        links.new(compare_a_sub.outputs[0], compare_a.inputs[0])

        compare_b_sub = new_math(nodes, "SUBTRACT", (-950, location_y - 240))
        links.new(abs_axis, compare_b_sub.inputs[0])
        links.new(abs_other_b, compare_b_sub.inputs[1])
        compare_b = new_math(nodes, "GREATER_THAN", (-760, location_y - 240), -1e-5)
        links.new(compare_b_sub.outputs[0], compare_b.inputs[0])

        mul_a = new_math(nodes, "MULTIPLY", (-570, location_y - 80))
        links.new(sign.outputs[0], mul_a.inputs[0])
        links.new(compare_a.outputs[0], mul_a.inputs[1])

        mul_b = new_math(nodes, "MULTIPLY", (-380, location_y - 160))
        links.new(mul_a.outputs[0], mul_b.inputs[0])
        links.new(compare_b.outputs[0], mul_b.inputs[1])
        return mul_b.outputs[0]

    return {
        "x_pos": mask_for("x", True, 420),
        "x_neg": mask_for("x", False, 120),
        "y_pos": mask_for("y", True, -180),
        "y_neg": mask_for("y", False, -480),
        "z_pos": mask_for("z", True, -780),
        "z_neg": mask_for("z", False, -1080),
    }


def build_projection_vector(nodes, links, coord_socket, min_index, size, output_extent, location):
    divide_voxel = new_math(nodes, "DIVIDE", location, size)
    links.new(coord_socket, divide_voxel.inputs[0])

    subtract_index = new_math(nodes, "SUBTRACT", (location[0] + 180, location[1]), float(min_index))
    links.new(divide_voxel.outputs[0], subtract_index.inputs[0])

    add_half = new_math(nodes, "ADD", (location[0] + 360, location[1]), 0.5)
    links.new(subtract_index.outputs[0], add_half.inputs[0])

    divide_extent = new_math(nodes, "DIVIDE", (location[0] + 540, location[1]), float(output_extent))
    links.new(add_half.outputs[0], divide_extent.inputs[0])
    return divide_extent.outputs[0]


def build_projection_sampler(nodes, links, separate_position, projection_name, projection_meta, texture_dir, location):
    u_axis = projection_meta["u_axis"].upper()
    v_axis = projection_meta["v_axis"].upper()
    width = projection_meta["width"]
    height = projection_meta["height"]

    u_socket = build_projection_vector(
        nodes,
        links,
        separate_position.outputs[u_axis],
        projection_meta["u_min_index"],
        projection_meta["voxel_size"],
        width,
        (location[0], location[1]),
    )
    v_socket = build_projection_vector(
        nodes,
        links,
        separate_position.outputs[v_axis],
        projection_meta["v_min_index"],
        projection_meta["voxel_size"],
        height,
        (location[0], location[1] - 180),
    )

    combine = nodes.new("ShaderNodeCombineXYZ")
    combine.location = (location[0] + 760, location[1] - 90)
    links.new(u_socket, combine.inputs["X"])
    links.new(v_socket, combine.inputs["Y"])
    combine.inputs["Z"].default_value = 0.0

    image_node = nodes.new("ShaderNodeTexImage")
    image_node.name = f"Projection {projection_name}"
    image_node.label = f"Projection {projection_name}"
    image_node.location = (location[0] + 980, location[1] - 90)
    raw_texture = projection_meta.get(RAW_TEXTURE_KEY)
    preview_texture = projection_meta.get(PREVIEW_TEXTURE_KEY)
    if raw_texture:
        image_node.image = load_image(texture_dir / raw_texture, non_color=True)
    elif preview_texture:
        image_node.image = load_image(texture_dir / preview_texture, non_color=False)
    else:
        raise FileNotFoundError(f"No texture entry for projection {projection_name}")
    image_node.interpolation = "Closest"
    image_node.extension = "CLIP"
    links.new(combine.outputs["Vector"], image_node.inputs["Vector"])

    mask_socket = None
    mask_texture = projection_meta.get(MASK_TEXTURE_KEY)
    if mask_texture:
        mask_node = nodes.new("ShaderNodeTexImage")
        mask_node.name = f"Projection {projection_name} Mask"
        mask_node.label = f"Projection {projection_name} Mask"
        mask_node.location = (location[0] + 980, location[1] - 270)
        mask_node.image = load_image(texture_dir / mask_texture, non_color=True)
        mask_node.interpolation = "Closest"
        mask_node.extension = "CLIP"
        links.new(combine.outputs["Vector"], mask_node.inputs["Vector"])

        separate_mask = nodes.new("ShaderNodeSeparateColor")
        separate_mask.location = (location[0] + 1200, location[1] - 270)
        links.new(mask_node.outputs["Color"], separate_mask.inputs["Color"])
        mask_socket = separate_mask.outputs["Red"]
    elif preview_texture:
        mask_socket = image_node.outputs["Alpha"]

    return image_node, mask_socket


def configure_viridis_ramp(color_ramp):
    elements = color_ramp.color_ramp.elements
    elements[0].position = 0.0
    elements[0].color = (0.267, 0.005, 0.329, 1.0)
    if len(elements) == 1:
        elements.new(1.0)
    elements[1].position = 1.0
    elements[1].color = (0.993, 0.906, 0.144, 1.0)
    for position, color in (
        (0.25, (0.230, 0.322, 0.545, 1.0)),
        (0.50, (0.128, 0.567, 0.551, 1.0)),
        (0.75, (0.369, 0.789, 0.383, 1.0)),
    ):
        element = elements.new(position)
        element.color = color


def ensure_material_for_metadata(metadata_path: Path):
    metadata = json.loads(metadata_path.read_text())
    material_name = f"SIM_Turns {metadata['site']} {metadata.get('scenario') or 'base'} {metadata['surface_kind']}"
    material = bpy.data.materials.get(material_name)
    if material is None:
        material = bpy.data.materials.new(material_name)
    material.use_nodes = True

    node_tree = material.node_tree
    clear_node_tree(node_tree)
    nodes = node_tree.nodes
    links = node_tree.links

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (1600, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (1320, 0)
    bsdf.inputs["Roughness"].default_value = 0.9
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    geometry = nodes.new("ShaderNodeNewGeometry")
    geometry.location = (-1700, 0)
    separate_position = nodes.new("ShaderNodeSeparateXYZ")
    separate_position.location = (-1460, 160)
    separate_normal = nodes.new("ShaderNodeSeparateXYZ")
    separate_normal.location = (-1460, -300)
    links.new(geometry.outputs["Position"], separate_position.inputs["Vector"])
    links.new(geometry.outputs["Normal"], separate_normal.inputs["Vector"])

    masks = build_normal_masks(nodes, links, separate_normal)
    texture_dir = metadata_path.parent

    projection_images = {}
    x_base = -180
    y_base = 780
    y_step = -420
    use_raw_textures = any(
        RAW_TEXTURE_KEY in projection_meta for projection_meta in metadata["projections"].values()
    )
    for index, projection_name in enumerate(["x_pos", "x_neg", "y_pos", "y_neg", "z_pos", "z_neg"]):
        projection_meta = metadata["projections"].get(projection_name)
        if projection_meta is None:
            continue

        projection_meta = {
            **projection_meta,
            "voxel_size": metadata["voxel_size"],
        }
        projection_images[projection_name] = build_projection_sampler(
            nodes,
            links,
            separate_position,
            projection_name,
            projection_meta,
            texture_dir,
            (x_base, y_base + index * y_step),
        )

    rgb_black = nodes.new("ShaderNodeRGB")
    rgb_black.location = (720, 320)
    rgb_black.outputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    current_color = rgb_black.outputs["Color"]

    mix_x = 960
    mix_y = 320
    for index, projection_name in enumerate(["x_pos", "x_neg", "y_pos", "y_neg", "z_pos", "z_neg"]):
        projection_nodes = projection_images.get(projection_name)
        if projection_nodes is None:
            continue
        image_node, mask_socket = projection_nodes

        alpha_mul = new_math(nodes, "MULTIPLY", (mix_x - 260, mix_y - index * 180), 1.0)
        links.new(masks[projection_name], alpha_mul.inputs[0])
        if mask_socket is not None:
            links.new(mask_socket, alpha_mul.inputs[1])
        else:
            alpha_mul.inputs[1].default_value = 1.0

        mix = nodes.new("ShaderNodeMixRGB")
        mix.blend_type = "MIX"
        mix.location = (mix_x, mix_y - index * 180)
        links.new(alpha_mul.outputs[0], mix.inputs["Fac"])
        links.new(current_color, mix.inputs["Color1"])
        links.new(image_node.outputs["Color"], mix.inputs["Color2"])
        current_color = mix.outputs["Color"]

    if use_raw_textures:
        separate_value = nodes.new("ShaderNodeSeparateColor")
        separate_value.location = (1180, 280)
        links.new(current_color, separate_value.inputs["Color"])

        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.location = (1420, 280)
        configure_viridis_ramp(ramp)
        links.new(separate_value.outputs["Red"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        links.new(current_color, bsdf.inputs["Base Color"])
    return material, metadata


def assign_material_to_matching_objects(material, metadata):
    matched = []
    if not AUTO_ASSIGN_BY_PREFIX:
        return matched

    candidate_prefixes = [Path(name).stem for name in metadata.get("ply_candidates", [])]
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if any(obj.name.startswith(prefix) for prefix in candidate_prefixes):
            if obj.data.materials:
                obj.data.materials[0] = material
            else:
                obj.data.materials.append(material)
            matched.append(obj.name)
    return matched


def main():
    metadata_files = load_metadata_files()
    if not metadata_files:
        raise FileNotFoundError("No rewilded sim-turn texture metadata files found.")

    for metadata_path in metadata_files:
        material, metadata = ensure_material_for_metadata(metadata_path)
        matched = assign_material_to_matching_objects(material, metadata)
        print(f"Material ready: {material.name}")
        print(f"Metadata: {metadata_path}")
        print(f"Matched objects: {matched}")


if __name__ == "__main__":
    main()
