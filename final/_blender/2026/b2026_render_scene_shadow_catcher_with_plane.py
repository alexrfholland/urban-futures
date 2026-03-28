from __future__ import annotations

import math
import os
import runpy
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data"
    / "blender"
    / "2026"
    / "baseline_renders"
    / "shadow_catcher_preview.png"
)
SUN_NAME = "BaselineShadowSun"
PLANE_NAME = "BaselineShadowPlane"
RECEIVER_NAME = "BaselineShadowTerrain"
PROXY_COLLECTION_NAME = "Baseline Shadow Proxies"
CLIPPER_SCRIPTS = [
    REPO_ROOT / "final" / "_blender" / "2026" / "b2026_clipbox_setup.py",
    REPO_ROOT / "final" / "_blender" / "2026" / "b2026_camera_clipboxes.py",
]


def env_str(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return value if value is not None and value != "" else default


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def parse_list(name: str) -> list[str]:
    raw = env_str(name, "")
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def render_mode() -> str:
    return env_str("B2026_RENDER_MODE", "shadow_catcher")


def setup_only() -> bool:
    return env_bool("B2026_SETUP_ONLY", False)


def get_scene() -> bpy.types.Scene:
    scene_name = env_str("B2026_SCENE_NAME", "city")
    scene = bpy.data.scenes.get(scene_name)
    if scene is None:
        raise ValueError(f"Scene '{scene_name}' was not found")
    return scene


def get_camera(scene: bpy.types.Scene) -> bpy.types.Object:
    camera_name = env_str("B2026_CAMERA_NAME", "")
    camera = bpy.data.objects.get(camera_name) if camera_name else scene.camera
    if camera is None or camera.type != "CAMERA":
        raise ValueError(
            f"Camera '{camera_name or '<scene camera>'}' was not found for scene '{scene.name}'"
        )
    return camera


def ensure_collection(scene: bpy.types.Scene, name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
    if collection.name not in scene.collection.children:
        scene.collection.children.link(collection)
    return collection


def world_bound_corners(obj: bpy.types.Object) -> list[Vector]:
    return [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]


def receiver_reference_object(scene: bpy.types.Scene) -> bpy.types.Object | None:
    requested_name = env_str("B2026_RECEIVER_SOURCE_OBJECT", "city_highResRoad.001")
    if requested_name:
        obj = scene.objects.get(requested_name)
        if obj is not None and obj.type == "MESH":
            return obj
    return None


def plane_reference_objects(scene: bpy.types.Scene) -> list[bpy.types.Object]:
    requested = parse_list("B2026_PLANE_REFERENCE_OBJECTS")
    if requested:
        objects = []
        for name in requested:
            obj = scene.objects.get(name)
            if obj is not None:
                objects.append(obj)
        return objects

    objects: list[bpy.types.Object] = []
    for obj in scene.objects:
        if obj.type != "MESH" or obj.data is None:
            continue
        if obj.name == PLANE_NAME or obj.name.endswith("_shadow_proxy"):
            continue
        if obj.hide_render:
            continue
        if len(obj.data.polygons) == 0:
            continue
        objects.append(obj)
    return objects


def scene_mesh_bounds(scene: bpy.types.Scene) -> tuple[Vector, Vector]:
    min_corner = Vector((float("inf"), float("inf"), float("inf")))
    max_corner = Vector((float("-inf"), float("-inf"), float("-inf")))
    found = False

    for obj in plane_reference_objects(scene):
        for corner in world_bound_corners(obj):
            min_corner.x = min(min_corner.x, corner.x)
            min_corner.y = min(min_corner.y, corner.y)
            min_corner.z = min(min_corner.z, corner.z)
            max_corner.x = max(max_corner.x, corner.x)
            max_corner.y = max(max_corner.y, corner.y)
            max_corner.z = max(max_corner.z, corner.z)
            found = True

    if not found:
        min_corner = Vector((-250.0, -250.0, 0.0))
        max_corner = Vector((250.0, 250.0, 0.0))
    return min_corner, max_corner


def ensure_white_ground_material() -> bpy.types.Material:
    material = bpy.data.materials.get("BaselineShadowGround")
    if material is None:
        material = bpy.data.materials.new("BaselineShadowGround")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    bsdf.inputs["Specular IOR Level"].default_value = 0.0
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def configure_world_fill(scene: bpy.types.Scene) -> None:
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    background = nodes.get("Background")
    if background is None:
        background = nodes.new("ShaderNodeBackground")
    background.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    background.inputs[1].default_value = env_float("B2026_WORLD_STRENGTH", 1.0)


def ensure_proxy_material() -> bpy.types.Material:
    material = bpy.data.materials.get("BaselineShadowCaster")
    if material is None:
        material = bpy.data.materials.new("BaselineShadowCaster")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    mix_shader = nodes.new("ShaderNodeMixShader")
    light_path = nodes.new("ShaderNodeLightPath")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    transparent.location = (-220, -40)
    bsdf.location = (-220, -180)
    light_path.location = (-440, 20)
    mix_shader.location = (0, 0)
    output.location = (220, 0)

    transparent.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bsdf.inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    bsdf.inputs["Specular IOR Level"].default_value = 0.0

    links.new(light_path.outputs["Is Camera Ray"], mix_shader.inputs["Fac"])
    links.new(bsdf.outputs["BSDF"], mix_shader.inputs[1])
    links.new(transparent.outputs["BSDF"], mix_shader.inputs[2])
    links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])
    return material


def ensure_shadow_plane(scene: bpy.types.Scene) -> bpy.types.Object:
    min_corner, max_corner = scene_mesh_bounds(scene)
    center = (min_corner + max_corner) * 0.5
    scale_multiplier = env_float("B2026_PLANE_SCALE_MULTIPLIER", 1.15)
    size_x = max(max_corner.x - min_corner.x, 10.0) * scale_multiplier
    size_y = max(max_corner.y - min_corner.y, 10.0) * scale_multiplier
    z_location = min_corner.z

    plane = scene.objects.get(PLANE_NAME)
    if plane is None:
        mesh = bpy.data.meshes.new(f"{PLANE_NAME}Mesh")
        plane = bpy.data.objects.new(PLANE_NAME, mesh)
        scene.collection.objects.link(plane)
        mesh.from_pydata(
            [
                (-0.5, -0.5, 0.0),
                (0.5, -0.5, 0.0),
                (0.5, 0.5, 0.0),
                (-0.5, 0.5, 0.0),
            ],
            [],
            [(0, 1, 2, 3)],
        )
        mesh.update()

    plane.location = (center.x, center.y, z_location)
    plane.rotation_euler = (0.0, 0.0, 0.0)
    plane.scale = (size_x, size_y, 1.0)
    plane.hide_render = False
    plane.visible_camera = True
    plane.visible_shadow = True
    if hasattr(plane, "is_shadow_catcher"):
        plane.is_shadow_catcher = render_mode() == "shadow_catcher"
    plane.pass_index = 1

    material = ensure_white_ground_material()
    plane.data.materials.clear()
    plane.data.materials.append(material)
    return plane


def cleanup_legacy_plane(scene: bpy.types.Scene) -> None:
    plane = scene.objects.get(PLANE_NAME)
    if plane is not None:
        bpy.data.objects.remove(plane, do_unlink=True)


def ensure_shadow_receiver(scene: bpy.types.Scene) -> bpy.types.Object:
    source_obj = receiver_reference_object(scene)
    if source_obj is None:
        return ensure_shadow_plane(scene)

    cleanup_legacy_plane(scene)
    receiver = scene.objects.get(RECEIVER_NAME)
    if receiver is None:
        mesh = source_obj.data.copy()
        receiver = bpy.data.objects.new(RECEIVER_NAME, mesh)
        scene.collection.objects.link(receiver)
    else:
        old_mesh = receiver.data if receiver.data is not None else None
        receiver.data = source_obj.data.copy()
        if receiver.data is not None:
            receiver.data.name = f"{RECEIVER_NAME}Mesh"
        if old_mesh is not None and old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)

    receiver.matrix_world = source_obj.matrix_world.copy()
    receiver.hide_render = False
    receiver.hide_viewport = False
    receiver.visible_camera = True
    receiver.visible_shadow = True
    if hasattr(receiver, "is_shadow_catcher"):
        receiver.is_shadow_catcher = render_mode() == "shadow_catcher"
    receiver.pass_index = 1

    material = ensure_white_ground_material()
    receiver.data.materials.clear()
    receiver.data.materials.append(material)
    return receiver


def ensure_sun(scene: bpy.types.Scene) -> bpy.types.Object:
    sun_object = bpy.data.objects.get(SUN_NAME)
    if sun_object is None or sun_object.type != "LIGHT":
        light_data = bpy.data.lights.new(name=SUN_NAME, type="SUN")
        sun_object = bpy.data.objects.new(SUN_NAME, light_data)
        scene.collection.objects.link(sun_object)
    light_data = sun_object.data
    light_data.type = "SUN"
    light_data.energy = env_float("B2026_SUN_ENERGY", 2.5)
    light_data.angle = math.radians(env_float("B2026_SUN_ANGLE_DEG", 2.0))
    sun_object.rotation_euler = (
        math.radians(env_float("B2026_SUN_ROT_X_DEG", 28.0)),
        math.radians(env_float("B2026_SUN_ROT_Y_DEG", 0.0)),
        math.radians(env_float("B2026_SUN_ROT_Z_DEG", 144.0)),
    )
    sun_object.hide_render = False
    sun_object.hide_viewport = False
    return sun_object


def run_clipper_scripts_if_requested() -> None:
    if not env_bool("B2026_RUN_CLIPPERS", False):
        return
    for script_path in CLIPPER_SCRIPTS:
        runpy.run_path(str(script_path), run_name="__main__")


def collection_info_nodes_for_object(obj: bpy.types.Object) -> list[bpy.types.Node]:
    nodes: list[bpy.types.Node] = []
    for modifier in obj.modifiers:
        if modifier.type != "NODES" or modifier.node_group is None:
            continue
        for node in modifier.node_group.nodes:
            if node.bl_idname == "GeometryNodeCollectionInfo":
                nodes.append(node)
    return nodes


def set_source_collection_render_visibility(obj_names: list[str], hide_render: bool) -> None:
    for obj_name in obj_names:
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            continue
        for node in collection_info_nodes_for_object(obj):
            collection = None
            if node.inputs:
                try:
                    collection = node.inputs[0].default_value
                except Exception:
                    collection = None
            if collection is None:
                continue
            for source_obj in collection.all_objects:
                source_obj.hide_render = hide_render


def clear_previous_shadow_proxies(scene: bpy.types.Scene) -> None:
    for obj in list(scene.objects):
        if obj.name.endswith("_shadow_proxy"):
            bpy.data.objects.remove(obj, do_unlink=True)


def attribute_values(mesh: bpy.types.Mesh, name: str, default: float = 0.0) -> list[float]:
    attr = mesh.attributes.get(name)
    if attr is None:
        return [default] * len(mesh.vertices)
    return [float(item.value) for item in attr.data]


def rotation_radians(values: list[float]) -> list[float]:
    if not values:
        return []
    if any(abs(value) > (math.tau + 1.0) for value in values):
        return [math.radians(value) for value in values]
    return values


def size_profile(size_value: int) -> tuple[str, tuple[float, float, float]]:
    profiles = {
        1: ("canopy", (2.5, 2.5, 2.0)),
        2: ("canopy", (3.5, 3.5, 2.6)),
        3: ("canopy", (5.0, 5.0, 3.4)),
        4: ("snag", (0.45, 0.45, 6.0)),
        5: ("fallen", (4.5, 0.55, 0.55)),
        6: ("fallen", (5.8, 0.7, 0.7)),
    }
    return profiles.get(size_value, ("canopy", (3.0, 3.0, 2.2)))


def create_point_shadow_proxy_mesh(
    source_obj: bpy.types.Object,
    proxy_name: str,
) -> bpy.types.Mesh:
    mesh = source_obj.data
    size_values = [int(round(value)) for value in attribute_values(mesh, "size", default=2.0)]
    rotation_values = rotation_radians(attribute_values(mesh, "rotation", default=0.0))

    bm = bmesh.new()
    for index, vertex in enumerate(mesh.vertices):
        world_co = source_obj.matrix_world @ vertex.co
        size_value = size_values[index] if index < len(size_values) else 2
        rotation_z = rotation_values[index] if index < len(rotation_values) else 0.0
        shape_kind, dims = size_profile(size_value)
        if shape_kind == "fallen":
            half_x, half_y, half_z = dims[0] * 0.5, dims[1] * 0.5, dims[2] * 0.5
            matrix = (
                Matrix.Translation(world_co + Vector((0.0, 0.0, half_z)))
                @ Matrix.Rotation(rotation_z, 4, "Z")
                @ Matrix.Diagonal((half_x, half_y, half_z, 1.0))
            )
            bmesh.ops.create_cube(bm, size=2.0, matrix=matrix)
        elif shape_kind == "snag":
            radius = dims[0]
            depth = dims[2]
            matrix = (
                Matrix.Translation(world_co + Vector((0.0, 0.0, depth * 0.5)))
                @ Matrix.Rotation(rotation_z, 4, "Z")
            )
            bmesh.ops.create_cone(
                bm,
                cap_ends=True,
                cap_tris=False,
                segments=8,
                radius1=radius,
                radius2=max(radius * 0.6, 0.15),
                depth=depth,
                matrix=matrix,
            )
        else:
            radius = dims[0]
            height = dims[2]
            matrix = Matrix.Translation(world_co + Vector((0.0, 0.0, height)))
            bmesh.ops.create_icosphere(
                bm,
                subdivisions=1,
                radius=radius,
                matrix=matrix,
            )

    proxy_mesh = bpy.data.meshes.new(f"{proxy_name}Mesh")
    bm.to_mesh(proxy_mesh)
    bm.free()
    proxy_mesh.update()
    return proxy_mesh


def create_shadow_proxies(scene: bpy.types.Scene, source_object_names: list[str]) -> list[bpy.types.Object]:
    if not source_object_names:
        return []

    clear_previous_shadow_proxies(scene)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    proxy_collection = ensure_collection(scene, PROXY_COLLECTION_NAME)
    proxy_material = ensure_proxy_material()
    proxies: list[bpy.types.Object] = []

    for source_name in source_object_names:
        source_obj = scene.objects.get(source_name)
        if source_obj is None:
            continue
        evaluated = source_obj.evaluated_get(depsgraph)
        generated_in_world_space = False
        mesh = bpy.data.meshes.new_from_object(
            evaluated, preserve_all_data_layers=True, depsgraph=depsgraph
        )
        proxy_name = f"{source_obj.name}_shadow_proxy"
        if len(mesh.polygons) == 0 and len(mesh.vertices) > 0:
            bpy.data.meshes.remove(mesh)
            mesh = create_point_shadow_proxy_mesh(source_obj, proxy_name)
            generated_in_world_space = True
        proxy = bpy.data.objects.new(proxy_name, mesh)
        proxy_collection.objects.link(proxy)
        proxy.matrix_world = Matrix.Identity(4) if generated_in_world_space else evaluated.matrix_world.copy()
        proxy.hide_render = False
        proxy.visible_camera = True
        proxy.visible_shadow = True
        proxy.pass_index = source_obj.pass_index
        proxy.data.materials.clear()
        proxy.data.materials.append(proxy_material)
        proxies.append(proxy)

    return proxies


def configure_visibility(scene: bpy.types.Scene, receiver: bpy.types.Object) -> None:
    ground_object_names = set(parse_list("B2026_HIDE_GROUND_OBJECTS"))
    for obj in scene.objects:
        if obj == receiver:
            continue
        if obj.name in ground_object_names:
            obj.hide_render = True

    if render_mode() == "shadow_only_beauty":
        for obj in scene.objects:
            if obj == receiver or obj.type in {"CAMERA", "LIGHT"}:
                continue
            if obj.hide_render:
                continue
            obj.visible_camera = False
            obj.visible_shadow = True


def enable_only_target_view_layer(scene: bpy.types.Scene) -> bpy.types.ViewLayer:
    requested_name = env_str("B2026_VIEW_LAYER_NAME", "")
    view_layer = scene.view_layers.get(requested_name) if requested_name else scene.view_layers[0]
    for candidate in scene.view_layers:
        candidate.use = candidate.name == view_layer.name
    view_layer.use_pass_shadow = True
    view_layer.use_pass_combined = True
    view_layer.cycles.use_pass_shadow_catcher = True
    return view_layer


def configure_compositor(scene: bpy.types.Scene, view_layer_name: str) -> None:
    scene.use_nodes = True
    node_tree = scene.node_tree
    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()

    render_node = nodes.new("CompositorNodeRLayers")
    render_node.scene = scene
    render_node.layer = view_layer_name
    render_node.location = (-400, 0)

    white_node = nodes.new("CompositorNodeRGB")
    white_node.location = (-400, 220)
    white_node.outputs[0].default_value = (1.0, 1.0, 1.0, 1.0)

    alpha_over = nodes.new("CompositorNodeAlphaOver")
    alpha_over.location = (-80, 80)
    alpha_over.use_premultiply = True

    composite = nodes.new("CompositorNodeComposite")
    composite.location = (220, 80)

    links.new(white_node.outputs[0], alpha_over.inputs[1])
    links.new(render_node.outputs["Shadow Catcher"], alpha_over.inputs[2])
    links.new(alpha_over.outputs[0], composite.inputs[0])


def configure_render(scene: bpy.types.Scene, output_path: Path) -> None:
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = env_int("B2026_RENDER_X", 1920)
    scene.render.resolution_y = env_int("B2026_RENDER_Y", 1080)
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(output_path)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Standard"
    if hasattr(scene.view_settings, "look"):
        scene.view_settings.look = "None"
    scene.render.film_transparent = False
    scene.render.use_compositing = render_mode() == "shadow_catcher"
    scene.cycles.samples = env_int("B2026_CYCLES_SAMPLES", 64)
    scene.cycles.use_denoising = True
    output_path.parent.mkdir(parents=True, exist_ok=True)


def save_mainfile_if_requested() -> None:
    if env_bool("B2026_SAVE_MAINFILE", False):
        bpy.ops.wm.save_mainfile()


def main() -> None:
    scene = get_scene()
    camera = get_camera(scene)
    run_clipper_scripts_if_requested()

    if setup_only():
        receiver = ensure_shadow_receiver(scene)
        sun = ensure_sun(scene)
        configure_world_fill(scene)
        receiver.hide_render = True
        receiver.visible_camera = False
        receiver.visible_shadow = False
        if hasattr(receiver, "is_shadow_catcher"):
            receiver.is_shadow_catcher = False
        sun.hide_render = True
        scene.camera = camera
        print(
            f"[shadow_catcher_with_plane] setup_only scene={scene.name} "
            f"camera={camera.name} receiver={receiver.name} sun={sun.name}",
            flush=True,
        )
        save_mainfile_if_requested()
        return

    if env_bool("B2026_UNHIDE_SOURCE_COLLECTIONS", False):
        set_source_collection_render_visibility(parse_list("B2026_SOURCE_OBJECTS"), False)

    proxies = []
    if env_bool("B2026_CREATE_SHADOW_PROXIES", False):
        proxies = create_shadow_proxies(scene, parse_list("B2026_SOURCE_OBJECTS"))

    receiver = ensure_shadow_receiver(scene)
    ensure_sun(scene)
    configure_world_fill(scene)
    view_layer = enable_only_target_view_layer(scene)
    configure_visibility(scene, receiver)
    if render_mode() == "shadow_catcher":
        configure_compositor(scene, view_layer.name)
    else:
        scene.use_nodes = False
    configure_render(scene, env_path("B2026_OUTPUT_PATH", DEFAULT_OUTPUT))
    scene.camera = camera

    print(
        f"[shadow_catcher_with_plane] mode={render_mode()} scene={scene.name} "
        f"view_layer={view_layer.name} camera={camera.name} receiver={receiver.name} proxies={len(proxies)} "
        f"output={scene.render.filepath}",
        flush=True,
    )
    bpy.ops.render.render(write_still=True, scene=scene.name)
    save_mainfile_if_requested()


if __name__ == "__main__":
    main()
