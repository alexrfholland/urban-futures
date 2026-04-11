"""Render isolated tree/log library PLYs as previews and face-AOV EXRs.

This script is intended for headless Blender runs. It builds a fresh scene,
imports tree/log PLY assets in small chunks, and renders:

- fast Workbench PNG previews for framing checks
- one multilayer EXR per asset with only face-level resource AOVs

Key rules for this workflow:
- one asset visible per render
- one shared orthographic isometric camera for the whole library
- the shared camera fit is derived from the largest spatial isometric footprint
  in the selected library set
- tree-instancer AOVs are intentionally excluded
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import bpy
from mathutils import Vector


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
CODE_ROOT = next(parent for parent in SCRIPT_PATH.parents if parent.name == "_futureSim_refactored")
REPO_ROOT = CODE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from _futureSim_refactored.paths import hook_log_ply_library_dir, hook_tree_ply_library_dir
except Exception:
    hook_log_ply_library_dir = None
    hook_tree_ply_library_dir = None

try:
    from _futureSim_refactored.blender.blenderv2.bV2_paths import iter_blender_input_roots
except Exception:
    iter_blender_input_roots = None


SCENE_NAME = "bV2_ply_library_assets"
VIEW_LAYER_NAME = "asset"
CAMERA_NAME = "bV2_ply_library_iso"
ASSET_COLLECTION_NAME = "asset_library"
MATERIAL_NAME = "BV2_LIBRARY_FACE_AOVS"
LOG_PATH = os.environ.get("BV2_LOG_PATH", "").strip()

RESOURCE_COLOURS = {
    "resource_other": (0.81, 0.81, 0.81, 1.0),
    "resource_dead branch": (1.00, 0.80, 0.00, 1.0),
    "resource_peeling bark": (1.00, 0.52, 0.75, 1.0),
    "resource_perch branch": (1.00, 0.80, 0.00, 1.0),
    "resource_epiphyte": (0.77, 0.89, 0.56, 1.0),
    "resource_fallen log": (0.56, 0.54, 0.75, 1.0),
    "resource_hollow": (0.81, 0.43, 0.85, 1.0),
}

RESOURCE_MASK_SPECS = (
    {
        "attr_name": "resource_other",
        "aov_name": "resource_none_mask",
        "label": "None",
        "color": RESOURCE_COLOURS["resource_other"],
    },
    {
        "attr_name": "resource_dead branch",
        "aov_name": "resource_dead_branch_mask",
        "label": "Dead Branch",
        "color": RESOURCE_COLOURS["resource_dead branch"],
    },
    {
        "attr_name": "resource_peeling bark",
        "aov_name": "resource_peeling_bark_mask",
        "label": "Peeling Bark",
        "color": RESOURCE_COLOURS["resource_peeling bark"],
    },
    {
        "attr_name": "resource_perch branch",
        "aov_name": "resource_perch_branch_mask",
        "label": "Perch Branch",
        "color": RESOURCE_COLOURS["resource_perch branch"],
    },
    {
        "attr_name": "resource_epiphyte",
        "aov_name": "resource_epiphyte_mask",
        "label": "Epiphyte",
        "color": RESOURCE_COLOURS["resource_epiphyte"],
    },
    {
        "attr_name": "resource_fallen log",
        "aov_name": "resource_fallen_log_mask",
        "label": "Fallen Log",
        "color": RESOURCE_COLOURS["resource_fallen log"],
    },
    {
        "attr_name": "resource_hollow",
        "aov_name": "resource_hollow_mask",
        "label": "Hollow",
        "color": RESOURCE_COLOURS["resource_hollow"],
    },
)

AOV_SPECS = (
    ("resource", "VALUE"),
    ("isSenescent", "VALUE"),
    ("isTerminal", "VALUE"),
    ("resource_tree_mask", "VALUE"),
    ("resource_colour", "COLOR"),
    *[(spec["aov_name"], "VALUE") for spec in RESOURCE_MASK_SPECS],
)

GPU_BACKEND_CANDIDATES = ("CUDA", "OPTIX", "HIP", "ONEAPI", "METAL")

# Camera angle config — set from CLI args at runtime.
# Defaults: profile 3/4 view (azimuth 315°, elevation 20°).
# The old true-isometric was azimuth=315°, elevation≈35.264°.
_camera_azimuth_deg: float = 315.0
_camera_elevation_deg: float = 20.0

_COMPASS_POINTS = (
    (0, "east"), (45, "northeast"), (90, "north"), (135, "northwest"),
    (180, "west"), (225, "southwest"), (270, "south"), (315, "southeast"),
)


def azimuth_to_compass(az_deg: float) -> str:
    """Map camera-position azimuth to compass label (az=0 → camera at +X → east)."""
    az = az_deg % 360
    return min(_COMPASS_POINTS, key=lambda p: min(abs(az - p[0]), 360 - abs(az - p[0])))[1]


@dataclass(frozen=True)
class AssetEntry:
    family: str
    path: Path

    @property
    def stem(self) -> str:
        return self.path.stem


@dataclass(frozen=True)
class CameraFit:
    largest_asset: str
    largest_asset_path: str
    largest_asset_size_bytes: int
    reference_bounds_min: tuple[float, float, float]
    reference_bounds_max: tuple[float, float, float]
    reference_size: tuple[float, float, float]
    ortho_scale: float
    distance: float
    direction: tuple[float, float, float]
    margin: float


@dataclass(frozen=True)
class SpatialSummary:
    asset: AssetEntry
    min_corner: tuple[float, float, float]
    max_corner: tuple[float, float, float]
    size: tuple[float, float, float]
    projected_width: float
    projected_height: float
    ortho_scale: float
    diagonal: float


def log(*args) -> None:
    message = " ".join(str(arg) for arg in args)
    timestamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(timestamped, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(timestamped + "\n")


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preview", "exr", "both"), default="both")
    parser.add_argument("--families", nargs="+", choices=("trees", "logs"), default=("trees", "logs"))
    parser.add_argument(
        "--tree-size-filter",
        nargs="+",
        choices=("small", "medium", "large", "senescing", "snag", "fallen", "decayed"),
        default=(),
    )
    parser.add_argument("--tree-dir", default="")
    parser.add_argument("--log-dir", default="")
    parser.add_argument(
        "--output-root",
        default=r"D:\2026 Arboreal Futures\data\renders\ply-library-face-aov",
    )
    parser.add_argument("--preview-limit", type=int, default=20)
    parser.add_argument("--asset-limit", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--preview-res", nargs=2, type=int, metavar=("X", "Y"), default=(1024, 1024))
    parser.add_argument("--preview-exposure", type=float, default=0.75)
    parser.add_argument("--preview-single-color", type=float, default=0.28)
    parser.add_argument("--exr-res", nargs=2, type=int, metavar=("X", "Y"), default=(2048, 2048))
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--margin", type=float, default=1.08)
    parser.add_argument("--camera-azimuth", nargs="+", type=float, default=[315.0],
                        help="Camera azimuth(s) in degrees (CCW from +X). Pass multiple for multi-side renders.")
    parser.add_argument("--fit-azimuths", nargs="+", type=float, default=None,
                        help="Azimuths to consider when computing shared camera fit. "
                             "Defaults to --camera-azimuth. Use when running parallel single-azimuth "
                             "processes that need the same framing.")
    parser.add_argument("--camera-elevation", type=float, default=20.0,
                        help="Camera elevation in degrees above XY plane. Old isometric≈35.264")
    parser.add_argument("--gpu-backend", default="CUDA")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--direction-suffix", action="store_true",
                        help="Always append compass direction to filenames, even for single azimuth.")
    parser.add_argument("--include-scalar-flags", action="store_true")
    return parser.parse_args(argv)


def iter_existing_bundle_roots() -> list[Path]:
    seen: set[Path] = set()
    roots: list[Path] = []
    if iter_blender_input_roots is not None:
        try:
            for root in iter_blender_input_roots():
                path = Path(root)
                if path not in seen and path.exists():
                    seen.add(path)
                    roots.append(path)
        except Exception:
            pass

    for candidate in (
        Path(r"D:\2026 Arboreal Futures\data"),
        REPO_ROOT / "data" / "revised" / "final",
    ):
        if candidate not in seen and candidate.exists():
            seen.add(candidate)
            roots.append(candidate)
    return roots


def resolve_family_dir(family: str, explicit_dir: str) -> Path:
    if explicit_dir:
        path = Path(explicit_dir).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Explicit {family} directory does not exist: {path}")
        return path

    if family == "trees" and hook_tree_ply_library_dir is not None:
        try:
            hook_path = Path(hook_tree_ply_library_dir())
            if any(hook_path.glob("*.ply")):
                return hook_path
        except Exception:
            pass
    if family == "logs" and hook_log_ply_library_dir is not None:
        try:
            hook_path = Path(hook_log_ply_library_dir())
            if any(hook_path.glob("*.ply")):
                return hook_path
        except Exception:
            pass

    folder_names = ("treeMeshesPly",) if family == "trees" else ("logMeshesPly", "logMeshesPLY")
    for root in iter_existing_bundle_roots():
        for folder_name in folder_names:
            candidate = root / folder_name
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Could not resolve PLY directory for family={family!r}")


def collect_assets(args: argparse.Namespace) -> tuple[list[AssetEntry], dict[str, str]]:
    assets: list[AssetEntry] = []
    resolved_dirs: dict[str, str] = {}
    tree_size_filter = {value.strip() for value in getattr(args, "tree_size_filter", ()) if value.strip()}
    for family in args.families:
        directory = resolve_family_dir(family, args.tree_dir if family == "trees" else args.log_dir)
        resolved_dirs[family] = str(directory)
        family_assets = [AssetEntry(family=family, path=path) for path in sorted(directory.glob("*.ply"))]
        if family == "trees" and tree_size_filter:
            family_assets = [
                asset
                for asset in family_assets
                if any(f".{size_name}_control." in asset.stem for size_name in tree_size_filter)
            ]
        assets.extend(family_assets)
    if not assets:
        raise FileNotFoundError("No PLY assets were found for the requested families")
    if args.asset_limit > 0:
        assets = assets[: args.asset_limit]
    return assets, resolved_dirs


def chunked(items: list[AssetEntry], chunk_size: int) -> list[list[AssetEntry]]:
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def set_scene_view_transform(scene: bpy.types.Scene) -> None:
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.display_settings.display_device = "sRGB"


def is_tree_reference_cohort_asset(asset: AssetEntry) -> bool:
    if asset.family != "trees":
        return False
    stem = asset.stem
    size_match = ".large_control." in stem or ".fallen_control." in stem
    control_match = ".reserve-tree_id." in stem or ".improved-tree_id." in stem
    return size_match and control_match


def bounds_cache_path(output_root: Path) -> Path:
    return output_root / "_bounds_cache.json"


def load_bounds_cache(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"version": 1, "assets": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "assets": {}}
    if not isinstance(payload, dict):
        return {"version": 1, "assets": {}}
    if not isinstance(payload.get("assets"), dict):
        payload["assets"] = {}
    payload.setdefault("version", 1)
    return payload


def save_bounds_cache(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_ascii_ply_bounds(filepath: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    with filepath.open("r", encoding="latin1", errors="replace") as handle:
        vertex_count: int | None = None
        property_names: list[str] = []
        in_vertex = False

        while True:
            line = handle.readline()
            if not line:
                break
            stripped = line.strip()
            if stripped.startswith("element vertex "):
                vertex_count = int(stripped.split()[-1])
                in_vertex = True
                continue
            if stripped.startswith("element ") and not stripped.startswith("element vertex "):
                in_vertex = False
                continue
            if in_vertex and stripped.startswith("property "):
                parts = stripped.split()
                property_names.append(" ".join(parts[2:]))
                continue
            if stripped == "end_header":
                break

        if vertex_count is None:
            raise ValueError(f"Could not find vertex count in {filepath}")

        x_index = property_names.index("x")
        y_index = property_names.index("y")
        z_index = property_names.index("z")
        min_x = min_y = min_z = float("inf")
        max_x = max_y = max_z = float("-inf")

        for _ in range(vertex_count):
            row = handle.readline()
            if not row:
                break
            values = row.split()
            x = float(values[x_index])
            y = float(values[y_index])
            z = float(values[z_index])
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if z < min_z:
                min_z = z
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y
            if z > max_z:
                max_z = z

    return (min_x, min_y, min_z), (max_x, max_y, max_z)


def get_cached_or_parsed_bounds(asset: AssetEntry, cache_payload: dict[str, object]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    cache_assets = cache_payload.setdefault("assets", {})
    assert isinstance(cache_assets, dict)
    cache_key = str(asset.path.resolve())
    stat = asset.path.stat()
    cached = cache_assets.get(cache_key)
    if isinstance(cached, dict):
        if cached.get("size_bytes") == stat.st_size and cached.get("mtime_ns") == stat.st_mtime_ns:
            min_corner = tuple(cached["min_corner"])
            max_corner = tuple(cached["max_corner"])
            return min_corner, max_corner

    min_corner, max_corner = parse_ascii_ply_bounds(asset.path)
    cache_assets[cache_key] = {
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "min_corner": list(min_corner),
        "max_corner": list(max_corner),
    }
    return min_corner, max_corner


def ensure_link(node_tree: bpy.types.NodeTree, from_socket, to_socket) -> None:
    for link in list(to_socket.links):
        if link.from_socket == from_socket:
            return
        node_tree.links.remove(link)
    node_tree.links.new(from_socket, to_socket)


def ensure_node(
    node_tree: bpy.types.NodeTree,
    bl_idname: str,
    name: str,
    label: str,
    location: tuple[float, float],
    parent=None,
):
    node = node_tree.nodes.get(name)
    if node is None or node.bl_idname != bl_idname:
        if node is not None:
            node_tree.nodes.remove(node)
        node = node_tree.nodes.new(bl_idname)
    node.name = name
    node.label = label
    node.location = location
    node.parent = parent
    return node


def ensure_rgb_node(
    node_tree: bpy.types.NodeTree,
    name: str,
    label: str,
    location: tuple[float, float],
    color: tuple[float, float, float, float],
    parent=None,
):
    node = ensure_node(node_tree, "ShaderNodeRGB", name, label, location, parent)
    node.outputs["Color"].default_value = color
    return node


def build_face_aov_material(include_scalar_flags: bool) -> bpy.types.Material:
    material = bpy.data.materials.get(MATERIAL_NAME)
    if material is None:
        material = bpy.data.materials.new(MATERIAL_NAME)
    material.use_nodes = True
    material.use_fake_user = True
    material.diffuse_color = (0.92, 0.92, 0.92, 1.0)

    node_tree = material.node_tree
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)

    output = ensure_node(node_tree, "ShaderNodeOutputMaterial", "Material Output", "Material Output", (1400.0, 0.0))
    emission = ensure_node(node_tree, "ShaderNodeEmission", "Preview Emission", "Preview Emission", (1120.0, 0.0))
    emission.inputs["Strength"].default_value = 1.0
    ensure_link(node_tree, emission.outputs["Emission"], output.inputs["Surface"])

    resource_attr = ensure_node(node_tree, "ShaderNodeAttribute", "Attr resource", "resource", (-900.0, 140.0))
    resource_attr.attribute_name = "resource"

    int_resource_attr = ensure_node(
        node_tree,
        "ShaderNodeAttribute",
        "Attr int_resource",
        "int_resource",
        (-900.0, -10.0),
    )
    int_resource_attr.attribute_name = "int_resource"

    scalar_max = ensure_node(node_tree, "ShaderNodeMath", "Math resource scalar", "resource scalar", (-650.0, 60.0))
    scalar_max.operation = "MAXIMUM"
    ensure_link(node_tree, resource_attr.outputs["Fac"], scalar_max.inputs[0])
    ensure_link(node_tree, int_resource_attr.outputs["Fac"], scalar_max.inputs[1])

    resource_aov = ensure_node(node_tree, "ShaderNodeOutputAOV", "AOV resource", "resource", (-360.0, 60.0))
    resource_aov.aov_name = "resource"
    ensure_link(node_tree, scalar_max.outputs["Value"], resource_aov.inputs["Value"])

    tree_mask = ensure_node(node_tree, "ShaderNodeMath", "Math tree mask", "resource_tree_mask", (-360.0, -120.0))
    tree_mask.operation = "GREATER_THAN"
    tree_mask.inputs[1].default_value = 0.5
    ensure_link(node_tree, scalar_max.outputs["Value"], tree_mask.inputs[0])

    tree_mask_aov = ensure_node(
        node_tree,
        "ShaderNodeOutputAOV",
        "AOV resource_tree_mask",
        "resource_tree_mask",
        (-80.0, -120.0),
    )
    tree_mask_aov.aov_name = "resource_tree_mask"
    ensure_link(node_tree, tree_mask.outputs["Value"], tree_mask_aov.inputs["Value"])

    if include_scalar_flags:
        for index, flag_name in enumerate(("isSenescent", "isTerminal")):
            y = -320.0 - (index * 140.0)
            attr = ensure_node(node_tree, "ShaderNodeAttribute", f"Attr {flag_name}", flag_name, (-650.0, y))
            attr.attribute_name = flag_name
            aov = ensure_node(node_tree, "ShaderNodeOutputAOV", f"AOV {flag_name}", flag_name, (-360.0, y))
            aov.aov_name = flag_name
            ensure_link(node_tree, attr.outputs["Fac"], aov.inputs["Value"])

    base = ensure_rgb_node(node_tree, "Preview Base", "Preview Base", (40.0, 260.0), (0.0, 0.0, 0.0, 1.0))
    current_colour_socket = base.outputs["Color"]

    for index, spec in enumerate(RESOURCE_MASK_SPECS):
        y = 260.0 - (index * 170.0)
        attr = ensure_node(node_tree, "ShaderNodeAttribute", f"Attr {spec['aov_name']}", spec["label"], (40.0, y))
        attr.attribute_name = spec["attr_name"]

        aov = ensure_node(node_tree, "ShaderNodeOutputAOV", f"AOV {spec['aov_name']}", spec["aov_name"], (340.0, y))
        aov.aov_name = spec["aov_name"]
        ensure_link(node_tree, attr.outputs["Fac"], aov.inputs["Value"])

        color = ensure_rgb_node(
            node_tree,
            f"Color {spec['aov_name']}",
            spec["label"],
            (620.0, y),
            spec["color"],
        )
        mix = ensure_node(node_tree, "ShaderNodeMix", f"Mix {spec['aov_name']}", spec["label"], (900.0, y))
        mix.data_type = "RGBA"
        mix.blend_type = "MIX"
        ensure_link(node_tree, attr.outputs["Fac"], mix.inputs["Factor"])
        ensure_link(node_tree, current_colour_socket, mix.inputs["A"])
        ensure_link(node_tree, color.outputs["Color"], mix.inputs["B"])
        current_colour_socket = mix.outputs["Result"]

    resource_colour_aov = ensure_node(
        node_tree,
        "ShaderNodeOutputAOV",
        "AOV resource_colour",
        "resource_colour",
        (1120.0, 260.0),
    )
    resource_colour_aov.aov_name = "resource_colour"
    ensure_link(node_tree, current_colour_socket, resource_colour_aov.inputs["Color"])
    ensure_link(node_tree, current_colour_socket, emission.inputs["Color"])
    return material


def ensure_view_layer_aovs(view_layer: bpy.types.ViewLayer, include_scalar_flags: bool) -> None:
    allowed = {"resource", "resource_tree_mask", "resource_colour"} | {spec["aov_name"] for spec in RESOURCE_MASK_SPECS}
    if include_scalar_flags:
        allowed |= {"isSenescent", "isTerminal"}

    while view_layer.aovs:
        view_layer.aovs.remove(view_layer.aovs[0])

    for name, aov_type in AOV_SPECS:
        if name not in allowed:
            continue
        aov = view_layer.aovs.add()
        aov.name = name
        aov.type = aov_type


def prepare_clean_scene(include_scalar_flags: bool) -> tuple[bpy.types.Scene, bpy.types.ViewLayer, bpy.types.Collection]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    if scene is None:
        raise RuntimeError("Blender did not create an initial scene")
    scene.name = SCENE_NAME

    while len(scene.view_layers) > 1:
        scene.view_layers.remove(scene.view_layers[-1])
    view_layer = scene.view_layers[0]
    view_layer.name = VIEW_LAYER_NAME
    ensure_view_layer_aovs(view_layer, include_scalar_flags=include_scalar_flags)

    asset_collection = bpy.data.collections.new(ASSET_COLLECTION_NAME)
    scene.collection.children.link(asset_collection)
    set_scene_view_transform(scene)
    scene.render.film_transparent = True
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    return scene, view_layer, asset_collection


def compute_bounds(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_corner = Vector(
        (
            min(point.x for point in corners),
            min(point.y for point in corners),
            min(point.z for point in corners),
        )
    )
    max_corner = Vector(
        (
            max(point.x for point in corners),
            max(point.y for point in corners),
            max(point.z for point in corners),
        )
    )
    return min_corner, max_corner


def center_asset_object(obj: bpy.types.Object) -> None:
    bpy.context.view_layer.update()
    min_corner, max_corner = compute_bounds(obj)
    center_xy = Vector(((min_corner.x + max_corner.x) * 0.5, (min_corner.y + max_corner.y) * 0.5, 0.0))
    delta = Vector((-center_xy.x, -center_xy.y, -min_corner.z))
    obj.location = obj.location + delta
    bpy.context.view_layer.update()


def assign_material(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    if obj.type != "MESH" or obj.data is None:
        return
    if len(obj.data.materials) == 0:
        obj.data.materials.append(material)
    else:
        for index in range(len(obj.data.materials)):
            obj.data.materials[index] = material
    obj.color = (0.92, 0.92, 0.92, 1.0)


def import_ply_object(filepath: Path, collection: bpy.types.Collection) -> bpy.types.Object:
    existing_names = {obj.name for obj in bpy.data.objects}
    bpy.ops.wm.ply_import(filepath=str(filepath))
    imported = [obj for obj in bpy.data.objects if obj.name not in existing_names]
    if not imported:
        raise RuntimeError(f"PLY import produced no new objects: {filepath}")
    obj = imported[0]
    if collection.objects.get(obj.name) is None:
        collection.objects.link(obj)
    for user_collection in list(obj.users_collection):
        if user_collection != collection:
            user_collection.objects.unlink(obj)
    obj.name = filepath.stem
    if obj.data is not None:
        obj.data.name = filepath.stem
    bpy.context.view_layer.objects.active = obj
    obj.select_set(False)
    return obj


def remove_objects(objects: list[bpy.types.Object]) -> None:
    meshes = []
    for obj in objects:
        if obj.data is not None:
            meshes.append(obj.data)
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    try:
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    except Exception:
        pass


def create_camera(scene: bpy.types.Scene) -> bpy.types.Object:
    data = bpy.data.cameras.new(CAMERA_NAME)
    data.type = "ORTHO"
    data.clip_start = 0.01
    data.clip_end = 10000.0
    camera = bpy.data.objects.new(CAMERA_NAME, data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    return camera


def get_camera_direction() -> Vector:
    """Camera offset direction from azimuth/elevation (spherical coords, Z-up)."""
    az = math.radians(_camera_azimuth_deg)
    el = math.radians(_camera_elevation_deg)
    x = math.cos(el) * math.cos(az)
    y = math.cos(el) * math.sin(az)
    z = math.sin(el)
    return Vector((x, y, z)).normalized()


def centered_bbox_corners(
    min_corner: tuple[float, float, float],
    max_corner: tuple[float, float, float],
) -> list[Vector]:
    min_v = Vector(min_corner)
    max_v = Vector(max_corner)
    center = (min_v + max_v) * 0.5
    return [
        Vector((x, y, z)) - center
        for x in (min_v.x, max_v.x)
        for y in (min_v.y, max_v.y)
        for z in (min_v.z, max_v.z)
    ]


def projected_extents_from_bounds(
    min_corner: tuple[float, float, float],
    max_corner: tuple[float, float, float],
) -> tuple[float, float]:
    rotation_inv = (-get_camera_direction()).to_track_quat("-Z", "Y").to_matrix().inverted()
    corners = [rotation_inv @ corner for corner in centered_bbox_corners(min_corner, max_corner)]
    width = max(point.x for point in corners) - min(point.x for point in corners)
    height = max(point.y for point in corners) - min(point.y for point in corners)
    return float(width), float(height)


def build_spatial_summary(
    asset: AssetEntry,
    min_corner: tuple[float, float, float],
    max_corner: tuple[float, float, float],
    resolution_x: int,
    resolution_y: int,
    margin: float,
) -> SpatialSummary:
    size_v = Vector((max_corner[0] - min_corner[0], max_corner[1] - min_corner[1], max_corner[2] - min_corner[2]))
    projected_width, projected_height = projected_extents_from_bounds(min_corner, max_corner)
    aspect = float(resolution_x) / float(resolution_y)
    ortho_scale = max(projected_width, projected_height * aspect) * margin
    return SpatialSummary(
        asset=asset,
        min_corner=min_corner,
        max_corner=max_corner,
        size=(round(size_v.x, 6), round(size_v.y, 6), round(size_v.z, 6)),
        projected_width=round(projected_width, 6),
        projected_height=round(projected_height, 6),
        ortho_scale=round(float(ortho_scale), 6),
        diagonal=round(float(size_v.length), 6),
    )


def choose_spatial_reference_asset(
    assets: list[AssetEntry],
    resolution_x: int,
    resolution_y: int,
    margin: float,
    cache_payload: dict[str, object],
) -> SpatialSummary:
    summaries: list[SpatialSummary] = []
    for index, asset in enumerate(assets, start=1):
        min_corner, max_corner = get_cached_or_parsed_bounds(asset, cache_payload)
        summaries.append(build_spatial_summary(asset, min_corner, max_corner, resolution_x, resolution_y, margin))
        if index % 25 == 0 or index == len(assets):
            log(f"Spatial scan progress: {index}/{len(assets)}")

    return max(
        summaries,
        key=lambda summary: (
            summary.ortho_scale,
            summary.projected_width * summary.projected_height,
            summary.diagonal,
            summary.asset.path.stat().st_size,
        ),
    )


def place_camera(camera: bpy.types.Object, target: Vector, distance: float, ortho_scale: float) -> None:
    direction = get_camera_direction()
    camera.location = target + (direction * distance)
    camera.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
    camera.data.ortho_scale = ortho_scale
    camera.data.clip_end = max(camera.data.clip_end, distance * 4.0)
    bpy.context.view_layer.update()


def compute_camera_fit(
    camera: bpy.types.Object,
    obj: bpy.types.Object,
    resolution_x: int,
    resolution_y: int,
    margin: float,
) -> CameraFit:
    min_corner, max_corner = compute_bounds(obj)
    target = Vector(
        (
            (min_corner.x + max_corner.x) * 0.5,
            (min_corner.y + max_corner.y) * 0.5,
            (min_corner.z + max_corner.z) * 0.5,
        )
    )
    size = max_corner - min_corner
    diagonal = size.length
    distance = max(diagonal * 2.5, 10.0)
    place_camera(camera, target, distance, ortho_scale=1.0)

    camera_matrix = camera.matrix_world.inverted()
    corners = [camera_matrix @ (obj.matrix_world @ Vector(corner)) for corner in obj.bound_box]
    width = max(point.x for point in corners) - min(point.x for point in corners)
    height = max(point.y for point in corners) - min(point.y for point in corners)
    aspect = float(resolution_x) / float(resolution_y)
    ortho_scale = max(width, height * aspect) * margin

    return CameraFit(
        largest_asset=obj.name,
        largest_asset_path=str(obj.get("bV2_source_ply", "")),
        largest_asset_size_bytes=int(obj.get("bV2_source_ply_size", 0)),
        reference_bounds_min=(round(min_corner.x, 6), round(min_corner.y, 6), round(min_corner.z, 6)),
        reference_bounds_max=(round(max_corner.x, 6), round(max_corner.y, 6), round(max_corner.z, 6)),
        reference_size=(round(size.x, 6), round(size.y, 6), round(size.z, 6)),
        ortho_scale=round(float(ortho_scale), 6),
        distance=round(float(distance), 6),
        direction=(round(get_camera_direction().x, 6), round(get_camera_direction().y, 6), round(get_camera_direction().z, 6)),
        margin=round(float(margin), 6),
    )


def camera_fit_from_spatial_summary(summary: SpatialSummary, margin: float) -> CameraFit:
    size = Vector(summary.size)
    distance = max(float(size.length) * 2.5, 10.0)
    return CameraFit(
        largest_asset=summary.asset.stem,
        largest_asset_path=str(summary.asset.path),
        largest_asset_size_bytes=int(summary.asset.path.stat().st_size),
        reference_bounds_min=tuple(round(value, 6) for value in summary.min_corner),
        reference_bounds_max=tuple(round(value, 6) for value in summary.max_corner),
        reference_size=summary.size,
        ortho_scale=summary.ortho_scale,
        distance=round(distance, 6),
        direction=(round(get_camera_direction().x, 6), round(get_camera_direction().y, 6), round(get_camera_direction().z, 6)),
        margin=round(float(margin), 6),
    )


def configure_cycles_device(scene: bpy.types.Scene, requested_backend: str) -> dict[str, object]:
    summary = {"device": "CPU", "backend": "", "devices": ()}
    if not hasattr(scene, "cycles"):
        return summary

    backend_token = requested_backend.strip().upper()
    candidates = GPU_BACKEND_CANDIDATES if backend_token in {"", "AUTO"} else (backend_token,)
    addon = bpy.context.preferences.addons.get("cycles")
    if addon is None:
        try:
            scene.cycles.device = "CPU"
        except Exception:
            pass
        return summary

    prefs = addon.preferences
    for backend in candidates:
        try:
            prefs.compute_device_type = backend
            try:
                prefs.get_devices()
            except Exception:
                pass
            devices = list(getattr(prefs, "devices", ()))
            gpu_devices = [device for device in devices if getattr(device, "type", "") != "CPU"]
            if not gpu_devices:
                continue
            for device in devices:
                device.use = getattr(device, "type", "") != "CPU"
            scene.cycles.device = "GPU"
            summary = {
                "device": "GPU",
                "backend": backend,
                "devices": tuple(f"{device.name}|{device.type}" for device in gpu_devices),
            }
            return summary
        except Exception:
            continue

    try:
        scene.cycles.device = "CPU"
    except Exception:
        pass
    return summary


def configure_preview_render(
    scene: bpy.types.Scene,
    resolution_x: int,
    resolution_y: int,
    exposure: float,
    single_color_value: float,
) -> None:
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "SINGLE"
    scene.display.shading.single_color = (single_color_value, single_color_value, single_color_value)
    scene.display.shading.show_object_outline = False
    scene.display.shading.show_backface_culling = False
    if hasattr(scene.display.shading, "show_shadows"):
        scene.display.shading.show_shadows = True
    if scene.world is None:
        scene.world = bpy.data.worlds.new("bV2_preview_world")
    scene.world.color = (1.0, 1.0, 1.0)
    set_scene_view_transform(scene)
    scene.view_settings.exposure = exposure


def configure_exr_render(scene: bpy.types.Scene, resolution_x: int, resolution_y: int, samples: int) -> None:
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "16"
    scene.render.image_settings.exr_codec = "ZIP"
    scene.render.film_transparent = True
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    set_scene_view_transform(scene)
    scene.view_settings.exposure = 0.0

    if hasattr(scene, "cycles"):
        scene.cycles.samples = samples
        scene.cycles.preview_samples = samples
        scene.cycles.use_denoising = False
        scene.cycles.max_bounces = 2
        scene.cycles.diffuse_bounces = 0
        scene.cycles.glossy_bounces = 0
        scene.cycles.transmission_bounces = 0
        scene.cycles.transparent_max_bounces = 2
        scene.cycles.volume_bounces = 0


def set_minimal_passes(view_layer: bpy.types.ViewLayer) -> None:
    """Preview mode: disable everything except combined."""
    for attr in (
        "use_pass_z",
        "use_pass_mist",
        "use_pass_normal",
        "use_pass_object_index",
        "use_pass_material_index",
        "use_pass_ambient_occlusion",
    ):
        if hasattr(view_layer, attr):
            setattr(view_layer, attr, False)
    if hasattr(view_layer, "use_pass_combined"):
        view_layer.use_pass_combined = True


def set_exr_passes(view_layer: bpy.types.ViewLayer) -> None:
    """EXR mode: enable AO, normals, depth, mist alongside combined."""
    if hasattr(view_layer, "use_pass_combined"):
        view_layer.use_pass_combined = True
    if hasattr(view_layer, "use_pass_z"):
        view_layer.use_pass_z = True
    if hasattr(view_layer, "use_pass_normal"):
        view_layer.use_pass_normal = True
    if hasattr(view_layer, "use_pass_ambient_occlusion"):
        view_layer.use_pass_ambient_occlusion = True
    if hasattr(view_layer, "use_pass_mist"):
        view_layer.use_pass_mist = True
    # Keep these off — not needed for library renders
    for attr in ("use_pass_object_index", "use_pass_material_index"):
        if hasattr(view_layer, attr):
            setattr(view_layer, attr, False)


def configure_mist(scene: bpy.types.Scene, camera_distance: float, ref_diagonal: float) -> None:
    """Set mist to span the tree volume so front-to-back depth reads clearly.

    start = camera_distance minus half the diagonal (front face of the bounding volume)
    depth  = the full diagonal (covers front-to-back extent of the tree)
    falloff = LINEAR preserves proportional distance, easily remapped in compositing
    """
    if scene.world is None:
        scene.world = bpy.data.worlds.new("bV2_library_world")
    mist = scene.world.mist_settings
    mist.start = max(0.0, camera_distance - ref_diagonal * 0.5)
    mist.depth = ref_diagonal
    mist.falloff = "LINEAR"


def preview_output_path(output_root: Path, asset: AssetEntry) -> Path:
    return output_root / "previews" / f"{asset.stem}.png"


def exr_output_path(output_root: Path, asset: AssetEntry) -> Path:
    return output_root / "exr" / f"{asset.stem}.exr"


def show_only(target: bpy.types.Object, objects: list[bpy.types.Object]) -> None:
    for obj in objects:
        visible = obj == target
        obj.hide_render = not visible
        obj.hide_viewport = not visible
    bpy.context.view_layer.update()


def fit_asset_to_shared_camera(camera: bpy.types.Object, obj: bpy.types.Object, fit: CameraFit) -> None:
    min_corner, max_corner = compute_bounds(obj)
    target = Vector(
        (
            (min_corner.x + max_corner.x) * 0.5,
            (min_corner.y + max_corner.y) * 0.5,
            (min_corner.z + max_corner.z) * 0.5,
        )
    )
    place_camera(camera, target, fit.distance, fit.ortho_scale)


def render_still(scene: bpy.types.Scene, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(filepath)
    bpy.ops.render.render(write_still=True, scene=scene.name, layer=VIEW_LAYER_NAME, use_viewport=False)


def write_manifest(output_root: Path, manifest: dict[str, object]) -> Path:
    path = output_root / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def import_and_prepare_asset(
    asset: AssetEntry,
    collection: bpy.types.Collection,
    material: bpy.types.Material,
) -> bpy.types.Object:
    obj = import_ply_object(asset.path, collection)
    obj["bV2_source_ply"] = str(asset.path)
    obj["bV2_source_ply_size"] = int(asset.path.stat().st_size)
    center_asset_object(obj)
    assign_material(obj, material)
    bpy.context.view_layer.update()
    return obj


def render_library(args: argparse.Namespace) -> dict[str, object]:
    global _camera_azimuth_deg, _camera_elevation_deg
    azimuths: list[float] = args.camera_azimuth
    fit_azimuths: list[float] = args.fit_azimuths or azimuths
    _camera_elevation_deg = args.camera_elevation

    assets, resolved_dirs = collect_assets(args)
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    cache_path = bounds_cache_path(output_root)
    cache_payload = load_bounds_cache(cache_path)
    reference_assets = [asset for asset in assets if is_tree_reference_cohort_asset(asset)]
    reference_scope = "reserve_or_improved_large_or_fallen_trees" if reference_assets else "all_selected_assets_fallback"

    # Compute worst-case spatial reference across all fit azimuths
    spatial_reference = None
    for az in fit_azimuths:
        _camera_azimuth_deg = az
        candidate = choose_spatial_reference_asset(
            reference_assets or assets,
            resolution_x=args.exr_res[0],
            resolution_y=args.exr_res[1],
            margin=args.margin,
            cache_payload=cache_payload,
        )
        if spatial_reference is None or candidate.ortho_scale > spatial_reference.ortho_scale:
            spatial_reference = candidate
    _camera_azimuth_deg = azimuths[0]

    save_bounds_cache(cache_path, cache_payload)
    largest_asset = spatial_reference.asset
    log("Resolved asset count:", len(assets))
    log("Resolved library directories:", resolved_dirs)
    log("Render azimuths:", [f"{az}°/{azimuth_to_compass(az)}" for az in azimuths])
    log("Fit azimuths:", [f"{az}°/{azimuth_to_compass(az)}" for az in fit_azimuths])
    log("Reference cohort count:", len(reference_assets), f"scope={reference_scope}")
    log(
        "Spatial reference asset:",
        largest_asset.path.name,
        f"ortho_scale={spatial_reference.ortho_scale}",
        f"projected_width={spatial_reference.projected_width}",
        f"projected_height={spatial_reference.projected_height}",
    )

    scene, view_layer, asset_collection = prepare_clean_scene(include_scalar_flags=args.include_scalar_flags)
    if args.mode in {"exr", "both"}:
        set_exr_passes(view_layer)
    else:
        set_minimal_passes(view_layer)
    material = build_face_aov_material(include_scalar_flags=args.include_scalar_flags)
    camera = create_camera(scene)
    fit = camera_fit_from_spatial_summary(spatial_reference, args.margin)
    ref_diagonal = spatial_reference.diagonal
    if args.mode in {"exr", "both"}:
        configure_mist(scene, fit.distance, ref_diagonal)
        log("Mist config: start=", round(max(0.0, fit.distance - ref_diagonal * 0.5), 2),
            "depth=", round(ref_diagonal, 2), "falloff=LINEAR")
    log("Shared camera fit:", fit)

    preview_remaining = math.inf if args.preview_limit == 0 else max(args.preview_limit, 0)
    preview_done = 0
    exr_done = 0
    skipped = 0
    exr_device_summary: dict[str, object] | None = None

    for batch_index, asset_batch in enumerate(chunked(assets, max(args.chunk_size, 1)), start=1):
        log(f"Import batch {batch_index}: size={len(asset_batch)}")
        imported: list[tuple[AssetEntry, bpy.types.Object]] = []
        for asset in asset_batch:
            imported.append((asset, import_and_prepare_asset(asset, asset_collection, material)))

        batch_objects = [obj for _, obj in imported]
        multi = len(azimuths) > 1 or args.direction_suffix
        for asset, obj in imported:
            show_only(obj, batch_objects)

            for az in azimuths:
                _camera_azimuth_deg = az
                fit_asset_to_shared_camera(camera, obj, fit)

                suffix = f"_{azimuth_to_compass(az)}" if multi else ""
                preview_path = output_root / "previews" / f"{asset.stem}{suffix}.png"
                exr_path = output_root / "exr" / f"{asset.stem}{suffix}.exr"
                need_preview = args.mode in {"preview", "both"} and preview_remaining > 0
                need_exr = args.mode in {"exr", "both"}

                if not args.force and need_preview and preview_path.exists():
                    need_preview = False
                if not args.force and need_exr and exr_path.exists():
                    need_exr = False
                if not need_preview and not need_exr:
                    skipped += 1
                    log("Skipping existing outputs for", f"{asset.stem}{suffix}")
                    continue

                if need_preview:
                    configure_preview_render(
                        scene,
                        args.preview_res[0],
                        args.preview_res[1],
                        args.preview_exposure,
                        args.preview_single_color,
                    )
                    render_still(scene, preview_path)
                    preview_done += 1
                    preview_remaining -= 1
                    log("Preview rendered:", preview_path)

                if need_exr:
                    configure_exr_render(scene, args.exr_res[0], args.exr_res[1], args.samples)
                    if exr_device_summary is None:
                        exr_device_summary = configure_cycles_device(scene, args.gpu_backend)
                        log("Cycles device:", exr_device_summary)
                    render_still(scene, exr_path)
                    exr_done += 1
                    log("EXR rendered:", exr_path)

        remove_objects(batch_objects)

    manifest = {
        "scene_name": SCENE_NAME,
        "view_layer": VIEW_LAYER_NAME,
        "families": list(args.families),
        "tree_size_filter": list(getattr(args, "tree_size_filter", ()) or ()),
        "resolved_dirs": resolved_dirs,
        "asset_count": len(assets),
        "reference_cohort_count": len(reference_assets),
        "reference_cohort_scope": reference_scope,
        "reference_ply": {
            "name": largest_asset.path.name,
            "path": str(largest_asset.path),
            "size_bytes": int(largest_asset.path.stat().st_size),
        },
        "reference_selection_method": "max_required_isometric_ortho_scale_from_spatial_bounds",
        "camera_fit": {
            "largest_asset": fit.largest_asset,
            "largest_asset_path": fit.largest_asset_path,
            "largest_asset_size_bytes": fit.largest_asset_size_bytes,
            "reference_bounds_min": fit.reference_bounds_min,
            "reference_bounds_max": fit.reference_bounds_max,
            "reference_size": fit.reference_size,
            "projected_width": spatial_reference.projected_width,
            "projected_height": spatial_reference.projected_height,
            "ortho_scale": fit.ortho_scale,
            "distance": fit.distance,
            "direction": fit.direction,
            "azimuths_deg": azimuths,
            "azimuth_labels": [azimuth_to_compass(az) for az in azimuths],
            "elevation_deg": args.camera_elevation,
            "margin": fit.margin,
        },
        "preview_resolution": list(args.preview_res),
        "preview_exposure": args.preview_exposure,
        "preview_single_color": args.preview_single_color,
        "exr_resolution": list(args.exr_res),
        "samples": args.samples,
        "chunk_size": args.chunk_size,
        "include_scalar_flags": bool(args.include_scalar_flags),
        "gpu_backend_request": args.gpu_backend,
        "gpu_device_summary": exr_device_summary or {"device": "not-used"},
        "outputs": {
            "preview_root": str((output_root / "previews").resolve()),
            "exr_root": str((output_root / "exr").resolve()),
            "preview_done": preview_done,
            "exr_done": exr_done,
            "skipped": skipped,
        },
        "bounds_cache": str(cache_path.resolve()),
    }
    manifest_path = write_manifest(output_root, manifest)
    log("Manifest written:", manifest_path)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = render_library(args)
    log(
        "DONE",
        f"preview_done={manifest['outputs']['preview_done']}",
        f"exr_done={manifest['outputs']['exr_done']}",
        f"skipped={manifest['outputs']['skipped']}",
    )


if __name__ == "__main__":
    main()
