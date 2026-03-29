from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import bpy  # type: ignore
except ImportError:
    bpy = None


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "_code-refactored"))

from refactor_code.paths import hook_state_vtk_latest_path

VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
CACHE_DIR = REPO_ROOT / "data" / "blender" / "2026" / "vtk_sim_layer_cache"
REPORT_PATH = REPO_ROOT / "data" / "blender" / "2026" / "vtk_year180_point_data_layers.md"
CHUNK_SIZE = 1_000_000
ONLY_CLIPPED_POINTS = False
SAVE_MAINFILE = True
ENABLE_LOCAL_NEIGHBOR_FALLBACK = True
SCENARIO = "positive"
YEAR = 180
ATTR_TURNS = "sim_Turns"
ATTR_NODES = "sim_Nodes"
ATTR_BIO = "scenario_bioEnvelope"
ATTR_BIO_SIMPLE = "scenario_bioEnvelopeSimple"
ATTR_MATCHED = "sim_Matched"

BIO_ENVELOPE_MAP = {
    "exoskeleton": 1,
    "brownRoof": 2,
    "otherGround": 3,
    "node-rewilded": 4,
    "rewilded": 4,
    "footprint-depaved": 5,
    "livingFacade": 6,
    "greenRoof": 7,
}

BIO_ENVELOPE_SIMPLE_MAP = {
    "brownRoof": 2,
    "livingFacade": 3,
    "greenRoof": 4,
}

LOCAL_NEIGHBOR_OFFSETS = np.array(
    [[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
    dtype=np.int64,
)

SITE_CONFIGS = {
    "trimmed-parade": {
        "vtk_path": hook_state_vtk_latest_path("trimmed-parade", SCENARIO, YEAR, 1),
        "object_names": ["trimmed-parade_base", "trimmed-parade_highResRoad"],
        "clip_box_name": "ClipBox",
    },
    "city": {
        "vtk_path": hook_state_vtk_latest_path("city", SCENARIO, YEAR, 1),
        "object_names": ["city_buildings.001", "city_highResRoad.001"],
        "clip_box_name": "City_ClipBox",
    },
}


def log(message: str) -> None:
    print(f"[vtk_sim_layers] {message}")


def helper_cache_paths(site: str) -> tuple[Path, Path]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{site}_{SCENARIO}_{YEAR}_sim_layers"
    return CACHE_DIR / f"{stem}.npz", CACHE_DIR / f"{stem}.json"


def _helper_main(argv: list[str]) -> int:
    import pyvista as pv

    if len(argv) != 4:
        raise SystemExit("Usage: b2026_transfer_vtk_sim_layers.py --helper <vtk_path> <cache_npz> <meta_json>")

    vtk_path = Path(argv[1])
    cache_path = Path(argv[2])
    meta_path = Path(argv[3])

    poly = pv.read(vtk_path)
    points = np.asarray(poly.points, dtype=np.float64)
    origin = points[0].astype(np.float64)
    grid_indices = np.rint(points - origin).astype(np.int32)
    index_min = grid_indices.min(axis=0)
    index_max = grid_indices.max(axis=0)
    dims = (index_max - index_min + 1).astype(np.int64)
    strides = np.array([dims[1] * dims[2], dims[2], 1], dtype=np.int64)
    shifted = (grid_indices - index_min).astype(np.int64)
    keys = shifted @ strides
    order = np.argsort(keys, kind="mergesort")

    sim_turns = np.asarray(poly.point_data[ATTR_TURNS], dtype=np.int32)
    sim_nodes = np.asarray(poly.point_data[ATTR_NODES], dtype=np.int32)
    bio_raw = np.asarray(poly.point_data["scenario_bioEnvelope"])
    bio_envelope = np.array([BIO_ENVELOPE_MAP.get(str(value), 0) for value in bio_raw], dtype=np.int32)
    bio_envelope_simple = np.array([BIO_ENVELOPE_SIMPLE_MAP.get(str(value), 1) for value in bio_raw], dtype=np.int32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        origin=origin,
        index_min=index_min.astype(np.int32),
        index_max=index_max.astype(np.int32),
        strides=strides,
        sorted_keys=keys[order].astype(np.int64),
        sim_turns_sorted=sim_turns[order],
        sim_nodes_sorted=sim_nodes[order],
        bio_envelope_sorted=bio_envelope[order],
        bio_envelope_simple_sorted=bio_envelope_simple[order],
    )

    metadata = {
        "vtk_path": str(vtk_path),
        "n_points": int(poly.n_points),
        "point_data_keys": list(poly.point_data.keys()),
        "point_data_dtypes": {key: str(np.asarray(poly.point_data[key]).dtype) for key in poly.point_data.keys()},
        "origin": origin.tolist(),
        "index_min": index_min.astype(int).tolist(),
        "index_max": index_max.astype(int).tolist(),
        "bio_envelope_map": BIO_ENVELOPE_MAP,
        "bio_envelope_simple_map": BIO_ENVELOPE_SIMPLE_MAP,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(cache_path)
    print(meta_path)
    return 0


def ensure_cache(site: str, vtk_path: Path) -> tuple[Path, Path]:
    if not VENV_PYTHON.exists():
        raise FileNotFoundError(f"Expected venv python was not found: {VENV_PYTHON}")

    cache_path, meta_path = helper_cache_paths(site)
    cmd = [
        str(VENV_PYTHON),
        str(Path(__file__).resolve()),
        "--helper",
        str(vtk_path),
        str(cache_path),
        str(meta_path),
    ]
    subprocess.run(cmd, check=True)
    return cache_path, meta_path


def write_report(site_reports: dict[str, dict]) -> None:
    lines = [
        "# VTK Point Data Layers",
        "",
        f"Scenario: `{SCENARIO}`",
        f"Year: `{YEAR}`",
        "",
        "## Blender point-cloud base and road targets",
        "",
        "These are the source point-cloud meshes that receive the transferred VTK data.",
        "The render-cube duplicates share the same mesh data, so they inherit these point attributes too.",
        "",
    ]
    for site, config in SITE_CONFIGS.items():
        lines.append(f"### {site}")
        for object_name in config["object_names"]:
            lines.append(f"- `{object_name}`")
        lines.append("")

    lines.extend(
        [
            "## Blender-transferred point attributes",
            "",
            "| Attribute | Compositor AOV | Source VTK layer | Type | Notes |",
            "| --- | --- | --- | --- | --- |",
            f"| `{ATTR_TURNS}` | `world_sim_turns` | `{ATTR_TURNS}` | integer | Direct transfer from VTK. Missing / unmatched points are `-1`. |",
            f"| `{ATTR_NODES}` | `world_sim_nodes` | `{ATTR_NODES}` | integer | Direct transfer from VTK. Missing / unmatched points are `-1`. |",
            f"| `{ATTR_BIO}` | `world_design_bioenvelope` | `scenario_bioEnvelope` | integer enum | Full bioenvelope mapping below. Unmatched points are `0`. |",
            f"| `{ATTR_BIO_SIMPLE}` | `world_design_bioenvelope_simple` | `scenario_bioEnvelope` | integer enum | Simplified bioenvelope mapping below. Unmatched points are `0`. |",
            f"| `{ATTR_MATCHED}` | `world_sim_matched` | derived | integer flag | `1` when a VTK voxel match was found, `0` when unmatched. |",
            "",
            "### Full `scenario_bioEnvelope` mapping",
            "",
            "| Value | Label |",
            "| --- | --- |",
            "| `0` | unmatched |",
        ]
    )
    for key, value in BIO_ENVELOPE_MAP.items():
        lines.append(f"| `{value}` | `{key}` |")
    lines.extend(
        [
            "",
            "### Simple `scenario_bioEnvelope` mapping",
            "",
            "| Value | Label |",
            "| --- | --- |",
            "| `0` | unmatched |",
            "| `1` | default / other |",
        ]
    )
    for key, value in BIO_ENVELOPE_SIMPLE_MAP.items():
        lines.append(f"| `{value}` | `{key}` |")
    lines.append("")

    for site, metadata in site_reports.items():
        lines.append(f"## {site}")
        lines.append(f"VTK: `{metadata['vtk_path']}`")
        lines.append(f"Points: `{metadata['n_points']}`")
        lines.append("")
        lines.append("| Layer | Dtype |")
        lines.append("| --- | --- |")
        for key in metadata["point_data_keys"]:
            lines.append(f"| `{key}` | `{metadata['point_data_dtypes'][key]}` |")
        lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    log(f"Wrote point-data report to {REPORT_PATH}")


def np_world_points_for_object(obj: bpy.types.Object) -> np.ndarray:
    vert_count = len(obj.data.vertices)
    coords = np.empty(vert_count * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", coords)
    coords = coords.reshape(-1, 3).astype(np.float64, copy=False)
    matrix = np.array(obj.matrix_world, dtype=np.float64)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    return coords @ rotation.T + translation


def clip_mask_for_points(
    world_points: np.ndarray,
    clip_box: bpy.types.Object | None,
) -> np.ndarray:
    if clip_box is None or not ONLY_CLIPPED_POINTS:
        return np.ones(len(world_points), dtype=bool)

    matrix_inv = np.array(clip_box.matrix_world.inverted(), dtype=np.float64)
    hom = np.ones((len(world_points), 4), dtype=np.float64)
    hom[:, :3] = world_points
    local = hom @ matrix_inv.T
    bounds = np.asarray(clip_box.bound_box, dtype=np.float64)
    mins = bounds.min(axis=0)
    maxs = bounds.max(axis=0)
    return np.all((local[:, :3] >= mins) & (local[:, :3] <= maxs), axis=1)


def ensure_int_attribute(mesh: bpy.types.Mesh, name: str):
    existing = mesh.attributes.get(name)
    if existing is not None:
        mesh.attributes.remove(existing)
    return mesh.attributes.new(name=name, type="INT", domain="POINT")


def assign_int_attribute(mesh: bpy.types.Mesh, name: str, values: np.ndarray) -> None:
    attr = ensure_int_attribute(mesh, name)
    attr.data.foreach_set("value", values.astype(np.int32, copy=False))


def local_neighbor_lookup(
    world_points: np.ndarray,
    origin: np.ndarray,
    index_min: np.ndarray,
    index_max: np.ndarray,
    strides: np.ndarray,
    sorted_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(world_points) == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int64)

    base_indices = np.floor(world_points - origin).astype(np.int64)
    candidate_indices = base_indices[:, None, :] + LOCAL_NEIGHBOR_OFFSETS[None, :, :]
    in_bounds = np.all((candidate_indices >= index_min) & (candidate_indices <= index_max), axis=2)

    shifted = candidate_indices - index_min
    candidate_keys = shifted @ strides
    flat_keys = candidate_keys.reshape(-1)
    flat_in_bounds = in_bounds.reshape(-1)
    safe_keys = flat_keys.copy()
    safe_keys[~flat_in_bounds] = 0

    flat_insertion = np.searchsorted(sorted_keys, safe_keys, side="left")
    flat_found = flat_in_bounds & (flat_insertion < len(sorted_keys)) & (sorted_keys[flat_insertion] == safe_keys)

    n_points = len(world_points)
    candidate_centers = origin + candidate_indices.astype(np.float64)
    distances_sq = np.sum((candidate_centers - world_points[:, None, :]) ** 2, axis=2)
    found_matrix = flat_found.reshape(n_points, -1)
    insertion_matrix = flat_insertion.reshape(n_points, -1)
    distances_sq[~found_matrix] = np.inf

    best_neighbor = np.argmin(distances_sq, axis=1)
    has_match = np.isfinite(distances_sq[np.arange(n_points), best_neighbor])
    best_lookup = insertion_matrix[np.arange(n_points), best_neighbor]
    return has_match, best_lookup.astype(np.int64, copy=False)


def transfer_cache_to_object(
    obj: bpy.types.Object,
    cache: dict[str, np.ndarray],
    clip_box: bpy.types.Object | None,
) -> dict[str, int]:
    world_points = np_world_points_for_object(obj)
    total = len(world_points)
    mask = clip_mask_for_points(world_points, clip_box)

    origin = cache["origin"]
    index_min = cache["index_min"].astype(np.int64)
    index_max = cache["index_max"].astype(np.int64)
    strides = cache["strides"].astype(np.int64)
    sorted_keys = cache["sorted_keys"].astype(np.int64)
    sim_turns_sorted = cache["sim_turns_sorted"].astype(np.int32)
    sim_nodes_sorted = cache["sim_nodes_sorted"].astype(np.int32)
    bio_envelope_sorted = cache["bio_envelope_sorted"].astype(np.int32)
    bio_envelope_simple_sorted = cache["bio_envelope_simple_sorted"].astype(np.int32)

    out_turns = np.full(total, -1, dtype=np.int32)
    out_nodes = np.full(total, -1, dtype=np.int32)
    out_bio = np.zeros(total, dtype=np.int32)
    out_bio_simple = np.zeros(total, dtype=np.int32)
    out_match = np.zeros(total, dtype=np.int32)

    for start in range(0, total, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total)
        chunk_points = world_points[start:end]
        chunk_mask = mask[start:end]
        if not np.any(chunk_mask):
            continue

        masked_points = chunk_points[chunk_mask]
        target_indices = np.rint(masked_points - origin).astype(np.int64)
        in_bounds = np.all((target_indices >= index_min) & (target_indices <= index_max), axis=1)
        if not np.any(in_bounds):
            continue

        masked_indices = np.flatnonzero(chunk_mask)
        valid_positions = masked_indices[in_bounds]
        valid_indices = target_indices[in_bounds] - index_min
        keys = valid_indices @ strides

        insertion = np.searchsorted(sorted_keys, keys, side="left")
        found = (insertion < len(sorted_keys)) & (sorted_keys[insertion] == keys)
        if not np.any(found):
            if not ENABLE_LOCAL_NEIGHBOR_FALLBACK:
                continue
        else:
            matched_positions = valid_positions[found] + start
            matched_lookup = insertion[found]
            out_turns[matched_positions] = sim_turns_sorted[matched_lookup]
            out_nodes[matched_positions] = sim_nodes_sorted[matched_lookup]
            out_bio[matched_positions] = bio_envelope_sorted[matched_lookup]
            out_bio_simple[matched_positions] = bio_envelope_simple_sorted[matched_lookup]
            out_match[matched_positions] = 1

        if ENABLE_LOCAL_NEIGHBOR_FALLBACK:
            fallback_points = masked_points[in_bounds][~found]
            fallback_positions = valid_positions[~found] + start
            fallback_has_match, fallback_lookup = local_neighbor_lookup(
                fallback_points,
                origin=origin,
                index_min=index_min,
                index_max=index_max,
                strides=strides,
                sorted_keys=sorted_keys,
            )
            if np.any(fallback_has_match):
                matched_positions = fallback_positions[fallback_has_match]
                matched_lookup = fallback_lookup[fallback_has_match]
                out_turns[matched_positions] = sim_turns_sorted[matched_lookup]
                out_nodes[matched_positions] = sim_nodes_sorted[matched_lookup]
                out_bio[matched_positions] = bio_envelope_sorted[matched_lookup]
                out_bio_simple[matched_positions] = bio_envelope_simple_sorted[matched_lookup]
                out_match[matched_positions] = 1

    assign_int_attribute(obj.data, ATTR_TURNS, out_turns)
    assign_int_attribute(obj.data, ATTR_NODES, out_nodes)
    assign_int_attribute(obj.data, ATTR_BIO, out_bio)
    assign_int_attribute(obj.data, ATTR_BIO_SIMPLE, out_bio_simple)
    assign_int_attribute(obj.data, ATTR_MATCHED, out_match)

    return {
        "points": total,
        "inside_scope": int(mask.sum()),
        "matched": int(out_match.sum()),
    }


def blender_main() -> int:
    if bpy is None:
        raise RuntimeError("This mode must be run from Blender.")

    site_reports: dict[str, dict] = {}
    processed_meshes: set[str] = set()
    results: list[tuple[str, str, dict[str, int]]] = []

    for site, config in SITE_CONFIGS.items():
        vtk_path = Path(config["vtk_path"])
        if not vtk_path.exists():
            raise FileNotFoundError(f"VTK not found for {site}: {vtk_path}")

        cache_path, meta_path = ensure_cache(site, vtk_path)
        cache = dict(np.load(cache_path))
        metadata = json.loads(meta_path.read_text())
        site_reports[site] = metadata

        clip_box = bpy.data.objects.get(config["clip_box_name"])
        for object_name in config["object_names"]:
            obj = bpy.data.objects.get(object_name)
            if obj is None:
                raise ValueError(f"Object '{object_name}' not found.")
            mesh_key = obj.data.name
            if mesh_key in processed_meshes:
                continue
            processed_meshes.add(mesh_key)
            summary = transfer_cache_to_object(obj, cache, clip_box)
            results.append((site, object_name, summary))
            log(
                f"{site}:{object_name} -> matched {summary['matched']:,} / {summary['inside_scope']:,}"
                f" scoped points ({summary['points']:,} total)"
            )

    write_report(site_reports)

    if SAVE_MAINFILE:
        bpy.ops.wm.save_mainfile()
        log("Saved blend file with transferred sim layers")

    print("Transfer complete.")
    for site, object_name, summary in results:
        print(
            f"{site} {object_name}: "
            f"{summary['matched']:,}/{summary['inside_scope']:,} scoped matches "
            f"({summary['points']:,} total points)"
        )
    return 0


def main() -> int:
    argv = sys.argv
    if "--helper" in argv:
        index = argv.index("--helper")
        return _helper_main(argv[index : index + 4])
    return blender_main()


if __name__ == "__main__":
    raise SystemExit(main())
