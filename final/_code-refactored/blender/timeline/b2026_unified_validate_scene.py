from __future__ import annotations

from pathlib import Path
import os
import re
import sys

import bpy


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import b2026_unified_scene_contract as unified_contract


def iter_layer_collection_tree(layer_collection):
    yield layer_collection
    for child in layer_collection.children:
        yield from iter_layer_collection_tree(child)


def get_layer_collection_by_name(view_layer: bpy.types.ViewLayer, collection_name: str):
    for layer_collection in iter_layer_collection_tree(view_layer.layer_collection):
        if layer_collection.collection.name == collection_name:
            return layer_collection
    return None


def resolve_scene() -> bpy.types.Scene:
    scene_name = os.environ.get("B2026_SCENE_NAME", "").strip()
    if scene_name:
        scene = bpy.data.scenes.get(scene_name)
        if scene is None:
            raise ValueError(f"Scene '{scene_name}' not found in {bpy.data.filepath}")
        return scene
    if bpy.context.scene is None:
        raise ValueError("No active Blender scene is available")
    return bpy.context.scene


def resolve_site(scene: bpy.types.Scene) -> str:
    site = os.environ.get("B2026_SITE_KEY", "").strip()
    if site:
        return site
    inferred = unified_contract.infer_site_from_scene_name(scene.name)
    if inferred is None:
        raise ValueError(f"Could not infer site from scene '{scene.name}'")
    return inferred


def add_issue(issues: list[str], message: str) -> None:
    issues.append(message)


def infer_single_state_year(site: str) -> int | None:
    year_raw = os.environ.get("B2026_SINGLE_STATE_YEAR", "").strip()
    if year_raw:
        return int(year_raw)

    patterns = (
        re.compile(rf"_{re.escape(site)}_yr(?P<year>\d+)_"),
        re.compile(r"__yr(?P<year>\d+)_"),
    )
    years: set[int] = set()

    for datablock in (bpy.data.collections, bpy.data.objects, bpy.data.node_groups):
        for item in datablock:
            name = item.name
            for pattern in patterns:
                match = pattern.search(name)
                if match:
                    years.add(int(match.group("year")))

    if not years:
        return None
    if len(years) == 1:
        return next(iter(years))
    raise ValueError(f"Could not infer a unique single-state year for site '{site}': {sorted(years)}")


def count_instancer_signature_hits(specs: tuple[dict[str, object], ...]) -> int:
    hits = 0
    for spec in specs:
        if bpy.data.objects.get(str(spec["point_object"])) is not None:
            hits += 1
        if bpy.data.collections.get(str(spec["positions_collection"])) is not None:
            hits += 1
        if bpy.data.collections.get(str(spec["models_collection"])) is not None:
            hits += 1
        if bpy.data.node_groups.get(str(spec["node_group"])) is not None:
            hits += 1
    return hits


def score_timeline_contract(site: str) -> int:
    contract = unified_contract.get_timeline_collection_contract(site)
    collection_names = {
        *contract["all_top"],
        *contract["child_defaults"].keys(),
    }
    object_names = set(unified_contract.get_expected_timeline_world_objects(site))
    score = sum(1 for name in collection_names if bpy.data.collections.get(name) is not None)
    score += sum(1 for name in object_names if bpy.data.objects.get(name) is not None)
    score += count_instancer_signature_hits(
        unified_contract.get_expected_instancer_specs(site, build_mode="timeline")
    )
    return score


def score_single_state_contract(site: str, year: int | None) -> int:
    if year is None:
        return 0

    contract = unified_contract.get_single_state_collection_contract(site, year)
    collection_names = {
        *contract["all_top"],
        *contract["positive_tree_log"],
        *contract["priority_tree_log"],
        *contract["trending_tree_log"],
        contract["positive_envelope_collection"],
        contract["trending_envelope_collection"],
    }
    object_names = set(unified_contract.get_expected_single_state_world_objects(site, year))
    score = sum(1 for name in collection_names if bpy.data.collections.get(name) is not None)
    score += sum(1 for name in object_names if bpy.data.objects.get(name) is not None)
    score += count_instancer_signature_hits(
        unified_contract.get_expected_instancer_specs(
            site,
            build_mode="single_state",
            year=year,
        )
    )
    return score


def resolve_validation_mode(scene: bpy.types.Scene, site: str) -> tuple[str, int | None]:
    mode_raw = os.environ.get("B2026_TIMELINE_BUILD_MODE", "").strip().lower()
    year = infer_single_state_year(site)
    if mode_raw:
        build_mode = unified_contract.get_build_mode()
        if build_mode == "single_state" and year is None:
            raise ValueError(
                "B2026_TIMELINE_BUILD_MODE=single_state but no single-state year could be inferred. "
                "Set B2026_SINGLE_STATE_YEAR or validate a built single-state scene."
            )
        return build_mode, year

    timeline_score = score_timeline_contract(site)
    single_state_score = score_single_state_contract(site, year)
    if single_state_score > timeline_score and single_state_score > 0:
        return "single_state", year
    if timeline_score > 0:
        return "timeline", year
    return unified_contract.get_build_mode(), year


def validate_view_layers(scene: bpy.types.Scene, issues: list[str]) -> None:
    actual_names = {view_layer.name for view_layer in scene.view_layers}
    missing = [name for name in unified_contract.STANDARD_VIEW_LAYERS if name not in actual_names]
    if missing:
        add_issue(issues, f"Missing required view layers: {', '.join(missing)}")

    aliases = [
        name
        for name in unified_contract.LEGACY_TIMELINE_ALIAS_VIEW_LAYERS
        if name in actual_names
    ]
    if aliases:
        add_issue(issues, f"Legacy alias view layers should not be required: {', '.join(aliases)}")


def validate_collection_excludes(
    scene: bpy.types.Scene,
    expected_by_layer: dict[str, dict[str, bool]],
    issues: list[str],
) -> None:
    for view_layer_name, expected in expected_by_layer.items():
        view_layer = scene.view_layers.get(view_layer_name)
        if view_layer is None:
            continue
        for collection_name, expected_excluded in expected.items():
            layer_collection = get_layer_collection_by_name(view_layer, collection_name)
            if layer_collection is None:
                add_issue(
                    issues,
                    f"[{view_layer_name}] Missing collection in layer tree: {collection_name}",
                )
                continue
            if bool(layer_collection.exclude) != bool(expected_excluded):
                add_issue(
                    issues,
                    f"[{view_layer_name}] Collection '{collection_name}' exclude="
                    f"{layer_collection.exclude} expected={expected_excluded}",
                )


def log_collection_excludes(
    scene: bpy.types.Scene,
    expected_by_layer: dict[str, dict[str, bool]],
) -> None:
    for view_layer_name, expected in expected_by_layer.items():
        view_layer = scene.view_layers.get(view_layer_name)
        if view_layer is None:
            print(f"VIEW_LAYER_LOG layer={view_layer_name} missing=1")
            continue

        actual_visible: list[str] = []
        expected_visible: list[str] = []
        unexpected_visible: list[str] = []
        missing_visible: list[str] = []

        for collection_name, expected_excluded in expected.items():
            layer_collection = get_layer_collection_by_name(view_layer, collection_name)
            if layer_collection is None:
                continue
            if not layer_collection.exclude:
                actual_visible.append(collection_name)
            if not expected_excluded:
                expected_visible.append(collection_name)

        actual_visible_set = set(actual_visible)
        for collection_name in actual_visible:
            if expected.get(collection_name, True):
                unexpected_visible.append(collection_name)
        for collection_name in expected_visible:
            if collection_name not in actual_visible_set:
                missing_visible.append(collection_name)

        print(
            "VIEW_LAYER_LOG "
            f"layer={view_layer_name} "
            f"actual_visible={','.join(actual_visible) or '-'} "
            f"expected_visible={','.join(expected_visible) or '-'} "
            f"unexpected_visible={','.join(unexpected_visible) or '-'} "
            f"missing_visible={','.join(missing_visible) or '-'}"
        )


def validate_objects_exist(object_names: tuple[str, ...], issues: list[str]) -> None:
    missing = [name for name in object_names if bpy.data.objects.get(name) is None]
    if missing:
        add_issue(issues, f"Missing expected objects: {', '.join(missing)}")


def validate_instancer_specs(specs: tuple[dict[str, object], ...], issues: list[str]) -> None:
    validated = 0
    for spec in specs:
        point_object_name = str(spec["point_object"])
        point_object = bpy.data.objects.get(point_object_name)
        positions_collection = bpy.data.collections.get(str(spec["positions_collection"]))
        models_collection = bpy.data.collections.get(str(spec["models_collection"]))
        node_group = bpy.data.node_groups.get(str(spec["node_group"]))
        modifier_name = str(spec["modifier_name"])

        if all(item is None for item in (point_object, positions_collection, models_collection, node_group)):
            continue

        validated += 1

        if positions_collection is None:
            add_issue(issues, f"Instancer positions collection missing: {spec['positions_collection']}")
        if models_collection is None:
            add_issue(issues, f"Instancer models collection missing: {spec['models_collection']}")
            continue
        if len(models_collection.objects) == 0:
            add_issue(issues, f"Instancer models collection is empty: {models_collection.name}")

        if point_object is None:
            add_issue(issues, f"Instancer point object missing: {point_object_name}")
            continue
        if point_object.type != "MESH" or getattr(point_object, "data", None) is None:
            add_issue(issues, f"Instancer point object is not a mesh: {point_object_name}")
            continue
        if len(point_object.data.vertices) == 0:
            add_issue(issues, f"Instancer point object has no locations: {point_object_name}")
        if positions_collection is not None and positions_collection.objects.get(point_object.name) is None:
            add_issue(
                issues,
                f"Instancer point object '{point_object.name}' is not linked to {positions_collection.name}",
            )

        modifier = point_object.modifiers.get(modifier_name)
        if modifier is None:
            add_issue(
                issues,
                f"Instancer point object '{point_object.name}' is missing modifier '{modifier_name}'",
            )
            continue
        if modifier.type != "NODES":
            add_issue(
                issues,
                f"Instancer modifier '{modifier.name}' on '{point_object.name}' is not Geometry Nodes",
            )
            continue

        if node_group is None:
            add_issue(issues, f"Instancer node group missing: {spec['node_group']}")
            continue
        if modifier.node_group != node_group:
            add_issue(
                issues,
                f"Instancer modifier '{modifier.name}' on '{point_object.name}' points to "
                f"{getattr(modifier.node_group, 'name', None)} expected {node_group.name}",
            )
            continue

        collection_targets = [
            node.inputs["Collection"].default_value
            for node in node_group.nodes
            if node.type == "COLLECTION_INFO"
            and hasattr(node, "inputs")
            and "Collection" in node.inputs
            and node.inputs["Collection"].default_value is not None
        ]
        if not collection_targets:
            add_issue(
                issues,
                f"Instancer node group '{node_group.name}' has no Collection Info target",
            )
            continue
        if models_collection not in collection_targets:
            add_issue(
                issues,
                f"Instancer node group '{node_group.name}' does not target models collection "
                f"'{models_collection.name}'",
            )

    if validated == 0:
        add_issue(issues, "No instancer point clouds were found to validate")


def validate_mode_specific_contract(
    scene: bpy.types.Scene,
    site: str,
    build_mode: str,
    year: int | None,
    issues: list[str],
) -> None:
    if build_mode == "single_state":
        if year is None:
            add_issue(issues, f"Could not infer single-state year for scene '{scene.name}'")
            return
        collection_contract = unified_contract.get_single_state_collection_contract(site, year)
        expected_by_layer = unified_contract.get_single_state_view_layer_expectations(site, year)
        required_collections = {
            *collection_contract["all_top"],
            *collection_contract["positive_tree_log"],
            *collection_contract["priority_tree_log"],
            *collection_contract["trending_tree_log"],
            collection_contract["positive_envelope_collection"],
            collection_contract["trending_envelope_collection"],
        }
        missing = [name for name in sorted(required_collections) if bpy.data.collections.get(name) is None]
        if missing:
            add_issue(issues, f"Missing expected collections: {', '.join(missing)}")
        log_collection_excludes(scene, expected_by_layer)
        validate_collection_excludes(scene, expected_by_layer, issues)
        validate_objects_exist(unified_contract.get_expected_single_state_world_objects(site, year), issues)
        validate_instancer_specs(
            unified_contract.get_expected_instancer_specs(
                site,
                build_mode=build_mode,
                year=year,
            ),
            issues,
        )
        return

    collection_contract = unified_contract.get_timeline_collection_contract(site)
    expected_by_layer = unified_contract.get_timeline_view_layer_expectations(site)
    required_collections = {
        *collection_contract["all_top"],
        *collection_contract["child_defaults"].keys(),
    }
    missing = [name for name in sorted(required_collections) if bpy.data.collections.get(name) is None]
    if missing:
        add_issue(issues, f"Missing expected collections: {', '.join(missing)}")
    log_collection_excludes(scene, expected_by_layer)
    validate_collection_excludes(scene, expected_by_layer, issues)
    validate_objects_exist(unified_contract.get_expected_timeline_world_objects(site), issues)
    validate_instancer_specs(
        unified_contract.get_expected_instancer_specs(
            site,
            build_mode=build_mode,
        ),
        issues,
    )


def main() -> None:
    scene = resolve_scene()
    site = resolve_site(scene)
    build_mode, year = resolve_validation_mode(scene, site)
    issues: list[str] = []

    validate_view_layers(scene, issues)
    validate_mode_specific_contract(scene, site, build_mode, year, issues)

    if issues:
        for issue in issues:
            print(f"VALIDATION_FAIL {issue}")
        raise ValueError(f"Unified scene validation failed with {len(issues)} issue(s)")

    print(
        f"VALIDATION_PASS scene={scene.name} site={site} "
        f"mode={build_mode} layers={len(unified_contract.STANDARD_VIEW_LAYERS)}"
    )


if __name__ == "__main__":
    main()
