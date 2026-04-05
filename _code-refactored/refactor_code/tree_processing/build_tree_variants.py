from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[3]
TREE_PROCESSING_DIR = REPO_ROOT / "final" / "tree_processing"
if str(TREE_PROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(TREE_PROCESSING_DIR))

import aa_tree_helper_functions  # noqa: E402
import combine_resource_treeMeshGenerator  # noqa: E402
import combined_redoSnags  # noqa: E402
import combined_tree_manager  # noqa: E402
import combined_voxelise_dfs  # noqa: E402


TREE_DATA_ROOT = REPO_ROOT / "data" / "revised" / "trees"
BASE_TREE_LIBRARY_ROOT = Path(
    os.environ.get(
        "TREE_TEMPLATE_BASE_ROOT",
        os.environ.get(
            "BASE_TREE_TEMPLATES_ROOT",
            str(REPO_ROOT / "_data-refactored" / "model-inputs" / "tree_libraries" / "base" / "trees"),
        ),
    )
).expanduser()
VARIANT_ROOT = Path(
    os.environ.get(
        "TREE_TEMPLATE_VARIANTS_ROOT",
        str(REPO_ROOT / "_data-refactored" / "model-inputs" / "tree_variants"),
    )
).expanduser()


RESOURCE_COLORS = {
    "other": "#DADADA",
    "dead branch": "#C67D3E",
    "peeling bark": "#D6B86A",
    "perch branch": "#F0E442",
    "epiphyte": "#62B36F",
    "fallen log": "#6CA8A6",
    "hollow": "#6B6B6B",
}

FALLEN_MODE_LABELS = {
    "canonical": "Keep canonical fallen templates",
    "nonpre-direct": "Use non-precolonial fallen templates directly",
    "nonpre-geometry-pre-attrs": "Use non-precolonial fallen geometry with precolonial attributes",
}

DECAYED_MODE_LABEL = "Use the smaller eucalyptus fallen templates as the decayed phase"

SNAG_MODE_LABELS = {
    "elm-models-new": "Use the newer elm-derived snag models",
    "elm-snags-old": "Use the older snag models with weaker resource mapping",
}

SNAGS_USE_ALIASES = {
    "elm-models-new": "elm-models-new",
    "elm-snags-old": "elm-snags-old",
    "regenerated": "elm-models-new",
    "updated-original-geometry": "elm-snags-old",
}


def _compatible_clean_mesh_absolute(surface: pv.PolyData, min_cells: int = 10) -> pv.PolyData:
    """PyVista 0.42-compatible fallback for the tree mesh generator."""
    conn = surface.connectivity()
    region_ids = conn.cell_data["RegionId"]
    unique_region_ids, counts = np.unique(region_ids, return_counts=True)
    keep_ids = unique_region_ids[counts >= min_cells]
    if len(keep_ids) == 0:
        return surface.connectivity(largest=True)
    mask = np.isin(region_ids, keep_ids)
    return conn.extract_cells(mask)


combine_resource_treeMeshGenerator.clean_meshABSOLUTE = _compatible_clean_mesh_absolute


def normalize_snags_use(snags_use: str) -> str:
    try:
        return SNAGS_USE_ALIASES[snags_use]
    except KeyError as exc:
        raise ValueError(f"Unsupported snags_use: {snags_use}") from exc


@dataclass(frozen=True)
class VariantConfig:
    variant_name: str
    fallens_use: str
    snags_use: str
    voxel_size: float
    save_template_pickle: bool
    build_voxel_tables: bool
    build_meshes: bool
    render_samples: bool


def eucalyptus_dict_to_dataframe(eucalyptus_templates: dict) -> pd.DataFrame:
    rows = []
    for (precolonial, size, control, tree_id), template in eucalyptus_templates.items():
        rows.append(
            {
                "precolonial": precolonial,
                "size": size,
                "control": control,
                "tree_id": tree_id,
                "template": template.copy(),
            }
        )
    return pd.DataFrame(rows)


def dataframe_rows_to_lookup(df: pd.DataFrame) -> dict[tuple[bool, str, str, int], pd.DataFrame]:
    lookup = {}
    for _, row in df.iterrows():
        lookup[(row["precolonial"], row["size"], row["control"], int(row["tree_id"]))] = row["template"].copy()
    return lookup


def transfer_attributes_to_donor_geometry(
    donor_geometry_template: pd.DataFrame,
    source_attribute_template: pd.DataFrame,
) -> pd.DataFrame:
    donor_geometry_template = donor_geometry_template.copy().reset_index(drop=True)
    source_attribute_template = source_attribute_template.copy().reset_index(drop=True)

    donor_points = donor_geometry_template[["x", "y", "z"]].values
    source_points = source_attribute_template[["x", "y", "z"]].values
    tree = cKDTree(source_points)
    _, matched_indices = tree.query(donor_points)

    out = donor_geometry_template[["x", "y", "z"]].copy()
    attribute_columns = sorted(
        (set(donor_geometry_template.columns) | set(source_attribute_template.columns)) - {"x", "y", "z"}
    )

    for col in attribute_columns:
        if col in source_attribute_template.columns:
            out[col] = source_attribute_template.iloc[matched_indices][col].values
        elif col in donor_geometry_template.columns:
            out[col] = donor_geometry_template[col].values

    out = aa_tree_helper_functions.verify_resources_columns(out)
    if "resource" not in out.columns:
        out = aa_tree_helper_functions.create_resource_column(out)
    return out


def relabel_as_fallen_log(template: pd.DataFrame) -> pd.DataFrame:
    template = template.copy()
    if "resource_fallen log" not in template.columns:
        template["resource_fallen log"] = 0
    template["resource_fallen log"] = 1
    template["stat_fallen log"] = 1
    template["resource"] = "fallen log"
    return template


def build_fallen_variant_rows(
    canonical_combined_templates: pd.DataFrame,
    eucalyptus_df: pd.DataFrame,
    fallens_use: str,
) -> tuple[pd.DataFrame, dict]:
    canonical_fallen = canonical_combined_templates[canonical_combined_templates["size"] == "fallen"].copy()
    if fallens_use == "canonical":
        canonical_fallen["template"] = canonical_fallen["template"].apply(relabel_as_fallen_log)
        return canonical_fallen, {"fallens_use": fallens_use, "mapping": {}, "added_non_precolonial_fallens": 0}

    euc_fallen = eucalyptus_df[eucalyptus_df["size"] == "fallen"].copy()
    non_pre_fallen = euc_fallen[euc_fallen["precolonial"] == False].copy()
    pre_fallen = canonical_fallen[canonical_fallen["precolonial"] == True].copy()
    tree_id_mapping = combined_voxelise_dfs.create_treeid_mapping(eucalyptus_df, target_sizes=["fallen"])

    replaced_pre_rows = []
    replacement_records = []

    euc_lookup = dataframe_rows_to_lookup(euc_fallen)

    for _, row in pre_fallen.iterrows():
        donor_tree_id = tree_id_mapping.get(int(row["tree_id"]))
        if donor_tree_id is None:
            continue

        donor_key = (False, "fallen", "improved-tree", int(donor_tree_id))
        donor_template = euc_lookup[donor_key].copy()

        if fallens_use == "nonpre-direct":
            replacement_template = donor_template
        elif fallens_use == "nonpre-geometry-pre-attrs":
            replacement_template = transfer_attributes_to_donor_geometry(
                donor_geometry_template=donor_template,
                source_attribute_template=row["template"],
            )
        else:
            raise ValueError(f"Unsupported fallens_use: {fallens_use}")

        replacement_template = relabel_as_fallen_log(replacement_template)
        new_row = row.copy()
        new_row["template"] = replacement_template
        replaced_pre_rows.append(new_row)
        replacement_records.append(
            {
                "precolonial_tree_id": int(row["tree_id"]),
                "donor_non_precolonial_tree_id": int(donor_tree_id),
                "target_precolonial": True,
                "fallens_use": fallens_use,
                "template_points": int(len(replacement_template)),
            }
        )

    if not non_pre_fallen.empty:
        non_pre_fallen = non_pre_fallen.copy()
        non_pre_fallen["template"] = non_pre_fallen["template"].apply(relabel_as_fallen_log)

    variant_rows = pd.concat([non_pre_fallen, pd.DataFrame(replaced_pre_rows)], ignore_index=True)
    variant_rows = variant_rows.sort_values(["precolonial", "tree_id"]).reset_index(drop=True)
    metadata = {
        "fallens_use": fallens_use,
        "fallens_use_label": FALLEN_MODE_LABELS[fallens_use],
        "fallen_mode": fallens_use,
        "fallen_mode_label": FALLEN_MODE_LABELS[fallens_use],
        "mapping": replacement_records,
        "added_non_precolonial_fallens": int(len(non_pre_fallen)),
    }
    return variant_rows, metadata


def build_snag_variant_rows(snags: dict[int, pd.DataFrame], snags_use: str) -> pd.DataFrame:
    snags_use = normalize_snags_use(snags_use)
    rows = []
    for tree_id, template in snags.items():
        rows.append(
            {
                "precolonial": False,
                "size": "snag",
                "control": "improved-tree",
                "tree_id": int(tree_id),
                "template": template.copy(),
            }
        )
    rows_df = pd.DataFrame(rows).sort_values(["tree_id"]).reset_index(drop=True)
    rows_df["snags_use"] = snags_use
    return rows_df


def build_decayed_variant_rows(
    canonical_combined_templates: pd.DataFrame,
    eucalyptus_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    canonical_pre_fallen = canonical_combined_templates[
        (canonical_combined_templates["size"] == "fallen") & (canonical_combined_templates["precolonial"] == True)
    ].copy()
    canonical_pre_fallen["size"] = "decayed"
    canonical_pre_fallen["template"] = canonical_pre_fallen["template"].apply(relabel_as_fallen_log)

    tree_id_mapping = combined_voxelise_dfs.create_treeid_mapping(eucalyptus_df, target_sizes=["fallen"])
    duplicated_false_rows = []
    duplication_records = []

    for _, row in canonical_pre_fallen.iterrows():
        donor_tree_id = tree_id_mapping.get(int(row["tree_id"]))
        if donor_tree_id is None:
            continue

        duplicated_row = row.copy()
        duplicated_row["precolonial"] = False
        duplicated_row["tree_id"] = int(donor_tree_id)
        duplicated_false_rows.append(duplicated_row)
        duplication_records.append(
            {
                "precolonial_tree_id": int(row["tree_id"]),
                "duplicated_non_precolonial_tree_id": int(donor_tree_id),
                "template_points": int(len(row["template"])),
            }
        )

    decayed_rows = pd.concat([canonical_pre_fallen, pd.DataFrame(duplicated_false_rows)], ignore_index=True)
    decayed_rows = decayed_rows.sort_values(["precolonial", "tree_id"]).reset_index(drop=True)

    metadata = {
        "decayed_use_label": DECAYED_MODE_LABEL,
        "source": "canonical-small-fallen-templates",
        "source_rows": int(len(canonical_pre_fallen)),
        "duplication_mapping": duplication_records,
        "row_count": int(len(decayed_rows)),
        "precolonial_true_rows": int(decayed_rows["precolonial"].eq(True).sum()),
        "precolonial_false_rows": int(decayed_rows["precolonial"].eq(False).sum()),
    }
    return decayed_rows, metadata


def build_variant_templates(config: VariantConfig) -> tuple[pd.DataFrame, dict]:
    canonical_combined_templates = pd.read_pickle(BASE_TREE_LIBRARY_ROOT / "template-library.base.pkl")
    normalized_snags_use = normalize_snags_use(config.snags_use)

    elm_templates, euc_templates, graph_dict, resource_df = combined_tree_manager.load_files()
    updated_snags, regenerated_snags = combined_redoSnags.process_snags(
        euc_templates, elm_templates, graph_dict, resource_df
    )
    eucalyptus_df = eucalyptus_dict_to_dataframe(euc_templates)

    variant_templates = canonical_combined_templates.copy()
    metadata = {
        "variant_name": config.variant_name,
        "fallens_use": config.fallens_use,
        "fallens_use_label": FALLEN_MODE_LABELS[config.fallens_use],
        "fallen_mode": config.fallens_use,
        "fallen_mode_label": FALLEN_MODE_LABELS[config.fallens_use],
        "snags_use": normalized_snags_use,
        "snags_use_label": SNAG_MODE_LABELS[normalized_snags_use],
        "snag_mode": normalized_snags_use,
        "snag_mode_label": SNAG_MODE_LABELS[normalized_snags_use],
        "snags_use_input": config.snags_use,
    }

    if config.fallens_use != "canonical":
        fallen_rows, fallen_metadata = build_fallen_variant_rows(
            canonical_combined_templates=canonical_combined_templates,
            eucalyptus_df=eucalyptus_df,
            fallens_use=config.fallens_use,
        )
        variant_templates = variant_templates[variant_templates["size"] != "fallen"].copy()
        variant_templates = pd.concat([variant_templates, fallen_rows], ignore_index=True)
        metadata["fallen_variant"] = fallen_metadata
    else:
        metadata["fallen_variant"] = {
            "fallens_use": config.fallens_use,
            "fallens_use_label": FALLEN_MODE_LABELS[config.fallens_use],
            "fallen_mode": config.fallens_use,
            "mapping": [],
            "added_non_precolonial_fallens": 0,
        }

    selected_snags = regenerated_snags if normalized_snags_use == "elm-models-new" else updated_snags
    snag_rows = build_snag_variant_rows(selected_snags, normalized_snags_use)
    snag_mask = (
        (variant_templates["precolonial"] == False)
        & (variant_templates["size"] == "snag")
        & (variant_templates["control"] == "improved-tree")
    )
    variant_templates = variant_templates[~snag_mask].copy()
    variant_templates = pd.concat([variant_templates, snag_rows.drop(columns=["snags_use"], errors="ignore")], ignore_index=True)
    variant_templates = variant_templates.sort_values(["precolonial", "size", "control", "tree_id"]).reset_index(drop=True)
    aa_tree_helper_functions.check_for_duplicates(variant_templates)

    metadata["snag_variant"] = {
        "snags_use": normalized_snags_use,
        "snags_use_label": SNAG_MODE_LABELS[normalized_snags_use],
        "snag_mode": normalized_snags_use,
        "snag_mode_label": SNAG_MODE_LABELS[normalized_snags_use],
        "snags_use_input": config.snags_use,
        "available_regenerated_tree_ids": sorted(int(tree_id) for tree_id in regenerated_snags.keys()),
        "available_updated_tree_ids": sorted(int(tree_id) for tree_id in updated_snags.keys()),
    }

    decayed_rows, decayed_metadata = build_decayed_variant_rows(
        canonical_combined_templates=canonical_combined_templates,
        eucalyptus_df=eucalyptus_df,
    )
    variant_templates = variant_templates[variant_templates["size"] != "decayed"].copy()
    variant_templates = pd.concat([variant_templates, decayed_rows], ignore_index=True)
    variant_templates = variant_templates.sort_values(["precolonial", "size", "control", "tree_id"]).reset_index(drop=True)
    aa_tree_helper_functions.check_for_duplicates(variant_templates)
    metadata["decayed_variant"] = decayed_metadata
    return variant_templates, metadata


def save_variant_tables(
    variant_templates: pd.DataFrame,
    config: VariantConfig,
    metadata: dict,
) -> tuple[Path | None, Path | None, Path]:
    variant_dir = VARIANT_ROOT / config.variant_name
    trees_dir = variant_dir / "trees"
    trees_dir.mkdir(parents=True, exist_ok=True)

    combined_path = trees_dir / "template-library.overrides-applied.pkl"
    if config.save_template_pickle:
        variant_templates.to_pickle(combined_path)
    else:
        combined_path = None

    voxel_path: Path | None = None
    if config.build_voxel_tables:
        voxelised_templates, adjustment_summary, all_resource_stats = combined_voxelise_dfs.process_trees(
            variant_templates, voxel_size=config.voxel_size, resetCount=True
        )
        voxel_path = trees_dir / f"combined_voxelSize_{config.voxel_size:g}_templateDF.pkl"
        voxelised_templates.to_pickle(voxel_path)
        compatibility_voxel_path = trees_dir / f"{config.voxel_size:g}_combined_voxel_templateDF.pkl"
        voxelised_templates.to_pickle(compatibility_voxel_path)
        adjustment_summary.to_csv(trees_dir / f"{config.voxel_size:g}_combined_voxel_adjustment_summary.csv", index=False)
        all_resource_stats.to_csv(trees_dir / f"{config.voxel_size:g}_combined_voxel_all_resource_stats.csv", index=False)

    fallen_rows = variant_templates[variant_templates["size"] == "fallen"].copy()
    if not fallen_rows.empty:
        fallen_summary = fallen_rows[["precolonial", "size", "control", "tree_id"]].copy()
        fallen_summary["template_points"] = fallen_rows["template"].apply(len).values
        fallen_summary.to_csv(trees_dir / "fallen_rows_summary.csv", index=False)

    snag_rows = variant_templates[(variant_templates["precolonial"] == False) & (variant_templates["size"] == "snag")].copy()
    if not snag_rows.empty:
        snag_summary = snag_rows[["precolonial", "size", "control", "tree_id"]].copy()
        snag_summary["template_points"] = snag_rows["template"].apply(len).values
        snag_summary.to_csv(trees_dir / "snag_rows_summary.csv", index=False)

    decayed_rows = variant_templates[variant_templates["size"] == "decayed"].copy()
    if not decayed_rows.empty:
        decayed_summary = decayed_rows[["precolonial", "size", "control", "tree_id"]].copy()
        decayed_summary["template_points"] = decayed_rows["template"].apply(len).values
        decayed_summary.to_csv(trees_dir / "decayed_rows_summary.csv", index=False)

    affected_rows = variant_templates[variant_templates["size"].isin(["fallen", "snag", "decayed"])].copy()
    template_edits_path = trees_dir / "template-library.selected-overrides.pkl"
    affected_rows.to_pickle(template_edits_path)
    if not affected_rows.empty:
        affected_summary = affected_rows[["precolonial", "size", "control", "tree_id"]].copy()
        affected_summary["template_points"] = affected_rows["template"].apply(len).values
        affected_summary.to_csv(trees_dir / "affected_rows_summary.csv", index=False)
        affected_summary.to_csv(trees_dir / "template-edits_summary.csv", index=False)

    with open(trees_dir / "variant_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return combined_path, voxel_path, template_edits_path


def build_variant_meshes(variant_templates: pd.DataFrame, variant_name: str) -> list[Path]:
    mesh_output_dir = VARIANT_ROOT / variant_name / "final" / "treeMeshes"
    mesh_output_dir.mkdir(parents=True, exist_ok=True)

    target_rows = variant_templates[variant_templates["size"].isin(["fallen", "snag", "decayed"])].copy()
    created_paths: list[Path] = []
    for _, row in target_rows.iterrows():
        combine_resource_treeMeshGenerator.process_template_row(row, mesh_output_dir)
        created_paths.append(
            mesh_output_dir
            / f"precolonial.{row['precolonial']}_size.{row['size']}_control.{row['control']}_id.{row['tree_id']}.vtk"
        )
    return created_paths


def render_mesh(mesh_path: Path, image_path: Path) -> None:
    if not mesh_path.exists():
        return

    mesh = pv.read(mesh_path)
    if mesh.n_points == 0:
        return

    image_path.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(off_screen=True, window_size=(1400, 1400))
    plotter.set_background("white")

    if "resource" in mesh.point_data:
        unique_resources = [str(value) for value in np.unique(mesh.point_data["resource"])]
        cmap = [RESOURCE_COLORS.get(value, "#777777") for value in unique_resources]
        plotter.add_mesh(
            mesh,
            scalars="resource",
            categories=True,
            cmap=cmap,
            render_points_as_spheres=False,
            show_scalar_bar=False,
        )
    else:
        plotter.add_mesh(mesh, color="#8c8c8c", show_scalar_bar=False)

    plotter.enable_eye_dome_lighting()
    plotter.view_isometric()
    plotter.camera.zoom(1.35)
    plotter.show(screenshot=str(image_path), auto_close=False)
    plotter.close()


def render_sample_meshes(mesh_paths: list[Path], variant_name: str) -> list[Path]:
    sample_dir = VARIANT_ROOT / variant_name / "renders"
    sample_dir.mkdir(parents=True, exist_ok=True)

    preferred_names = [
        "precolonial.True_size.fallen_control.improved-tree_id.15.vtk",
        "precolonial.True_size.fallen_control.improved-tree_id.11.vtk",
        "precolonial.False_size.snag_control.improved-tree_id.11.vtk",
        "precolonial.False_size.snag_control.improved-tree_id.14.vtk",
    ]
    mesh_lookup = {path.name: path for path in mesh_paths if path.exists()}
    selected_paths = [mesh_lookup[name] for name in preferred_names if name in mesh_lookup]
    if not selected_paths:
        selected_paths = mesh_paths[:4]

    rendered = []
    for mesh_path in selected_paths:
        image_path = sample_dir / f"{mesh_path.stem}.png"
        render_mesh(mesh_path, image_path)
        rendered.append(image_path)
    return rendered


def parse_args() -> VariantConfig:
    parser = argparse.ArgumentParser(description="Build non-destructive tree-template variants for fallen/snag testing.")
    parser.add_argument("--variant-name", required=True)
    parser.add_argument(
        "--fallens-use",
        "--fallen-mode",
        dest="fallens_use",
        choices=["canonical", "nonpre-direct", "nonpre-geometry-pre-attrs"],
        default="nonpre-direct",
    )
    parser.add_argument(
        "--snags-use",
        "--snag-mode",
        dest="snags_use",
        choices=["elm-models-new", "elm-snags-old", "regenerated", "updated-original-geometry"],
        default="elm-models-new",
    )
    parser.add_argument("--voxel-size", type=float, default=1.0)
    parser.add_argument("--save-template-pickle", action="store_true")
    parser.add_argument("--build-voxel-tables", action="store_true")
    parser.add_argument("--build-meshes", action="store_true")
    parser.add_argument("--render-samples", action="store_true")
    args = parser.parse_args()
    return VariantConfig(
        variant_name=args.variant_name,
        fallens_use=args.fallens_use,
        snags_use=args.snags_use,
        voxel_size=args.voxel_size,
        save_template_pickle=args.save_template_pickle,
        build_voxel_tables=args.build_voxel_tables,
        build_meshes=args.build_meshes,
        render_samples=args.render_samples,
    )


def main() -> None:
    config = parse_args()
    variant_templates, metadata = build_variant_templates(config)
    combined_path, voxel_path, template_edits_path = save_variant_tables(variant_templates, config, metadata)

    result = {
        "variant_name": config.variant_name,
        "combined_template_path": str(combined_path) if combined_path else None,
        "voxelised_template_path": str(voxel_path) if voxel_path else None,
        "template_edits_path": str(template_edits_path),
        "mesh_paths": [],
        "sample_renders": [],
    }

    mesh_paths: list[Path] = []
    if config.build_meshes:
        mesh_paths = build_variant_meshes(variant_templates, config.variant_name)
        result["mesh_paths"] = [str(path) for path in mesh_paths]

    if config.render_samples and mesh_paths:
        rendered = render_sample_meshes(mesh_paths, config.variant_name)
        result["sample_renders"] = [str(path) for path in rendered]

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
