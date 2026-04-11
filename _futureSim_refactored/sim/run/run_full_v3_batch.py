"""
Full V3 batch runner.

This script is intentionally non-interactive so validation can be repeated
exactly under explicit environment-variable roots.

Expected env vars (see _documentation-refactored/scenario_engine_v3_validation.md):
- REFACTOR_RUN_OUTPUT_ROOT
- TREE_TEMPLATE_ROOT
- EXPORT_ALL_POINTDATA_VARIABLES

Typical usage:
  uv run python _futureSim_refactored/sim/run/run_full_v3_batch.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

CODE_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "_futureSim_refactored")
REPO_ROOT = CODE_ROOT.parent

if str(CODE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT.parent))

from _futureSim_refactored.sim.baseline import baseline_v3
from _futureSim_refactored.sim.generate_interim_state_data import a_scenario_runscenario
from _futureSim_refactored.sim.generate_interim_state_data.engine_v3 import log_run_stats
from _futureSim_refactored.paths import scenario_output_root
from _futureSim_refactored.sim.generate_vtk_and_nodeDFs import (
    a_info_gather_capabilities,
    a_scenario_generateVTKs,
    a_scenario_urban_elements_count,
)
from _futureSim_refactored.sim.setup import a_scenario_initialiseDS, params_v3

from _futureSim_refactored.paths import (
    engine_output_validation_dir,
)
from _futureSim_refactored.outputs.report import render_proposal_v4, render_debug_recruit
from _futureSim_refactored.blender.bexport import export_rewilded_envelopes
from _futureSim_refactored.sim.v4_indicator_extract import compute_indicators, format_site_table, INDICATOR_ORDER, write_v4_indicator_csv
from _futureSim_refactored.sim.run.run_log import append_run_log

import pandas as pd

SITES = ["trimmed-parade", "city", "uni"]
SCENARIOS = ["positive", "trending"]
YEARS = params_v3.generate_timesteps(interval=30)  # [0,10,30,60,90,120,150,180]
VOXEL_SIZE = 1


def _run_config() -> dict:
    keys = [
        "REFACTOR_RUN_OUTPUT_ROOT",
        "TREE_TEMPLATE_ROOT",
        "EXPORT_ALL_POINTDATA_VARIABLES",
        "REFACTOR_OUTPUT_MODE",
    ]
    return {key: os.environ.get(key) for key in keys}


def _reject_legacy_root_envs() -> None:
    legacy_keys = [
        "REFACTOR_SCENARIO_OUTPUT_ROOT",
        "REFACTOR_ENGINE_OUTPUT_ROOT",
        "REFACTOR_STATISTICS_ROOT",
    ]
    configured = {key: os.environ.get(key) for key in legacy_keys if os.environ.get(key)}
    if configured:
        details = ", ".join(f"{key}={value}" for key, value in configured.items())
        raise RuntimeError(
            "Legacy root env vars are no longer supported. "
            f"Unset them and use REFACTOR_RUN_OUTPUT_ROOT only. Found: {details}"
        )


def _parse_csv_arg(raw: str | None, default: list[str]) -> list[str]:
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_years(raw: str | None) -> list[int]:
    if not raw:
        return YEARS
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full v3 batch, a node-only CSV pass, or a saved-CSV VTK pass.")
    parser.add_argument("--sites", help="Comma-separated sites. Default: all canonical sites.")
    parser.add_argument("--scenarios", help="Comma-separated scenarios. Default: positive,trending.")
    parser.add_argument("--years", help="Comma-separated assessed years. Default: canonical v3 years.")
    parser.add_argument("--voxel-size", type=int, default=VOXEL_SIZE, help="Voxel size. Default: 1.")
    parser.add_argument(
        "--node-only",
        action="store_true",
        help="Run scenario CSV/dataframe outputs only and skip VTK/baseline/capability generation.",
    )
    parser.add_argument(
        "--vtk-only",
        action="store_true",
        help="Skip scenario CSV generation, load saved scenario CSVs, and run downstream VTK work only.",
    )
    parser.add_argument(
        "--compile-stats-only",
        action="store_true",
        help="Run the stats pass from final state VTKs and write/merge stats CSVs only; do no scenario or VTK generation.",
    )
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="Generate baselines only; do no scenario, VTK, or stats work.",
    )
    parser.add_argument(
        "--regenerate-baselines",
        action="store_true",
        help="Regenerate baseline outputs after the site/scenario pass. Default: off.",
    )
    parser.add_argument(
        "--multiple-agent",
        action="store_true",
        help=(
            "Per-slice batch mode for parallel agent work. Runs the requested site/scenario/year slice "
            "and skips the cross-state capability pass so another agent can compile it later."
        ),
    )
    parser.add_argument(
        "--save-raw-vtk",
        action="store_true",
        help="Save raw scenario state VTKs during VTK generation.",
    )
    parser.add_argument(
        "--description",
        default="",
        help="One-sentence description appended to the run log.",
    )
    return parser.parse_args()


def _write_v4_indicator_csv(polydata, site: str, scenario: str, year: int, is_baseline: bool = False) -> None:
    """Write per-state V4 indicator counts to a CSV alongside the V3 stats."""
    path = write_v4_indicator_csv(polydata, site, scenario, year, is_baseline=is_baseline)
    print(f"  V4 indicators → {path}")


def _prepare_subset_dataset(site: str, voxel_size: int, *, write_cache: bool = True):
    possibility_space_ds = a_scenario_initialiseDS.initialize_dataset(site, voxel_size, write_cache=write_cache)
    tree_df, pole_df, log_df = a_scenario_initialiseDS.load_node_dataframes(site, voxel_size)
    tree_df, possibility_space_ds = a_scenario_initialiseDS.PreprocessData(tree_df, possibility_space_ds, None)
    possibility_space_ds, _initial_poly = a_scenario_initialiseDS.further_xarray_processing(possibility_space_ds)

    if log_df is not None:
        log_df = a_scenario_initialiseDS.log_processing(log_df, possibility_space_ds)
    if pole_df is not None:
        pole_df = a_scenario_initialiseDS.pole_processing(pole_df, None, possibility_space_ds)
    return possibility_space_ds, tree_df, pole_df, log_df


def run_site_scenario(
    site: str,
    scenario: str,
    years: list[int],
    *,
    voxel_size: int = 1,
    node_only: bool = False,
    vtk_only: bool = False,
    multiple_agent: bool = False,
    save_raw_vtk: bool = False,
    indicator_counts: dict | None = None,
) -> None:
    print(f"\n===== V3: {site} / {scenario} =====\n")

    possibility_space_ds, treeDF, poleDF, logDF = _prepare_subset_dataset(
        site,
        voxel_size,
        write_cache=not (vtk_only or multiple_agent),
    )

    # Set up recruit telemetry path (delete old file so appends start fresh)
    site_dir = scenario_output_root("validation") / site
    site_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = site_dir / f"{site}_{scenario}_recruit_telemetry.csv"
    if not vtk_only and telemetry_path.exists():
        telemetry_path.unlink()

    current_tree_df = treeDF.copy()
    previous_year = 0

    for year in years:
        print(f"\n----- V3: {site} / {scenario} / yr{year} -----\n")

        if vtk_only:
            treeDF_scenario, logDF_scenario, poleDF_scenario = a_scenario_generateVTKs.load_scenario_dataframes(
                site, scenario, year, voxel_size
            )
            if treeDF_scenario is None:
                print(f"Skipping {site} / {scenario} / yr{year}: saved scenario CSVs not found")
                previous_year = year
                continue
        else:
            treeDF_scenario, logDF_scenario, poleDF_scenario = a_scenario_runscenario.run_scenario(
                site,
                scenario,
                year,
                voxel_size,
                current_tree_df,
                possibility_space_ds,
                logDF,
                poleDF,
                previous_year=previous_year,
                recruit_telemetry_path=telemetry_path,
            )

        if node_only:
            print(f"Node-only mode: skipping VTK generation for {site} / {scenario} / yr{year}")
        else:
            vtk_file, state_polydata = a_scenario_generateVTKs.generate_vtk(
                site,
                scenario,
                year,
                voxel_size,
                possibility_space_ds.copy(deep=True),
                treeDF_scenario,
                logDF_scenario,
                poleDF_scenario,
                enable_visualization=False,
                return_polydata=True,
                save_raw_vtk=save_raw_vtk,
            )

            state_polydata = a_scenario_urban_elements_count.process_scenario_polydata(
                state_polydata,
                site=site,
                voxel_size=voxel_size,
                scenario=scenario,
                year=year,
                save_path=None,
                enable_visualization=False,
            )

            a_info_gather_capabilities.process_polydata(
                state_polydata,
                site,
                scenario,
                year,
                voxel_size=voxel_size,
                save_vtk=True,
                save_stats=False,
                output_mode="validation",
            )

            if vtk_file:
                print(f"Saved scenario VTK (manager path): {vtk_file}")

            # Inline proposal render (proposal-and-interventions, from in-memory polydata)
            try:
                render_proposal_v4.render_target(
                    site, scenario, year, vtk_path=None, output_mode="validation",
                    mesh=state_polydata,
                )
                print(f"  Rendered proposal: {site}/{scenario}/yr{year}")
            except Exception as e:
                print(f"  Proposal render failed: {e}")

            # Inline debug recruit render (years 10, 60, 180 only)
            if year in (10, 60, 180):
                try:
                    render_debug_recruit.render_debug_target(
                        site, scenario, year, vtk_path=None,
                        output_mode="validation", mesh=state_polydata,
                    )
                    print(f"  Rendered debug recruit: {site}/{scenario}/yr{year}")
                except Exception as e:
                    print(f"  Debug recruit render failed: {e}")

            # Inline bioenvelope export (from in-memory polydata)
            try:
                export_rewilded_envelopes.export_target(
                    site, scenario, year, vtk_path=None,
                    output_mode="validation", voxel_size=voxel_size, mesh=state_polydata,
                )
                print(f"  Exported bioenvelope: {site}/{scenario}/yr{year}")
            except Exception as e:
                print(f"  Bioenvelope export failed: {e}")

            # Write per-state V4 indicator CSV
            _write_v4_indicator_csv(state_polydata, site, scenario, year)

            # Collect V4 indicator counts for the final year
            if indicator_counts is not None and year == max(years):
                indicator_counts[(site, scenario)] = compute_indicators(state_polydata)

        current_tree_df = treeDF_scenario.copy()
        previous_year = year

    # Log recruit and size stats after all years complete
    log_run_stats(site, scenario, years, voxel_size=voxel_size, output_mode="validation")


def run_baselines(sites: list[str], *, voxel_size: int = 1) -> None:
    print("\n===== V3: Baselines =====\n")
    for site in sites:
        print(f"\n----- V3: baseline / {site} -----\n")
        artifacts = baseline_v3.generate_baseline(site=site, voxel_size=voxel_size, visualize=False)
        baseline_polydata = a_scenario_urban_elements_count.process_baseline_polydata(
            artifacts.combined_polydata,
            site=site,
            voxel_size=voxel_size,
            save_path=None,
        )
        a_info_gather_capabilities.process_polydata(
            baseline_polydata,
            site,
            "baseline",
            -180,
            voxel_size=voxel_size,
            save_vtk=True,
            save_stats=False,
            output_mode="validation",
        )
        _write_v4_indicator_csv(baseline_polydata, site, "baseline", -180, is_baseline=True)


def run_capabilities(sites: list[str], scenarios: list[str], years: list[int], *, voxel_size: int = 1) -> None:
    print("\n===== V3: Stats Pass =====\n")
    for site in sites:
        indicator_df, action_df = a_info_gather_capabilities.process_site(
            site,
            scenarios=scenarios,
            years=years,
            voxel_size=voxel_size,
            save_vtk=False,
            include_baseline=True,
            output_mode="validation",
        )
        print(
            f"Stats pass complete: {site} "
            f"(indicator_rows={len(indicator_df)}, action_rows={len(action_df)})"
        )


def main() -> None:
    _reject_legacy_root_envs()
    args = parse_args()
    sites = _parse_csv_arg(args.sites, SITES)
    scenarios = _parse_csv_arg(args.scenarios, SCENARIOS)
    years = _parse_years(args.years)

    validation_dir = engine_output_validation_dir("validation")
    validation_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_meta_path = validation_dir / f"v3_full_run_metadata_{stamp}.json"
    run_meta_path.write_text(
        json.dumps(
            {
                "timestamp": stamp,
                "sites": sites,
                "scenarios": scenarios,
                "years": years,
                "voxel_size": args.voxel_size,
                "node_only": args.node_only,
                "vtk_only": args.vtk_only,
                "compile_stats_only": args.compile_stats_only,
                "baselines_only": args.baselines_only,
                "regenerate_baselines": args.regenerate_baselines,
                "multiple_agent": args.multiple_agent,
                "save_raw_vtk": args.save_raw_vtk,
                "env": _run_config(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote run metadata: {run_meta_path}")

    # Log to the run log only when an explicit root is set
    _explicit_root = os.environ.get("REFACTOR_RUN_OUTPUT_ROOT")
    _mode = "node-only" if args.node_only else "vtk-only" if args.vtk_only else "stats-only" if args.compile_stats_only else "baselines-only" if args.baselines_only else "full"
    _name = f"{'-'.join(sites)}_{'-'.join(scenarios)}_{_mode}"
    append_run_log(
        name=_name,
        output_root=_explicit_root or "default",
        description=args.description or f"{_mode} run for {', '.join(sites)} / {', '.join(scenarios)} yrs {years}",
    )

    if args.compile_stats_only:
        print("\n===== V3: Compile-stats-only mode =====\n")
        run_capabilities(sites, scenarios, years, voxel_size=args.voxel_size)
        return

    if args.baselines_only:
        print("\n===== V3: Baselines-only mode =====\n")
        run_baselines(sites, voxel_size=args.voxel_size)
        return

    # Collect V4 indicator counts per (site, scenario) during VTK generation
    indicator_counts: dict[tuple[str, str], dict] = {}

    for site in sites:
        for scenario in scenarios:
            run_site_scenario(
                site,
                scenario,
                years,
                voxel_size=args.voxel_size,
                node_only=args.node_only,
                vtk_only=args.vtk_only,
                multiple_agent=args.multiple_agent,
                save_raw_vtk=args.save_raw_vtk,
                indicator_counts=indicator_counts,
            )

    if args.node_only:
        print("\n===== V3: Node-only mode, skipping baselines and capability pass =====\n")
        return

    if args.regenerate_baselines:
        run_baselines(sites, voxel_size=args.voxel_size)
    else:
        print("\n===== V3: Baseline regeneration disabled =====\n")

    if args.multiple_agent:
        print("\n===== V3: Multiple-agent mode, stats remain for later explicit compilation =====\n")
        return

    # Write V4 indicator comparison if we collected counts during VTK generation
    if indicator_counts:
        import pyvista as pv
        from _futureSim_refactored.paths import refactor_run_output_root
        root = refactor_run_output_root("validation")
        vtk_dir = root.parent / "output" / "vtks" if root else None

        all_lines = ["# V4 Indicator Comparisons (voxel counts, yr 180)", ""]
        for site in sites:
            # Load baseline indicators from VTK
            baseline_vtk_path = vtk_dir / site / f"{site}_baseline_1_state_with_indicators.vtk" if vtk_dir else None
            if baseline_vtk_path and baseline_vtk_path.exists():
                baseline_counts = compute_indicators(pv.read(baseline_vtk_path), is_baseline=True)
            else:
                baseline_counts = {}
            pos_counts = indicator_counts.get((site, "positive"), {})
            trend_counts = indicator_counts.get((site, "trending"), {})
            if pos_counts or trend_counts:
                all_lines.extend(format_site_table(site, baseline_counts, pos_counts, trend_counts))

        if root:
            comparison_dir = root.parents[1] / "comparison"
            comparison_dir.mkdir(parents=True, exist_ok=True)
            output_path = comparison_dir / "v4_indicator_comparison.md"
            output_path.write_text("\n".join(all_lines) + "\n")
            print(f"\nWrote V4 indicator comparison: {output_path}")

    print("\n===== V3: Generation complete, skipping stats pass by default =====\n")
    print("Run --compile-stats-only to build per-state and merged stats from final state_with_indicators VTKs.\n")


if __name__ == "__main__":
    main()
