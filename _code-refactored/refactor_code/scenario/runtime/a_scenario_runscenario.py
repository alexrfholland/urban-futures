import sys
from pathlib import Path


RUNTIME_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = REPO_ROOT / "_code-refactored"
FINAL_DIR = REPO_ROOT / "final"
TREE_PROCESSING_DIR = CODE_ROOT / "refactor_code" / "tree_processing"

for import_root in (RUNTIME_DIR, TREE_PROCESSING_DIR, FINAL_DIR, CODE_ROOT):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

from refactor_code.scenario.engine_v3 import calculate_under_node_treatment_status, run_scenario, run_timestep


if __name__ == "__main__":
    import a_scenario_initialiseDS

    default_sites = ["trimmed-parade", "city", "uni"]
    default_scenarios = ["positive", "trending"]
    default_voxel_size = 1
    all_years = [0, 10, 30, 60, 90, 120, 150, 180]

    sites_input = input(f"Enter site(s) to process (comma-separated) or press Enter for default {default_sites}: ")
    sites = [site.strip() for site in (sites_input.split(",") if sites_input else default_sites)]

    scenarios_input = input(
        f"Enter scenario(s) to process (comma-separated) or press Enter for default {default_scenarios}: "
    )
    scenarios = [scenario.strip() for scenario in (scenarios_input.split(",") if scenarios_input else default_scenarios)]

    voxel_size_input = input(f"Enter voxel size (default {default_voxel_size}): ")
    voxel_size = int(voxel_size_input) if voxel_size_input else default_voxel_size

    years_input = input(f"Enter years to simulate (comma-separated) or press Enter for default {all_years}: ")
    years = [int(year) for year in years_input.split(",")] if years_input else all_years

    for site in sites:
        for scenario in scenarios:
            tree_df, pole_df, log_df, possibility_space_ds = a_scenario_initialiseDS.initialize_scenario_data(site, voxel_size)
            previous_year = 0
            for year in years:
                print(f"\n--- Running simulation for {site} / {scenario} / {year} ---\n")
                tree_df, _, _ = run_scenario(
                    site=site,
                    scenario=scenario,
                    year=year,
                    voxel_size=voxel_size,
                    treeDF=tree_df,
                    possibility_space_ds=possibility_space_ds,
                    logDF=log_df,
                    poleDF=pole_df,
                    previous_year=previous_year,
                )
                previous_year = year
