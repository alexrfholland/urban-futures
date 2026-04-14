"""
Modular VTK stat extractor.

Loads each state VTK once and runs all registered schemas against it,
saving per-state CSVs.  Schemas are defined in sibling modules and
registered via SCHEMA_REGISTRY.

In-pipeline usage (mesh already in memory):
    from _futureSim_refactored.outputs.stats.vtk_to_stat_counts import process_state
    process_state(mesh, site, scenario, year, output_root=root)

Standalone batch (reads VTKs from disk):
    uv run python -m _futureSim_refactored.outputs.stats.vtk_to_stat_counts \
        --root _data-refactored/model-outputs/generated-states/4.11

    # Subset of schemas:
    uv run python -m ... --root 4.11 --schemas v4_indicators,v4_decisions
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pyvista as pv

CODE_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "_futureSim_refactored")
REPO_ROOT = CODE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _futureSim_refactored.sim.setup import params_v3

SITES = ["trimmed-parade", "city", "uni"]
SCENARIOS = ["positive", "trending"]
DEFAULT_YEARS = params_v3.generate_timesteps(interval=30)  # [0, 1, 10, 30, 60, 90, 120, 150, 180]


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------

@dataclass
class Schema:
    """A named stat extraction schema.

    extract:
        Callable(mesh, site, scenario, year, **ctx) -> list[dict]
        Each dict becomes one CSV row.  The function receives:
          - mesh: pv.PolyData (the state VTK)
          - site, scenario, year: state identity
          - **ctx: extra context (currently ``root`` — the run output root,
            useful for schemas that also need the nodeDF)
    columns:
        Ordered column names for the output CSV.  If None the columns are
        inferred from the first row returned by extract().
    """
    name: str
    extract: Callable[..., list[dict]]
    columns: list[str] | None = None


# Global registry — schemas self-register on import.
SCHEMA_REGISTRY: dict[str, Schema] = {}


def register(schema: Schema) -> Schema:
    """Add a schema to the global registry."""
    SCHEMA_REGISTRY[schema.name] = schema
    return schema


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def state_vtk_path(root: Path, site: str, scenario: str, year: int) -> Path:
    return root / "output" / "vtks" / site / f"{site}_{scenario}_1_yr{year}_state_with_indicators.vtk"


def per_state_csv_path(root: Path, site: str, scenario: str, year: int, schema_name: str) -> Path:
    if scenario == "baseline":
        filename = f"{site}_baseline_1_{schema_name}.csv"
    else:
        filename = f"{site}_{scenario}_1_yr{year}_{schema_name}.csv"
    return root / "output" / "stats" / "per-state" / site / filename


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def process_state(
    mesh_or_path: pv.PolyData | str | Path,
    site: str,
    scenario: str,
    year: int,
    *,
    output_root: str | Path,
    schemas: list[str] | None = None,
    is_baseline: bool = False,
) -> list[Path]:
    """Run schemas against a single state, save CSVs, return written paths.

    Args:
        mesh_or_path: In-memory PolyData or path to a VTK file.
        site, scenario, year: State identity.
        output_root: Run output root (e.g. ``_data-refactored/.../4.11``).
        schemas: Which schemas to run (names).  None = all registered.
        is_baseline: Passed through to extract functions as ``ctx['is_baseline']``.
    """
    root = Path(output_root)

    if isinstance(mesh_or_path, (str, Path)):
        path = Path(mesh_or_path)
        if not path.exists():
            print(f"  skip {site}/{scenario}/yr{year}: VTK not found at {path}")
            return []
        mesh = pv.read(str(path))
    else:
        mesh = mesh_or_path

    schema_names = schemas or list(SCHEMA_REGISTRY.keys())
    written: list[Path] = []

    for name in schema_names:
        schema = SCHEMA_REGISTRY.get(name)
        if schema is None:
            print(f"  warning: unknown schema '{name}', skipping")
            continue

        rows = schema.extract(mesh, site, scenario, year, root=root, is_baseline=is_baseline)
        if not rows:
            continue

        csv_path = per_state_csv_path(root, site, scenario, year, name)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(rows, columns=schema.columns) if schema.columns else pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        written.append(csv_path)

    return written


def process_batch(
    root: str | Path,
    *,
    sites: list[str] | None = None,
    scenarios: list[str] | None = None,
    years: list[int] | None = None,
    schemas: list[str] | None = None,
    include_baselines: bool = True,
) -> list[Path]:
    """Batch-process all states under a run root.  One VTK load per state."""
    root = Path(root)
    sites = sites or SITES
    scenarios = scenarios or SCENARIOS
    years = years or DEFAULT_YEARS

    written: list[Path] = []

    # Baselines
    if include_baselines:
        for site in sites:
            vtk_path = root / "output" / "vtks" / site / f"{site}_baseline_1_state_with_indicators.vtk"
            if not vtk_path.exists():
                print(f"  skip baseline/{site}: VTK not found")
                continue
            print(f"  baseline/{site}")
            mesh = pv.read(str(vtk_path))
            paths = process_state(mesh, site, "baseline", -180, output_root=root, schemas=schemas, is_baseline=True)
            written.extend(paths)

    # Scenario states
    for site in sites:
        for scenario in scenarios:
            for year in years:
                vtk_path = state_vtk_path(root, site, scenario, year)
                if not vtk_path.exists():
                    print(f"  skip {site}/{scenario}/yr{year}: VTK not found")
                    continue
                print(f"  {site}/{scenario}/yr{year}")
                mesh = pv.read(str(vtk_path))
                paths = process_state(mesh, site, scenario, year, output_root=root, schemas=schemas)
                written.extend(paths)

    return written


# ---------------------------------------------------------------------------
# Schema imports — each module registers itself on import
# ---------------------------------------------------------------------------

def _import_schemas():
    """Import all built-in schema modules so they self-register."""
    from _futureSim_refactored.outputs.stats import schema_v4_indicators  # noqa: F401
    from _futureSim_refactored.outputs.stats import schema_v4_interventions  # noqa: F401
    from _futureSim_refactored.outputs.stats import schema_v4_decisions  # noqa: F401


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    # Import schemas into the *canonical* module's registry (not __main__'s).
    # When run via ``python -m``, this file is loaded as __main__ *and* as
    # the canonical module.  Schema modules import ``register`` from the
    # canonical path, so they populate that copy's SCHEMA_REGISTRY.  We
    # must use that same copy here.
    _import_schemas()
    import _futureSim_refactored.outputs.stats.vtk_to_stat_counts as _canonical
    registry = _canonical.SCHEMA_REGISTRY

    parser = argparse.ArgumentParser(description="Extract VTK stat counts using registered schemas.")
    parser.add_argument("--root", required=True, help="Run output root, e.g. _data-refactored/model-outputs/generated-states/4.11")
    parser.add_argument("--schemas", type=str, default=None, help="Comma-separated schema names (default: all)")
    parser.add_argument("--sites", type=str, default=None)
    parser.add_argument("--scenarios", type=str, default=None)
    parser.add_argument("--years", type=str, default=None)
    parser.add_argument("--no-baselines", action="store_true")
    args = parser.parse_args()

    schema_list = [s.strip() for s in args.schemas.split(",")] if args.schemas else None
    site_list = [s.strip() for s in args.sites.split(",")] if args.sites else None
    scenario_list = [s.strip() for s in args.scenarios.split(",")] if args.scenarios else None
    year_list = [int(y.strip()) for y in args.years.split(",")] if args.years else None

    print(f"Schemas: {schema_list or list(registry.keys())}")
    print(f"Root: {args.root}")

    written = _canonical.process_batch(
        args.root,
        sites=site_list,
        scenarios=scenario_list,
        years=year_list,
        schemas=schema_list,
        include_baselines=not args.no_baselines,
    )
    print(f"\nWrote {len(written)} CSVs")


if __name__ == "__main__":
    main()
