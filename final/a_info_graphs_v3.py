from __future__ import annotations

"""
V3-safe wrapper around `a_info_graphs.py`.

This keeps the legacy generator unchanged while letting us render against the
v3 indicator CSVs and write to a v3-only output root.
"""

from pathlib import Path
import importlib.util
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_SCRIPT = Path(__file__).resolve().with_name("a_info_graphs.py")
V3_DATA_DIR = REPO_ROOT / "data" / "revised" / "final-v3" / "output" / "csv"
V3_PLOT_DIR = REPO_ROOT / "_statistics-refactored-v3" / "plots" / "capability-indicator-streamgraphs"


def _load_legacy_module():
    spec = importlib.util.spec_from_file_location("a_info_graphs_legacy", LEGACY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load legacy script: {LEGACY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = _load_legacy_module()
    module.DATA_DIR = V3_DATA_DIR
    module.PLOT_DIR = V3_PLOT_DIR
    V3_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    module.generate_stream_graph(
        sites=["trimmed-parade", "city", "uni"],
        voxel_size=1,
        color_by="capability",
        save=True,
        show=False,
        combined_aggregated=True,
    )


if __name__ == "__main__":
    main()
