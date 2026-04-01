from __future__ import annotations

import os
from pathlib import Path

from render_edge_lab_current_mist import main as render_current_mist_main


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "data"
    / "blender"
    / "2026"
    / "edge_detection_lab"
    / "outputs"
    / "edge_lab_final_template_mist_remapped"
)


def main() -> None:
    os.environ.setdefault("EDGE_LAB_MIST_VARIANT_PRESET", "kirschremap")
    os.environ.setdefault("EDGE_LAB_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))
    render_current_mist_main()


if __name__ == "__main__":
    main()
