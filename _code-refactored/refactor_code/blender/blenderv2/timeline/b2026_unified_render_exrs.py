from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from b2026_unified_runtime import run_local_script


def main() -> None:
    run_local_script(
        "b2026_timeline_render_lightweight_isolated_exrs_generic.py",
        env_updates={"B2026_SETUP_ONLY": "0"},
    )


if __name__ == "__main__":
    main()
