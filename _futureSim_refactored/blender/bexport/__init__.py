from __future__ import annotations

import sys
from pathlib import Path


BLENDER_EXPORT_DIR = Path(__file__).resolve().parent
CODE_ROOT = next(parent for parent in BLENDER_EXPORT_DIR.parents if parent.name == "_futureSim_refactored")
REPO_ROOT = CODE_ROOT.parent
FINAL_DIR = REPO_ROOT / "final"
TREE_PROCESSING_DIR = CODE_ROOT / "input_processing" / "tree_processing"

for import_root in (BLENDER_EXPORT_DIR, TREE_PROCESSING_DIR, FINAL_DIR, CODE_ROOT):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)
