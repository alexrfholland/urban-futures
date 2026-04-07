from __future__ import annotations

import sys
from pathlib import Path


BLENDER_EXPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT = BLENDER_EXPORT_DIR.parents[3]
CODE_ROOT = REPO_ROOT / "_code-refactored"
FINAL_DIR = REPO_ROOT / "final"
TREE_PROCESSING_DIR = CODE_ROOT / "refactor_code" / "tree_processing"

for import_root in (BLENDER_EXPORT_DIR, TREE_PROCESSING_DIR, FINAL_DIR, CODE_ROOT):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)
