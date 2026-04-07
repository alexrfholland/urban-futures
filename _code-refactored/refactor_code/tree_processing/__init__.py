"""Tree-processing refactor helpers and non-destructive variant builders."""

from __future__ import annotations

import sys
from pathlib import Path


TREE_PROCESSING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TREE_PROCESSING_DIR.parents[3]
CODE_ROOT = REPO_ROOT / "_code-refactored"
FINAL_DIR = REPO_ROOT / "final"
BLENDER_EXPORT_DIR = CODE_ROOT / "refactor_code" / "blender_export"

for import_root in (TREE_PROCESSING_DIR, BLENDER_EXPORT_DIR, FINAL_DIR, CODE_ROOT):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)
